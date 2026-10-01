"""Lean 4 compiler abstraction + default lake-env-lean implementation."""
from __future__ import annotations

import asyncio
import json
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

# goedel_lean project root — where lakefile.toml lives (depends on the real
# LeanArchitect package, see goedel_lean/lakefile.toml)
LEAN_PROJECT_ROOT = Path(__file__).parent.parent / "goedel_lean"

# Standard imports prepended to snippet-mode compilation.
# GoedelArch provides #validate_blueprint; Architect provides @[blueprint] / sorry_using.
MATHLIB_HEADER = "import Mathlib\nimport Architect\n\n"
GOEDEL_HEADER = "import Mathlib\nimport Architect\nimport GoedelArch\n\n"


@dataclass
class CompilerResult:
    success: bool
    goals: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_output: str = ""
    # True only when `success` came from an actual Lean compile (`lake env
    # lean` / LeanREPL). Some fallback paths (e.g. a structural-only check
    # when the target repo isn't on disk) report success without ever
    # invoking Lean - those must set this to False so callers can tell a
    # genuinely-validated result from a give-up/best-effort one.
    validated: bool = True

    @property
    def safeguard_rejected(self) -> bool:
        return "Safeguard rejected" in self.raw_output

    @property
    def has_sorry(self) -> bool:
        return any(
            "declaration uses" in w and ("sorry" in w or "admit" in w)
            for w in self.warnings
        )


_DECL_RE = re.compile(r"\b(?:noncomputable\s+def|def|lemma|theorem|abbrev)\s+\w+.*", re.DOTALL)
_SORRY_USING_RE = re.compile(r":=\s*by\s*sorry_using\s*\[[^\]]*\]", re.DOTALL)

# Both prover_system.md and blueprint_system.md tell the model these are
# forbidden — `axiom` lets it assert its own goal as true with zero proof,
# `native_decide` trusts a compiled binary instead of the kernel checker.
# Nothing previously enforced this; it was an honor-system instruction only.
_FORBIDDEN_CONSTRUCT_RE = re.compile(r"\baxiom\b|\bnative_decide\b")
# A prover submission containing sorry/admit is incomplete by definition;
# the smoke run showed each one previously burned a FULL Lean elaboration
# before the post-compile sorry check rejected it. Checked BEFORE Lean
# runs (check() only - check_blueprint legitimately contains sorry_using).
_INCOMPLETE_PROOF_RE = re.compile(r"\bsorry\b|\badmit\b|\bsorry_using\b")


class MalformedProofBody(ValueError):
    """A proof body that cannot be normalized into the canonical `by ...`
    form (e.g. a `:= := by ...` double-assignment leftover)."""


def normalize_proof_body(body: str) -> str:
    """Canonical internal proof-body form: starts with `by` (or a bare
    term), NEVER with `:=`.

    Exactly one representation exists at rest (proved caches, checkpoints,
    ProofResult payloads); every site that splices a proof into a
    declaration adds the `:= ` assignment itself via format_proof_assign().
    This makes the historical `:= := by ...` double-assembly bug impossible:
    a leading `:=` is stripped here (model-facing schemas ask for `:= by`),
    and a body that still starts with `:=` after one strip is a double
    leftover that must fail loudly instead of silently compiling garbage.
    """
    b = (body or "").strip()
    if b.startswith(":="):
        b = b[2:].lstrip()
    if b.startswith(":="):
        raise MalformedProofBody(
            f"proof body still starts with ':=' after normalization "
            f"(double-assignment leftover): {b[:60]!r}"
        )
    return b


def format_proof_assign(body: str) -> str:
    """Render a proof body with exactly one leading `:= ` assignment."""
    return ":= " + normalize_proof_body(body)


def _assemble_node_attempt(node_decl: str, aux_lemmas: str, proof_body: str) -> str:
    """Build a standalone compilable file from a blueprint node's declaration.

    Strips the `@[blueprint ...]` attribute and swaps `:= by sorry_using [...]`
    for `:= ` + the prover's tactic proof (normalized - the model-facing tool
    schema historically asked for a `:= by`-prefixed body, which used to
    re-assemble as `:= := by ...` and could never compile).
    """
    m = _DECL_RE.search(node_decl)
    decl_text = m.group(0) if m else node_decl
    decl_text = _SORRY_USING_RE.sub(lambda _: f":= {normalize_proof_body(proof_body)}", decl_text, count=1)
    parts = [MATHLIB_HEADER.rstrip("\n")]
    if aux_lemmas.strip():
        parts.append(aux_lemmas.strip())
    parts.append(decl_text)
    return "\n\n".join(parts) + "\n"


def _reject_forbidden_constructs(code: str) -> CompilerResult | None:
    """First-layer fast filter over code about to be compiled. This is NOT a
    security sandbox (Lean metaprogramming is Turing-complete); it only
    enforces the proof-integrity bans the prompts already state. See
    README's known limitations."""
    hit = _FORBIDDEN_CONSTRUCT_RE.search(code)
    if hit is None:
        return None
    return CompilerResult(
        success=False,
        errors=[f"Safeguard rejected: forbidden construct `{hit.group(0)}` is not allowed."],
        raw_output="Safeguard rejected",
    )


class AbstractLeanCompiler(ABC):
    """Interface every Lean compiler backend must implement."""

    @abstractmethod
    def check(self, lean_code: str, **kwargs) -> CompilerResult: ...

    def check_blueprint(self, lean_code: str, target_name: str) -> CompilerResult:
        """Validate a @[blueprint]-annotated file. Default: no-op (just compile)."""
        return self.check(lean_code)

    async def check_async(self, lean_code: str, **kwargs) -> CompilerResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.check, lean_code)


class LeanCompiler(AbstractLeanCompiler):
    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or LEAN_PROJECT_ROOT

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        lean_code: str,
        prepend_header: bool = False,
        aux_lemmas: str = "",
        node_decl: str = "",
        **_,
    ) -> CompilerResult:
        """Compile a Lean snippet and return structured feedback.

        If `prepend_header` is True, `import Mathlib\\nimport Architect` is
        prepended automatically.  For blueprint and prover code the imports
        are already present in the LLM-generated text, so False is the default.

        If `node_decl` is given, `lean_code` is treated as a bare proof body
        (tactics starting with `by`, no theorem header) that gets substituted
        into `node_decl`'s `:= by sorry_using [...]` tail — mirroring the
        VSBLeanCompiler contract where the prover only ever submits tactics.
        `aux_lemmas` (if any) is compiled ahead of the assembled declaration.
        """
        if node_decl:
            code = _assemble_node_attempt(node_decl, aux_lemmas, lean_code)
        else:
            code = (MATHLIB_HEADER + lean_code) if prepend_header else lean_code

        forbidden = _reject_forbidden_constructs(code)
        if forbidden is not None:
            return forbidden
        if _INCOMPLETE_PROOF_RE.search(lean_code) or _INCOMPLETE_PROOF_RE.search(aux_lemmas):
            return CompilerResult(
                success=False,
                errors=["Safeguard rejected: submission contains sorry/admit - an "
                        "incomplete proof. It was rejected WITHOUT compiling. Submit "
                        "only complete proofs; if you cannot finish, analyze the "
                        "goal and try a genuinely different approach."],
                raw_output="Safeguard rejected",
            )

        result = self._run_lean(code)
        if result.success and result.has_sorry:
            return CompilerResult(
                success=False,
                goals=result.goals,
                errors=result.errors + ["Proof contains `sorry` — not a complete proof."],
                warnings=result.warnings,
                raw_output=result.raw_output,
            )
        return result

    def check_blueprint(self, lean_code: str, target_name: str) -> CompilerResult:
        """Compile blueprint code and run #validate_blueprint <target>.

        Prepends import GoedelArch (which provides #validate_blueprint) if not
        already present, then appends the validation command.
        """
        if "import GoedelArch" not in lean_code:
            # Insert GoedelArch import after the existing Architect import
            code = lean_code.replace(
                "import Architect",
                "import Architect\nimport GoedelArch",
                1,
            )
            if "import GoedelArch" not in code:
                code = GOEDEL_HEADER + lean_code
        else:
            code = lean_code
        # Blueprint code previously bypassed the forbidden-construct filter
        # entirely (it calls _run_lean directly) - a blueprint containing
        # `axiom` or `native_decide` would sail through Phase 1 validation.
        forbidden = _reject_forbidden_constructs(code)
        if forbidden is not None:
            return forbidden
        code = code.rstrip() + f"\n\n#validate_blueprint {target_name}\n"
        return self._run_lean(code)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_lean(self, code: str) -> CompilerResult:
        with tempfile.NamedTemporaryFile(suffix=".lean", mode="w", delete=False) as f:
            f.write(code)
            tmp_path = f.name
        try:
            result = subprocess.run(
                ["lake", "env", "lean", "--json", tmp_path],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=180,
            )
            return self._parse_output(result.stdout + result.stderr, result.returncode)
        except subprocess.TimeoutExpired:
            return CompilerResult(success=False, errors=["Lean compilation timed out after 180s"])
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _parse_output(self, raw: str, returncode: int) -> CompilerResult:
        goals: list[str] = []
        errors: list[str] = []
        warnings: list[str] = []

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                severity = msg.get("severity", "")
                # Lean 4 --json uses "data" for the message text
                text = msg.get("data") or msg.get("message") or ""
                if severity == "error":
                    errors.append(text)
                elif severity == "warning":
                    warnings.append(text)
                elif severity == "information":
                    if "⊢" in text or "goals" in text.lower():
                        goals.append(text)
            except json.JSONDecodeError:
                # Non-JSON line (shouldn't happen with --json, but be safe)
                if re.search(r"\berror\b", line, re.IGNORECASE):
                    errors.append(line)

        # sorry_using sorries are expected; don't treat them as errors
        errors = [e for e in errors if "declaration uses `sorry`" not in e]

        return CompilerResult(
            success=returncode == 0 and not errors,
            goals=goals,
            errors=errors,
            warnings=warnings,
            raw_output=raw,
        )
