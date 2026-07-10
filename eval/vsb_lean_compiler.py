"""
VeriSoftBench adapter: wraps LeanREPL so it implements AbstractLeanCompiler.

The standard LeanCompiler runs `lake env lean` on arbitrary Lean snippets.
VeriSoftBench uses a Docker-managed LeanREPL that compiles proofs inside the
actual repository environment. This adapter bridges the two interfaces so
src/prover.py can work with either backend.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

VSB_ROOT = Path(__file__).parent.parent.parent / "VeriSoftBench"
VSB_LEAN_SRC = VSB_ROOT / "data" / "lean_repos"
sys.path.insert(0, str(VSB_ROOT))

from core.lean_interface import LeanREPL
from lean_compiler import AbstractLeanCompiler, CompilerResult
from blueprint import strip_blueprint_attr, lemma_to_theorem

BLUEPRINT_COMPILE_TIMEOUT = 120  # seconds

_SORRY_WARNING_RE = re.compile(r":(\d+):\d+:\s*warning:\s*declaration uses '(?:sorry|admit)'")


def _own_declaration_uses_sorry(combined_output: str, target_line: int) -> bool:
    """True only if a sorry/admit warning's line number is at or after
    `target_line` (where our own declaration starts) - warnings for earlier,
    unrelated content in the file don't count."""
    return any(
        int(m.group(1)) >= target_line
        for m in _SORRY_WARNING_RE.finditer(combined_output)
    )


# A Lean diagnostic header: `<file>:<line>:<col>: <severity>[:( ] ...`. The
# `[\(:]` tolerates both `error:` and the rarer `error(` code form.
_DIAG_HEADER_RE = re.compile(r"^.*?:\d+:\d+:\s*(error|warning|info)[\(:]")


def _extract_errors(combined_output: str) -> list[str]:
    """Return full, multi-line error diagnostics from raw Lean output.

    A Lean error is a *block*: a `... : error: <header>` line followed by
    indented/unindented continuation lines carrying the actual diagnostic
    (`has type X but is expected to have type Y`, the offending term, the
    unsolved goal, etc.) up to the next diagnostic header. The old code kept
    only the header line and discarded every continuation, so a proof that
    failed with `type mismatch` reached the model as the bare string
    "type mismatch" with no expected/actual types - blinding it and causing
    it to thrash for its whole tool budget on otherwise one-line goals
    (observed on TM.progress's trivially-true base cases). Each returned
    element is one error's complete block, warnings/info excluded.
    """
    blocks: list[str] = []
    current: list[str] | None = None
    current_is_error = False
    for line in combined_output.splitlines():
        m = _DIAG_HEADER_RE.match(line)
        if m:
            if current is not None and current_is_error:
                blocks.append("\n".join(current).rstrip())
            current = [line]
            current_is_error = m.group(1) == "error"
        elif current is not None:
            current.append(line)
    if current is not None and current_is_error:
        blocks.append("\n".join(current).rstrip())
    return blocks


def _node_signature(node_decl: str) -> str:
    """Bare theorem/lemma signature (no body) for a blueprint node's own
    declaration - shares BlueprintNode.signature()'s exact helpers
    (src/blueprint.py) so a fix to one can't silently leave the other
    unfixed: strip the `@[blueprint ...]` attribute, normalize `lemma` to
    `theorem` (not every repo has Mathlib/Batteries), and cut off at the
    `:=` that starts the `sorry_using [...]` body."""
    text = lemma_to_theorem(strip_blueprint_attr(node_decl))
    return text.split(":=", 1)[0].strip()


class VSBLeanCompiler(AbstractLeanCompiler):
    """
    Compiles a proof attempt by calling LeanREPL.verify_proof() with the
    full VeriSoftBench theorem context (repo, file path, local ctx, stmt).

    The prover calls check(proof_body, aux_lemmas=...) — note: proof_body is
    JUST the tactic block starting with `by`, not a full Lean file.

    Blueprint validation (check_blueprint) compiles the full @[blueprint]-annotated
    skeleton via `lake env lean` in the repo's environment. This requires
    LeanArchitect to be in the repo's lakefile (import Architect must resolve).
    """

    def __init__(
        self,
        lean_repl: LeanREPL,
        theorem_entry: dict,
        call_prefix: str = "ga",
    ) -> None:
        self.lean_repl = lean_repl
        self.theorem_entry = theorem_entry
        self.call_prefix = call_prefix
        self._count = 0
        # Each blueprint node gets its own VSBLeanCompiler instance (see
        # compiler_factory in orchestrator._prove_one), but the temp filename
        # was built only from entry["thm_name"] (the same for every node)
        # plus a per-instance counter that restarts at 1 for each node - two
        # nodes proving concurrently could collide on the exact same temp
        # file path and corrupt each other's compile. uuid4 makes every
        # instance's temp files unique regardless of concurrency.
        import uuid
        self._instance_id = uuid.uuid4().hex[:8]

    def check_blueprint(self, lean_code: str, target_name: str) -> CompilerResult:
        """
        Compile a @[blueprint]-annotated skeleton in the repo's Lean environment.

        Prepends `import Architect` plus all imports from the theorem's local_ctx
        so the blueprint's type signatures can elaborate correctly.
        Writes a temp .lean file, runs `lake env lean`, then deletes the file.
        """
        entry = self.theorem_entry
        lean_root = entry.get("lean_root", "")
        repo_root = VSB_LEAN_SRC / lean_root
        if not repo_root.exists():
            # Fall back to structural check if repo not found. This is a
            # regex/structural sanity check, NOT a real Lean compile -
            # validated=False so callers can tell this apart from a genuine
            # `lake env lean` pass.
            errors = _validate_blueprint_structure(lean_code, target_name)
            if errors:
                return CompilerResult(success=False, errors=errors, validated=False)
            return CompilerResult(success=True, validated=False)

        # Build a self-contained file: deduplicated imports + blueprint body
        local_ctx = entry.get("verif_local_ctxs") or entry.get("local_ctx", "")
        full_lean = _build_blueprint_file(
            lean_code, local_ctx,
            rel_path=entry.get("rel_path", ""),
            target_name=target_name,
        )

        tmp = repo_root / f"_blueprint_check_{abs(hash(lean_code)) % 1_000_000}.lean"
        try:
            tmp.write_text(full_lean, encoding="utf-8")
            result = subprocess.run(
                ["lake", "env", "lean", str(tmp)],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=BLUEPRINT_COMPILE_TIMEOUT,
            )
            combined = result.stdout + result.stderr
            errors = _extract_errors(combined)
            # sorry warnings are expected in blueprints — ignore them
            if errors:
                return CompilerResult(success=False, errors=errors)
            return CompilerResult(success=True)
        except subprocess.TimeoutExpired:
            return CompilerResult(
                success=False, errors=["Blueprint compilation timed out"]
            )
        except Exception as exc:
            return CompilerResult(success=False, errors=[str(exc)])
        finally:
            tmp.unlink(missing_ok=True)

    def check(self, proof_body: str, aux_lemmas: str = "", node_decl: str = "", **_) -> CompilerResult:
        """Verify proof_body against the configured theorem entry.

        Uses `lake env lean <file>` directly (not `lake build`) to avoid
        replaying thousands of cached modules on every call.

        `node_decl` (if given) is the specific blueprint node's own
        declaration - its signature is what actually gets checked. Without
        it, every node would be checked against `entry["thm_stmt"]` (the
        *root* theorem's statement) regardless of which sub-lemma is being
        proven, which silently checks the wrong goal for every non-root node.
        """
        if not proof_body.strip():
            return CompilerResult(success=False, errors=["proof_body is empty"])

        self._count += 1
        entry = self.theorem_entry
        lean_root = entry.get("lean_root", "")
        repo_root = VSB_LEAN_SRC / lean_root
        thm_stmt = _node_signature(node_decl) if node_decl.strip() else entry["thm_stmt"]

        if not repo_root.exists():
            # Fall back to LeanREPL if repo not on disk
            try:
                success, error_msg = self.lean_repl.verify_proof(
                    thm_name=entry["thm_name"],
                    repo_name=lean_root,
                    rel_path=entry["rel_path"],
                    local_context=entry.get("verif_local_ctxs") or entry.get("local_ctx", ""),
                    theorem_stmt=thm_stmt,
                    theorem_proof=proof_body,
                    proof_id=f"{self.call_prefix}_{self._instance_id}_{self._count}",
                    aux_lemmas=aux_lemmas or "",
                    suffix=entry.get("suffix", ""),
                )
            except Exception as exc:
                return CompilerResult(success=False, errors=[f"LeanREPL error: {exc}"])
            if success:
                return CompilerResult(success=True)
            return CompilerResult(success=False, errors=[error_msg or "Compilation failed"])

        # Fast path: write temp file and run `lake env lean` directly.
        # This skips `lake build`'s full dependency-graph replay (~3000 jobs)
        # and only elaborates the single file in the lake environment.
        import utils.utils as vsb_utils
        local_ctx = entry.get("verif_local_ctxs") or entry.get("local_ctx", "")
        suffix    = entry.get("suffix", "")

        content = vsb_utils.format_generated_lean(
            local_ctx, thm_stmt, proof_body, aux_lemmas or "", suffix
        )

        proof_id = f"{self.call_prefix}_{self._instance_id}_{self._count}"
        from pathlib import Path as _Path
        rel = _Path(entry["rel_path"])
        tmp_name = f"{rel.stem}_{vsb_utils.clean_thm_name(entry['thm_name'])}_v{proof_id}.lean"
        tmp = repo_root / rel.parent / tmp_name

        try:
            tmp.write_text(content, encoding="utf-8")
            result = subprocess.run(
                ["lake", "env", "lean", str(tmp)],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=120,
            )
            combined = result.stdout + result.stderr
            # Check for sorry/admit in OUR target declaration only. `local_ctx`
            # is the real file's preceding content and can itself contain
            # unrelated pre-existing `sorry`s (e.g. an earlier unsolved
            # example in a course-style repo) - a blanket substring search
            # over the whole compiler output would treat those as if our own
            # proof failed, even when our declaration compiled cleanly.
            target_pos = content.rfind(thm_stmt)
            target_line = content[:target_pos].count("\n") + 1 if target_pos != -1 else 0
            if _own_declaration_uses_sorry(combined, target_line):
                return CompilerResult(success=False, errors=["declaration uses 'sorry' — proof incomplete"])
            errors = _extract_errors(combined)
            if errors:
                return CompilerResult(success=False, errors=errors)
            return CompilerResult(success=True)
        except subprocess.TimeoutExpired:
            return CompilerResult(success=False, errors=["Compilation timed out"])
        except Exception as exc:
            return CompilerResult(success=False, errors=[str(exc)])
        finally:
            tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Blueprint file builder
# ---------------------------------------------------------------------------

def _build_blueprint_file(lean_code: str, local_ctx: str, rel_path: str = "", target_name: str = "") -> str:
    """
    Build a plain-Lean file that type-checks the blueprint's statements inside the
    repo environment, without requiring LeanArchitect.

    Strategy:
    - Import the already-built repo module (derived from rel_path) so every symbol
      the blueprint references — including definitions local to the repo file, which
      are NOT in mathlib — resolves. This replaces the old approach of inlining
      local_ctx, which was fragile around namespaces and universe variables.
    - Strip @[blueprint ...] attribute blocks (they need LeanArchitect).
    - Replace `sorry_using [...]` with plain `sorry` (the dependency graph is already
      tracked on the Python side, so the annotation is not needed for type-checking).
    - Rename the target theorem so it does not clash with the real one pulled in via
      import.
    - Drop the blueprint body's own import lines; we supply imports here.

    Falls back to inlining local_ctx when rel_path is unavailable (repo not on disk).
    """
    # 1. Strip @[blueprint ...] attribute blocks from the blueprint code
    stripped = strip_blueprint_attr(lean_code)

    # 2. Replace sorry_using [...] with plain sorry (handles `by sorry_using [...]`
    #    and any other position)
    stripped = re.sub(r"sorry_using\s*\[[^\]]*\]", "sorry", stripped)

    # 3. Rename the target theorem/lemma so it does not clash with the real theorem
    #    imported from the repo module
    if target_name:
        short_name = target_name.split(".")[-1]
        for pat in [re.escape(target_name), re.escape(short_name)]:
            stripped = re.sub(
                r"\b(theorem|lemma)\s+(" + pat + r")\b",
                lambda m: f"{m.group(1)} {m.group(2)}_blueprint_check",
                stripped,
            )

    # 4. Remove import lines from the blueprint body — we supply imports below
    body_lines = [l for l in stripped.splitlines() if not l.startswith("import ")]
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    body = "\n".join(body_lines)

    # 4b. `lemma` is a Mathlib/Batteries syntax extension, not core Lean. When we import
    #     only the repo module (which may be core-only, importing no mathlib), `lemma`
    #     will not parse. `theorem` is the core equivalent, so normalise declarations.
    body = lemma_to_theorem(body)

    # 5. Build the header. Import ONLY the built repo module: this reproduces the exact
    #    environment the target theorem lives in (the module's own imports plus its local
    #    declarations). Deliberately do NOT add `import Mathlib` — doing so injects symbols
    #    the real theorem does not have and can double-define classes the repo file
    #    declares itself (e.g. LawfulMonadStateOf also exists in Batteries). Fall back to
    #    inlining local_ctx when rel_path is unavailable (repo not on disk).
    if rel_path:
        module = rel_path[:-len(".lean")] if rel_path.endswith(".lean") else rel_path
        module = module.replace("/", ".")
        # Universe variables are file-local and are NOT inherited through import, so
        # re-declare the common ones the blueprint may use.
        return f"import {module}\n\nuniverse u v w\n\n{body}"

    preamble = local_ctx.replace("\\n", "\n").rstrip()
    return f"{preamble}\n\n{body}"


# ---------------------------------------------------------------------------
# Structural validator (fallback when repo not on disk)
# ---------------------------------------------------------------------------

_NODE_RE = re.compile(
    r"@\[blueprint[^\]]*\]\s*\n\s*(?:noncomputable\s+)?(?:def|lemma|theorem|abbrev)\s+(\w+)",
    re.DOTALL,
)
_SORRY_USING_RE = re.compile(r"sorry_using\s*\[([^\]]*)\]")
_THEOREM_RE = re.compile(r"theorem\s+(\w+)")


def _validate_blueprint_structure(lean_code: str, target_name: str) -> list[str]:
    errors: list[str] = []

    declared: list[str] = _NODE_RE.findall(lean_code)
    if not declared:
        return ["No @[blueprint]-annotated declarations found."]

    declared_set = set(declared)

    if target_name not in declared_set:
        if not _THEOREM_RE.search(lean_code):
            errors.append(f"Main theorem '{target_name}' not found in blueprint.")

    deps: dict[str, list[str]] = {n: [] for n in declared}
    blocks = list(_NODE_RE.finditer(lean_code))
    for i, m in enumerate(blocks):
        name = m.group(1)
        block_text = lean_code[m.start(): blocks[i + 1].start() if i + 1 < len(blocks) else len(lean_code)]
        su = _SORRY_USING_RE.search(block_text)
        if su:
            raw_deps = [d.strip() for d in su.group(1).split(",") if d.strip()]
            for dep in raw_deps:
                if dep not in declared_set:
                    errors.append(f"Node '{name}' depends on undeclared '{dep}'.")
                else:
                    deps[name].append(dep)

    visiting: set[str] = set()
    visited: set[str] = set()

    def has_cycle(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for dep in deps.get(node, []):
            if has_cycle(dep):
                return True
        visiting.discard(node)
        visited.add(node)
        return False

    for n in declared:
        if has_cycle(n):
            errors.append(f"Cycle detected involving node '{n}'.")
            break

    return errors
