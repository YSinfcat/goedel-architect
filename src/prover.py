"""Phase 2: Per-node tool-equipped prover.

Uses the OpenAI Responses API (stateful, previous_response_id chaining).
Three tools: lean_compile, repo_search, mathlib_search.

Compiler backend is injectable — pass a VSBLeanCompiler for VeriSoftBench or
LeanCompiler for standalone Lean projects.

Returns one of four structured signals per the paper:
    solved | statement_wrong | proof_too_hard | formally_negated
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from lean_compiler import AbstractLeanCompiler, CompilerResult
from llm_client import make_client
from mathlib_retrieval import MathlibRetrieval
from goedel_prompts import load, render
from tracer import NullTracer, TraceEvent


def _responses_reasoning_kwargs(model: str) -> dict:
    """Return reasoning kwarg for models that support it in the Responses API."""
    if model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4"):
        return {"reasoning": {"effort": "low"}}
    return {}


def _force_lean_compile_kwargs() -> dict:
    """`tools`/`tool_choice` kwargs that force the next call to be lean_compile.

    Not OpenAI's named-function tool_choice shape ({"type": "function", "name":
    ...}) - Fireworks' Responses API rejects that with a 400 (verified against
    the live endpoint), accepting only the bare "required"/"auto"/"none"/"any"
    literals, which force *some* tool call but can't pin down which one; a
    Fireworks-hosted model kept picking repo_search again instead of ever
    reaching lean_compile. Restricting the visible `tools` list to just
    lean_compile makes "required" unambiguous on every provider, and as a
    side effect also stops a model from hallucinating a nonexistent tool
    (observed: a Fireworks-hosted model repeatedly invented an "open_file"
    tool that was never declared).
    """
    return {"tools": [TOOLS[0]], "tool_choice": "required"}

try:
    from repo_retrieval import RepoRetrieval
    _HAS_REPO_RETRIEVAL = True
except ImportError:
    _HAS_REPO_RETRIEVAL = False

PROVER_SYSTEM_PROMPT = load("prover_system")
PROVER_USER_TEMPLATE = load("prover_user")

# From Appendix A: "each node retries up to 4 times." The paper's mechanism
# (discrete full-code resubmissions) differs from this loop (one continuous
# multi-turn conversation), so this matches the stated budget number, not
# the exact retry semantics — a true match would need a different loop shape.
# Token budget capped to 32,000 (below the paper's 65,536) to control cost.
#
# MAX_TOOL_CALLS raised 4 -> 8 (above the paper's own number) after observing
# twice in VSB smoke tests that the model converged on a materially better
# proof strategy right as the 4-call budget ran out, with the improved draft
# never reaching lean_compile at all.
MAX_TOKENS = 64_000
MAX_TOOL_CALLS = 8
NEGATION_PROBE_CALLS = 4

SYSTEM_SUFFIX = """
## Tool-First Workflow

You have three tools: lean_compile, repo_search, mathlib_search.

**Key insight**: the prompt already contains this file's own preceding
declarations in <local_ctx> (definitions in full, lemma/theorem PROOFS
elided). Read it first — some of what you need is likely already visible
there. Anything from *outside* this file (other modules, or same-file
lemma statements local_ctx happened to omit) is NOT pre-loaded — you must
fetch it yourself via repo_search (project-specific) or mathlib_search
(Mathlib).

Workflow:
1. Draft a proof using the visible repo definitions.
2. Call lean_compile with your proof_body (starting with `:= by` — the harness
   appends proof_body directly after the bare theorem signature, so the leading
   `:=` is required or the submission fails to parse). proof_body is TACTIC
   MODE ONLY: `#check`/`#eval`/`#print` and other `#`-commands are top-level
   commands, not tactics, and will hard-fail with "unexpected token" if placed
   there — do not use them to inspect a type mid-proof; instead reason from
   what's already visible in <local_ctx>, or call repo_search/mathlib_search.
3. Read errors, adjust, call lean_compile again.
4. When you need a lemma whose name you do NOT already know:
   - call repo_search for project-specific lemmas
   - call mathlib_search for general Mathlib lemmas
5. Once lean_compile returns SUCCESSFUL, output:
   <lean4_proof>:= by\n  ...\n</lean4_proof>

Prefer lean_compile over search — faster to try a tactic and read the error.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "lean_compile",
        "description": (
            "Compile and verify a proof attempt. "
            "Returns 'Compilation SUCCESSFUL' or detailed Lean error messages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "proof_body": {
                    "type": "string",
                    "description": (
                        "Proof term starting with ':= by' (the leading ':=' is required — "
                        "this gets appended directly after the bare theorem signature). "
                        "Do NOT include the theorem declaration."
                    ),
                },
                "aux_lemmas": {
                    "type": "string",
                    "description": "Optional helper lemma declarations to define before the target theorem.",
                },
            },
            "required": ["proof_body"],
        },
    },
    {
        "type": "function",
        "name": "repo_search",
        "description": (
            "Semantic search over the target repository's .lean files. "
            "Use BEFORE mathlib_search for project-specific lemmas, induction principles, or coercions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language or identifier fragment."},
                "k": {"type": "integer", "description": "Number of results (default 10).", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "mathlib_search",
        "description": "Semantic search over Mathlib for general library lemmas.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
]


# ---------------------------------------------------------------------------
# Result types (unchanged from paper)
# ---------------------------------------------------------------------------

class ProofSignal(str, Enum):
    SOLVED = "solved"
    STATEMENT_WRONG = "statement_wrong"
    PROOF_TOO_HARD = "proof_too_hard"
    FORMALLY_NEGATED = "formally_negated"
    # Not one of the paper's four signals: an infra/tooling failure (timeout,
    # unhandled exception) rather than a genuine "the model tried and
    # couldn't" verdict. Kept distinct so refinement (and human diagnosis)
    # doesn't treat a broken harness as evidence the sub-goal is hard.
    INFRA_ERROR = "infra_error"


@dataclass
class ProverResult:
    signal: ProofSignal
    proof_body: str = ""
    analysis: str = ""
    suggested_fix: str = ""
    lean_errors: list[str] = field(default_factory=list)

    def diagnosis_block(self, node_name: str) -> str:
        if self.signal == ProofSignal.FORMALLY_NEGATED:
            return (
                f"/- Diagnosis\n## Diagnosis\nFORMALLY_NEGATED\n\n"
                f"## Analysis\n{self.analysis}\n\n"
                f"## Counterexample Proof\n```lean\n{self.proof_body}\n```\n\n"
                f"## Suggested Fix\n{self.suggested_fix}\n-/"
            )
        return (
            f"/- Diagnosis\n## Diagnosis\n{self.signal.value.upper()}\n\n"
            f"## Analysis\n{self.analysis}\n\n"
            f"## Suggested Fix\n{self.suggested_fix}\n-/"
        )


# ---------------------------------------------------------------------------
# Prover
# ---------------------------------------------------------------------------

class GoedelProver:
    """
    Phase 2 per-node tool-equipped prover.

    Inject a compiler backend (VSBLeanCompiler for VeriSoftBench, LeanCompiler
    for standalone) and optionally a RepoRetrieval for repo_search.
    """

    def __init__(
        self,
        model_id: str = "gpt-4o",
        retrieval: MathlibRetrieval | None = None,
        tracer=None,
        api_timeout_s: float = 120.0,
        max_tool_calls: int | None = None,
    ):
        self.model_id = model_id
        # Bounds each individual Responses API call so a stuck request can't
        # hang a node indefinitely; the orchestrator's node_timeout_s bounds
        # the whole multi-turn tool loop on top of this.
        self.client = make_client(model_id, timeout=api_timeout_s)
        self.retrieval = retrieval or MathlibRetrieval()
        self.tracer = tracer or NullTracer()
        # Per-instance override of the module-level MAX_TOOL_CALLS budget -
        # used by orchestrator.py's cascade to give an escalated (expensive)
        # attempt a much tighter budget (e.g. 1: a single lean_compile call,
        # no fix-and-retry loop) than the cheap cascade attempt gets.
        self.max_tool_calls = max_tool_calls if max_tool_calls is not None else MAX_TOOL_CALLS

    def _emit_usage(self, node_name: str, response) -> None:
        """Log token usage from a Responses API response. Field names differ
        from chat.completions (input_tokens/output_tokens vs prompt/
        completion), so this is a separate normalizer from blueprint.py's
        _emit_usage rather than a shared one."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt = getattr(usage, "input_tokens", 0)
        completion = getattr(usage, "output_tokens", 0)
        total = getattr(usage, "total_tokens", None) or (prompt + completion)
        self.tracer.emit(TraceEvent(
            kind="llm_usage",
            thm_name=node_name,
            args={
                "phase": "phase2", "model": self.model_id,
                "prompt_tokens": prompt, "completion_tokens": completion,
                "total_tokens": total,
            },
        ))

    def prove_node(
        self,
        compiler: AbstractLeanCompiler,
        node_name: str,
        node_stmt: str,
        sys_prompt: str = "",
        user_prompt: str = "",
        nl_statement: str = "",
        nl_proof_sketch: str = "",
        repo_retrieval=None,
        parent_lemma_decls: str = "",
    ) -> ProverResult:
        """Attempt to prove a single node, timing it and emitting a final_verify trace event."""
        t0 = time.time()
        result = self._prove_node_inner(
            compiler, node_name, node_stmt, sys_prompt, user_prompt,
            nl_statement, nl_proof_sketch, repo_retrieval,
            parent_lemma_decls=parent_lemma_decls,
        )
        self.tracer.emit(TraceEvent(
            kind="final_verify",
            thm_name=node_name,
            ok=result.signal == ProofSignal.SOLVED,
            args={
                "wall_time_s": time.time() - t0,
                "proof": result.proof_body,
                "error": result.analysis,
            },
        ))
        return result

    def _prove_node_inner(
        self,
        compiler: AbstractLeanCompiler,
        node_name: str,
        node_stmt: str,
        sys_prompt: str = "",
        user_prompt: str = "",
        nl_statement: str = "",
        nl_proof_sketch: str = "",
        repo_retrieval=None,
        parent_lemma_decls: str = "",
    ) -> ProverResult:
        """Attempt to prove a single node using the Responses API tool loop."""
        # Stashed on self rather than threaded through every _process_response /
        # _probe_negation call - one GoedelProver instance proves exactly one
        # node (see the module-level prove_node() factory), so this is safe.
        self._parent_lemma_decls = parent_lemma_decls
        augmented_sys = (sys_prompt or PROVER_SYSTEM_PROMPT).strip() + "\n\n" + SYSTEM_SUFFIX.strip()

        if not user_prompt:
            user_prompt = render(
                PROVER_USER_TEMPLATE,
                canonical_stmt=node_stmt,
                nl_statement=nl_statement,
                nl_proof_sketch=nl_proof_sketch,
                parent_proofs="",
            )

        self.tracer.emit(TraceEvent(
            kind="theorem_start",
            thm_name=node_name,
            args={"thm_stmt": node_stmt},
        ))

        # Force first call to lean_compile
        response = self.client.responses.create(
            model=self.model_id,
            instructions=augmented_sys,
            input=user_prompt,
            max_output_tokens=MAX_TOKENS,
            **_force_lean_compile_kwargs(),
            **_responses_reasoning_kwargs(self.model_id),
        )
        self._emit_usage(node_name, response)

        tool_calls_used = 0
        best_proof_body = ""
        last_text = ""
        tool_results: list[dict] = []
        last_compile_ok = False
        all_lean_errors: list[str] = []
        last_errors: list[str] = []
        # Signatures of repo_search hits this node has seen so far, auto-spliced
        # into every lean_compile call's aux_lemmas (see _process_response) -
        # repo_search finding a real repo lemma doesn't put it in scope for the
        # compiler on its own (that only reflects what's textually in the
        # compiled unit), and a model citing it by name otherwise just gets
        # "unknown identifier" regardless of how correct its proof is.
        discovered_decls: dict[str, str] = {}

        while tool_calls_used < self.max_tool_calls:
            tool_results, text, proof, compile_ok, tools_called, compile_errors = self._process_response(
                response, compiler, node_name, repo_retrieval, tool_calls_used, node_decl=node_stmt,
                discovered_decls=discovered_decls,
            )
            all_lean_errors.extend(compile_errors)
            # Classification must react to the MOST RECENT compile attempt only:
            # an early draft's "type mismatch" (later abandoned for a completely
            # different approach) must not keep tainting the verdict just
            # because all_lean_errors accumulates across the whole tool loop.
            if compile_errors:
                last_errors = compile_errors
            if text:
                last_text = text
            if proof:
                best_proof_body = proof
            if compile_ok:
                last_compile_ok = True
                return ProverResult(signal=ProofSignal.SOLVED, proof_body=proof)

            had_search = any(t in ("repo_search", "mathlib_search") for t in tools_called)
            had_compile = "lean_compile" in tools_called
            tool_calls_used += len(tool_results)

            if not tool_results:
                break
            if tool_calls_used >= self.max_tool_calls:
                break

            next_tool_kwargs = (
                _force_lean_compile_kwargs() if (had_search and not had_compile)
                else {"tools": TOOLS, "tool_choice": "required"}
            )

            response = self.client.responses.create(
                model=self.model_id,
                previous_response_id=response.id,
                input=tool_results,
                max_output_tokens=MAX_TOKENS,
                **next_tool_kwargs,
                **_responses_reasoning_kwargs(self.model_id),
            )
            self._emit_usage(node_name, response)

        # Drain any pending tool calls
        if tool_results:
            response = self.client.responses.create(
                model=self.model_id,
                previous_response_id=response.id,
                input=tool_results,
                tools=TOOLS,
                tool_choice="none",
                max_output_tokens=MAX_TOKENS,
                **_responses_reasoning_kwargs(self.model_id),
            )
            self._emit_usage(node_name, response)
            _, drain_text, drain_proof, _, _, drain_errors = self._process_response(
                response, compiler, node_name, repo_retrieval, tool_calls_used,
                discovered_decls=discovered_decls,
            )
            all_lean_errors.extend(drain_errors)
            if drain_errors:
                last_errors = drain_errors
            if drain_text:
                last_text = drain_text
            if drain_proof:
                best_proof_body = drain_proof

        # Ask for final answer if no proof tag found
        if not best_proof_body:
            response = self.client.responses.create(
                model=self.model_id,
                previous_response_id=response.id,
                input="Output your best proof: <lean4_proof>:= by\n  ...\n</lean4_proof>",
                max_output_tokens=MAX_TOKENS,
                **_responses_reasoning_kwargs(self.model_id),
            )
            self._emit_usage(node_name, response)
            _, last_text, best_proof_body, _, _, final_errors = self._process_response(
                response, compiler, node_name, repo_retrieval, tool_calls_used,
                discovered_decls=discovered_decls,
            )
            all_lean_errors.extend(final_errors)
            if final_errors:
                last_errors = final_errors

        # Probe negation if we couldn't prove it
        negation = self._probe_negation(compiler, node_name, response.id, MAX_TOKENS, discovered_decls)
        if negation:
            return negation

        if best_proof_body:
            signal = _classify_failure(last_errors, last_text)
            return ProverResult(signal=signal, proof_body=best_proof_body,
                                analysis=last_text[:500], lean_errors=all_lean_errors)
        return ProverResult(signal=_classify_failure(last_errors, last_text),
                            analysis=last_text[:500], lean_errors=all_lean_errors)

    # ------------------------------------------------------------------
    # Response processing
    # ------------------------------------------------------------------

    def _process_response(
        self,
        response,
        compiler: AbstractLeanCompiler,
        node_name: str,
        repo_retrieval,
        tool_calls_so_far: int,
        node_decl: str = "",
        discovered_decls: dict[str, str] | None = None,
    ) -> tuple[list[dict], str, str, bool, list[str], list[str]]:
        tool_results: list[dict] = []
        last_text = ""
        best_proof = ""
        compiled_proof = ""   # proof body that actually compiled — never overwritten by message text
        any_compile_ok = False
        tools_called: list[str] = []
        compile_errors: list[str] = []
        turn = tool_calls_so_far + 1

        for item in response.output:
            if item.type == "function_call":
                fn = item.name
                args = json.loads(item.arguments)
                tools_called.append(fn)

                self.tracer.emit(TraceEvent(
                    kind="tool_call", thm_name=node_name, turn=turn,
                    call_id=item.call_id, tool_name=fn, args=args,
                ))

                if fn == "lean_compile":
                    proof_body = args.get("proof_body", "")
                    aux = args.get("aux_lemmas", "")
                    # Splice in already-proved sibling lemmas as real declarations
                    # (see BlueprintNode.signature) so the model can reference them
                    # by name instead of hitting "unknown identifier".
                    parent_decls = getattr(self, "_parent_lemma_decls", "")
                    full_aux = "\n\n".join(p for p in (parent_decls, aux) if p.strip())
                    cr = compiler.check(proof_body, aux_lemmas=full_aux, node_decl=node_decl)
                    if not cr.success and discovered_decls:
                        # Self-heal a missing-context gap: a repo_search hit found
                        # a real repo lemma's signature, but that alone doesn't
                        # put it in scope for the compiler (only the compiled
                        # unit's own text does) - only inject a signature-only
                        # stub (`:= sorry`, never the real proof - that would
                        # reintroduce the exact advantage the raw-file-read leak
                        # fix removed) for a name the compiler *just this attempt*
                        # proved missing, so a name that's already legitimately in
                        # scope is never redeclared (that caused a real
                        # 'X has already been declared' regression when this was
                        # tried proactively for every search hit up front).
                        missing = set(re.findall(r"unknown identifier '([^']+)'", "\n".join(cr.errors)))
                        healable = {
                            stub for name, stub in discovered_decls.items()
                            if name in missing or any(m.rsplit(".", 1)[-1] == name for m in missing)
                        }
                        if healable:
                            healed_aux = "\n\n".join(p for p in (full_aux, *healable) if p.strip())
                            retry = compiler.check(proof_body, aux_lemmas=healed_aux, node_decl=node_decl)
                            if retry.success:
                                cr = retry
                    if cr.success:
                        result = "Compilation SUCCESSFUL. Proof is correct."
                        any_compile_ok = True
                        compiled_proof = proof_body
                        best_proof = proof_body
                    else:
                        errs = "\n".join(cr.errors)
                        result = f"Compilation FAILED.\n{errs}\n\nFix errors and call lean_compile again."
                        compile_errors.extend(cr.errors)

                elif fn == "repo_search" and repo_retrieval is not None:
                    hits = repo_retrieval.search(args["query"], int(args.get("k", 10)))
                    result = "\n\n".join(h.format() for h in hits) or "No results in repo."
                    if discovered_decls is not None:
                        for h in hits:
                            discovered_decls.setdefault(h.name, f"{h.kind} {h.name} {h.signature} := sorry")

                elif fn == "mathlib_search":
                    hits = self.retrieval.search(args["query"], int(args.get("k", 10)))
                    result = "\n\n".join(h.format() for h in hits) or "No results found."

                else:
                    result = f"Tool unavailable: {fn}. Valid tools: lean_compile, repo_search, mathlib_search."

                self.tracer.emit(TraceEvent(
                    kind="tool_result", thm_name=node_name, turn=turn,
                    call_id=item.call_id, tool_name=fn,
                    result=result[:500], ok=(fn == "lean_compile" and any_compile_ok),
                ))

                tool_results.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": result,
                })

            elif item.type == "message":
                for block in item.content:
                    text = getattr(block, "text", None)
                    if text:
                        last_text = text
                        extracted = _extract_proof_body(text)
                        # Only use message-extracted proof if no compile succeeded yet;
                        # compiled_proof is authoritative and must not be overwritten.
                        if extracted and not any_compile_ok:
                            best_proof = extracted
                        self.tracer.emit(TraceEvent(
                            kind="model_text", thm_name=node_name, turn=turn,
                            result=text[:500],
                        ))

        # Prefer the proof that actually compiled over anything extracted from text
        return tool_results, last_text, compiled_proof or best_proof, any_compile_ok, tools_called, compile_errors

    # ------------------------------------------------------------------
    # Negation probe (Section 4.3 / Figure 1)
    # ------------------------------------------------------------------

    def _probe_negation(
        self,
        compiler: AbstractLeanCompiler,
        node_name: str,
        previous_response_id: str,
        max_tokens: int,
        discovered_decls: dict[str, str] | None = None,
    ) -> ProverResult | None:
        prompt = (
            f"You could not prove the statement. Try to show it is FALSE.\n"
            f"Prove `neg_{node_name}` showing `¬ (conclusion)` with the same hypotheses. "
            "Tactics: `omega`, `decide`, `norm_num`, `push_neg; linarith`, `simp`, `native_decide`.\n"
            "Call lean_compile. If it succeeds, the original statement is formally refuted."
        )
        response = self.client.responses.create(
            model=self.model_id,
            previous_response_id=previous_response_id,
            input=prompt,
            max_output_tokens=max_tokens,
            **_force_lean_compile_kwargs(),
            **_responses_reasoning_kwargs(self.model_id),
        )
        self._emit_usage(node_name, response)

        for _ in range(NEGATION_PROBE_CALLS):
            results: list[dict] = []
            for item in response.output:
                if item.type != "function_call":
                    continue
                args = json.loads(item.arguments)
                if item.name == "lean_compile":
                    parent_decls = getattr(self, "_parent_lemma_decls", "")
                    proof_body = args.get("proof_body", "")
                    cr = compiler.check(proof_body, aux_lemmas=parent_decls)
                    if not cr.success and discovered_decls:
                        missing = set(re.findall(r"unknown identifier '([^']+)'", "\n".join(cr.errors)))
                        healable = {
                            stub for name, stub in discovered_decls.items()
                            if name in missing or any(m.rsplit(".", 1)[-1] == name for m in missing)
                        }
                        if healable:
                            healed_aux = "\n\n".join(p for p in (parent_decls, *healable) if p.strip())
                            retry = compiler.check(proof_body, aux_lemmas=healed_aux)
                            if retry.success:
                                cr = retry
                    if cr.success:
                        return ProverResult(
                            signal=ProofSignal.FORMALLY_NEGATED,
                            proof_body=args.get("proof_body", ""),
                            analysis=f"{node_name} is formally refuted — a proof of ¬(statement) was found.",
                            suggested_fix="The statement is mathematically FALSE. Revise it.",
                        )
                    output = f"Compilation FAILED.\n" + "\n".join(cr.errors)
                else:
                    output = "Tool unavailable in negation probe."
                results.append({"type": "function_call_output", "call_id": item.call_id, "output": output})

            if not results:
                break
            response = self.client.responses.create(
                model=self.model_id,
                previous_response_id=response.id,
                input=results,
                tools=TOOLS,
                tool_choice="auto",
                max_output_tokens=max_tokens,
                **_responses_reasoning_kwargs(self.model_id),
            )
            self._emit_usage(node_name, response)

        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_proof_body(text: str) -> str:
    import re
    m = re.search(r"<lean4_proof>(.*?)</lean4_proof>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _classify_failure(errors: list[str], analysis: str) -> ProofSignal:
    """A failed proof attempt is PROOF_TOO_HARD by default.

    STATEMENT_WRONG is expensive to be wrong about: refinement's prompt
    reacts to it by rewriting or dropping the lemma's statement (see
    prompts/refinement_system.md), so a false STATEMENT_WRONG destroys a
    perfectly good sub-goal and churns the whole refinement loop. It
    therefore requires actual evidence the *statement* is false - not merely
    that some proof term failed to typecheck.

    A Lean "type mismatch" is NOT such evidence. It's the compiler's generic
    complaint that a supplied term had the wrong shape, and it fires
    constantly on ordinary wrong-tactic attempts against perfectly true
    goals. Observed concretely on TM.progress: `progress_tru`, `progress_zro`,
    `succ_value_of_nvalue`, and `iszero_step_of_nvalue` are all trivially true,
    yet every attempt tripped "type mismatch" (because `value` is a nested
    `Or` needing a double `Or.inl`, and the model kept trying a single
    injection) and all four were wrongly marked statement_wrong, sending
    refinement off to "fix" statements that were already correct. So a bare
    "type mismatch" - in the compiler errors OR echoed in the model's own
    commentary - is deliberately NOT treated as a statement-falsity signal.

    Machine-checked statement-falsity is the negation probe's job
    (FORMALLY_NEGATED), handled before this function is ever reached. Here we
    only escalate to STATEMENT_WRONG when the model itself explicitly concludes
    the statement is false / exhibits a counterexample in its own reasoning.
    """
    # (?<!\.) excludes dotted-qualified identifiers like `Tm.false`/`Bool.false`
    # (a real AST constructor, not a claim the theorem is false) - \b alone
    # only guards underscore-joined identifiers (`t_false`) since `.` is a
    # non-word char and still satisfies \b on its own.
    if re.search(r"(?<!\.)\b(false|counterexample)\b", analysis.lower()):
        return ProofSignal.STATEMENT_WRONG
    return ProofSignal.PROOF_TOO_HARD


# ---------------------------------------------------------------------------
# Backward-compat function (used by orchestrator.py)
# ---------------------------------------------------------------------------

def prove_node(
    node_name: str,
    canonical_stmt: str,
    parent_proofs: dict[str, str],
    parent_lemma_decls: str,
    compiler: AbstractLeanCompiler,
    retrieval: MathlibRetrieval,
    model: str = "gpt-4o",
    node_statement_nl: str = "",
    node_proof_sketch_nl: str = "",
    repo_retrieval=None,
    tracer=None,
    api_timeout_s: float = 120.0,
    max_tool_calls: int | None = None,
) -> ProverResult:
    parent_block = "\n\n".join(
        f"```lean\n-- {n}\n{p}\n```" for n, p in parent_proofs.items()
    )
    user_prompt = render(
        PROVER_USER_TEMPLATE,
        canonical_stmt=canonical_stmt,
        nl_statement=node_statement_nl,
        nl_proof_sketch=node_proof_sketch_nl,
        parent_proofs=parent_block,
    )
    prover = GoedelProver(model_id=model, retrieval=retrieval, tracer=tracer,
                           api_timeout_s=api_timeout_s, max_tool_calls=max_tool_calls)
    return prover.prove_node(
        compiler=compiler,
        node_name=node_name,
        node_stmt=canonical_stmt,
        user_prompt=user_prompt,
        repo_retrieval=repo_retrieval,
        parent_lemma_decls=parent_lemma_decls,
    )
