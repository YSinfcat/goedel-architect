"""Phase 2: Per-node tool-equipped prover.

Uses the OpenAI-compatible chat.completions API (stateless — the running
`messages` list is threaded through explicitly), so it works against OpenAI,
Fireworks, or Mistral (see llm_client.make_client). Three tools: lean_compile,
repo_search, mathlib_search.

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


def _chat_reasoning_kwargs(model: str) -> dict:
    """Return reasoning_effort kwarg for models that support it (gpt-5.x/o-series).

    Only safe to pass on calls that omit `tools` — chat.completions rejects
    function tools + reasoning_effort together for these models ("Function
    tools with reasoning_effort are not supported ... Please use /v1/responses
    instead"), unlike the Responses API this loop used to call.
    """
    if model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4"):
        return {"reasoning_effort": "low"}
    return {}

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

**Key insight**: the prompt already contains repo context in <used_repo_defs>,
<repo_lemmas>, and <local_ctx>. Read these first — the lemmas you need are
likely already visible.

Workflow:
1. Draft a proof using the visible repo definitions.
2. Call lean_compile with your proof_body — the tactic block starting with
   `by` (a `:= by`-prefixed body is also accepted; the harness normalizes
   either form and owns the single `:=` assignment when splicing).
3. Read errors, adjust, call lean_compile again.
4. When you need a lemma whose name you do NOT already know:
   - call repo_search for project-specific lemmas
   - call mathlib_search for general Mathlib lemmas
5. Once lean_compile returns SUCCESSFUL, output:
   <lean4_proof>by\n  ...\n</lean4_proof>

Prefer lean_compile over search — faster to try a tactic and read the error.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
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
                            "Tactic block starting with 'by' (a ':= by' prefix is also "
                            "accepted and normalized away - do NOT include the theorem "
                            "declaration)."
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
    },
    {
        "type": "function",
        "function": {
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
    },
    {
        "type": "function",
        "function": {
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
    },
]


# ---------------------------------------------------------------------------
# Result types (unchanged from paper)
# ---------------------------------------------------------------------------

class ProofSignal(str, Enum):
    SOLVED = "solved"
    # Legacy value kept ONLY so checkpoints written before the downgrade can
    # still be deserialized. Nothing produces it anymore: a model *saying* the
    # statement looks false is not evidence, so new results use
    # MODEL_SUSPECTS_WRONG (advisory) and only the (default-off, experimental)
    # negation probe can still emit FORMALLY_NEGATED.
    STATEMENT_WRONG = "statement_wrong"
    MODEL_SUSPECTS_WRONG = "model_suspects_wrong"
    PROOF_TOO_HARD = "proof_too_hard"
    FORMALLY_NEGATED = "formally_negated"
    # Not one of the paper's four signals: an infra/tooling failure (timeout,
    # unhandled exception) rather than a genuine "the model tried and
    # couldn't" verdict. Kept distinct so refinement (and human diagnosis)
    # doesn't treat a broken harness as evidence the sub-goal is hard.
    INFRA_ERROR = "infra_error"

    @property
    def advisory_statement_suspect(self) -> bool:
        """True for the downgraded text-based 'statement looks false' signals
        (current MODEL_SUSPECTS_WRONG and legacy STATEMENT_WRONG checkpoints).
        Refinement must treat these as diagnostics, never as license to
        rewrite the mathematical statement."""
        return self in (ProofSignal.MODEL_SUSPECTS_WRONG, ProofSignal.STATEMENT_WRONG)


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
        model_id: str = "labs-leanstral-1-5",
        retrieval: MathlibRetrieval | None = None,
        tracer=None,
        api_timeout_s: float = 120.0,
        max_tool_calls: int | None = None,
        enable_negation_probe: bool = False,
    ):
        self.model_id = model_id
        # Bounds each individual chat.completions call so a stuck request
        # can't hang a node indefinitely; the orchestrator's node_timeout_s
        # bounds the whole multi-turn tool loop on top of this.
        self.client = make_client(model_id, timeout=api_timeout_s)
        self.retrieval = retrieval or MathlibRetrieval()
        self.tracer = tracer or NullTracer()
        # Per-instance override of the module-level MAX_TOOL_CALLS budget -
        # used by orchestrator.py's cascade to give an escalated (expensive)
        # attempt a much tighter budget (e.g. 1: a single lean_compile call,
        # no fix-and-retry loop) than the cheap cascade attempt gets.
        self.max_tool_calls = max_tool_calls if max_tool_calls is not None else MAX_TOOL_CALLS
        # The negation probe claims FORMALLY_NEGATED, but it never actually
        # compiles a `Not <node proposition>` goal - it feeds the model's
        # proof_body to the same checker as a POSITIVE proof (no negated
        # node_decl is constructed), so a positive proof of a TRUE statement
        # passes and gets mislabeled as a formal refutation. This poisoned
        # refinement (which treated FORMALLY_NEGATED as machine-checked and
        # locked the node). Default-off until the probe constructs the real
        # negated goal via the Lean meta API; enable explicitly only for
        # experiments that treat its output as MODEL_SUSPECTS_FALSE-grade.
        self.enable_negation_probe = enable_negation_probe

    def _emit_usage(self, node_name: str, response) -> None:
        """Log token usage from a chat.completions response."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt = getattr(usage, "prompt_tokens", 0)
        completion = getattr(usage, "completion_tokens", 0)
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
        """Attempt to prove a single node using the chat.completions tool loop."""
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

        # `messages` is mutated in place by _process_response (assistant
        # tool-call message + tool result messages appended each turn) - it
        # replaces the Responses API's previous_response_id chaining.
        messages: list[dict] = [
            {"role": "system", "content": augmented_sys},
            {"role": "user", "content": user_prompt},
        ]

        # Force first call to lean_compile
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": "lean_compile"}},
            max_completion_tokens=MAX_TOKENS,
        )
        self._emit_usage(node_name, response)

        tool_calls_used = 0
        best_proof_body = ""
        last_text = ""
        had_tool_calls = False
        last_compile_ok = False
        all_lean_errors: list[str] = []

        while tool_calls_used < self.max_tool_calls:
            had_tool_calls, text, proof, compile_ok, tools_called, compile_errors = self._process_response(
                response, messages, compiler, node_name, repo_retrieval, tool_calls_used, node_decl=node_stmt
            )
            all_lean_errors.extend(compile_errors)
            if text:
                last_text = text
            if proof:
                best_proof_body = proof
            if compile_ok:
                last_compile_ok = True
                return ProverResult(signal=ProofSignal.SOLVED, proof_body=proof)

            had_search = any(t in ("repo_search", "mathlib_search") for t in tools_called)
            had_compile = "lean_compile" in tools_called
            tool_calls_used += len(tools_called)

            if not had_tool_calls:
                break
            if tool_calls_used >= self.max_tool_calls:
                break

            next_choice = (
                {"type": "function", "function": {"name": "lean_compile"}}
                if (had_search and not had_compile) else "required"
            )

            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                tools=TOOLS,
                tool_choice=next_choice,
                max_completion_tokens=MAX_TOKENS,
            )
            self._emit_usage(node_name, response)

        # Drain any pending tool calls (get the model's reaction to the last
        # round of tool results without letting it call more tools)
        if had_tool_calls:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                tools=TOOLS,
                tool_choice="none",
                max_completion_tokens=MAX_TOKENS,
            )
            self._emit_usage(node_name, response)
            _, drain_text, drain_proof, _, _, drain_errors = self._process_response(
                response, messages, compiler, node_name, repo_retrieval, tool_calls_used, node_decl=node_stmt
            )
            all_lean_errors.extend(drain_errors)
            if drain_text:
                last_text = drain_text
            if drain_proof:
                best_proof_body = drain_proof

        # Ask for final answer if no proof tag found
        if not best_proof_body:
            messages.append({
                "role": "user",
                "content": "Output your best proof: <lean4_proof>:= by\n  ...\n</lean4_proof>",
            })
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_completion_tokens=MAX_TOKENS,
                **_chat_reasoning_kwargs(self.model_id),
            )
            self._emit_usage(node_name, response)
            _, last_text, best_proof_body, _, _, final_errors = self._process_response(
                response, messages, compiler, node_name, repo_retrieval, tool_calls_used, node_decl=node_stmt
            )
            all_lean_errors.extend(final_errors)

        # Probe negation if we couldn't prove it. Default-off: the probe
        # doesn't construct a real `Not P` goal (see __init__), so a positive
        # proof of a true statement can pass it and be mislabeled
        # FORMALLY_NEGATED - which refinement then treats as ground truth.
        if self.enable_negation_probe:
            negation = self._probe_negation(compiler, node_name, messages, MAX_TOKENS)
            if negation:
                return negation

        if best_proof_body:
            signal = _classify_failure(last_text)
            return ProverResult(signal=signal, proof_body=best_proof_body,
                                analysis=last_text[:500], lean_errors=all_lean_errors)
        return ProverResult(signal=_classify_failure(last_text),
                            analysis=last_text[:500], lean_errors=all_lean_errors)

    # ------------------------------------------------------------------
    # Response processing
    # ------------------------------------------------------------------

    def _process_response(
        self,
        response,
        messages: list[dict],
        compiler: AbstractLeanCompiler,
        node_name: str,
        repo_retrieval,
        tool_calls_so_far: int,
        node_decl: str = "",
    ) -> tuple[bool, str, str, bool, list[str], list[str]]:
        """Append this turn's assistant + tool-result messages to `messages`
        in place (chat.completions is stateless - the caller re-sends the
        full history each call) and return what happened this turn.

        Returns (had_tool_calls, last_text, proof, any_compile_ok,
        tools_called, compile_errors) - had_tool_calls replaces the old
        Responses-API tool_results list as the "was anything called this
        turn" signal.
        """
        msg = response.choices[0].message
        last_text = ""
        best_proof = ""
        compiled_proof = ""   # proof body that actually compiled — never overwritten by message text
        any_compile_ok = False
        tools_called: list[str] = []
        compile_errors: list[str] = []
        turn = tool_calls_so_far + 1

        assistant_msg: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if msg.content:
            last_text = msg.content
            self.tracer.emit(TraceEvent(
                kind="model_text", thm_name=node_name, turn=turn,
                result=msg.content[:500],
            ))

        for tc in msg.tool_calls or []:
            fn = tc.function.name
            args = _safe_tool_args(tc)
            tools_called.append(fn)

            self.tracer.emit(TraceEvent(
                kind="tool_call", thm_name=node_name, turn=turn,
                call_id=tc.id, tool_name=fn, args=args,
            ))

            if "__parse_error__" in args:
                # Malformed tool-argument JSON: tell the model instead of
                # crashing the node's whole tool loop.
                result = args["__parse_error__"] + "\nRe-call the tool with valid JSON arguments."
                self.tracer.emit(TraceEvent(
                    kind="tool_result", thm_name=node_name, turn=turn,
                    call_id=tc.id, result=result[:500],
                ))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                continue

            if fn == "lean_compile":
                proof_body = args.get("proof_body", "")
                aux = args.get("aux_lemmas", "")
                # Splice in already-proved sibling lemmas as real declarations
                # (see BlueprintNode.signature) so the model can reference them
                # by name instead of hitting "unknown identifier".
                parent_decls = getattr(self, "_parent_lemma_decls", "")
                full_aux = f"{parent_decls}\n\n{aux}".strip() if parent_decls else aux
                cr = compiler.check(proof_body, aux_lemmas=full_aux, node_decl=node_decl)
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
                hits = repo_retrieval.search(args["query"], args.get("k", 10))
                result = "\n\n".join(h.format() for h in hits) or "No results in repo."

            elif fn == "mathlib_search":
                hits = self.retrieval.search(args["query"], args.get("k", 10))
                result = "\n\n".join(h.format() for h in hits) or "No results found."

            else:
                result = f"Tool unavailable: {fn}"

            self.tracer.emit(TraceEvent(
                kind="tool_result", thm_name=node_name, turn=turn,
                call_id=tc.id, tool_name=fn,
                result=result[:500], ok=(fn == "lean_compile" and any_compile_ok),
            ))

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        # Only use message-extracted proof if no compile succeeded yet;
        # compiled_proof is authoritative and must not be overwritten.
        if last_text and not any_compile_ok:
            extracted = _extract_proof_body(last_text)
            if extracted:
                best_proof = extracted

        # Prefer the proof that actually compiled over anything extracted from text
        return bool(msg.tool_calls), last_text, compiled_proof or best_proof, any_compile_ok, tools_called, compile_errors

    # ------------------------------------------------------------------
    # Negation probe (Section 4.3 / Figure 1)
    # ------------------------------------------------------------------

    def _probe_negation(
        self,
        compiler: AbstractLeanCompiler,
        node_name: str,
        messages: list[dict],
        max_tokens: int,
    ) -> ProverResult | None:
        prompt = (
            f"You could not prove the statement. Try to show it is FALSE.\n"
            f"Prove `neg_{node_name}` showing `¬ (conclusion)` with the same hypotheses. "
            "Tactics: `omega`, `decide`, `norm_num`, `push_neg; linarith`, `simp`, `native_decide`.\n"
            "Call lean_compile. If it succeeds, the original statement is formally refuted."
        )
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": "lean_compile"}},
            max_completion_tokens=max_tokens,
        )
        self._emit_usage(node_name, response)

        for _ in range(NEGATION_PROBE_CALLS):
            msg = response.choices[0].message
            assistant_msg: dict = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            if not msg.tool_calls:
                break

            for tc in msg.tool_calls:
                if tc.function.name == "lean_compile":
                    args = _safe_tool_args(tc)
                    if "__parse_error__" in args:
                        output = args["__parse_error__"] + "\nRe-call lean_compile with valid JSON arguments."
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
                        continue
                    parent_decls = getattr(self, "_parent_lemma_decls", "")
                    cr = compiler.check(args.get("proof_body", ""), aux_lemmas=parent_decls)
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
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})

            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_completion_tokens=max_tokens,
            )
            self._emit_usage(node_name, response)

        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_tool_args(tc) -> dict:
    """Parse a tool call's arguments JSON defensively. A model emitting
    malformed JSON used to raise straight out of the tool loop and kill the
    whole node (and, pre-orchestrator-guard, the whole wave); instead the
    caller feeds the parse error back to the model as the tool result so it
    can retry with valid JSON."""
    try:
        parsed = json.loads(tc.function.arguments or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError) as exc:
        return {"__parse_error__": f"tool arguments were not valid JSON: {exc}"}


def _extract_proof_body(text: str) -> str:
    import re
    m = re.search(r"<lean4_proof>(.*?)</lean4_proof>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _classify_failure(analysis: str) -> ProofSignal:
    """The model's own text mentioning "false"/"counterexample" downgrades to
    MODEL_SUSPECTS_WRONG: an advisory diagnostic, NOT evidence. Only the
    negation probe (default-off, experimental) can emit FORMALLY_NEGATED, and
    refinement must not rewrite a mathematical statement on the model's word
    alone. A compiler "type mismatch" was never such evidence either - it's
    one of the most generic Lean elaboration errors and fires for merely-
    wrong tactics on true theorems just as often as on false ones."""
    analysis_lower = analysis.lower()
    if re.search(r"\b(false|counterexample)\b", analysis_lower):
        return ProofSignal.MODEL_SUSPECTS_WRONG
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
    model: str = "labs-leanstral-1-5",
    node_statement_nl: str = "",
    node_proof_sketch_nl: str = "",
    repo_retrieval=None,
    tracer=None,
    api_timeout_s: float = 120.0,
    max_tool_calls: int | None = None,
    enable_negation_probe: bool = False,
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
                           api_timeout_s=api_timeout_s, max_tool_calls=max_tool_calls,
                           enable_negation_probe=enable_negation_probe)
    return prover.prove_node(
        compiler=compiler,
        node_name=node_name,
        node_stmt=canonical_stmt,
        user_prompt=user_prompt,
        repo_retrieval=repo_retrieval,
        parent_lemma_decls=parent_lemma_decls,
    )
