"""Deterministic tactic portfolio (review IV.5): cheap first attempts
before any LLM call.

Many blueprint nodes are one-liners a fixed tactic closes outright (`simp`
on a definitional unfolding, `omega` on Presburger arithmetic, `decide` on
a decidable finite goal). Asking an LLM to rediscover `by omega` burns
tokens, latency, and a slot of the node's tool-call budget. The portfolio
tries an ordered list of single tactics through the SAME compile contract
the prover uses (bare `by <tactic>` body + node_decl + aux declarations);
the first tactic that compiles closes the node at zero LLM cost.

Failure costs one Lean elaboration per tactic (bounded by the compiler's
own timeout) and is not wasted silently: every attempt is emitted to the
trace, so post-mortems can see the goal survived the portfolio.

Selection stays static in v1 - ordering the portfolio by node kind /
error history (the review's type-directed table) needs those signals
plumbed through first.
"""
from __future__ import annotations

from lean_compiler import AbstractLeanCompiler
from prover import ProverResult, ProofSignal
from tracer import NullTracer, TraceEvent

# Ordered cheap-first. Deliberately no `exact?`/`apply?` (interactive
# suggestion commands, not scriptable closers) and no `native_decide`
# (forbidden construct - would be rejected by the safeguard anyway).
DEFAULT_TACTIC_PORTFOLIO: tuple[str, ...] = (
    "trivial",
    "simp",
    "simp_all",
    "aesop",
    "omega",
    "norm_num",
    "decide",
    "linarith",
)


def run_tactic_portfolio(
    compiler: AbstractLeanCompiler,
    node_decl: str,
    aux_lemmas: str,
    node_name: str,
    tactics: tuple[str, ...] | list[str] = DEFAULT_TACTIC_PORTFOLIO,
    tracer=None,
) -> ProverResult | None:
    """Try each tactic as a bare `by <tactic>` proof for the node.

    Returns a SOLVED ProverResult on the first compiling tactic, or None
    when the portfolio is exhausted (the caller falls through to the LLM).
    """
    tracer = tracer or NullTracer()
    attempts: list[str] = []
    for tactic in tactics:
        proof = f"by {tactic}"
        result = compiler.check(proof, aux_lemmas=aux_lemmas, node_decl=node_decl)
        attempts.append(f"{tactic}: {'ok' if result.success else 'no'}")
        tracer.emit(TraceEvent(
            kind="tactic_portfolio", thm_name=node_name,
            ok=result.success,
            args={"tactic": tactic},
            result=(None if result.success else "; ".join(result.errors[:2])),
        ))
        if result.success:
            return ProverResult(
                signal=ProofSignal.SOLVED,
                proof_body=proof,
                analysis=f"closed by the deterministic tactic portfolio ({tactic}) "
                         f"before any model call; attempts: {', '.join(attempts)}",
            )
    tracer.emit(TraceEvent(
        kind="tactic_portfolio", thm_name=node_name, ok=False,
        args={"tactic": "<exhausted>"},
        result=f"portfolio exhausted without closing the goal: {', '.join(attempts)}",
    ))
    return None
