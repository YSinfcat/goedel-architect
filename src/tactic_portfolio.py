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

import hashlib

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


def portfolio_cache_key(node_decl: str, node_cache_key: str,
                         aux_lemmas: str, tactics) -> str:
    """Cache key for one portfolio sweep: the node's exact mathematical
    shape (signature + dependency set), the sibling-lemma context the
    tactics compile against, and the tactic list itself. A refinement
    round that echoes an unchanged node reuses the previous sweep instead
    of re-paying one Lean elaboration per tactic (the smoke trace showed
    iteration 2 repeating iteration 1's full 8-tactic sweep verbatim)."""
    h = hashlib.sha256()
    h.update(node_cache_key.encode())
    h.update(b"\x00aux:")
    h.update(hashlib.sha256(aux_lemmas.encode()).digest())
    h.update(b"\x00tactics:")
    h.update(",".join(tactics).encode())
    return h.hexdigest()


def run_tactic_portfolio(
    compiler: AbstractLeanCompiler,
    node_decl: str,
    aux_lemmas: str,
    node_name: str,
    tactics: tuple[str, ...] | list[str] = DEFAULT_TACTIC_PORTFOLIO,
    tracer=None,
    cache: dict | None = None,
    node_cache_key: str = "",
) -> tuple[ProverResult | None, list[str]]:
    """Try each tactic as a bare `by <tactic>` proof for the node.

    Returns (result, failed_tactics): a SOLVED ProverResult plus [] on the
    first compiling tactic, or None plus the list of tactics that were
    tried and rejected. The failure list is fed into the subsequent LLM
    prompt (see prover._portfolio_note) so the model doesn't burn tool
    calls rediscovering that `simp` doesn't close this goal.
    """
    tracer = tracer or NullTracer()
    if cache is not None and node_cache_key:
        key = portfolio_cache_key(node_decl, node_cache_key, aux_lemmas, tactics)
        if key in cache:
            cached_result, cached_failed = cache[key]
            tracer.emit(TraceEvent(
                kind="tactic_portfolio", thm_name=node_name, ok=cached_result is not None,
                args={"tactic": "<cache-hit>"},
                result=(f"sweep reused from cache: {len(cached_failed)} tactics "
                        f"already rejected on this exact node shape"),
            ))
            return cached_result, cached_failed
    attempts: list[str] = []
    failed: list[str] = []
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
            solved = ProverResult(
                signal=ProofSignal.SOLVED,
                proof_body=proof,
                analysis=f"closed by the deterministic tactic portfolio ({tactic}) "
                         f"before any model call; attempts: {', '.join(attempts)}",
            ), failed
            if cache is not None and node_cache_key:
                cache[portfolio_cache_key(node_decl, node_cache_key, aux_lemmas, tactics)] = solved
            return solved
        failed.append(tactic)
    tracer.emit(TraceEvent(
        kind="tactic_portfolio", thm_name=node_name, ok=False,
        args={"tactic": "<exhausted>"},
        result=f"portfolio exhausted without closing the goal: {', '.join(attempts)}",
    ))
    if cache is not None and node_cache_key:
        cache[portfolio_cache_key(node_decl, node_cache_key, aux_lemmas, tactics)] = (None, failed)
    return None, failed
