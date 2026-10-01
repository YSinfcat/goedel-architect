"""Hard budgets for a single prove_theorem run (review IV.6).

Experiments without ceilings leak money two ways: a node whose cheap
attempt timed out keeps burning tokens in its abandoned thread, and a
refinement loop oscillating between two graph shapes re-buys Phase 2
forever. A Budget is a thread-safe counter set the pipeline checks
between iterations and the prover/compiler layers feed on every spend:

    budget = Budget(max_total_tokens=2_000_000,
                    max_compile_calls=400,
                    max_wall_time_s=3600)
    prove_theorem(..., budget=budget)

When a ceiling is hit, `exhausted` turns True with a human-readable
`stop_reason`; the pipeline stops at the next checkpoint boundary and
stamps ProofResult.stopped_reason, so a budget-stopped run is never
confusable with a mathematical failure.

Cost (USD) needs a per-model price table the harness does not have - the
hook is here (price_table: model -> ($/1k-in, $/1k-out)), tracking
activates only when one is supplied, and tokens remain the primary unit.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from lean_compiler import AbstractLeanCompiler


@dataclass
class Budget:
    max_total_tokens: int | None = None
    max_compile_calls: int | None = None
    max_wall_time_s: float | None = None
    max_cost_usd: float | None = None
    # model -> (usd per 1k prompt tokens, usd per 1k completion tokens).
    # Without a table, cost stays 0.0 and only token ceilings apply.
    price_table: dict[str, tuple[float, float]] | None = None

    spent_tokens: int = 0
    spent_cost_usd: float = 0.0
    compile_calls: int = 0
    stop_reason: str = ""
    _t0: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def spend_tokens(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.spent_tokens += prompt_tokens + completion_tokens
            if self.price_table and model in self.price_table:
                pin, pout = self.price_table[model]
                self.spent_cost_usd += (prompt_tokens * pin + completion_tokens * pout) / 1000.0

    def spend_compile(self) -> None:
        with self._lock:
            self.compile_calls += 1

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._t0

    @property
    def exhausted(self) -> bool:
        """True when any ceiling is hit (sticky: records the first reason)."""
        with self._lock:
            if self.stop_reason:
                return True
        reason = ""
        if self.max_total_tokens is not None and self.spent_tokens >= self.max_total_tokens:
            reason = f"token budget exhausted ({self.spent_tokens} >= {self.max_total_tokens})"
        elif self.max_compile_calls is not None and self.compile_calls >= self.max_compile_calls:
            reason = f"compile-call budget exhausted ({self.compile_calls} >= {self.max_compile_calls})"
        elif self.max_wall_time_s is not None and self.elapsed_s >= self.max_wall_time_s:
            reason = f"wall-time budget exhausted ({self.elapsed_s:.0f}s >= {self.max_wall_time_s}s)"
        elif (self.max_cost_usd is not None and self.price_table
              and self.spent_cost_usd >= self.max_cost_usd):
            reason = f"cost budget exhausted (${self.spent_cost_usd:.2f} >= ${self.max_cost_usd:.2f})"
        if reason:
            with self._lock:
                if not self.stop_reason:
                    self.stop_reason = reason
            return True
        return False

    def report(self) -> str:
        return (f"tokens={self.spent_tokens} compiles={self.compile_calls} "
                f"cost=${self.spent_cost_usd:.2f} wall={self.elapsed_s:.0f}s")


class BudgetedCompiler(AbstractLeanCompiler):
    """Delegate that counts every Lean elaboration against a Budget -
    node attempts, blueprint validations, and the pipeline's final
    verification all funnel through check()/check_blueprint(), so one
    wrapper sees them all."""

    def __init__(self, inner: AbstractLeanCompiler, budget: Budget):
        self.inner = inner
        self.budget = budget

    def check(self, lean_code: str, **kwargs):
        self.budget.spend_compile()
        return self.inner.check(lean_code, **kwargs)

    def check_blueprint(self, lean_code: str, target_name: str):
        self.budget.spend_compile()
        return self.inner.check_blueprint(lean_code, target_name)

    async def check_async(self, lean_code: str, **kwargs):
        self.budget.spend_compile()
        return await self.inner.check(lean_code, **kwargs)
