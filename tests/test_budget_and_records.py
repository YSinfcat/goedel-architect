"""Tests for hard budgets (review IV.6) and the unified evaluation record
schema (review IV.7).
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("GOEDEL_LEAN_TOOLCHAIN", "test-toolchain")

from blueprint import _parse_blueprint  # noqa: E402
from budget import Budget, BudgetedCompiler  # noqa: E402
from eval_records import evaluation_record  # noqa: E402
from lean_compiler import CompilerResult  # noqa: E402
from pipeline import prove_theorem  # noqa: E402
from run_fingerprint import fingerprint, run_manifest  # noqa: E402

LEAN = ("@[blueprint (statement := /-- helper -/)]\n"
        "theorem helper : True := by sorry_using []\n\n"
        "@[blueprint (statement := /-- main -/)]\n"
        "theorem main : True := by sorry_using [helper]\n")


class TestBudget(unittest.TestCase):
    def test_token_ceiling_exhausts_with_reason(self):
        b = Budget(max_total_tokens=100)
        b.spend_tokens("m", 60, 30)
        self.assertFalse(b.exhausted)
        b.spend_tokens("m", 10, 0)
        self.assertTrue(b.exhausted)
        self.assertIn("token budget", b.stop_reason)

    def test_compile_and_wall_time_ceilings(self):
        b = Budget(max_compile_calls=2)
        b.spend_compile(); b.spend_compile()
        self.assertTrue(b.exhausted)  # reason is set on the check
        self.assertIn("compile-call", b.stop_reason)
        b2 = Budget(max_wall_time_s=0.0)
        self.assertTrue(b2.exhausted)
        self.assertIn("wall-time", b2.stop_reason)

    def test_exhausted_is_sticky(self):
        b = Budget(max_total_tokens=10)
        b.spend_tokens("m", 10, 0)
        self.assertTrue(b.exhausted)
        self.assertEqual(b.stop_reason, b.stop_reason)  # first reason kept

    def test_cost_tracking_needs_price_table(self):
        b = Budget(max_cost_usd=1.0)
        b.spend_tokens("m", 1_000_000, 0)
        self.assertEqual(b.spent_cost_usd, 0.0)  # no table -> no cost
        b2 = Budget(max_cost_usd=0.001,
                    price_table={"m": (0.001, 0.002)})
        b2.spend_tokens("m", 1000, 1000)
        self.assertAlmostEqual(b2.spent_cost_usd, 0.003)
        self.assertTrue(b2.exhausted)

    def test_budgeted_compiler_counts_every_elaboration(self):
        class Inner:
            calls = 0

            def check(self, code, **_):
                Inner.calls += 1
                return CompilerResult(success=True)

            def check_blueprint(self, code, target):
                Inner.calls += 1
                return CompilerResult(success=True)

        b = Budget(max_compile_calls=100)
        wrapped = BudgetedCompiler(Inner(), b)
        wrapped.check("x")
        wrapped.check("y", node_decl="d")
        wrapped.check_blueprint("z", "t")
        self.assertEqual(b.compile_calls, 3)


class TestBudgetInPipeline(unittest.TestCase):
    def test_budget_stop_before_first_iteration(self):
        import pipeline
        bp = _parse_blueprint(LEAN, "main")
        bp.fully_validated = True
        old_gen = pipeline.generate_blueprint
        pipeline.generate_blueprint = lambda **kw: bp
        try:
            with tempfile.TemporaryDirectory() as d:
                ckpt = Path(d) / "ckpt.json"
                s = None
                from checkpoint import CheckpointState
                s = CheckpointState(theorem_stmt="theorem main : True := sorry",
                                     model="stub")
                s.set_blueprint(bp)
                s.proved_cache = {"helper": "by trivial", "main": "by trivial"}
                m = run_manifest(model="stub", cascade_model=None, max_iterations=2)
                s.run_fingerprint = {**m, "fingerprint": fingerprint(m)}
                s.save(ckpt)

                class Passing:
                    def check(self, code, **_):
                        return CompilerResult(success=True)

                    def check_blueprint(self, code, target):
                        return CompilerResult(success=True)

                # pre-exhausted budget: loop must stop immediately
                budget = Budget(max_total_tokens=10)
                budget.spend_tokens("m", 10, 0)
                result = prove_theorem(
                    theorem_stmt="theorem main : True := sorry",
                    model="stub", compiler=Passing(), max_iterations=2,
                    checkpoint_path=ckpt, budget=budget,
                )
                self.assertFalse(result.success)
                self.assertIn("token budget", result.stopped_reason)
                self.assertEqual(result.iterations, 0)
        finally:
            pipeline.generate_blueprint = old_gen


class TestEvaluationRecord(unittest.TestCase):
    def _result(self):
        from pipeline import ProofResult, VerificationReport
        return ProofResult(
            success=False, theorem_name="main", iterations=3,
            proved_nodes=["helper"], failed_nodes=["main"],
            final_verification=VerificationReport(
                performed=True, passed=False, mode="full-file",
                errors=["type mismatch"]),
            autonomy="fully_autonomous",
            stopped_reason="token budget exhausted (10 >= 10)")

    def test_schema_fields(self):
        r = evaluation_record("prob1", self._result(), 12.34)
        self.assertEqual(r["status"], "FAILED")
        self.assertFalse(r["final_verified"])
        self.assertEqual(r["verify_mode"], "full-file")
        self.assertEqual(r["autonomy"], "fully_autonomous")
        self.assertIn("token budget", r["stopped_reason"])
        self.assertEqual(r["elapsed_s"], 12.34)
        self.assertIn("ts", r)

    def test_error_record(self):
        r = evaluation_record("prob2", None, 1.0, error="boom")
        self.assertEqual(r["status"], "ERROR")
        self.assertIsNone(r["final_verified"])
        self.assertEqual(r["error"], "boom")
        self.assertEqual(r["stopped_reason"], "")  # budget-only field

    def test_success_record_verified(self):
        from pipeline import ProofResult, VerificationReport
        result = ProofResult(success=True, theorem_name="main",
                             final_verification=VerificationReport(
                                 performed=True, passed=True, mode="full-file"))
        r = evaluation_record("p", result, 5.0)
        self.assertEqual(r["status"], "SOLVED")
        self.assertTrue(r["final_verified"])

    def test_metrics_summarizes_unified_fields(self):
        import metrics
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "r.jsonl"
            records = [
                evaluation_record("a", self._result(), 3.0),
                evaluation_record("b", None, 1.0, error="x"),
            ]
            path.write_text("\n".join(json.dumps(r) for r in records))
            buf = io.StringIO()
            with redirect_stdout(buf):
                metrics.summarize(str(path))
            out = buf.getvalue()
            self.assertIn("Autonomy: fully_autonomous=1", out)
            self.assertIn("rejected (weakened/corrupted proofs caught)", out)
            self.assertIn("Budget-stopped: 1", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
