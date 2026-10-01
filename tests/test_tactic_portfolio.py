"""Tests for the deterministic tactic portfolio (review IV.5).

  - A tactic that compiles closes the node with a SOLVED result and the
    canonical `by <tactic>` proof body; earlier failures are recorded.
  - An exhausted portfolio returns None so the caller falls through to
    the LLM.
  - End-to-end: prove_dag with the portfolio enabled solves a node with a
    compiler that only accepts one specific tactic - zero LLM involved.
  - The portfolio participates in the run fingerprint (experiments with
    and without it are not resumable against each other).
"""
from __future__ import annotations

import asyncio
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("GOEDEL_LEAN_TOOLCHAIN", "test-toolchain")

from blueprint import _parse_blueprint  # noqa: E402
from lean_compiler import CompilerResult  # noqa: E402
from orchestrator import prove_dag  # noqa: E402
from prover import ProofSignal  # noqa: E402
from run_fingerprint import fingerprint, run_manifest  # noqa: E402
from tactic_portfolio import run_tactic_portfolio  # noqa: E402

LEAN = ("@[blueprint (statement := /-- s -/)]\n"
        "theorem main : True := by sorry_using []\n")


class OmegaOnlyCompiler:
    """Accepts exactly `by omega` via the node_decl contract; everything
    else fails - mimics a Presburger goal no other cheap tactic closes."""

    def check(self, code, aux_lemmas="", node_decl="", **_):
        if node_decl and code.strip() == "by omega":
            return CompilerResult(success=True)
        return CompilerResult(success=False, errors=[f"stub rejects {code.strip()!r}"])

    def check_blueprint(self, code, target):
        return CompilerResult(success=True)


class AllFailCompiler:
    def check(self, code, **_):
        return CompilerResult(success=False, errors=["stub: nothing works"])

    def check_blueprint(self, code, target):
        return CompilerResult(success=True)


class TestRunTacticPortfolio(unittest.TestCase):
    def test_hit_returns_solved_with_canonical_body(self):
        result, failed = run_tactic_portfolio(
            OmegaOnlyCompiler(), LEAN, "", "main",
            tactics=["simp", "aesop", "omega"],
        )
        self.assertIsNotNone(result)
        self.assertEqual(failed, ["simp", "aesop"])
        self.assertEqual(result.signal, ProofSignal.SOLVED)
        self.assertEqual(result.proof_body, "by omega")
        self.assertIn("simp", result.analysis)  # prior attempts recorded

    def test_exhaustion_returns_none(self):
        result, failed = run_tactic_portfolio(
            AllFailCompiler(), LEAN, "", "main", tactics=["simp", "omega"],
        )
        self.assertIsNone(result)
        self.assertEqual(failed, ["simp", "omega"])


class TestPortfolioInDag(unittest.TestCase):
    def test_portfolio_solves_node_without_llm(self):
        bp = _parse_blueprint(LEAN, "main")
        # main is NOT in proved_cache -> _prove_one runs; without the
        # portfolio this would construct a GoedelProver and hit the (stub
        # key) API - with it, `by omega` closes the node first.
        result = asyncio.run(prove_dag(
            blueprint=bp, compiler=OmegaOnlyCompiler(), retrieval=None,
            tactic_portfolio=["simp", "omega"],
        ))
        self.assertTrue(result.all_proved())
        self.assertIn("main", result.proved)

    def test_disabled_by_default(self):
        import orchestrator
        bp = _parse_blueprint(LEAN, "main")
        # tactic_portfolio=None (default): the portfolio never runs, so the
        # node goes to the LLM path. Replace prove_node with a recorder to
        # observe that WITHOUT any network or API client construction.
        calls = []

        def fake_prove_node(**kwargs):
            calls.append(kwargs.get("node_name"))
            from prover import ProverResult
            return ProverResult(signal=ProofSignal.INFRA_ERROR, analysis="stubbed")

        original = orchestrator.prove_node
        orchestrator.prove_node = fake_prove_node
        try:
            result = asyncio.run(prove_dag(
                blueprint=bp, compiler=OmegaOnlyCompiler(), retrieval=None,
            ))
        finally:
            orchestrator.prove_node = original
        self.assertEqual(calls, ["main"])  # LLM path taken, not portfolio
        self.assertFalse(result.all_proved())


class TestPortfolioInFingerprint(unittest.TestCase):
    def test_portfolio_changes_fingerprint(self):
        off = run_manifest(model="m1")
        on = run_manifest(model="m1", tactic_portfolio=["simp"])
        self.assertNotEqual(fingerprint(off), fingerprint(on))
        self.assertIsNone(off["pipeline_config"]["tactic_portfolio"])
        self.assertEqual(on["pipeline_config"]["tactic_portfolio"], ["simp"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
