"""Tests for the human-intervention layer (review IV.3).
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

os.environ.setdefault("GOEDEL_LEAN_TOOLCHAIN", "test-toolchain")

import interventions  # noqa: E402
from blueprint import _parse_blueprint  # noqa: E402
from checkpoint import CheckpointState  # noqa: E402
from lean_compiler import CompilerResult  # noqa: E402
from pipeline import _invalidate_stale_proofs, prove_theorem  # noqa: E402
from run_fingerprint import fingerprint, run_manifest  # noqa: E402

LEAN = ("@[blueprint (statement := /-- helper -/)]\n"
        "theorem helper : True := by sorry_using []\n\n"
        "@[blueprint (statement := /-- main -/)]\n"
        "theorem main : True := by sorry_using [helper]\n")


def _state() -> CheckpointState:
    bp = _parse_blueprint(LEAN, "main")
    bp.fully_validated = True
    s = CheckpointState(theorem_stmt="theorem main : True := sorry", model="stub")
    s.set_blueprint(bp)
    return s


class TestInterventions(unittest.TestCase):
    def test_set_proof_canonicalizes_and_journals(self):
        s = _state()
        interventions.set_proof(s, "helper", ":= by exact trivial", reason="hand proof")
        self.assertEqual(s.proved_cache["helper"], "by exact trivial")
        self.assertEqual(s.proof_sources["helper"], "human")
        rec = s.interventions[-1]
        self.assertEqual(rec["type"], "set-proof")
        self.assertEqual(rec["node"], "helper")
        self.assertEqual(rec["reason"], "hand proof")
        self.assertTrue(rec["operator"])

    def test_lock_survives_stale_invalidation(self):
        s = _state()
        s.proved_cache = {"helper": "by trivial"}
        s.proof_cache_keys = {"helper": "old-shape"}
        s.locked_nodes = ["helper"]
        pruned = _invalidate_stale_proofs(s.get_blueprint(), s.proved_cache,
                                          s.proof_cache_keys, s.locked_nodes)
        self.assertIn("helper", pruned)
        # without the lock it would be dropped
        pruned = _invalidate_stale_proofs(s.get_blueprint(), s.proved_cache,
                                          s.proof_cache_keys, [])
        self.assertNotIn("helper", pruned)

    def test_retry_clears_and_reopens_finished_checkpoint(self):
        s = _state()
        s.proved_cache = {"helper": "by trivial"}
        s.done = True
        s.success = False
        interventions.retry(s, "helper", reason="try new model")
        self.assertNotIn("helper", s.proved_cache)
        self.assertFalse(s.done)

    def test_autonomy_labels(self):
        self.assertEqual(interventions.autonomy_label(_state()),
                         "fully_autonomous")
        s = _state()
        interventions.lock(s, "helper", reason="")
        self.assertEqual(interventions.autonomy_label(s), "human_guided")
        interventions.set_proof(s, "helper", "by trivial")
        self.assertEqual(interventions.autonomy_label(s), "human_written")


class TestHumanProofThroughPipeline(unittest.TestCase):
    def test_human_proof_still_requires_final_verification(self):
        # A human-set proof flows through the SAME success criterion: with
        # a verifier that rejects everything, the run must NOT succeed.
        import blueprint as bp_mod
        import llm_client
        import pipeline
        import prover as prover_mod
        import refinement as ref_mod
        # Hide the developer's live gateway config so nothing routes
        # through the real custom endpoint during this test.
        godel_env = {k: os.environ.pop(k, None) for k in
                     ("GOEDEL_BASE_URL", "GOEDEL_API_KEY",
                      "GOEDEL_API_STYLE", "GOEDEL_SESSION_ID")}
        llm_client._CLIENT_CACHE.clear()
        bp = _parse_blueprint(LEAN, "main")
        bp.fully_validated = True
        old_gen = pipeline.generate_blueprint
        pipeline.generate_blueprint = lambda **kw: bp
        # Each caller module holds its own `from llm_client import
        # make_client` reference - patch all three call sites' modules
        # (blueprint, refinement, prover) instead of the OpenAI class.
        class _StubCompletions:
            def create(self, **kw):
                class _Msg:
                    content = ""
                    tool_calls = None
                class _Choice:
                    message = _Msg()
                    finish_reason = "stop"
                class _Resp:
                    choices = [_Choice()]
                return _Resp()
        class _StubChat:
            completions = _StubCompletions()
        class _StubClient:
            chat = _StubChat
        _stub_factory = lambda model, timeout=None: _StubClient()
        old_bp_mc, old_ref_mc, old_prover_mc = (bp_mod.make_client,
                                                ref_mod.make_client,
                                                prover_mod.make_client)
        bp_mod.make_client = _stub_factory
        ref_mod.make_client = _stub_factory
        prover_mod.make_client = _stub_factory
        try:
            with tempfile.TemporaryDirectory() as d:
                ckpt = Path(d) / "ckpt.json"
                s = _state()
                m = run_manifest(model="stub", cascade_model=None, max_iterations=2)
                s.run_fingerprint = {**m, "fingerprint": fingerprint(m)}
                interventions.set_proof(s, "helper", "by trivial", reason="hand")
                interventions.set_proof(s, "main", "by trivial", reason="hand")
                s.save(ckpt)

                class Failing:
                    def check(self, code, **_):
                        return CompilerResult(success=False, errors=["stub: no"])

                    def check_blueprint(self, code, target):
                        return CompilerResult(success=True)

                result = prove_theorem(
                    theorem_stmt="theorem main : True := sorry",
                    model="stub", compiler=Failing(), max_iterations=2,
                    checkpoint_path=ckpt,
                )
                self.assertFalse(result.success)
                self.assertEqual(result.autonomy, "human_written")
        finally:
            pipeline.generate_blueprint = old_gen
            bp_mod.make_client = old_bp_mc
            ref_mod.make_client = old_ref_mc
            prover_mod.make_client = old_prover_mc
            llm_client._CLIENT_CACHE.clear()
            for k, v in godel_env.items():
                if v is not None:
                    os.environ[k] = v

    def test_status_lists_everything(self):
        s = _state()
        s.proved_cache = {"helper": "by trivial"}
        s.proof_sources = {"helper": "human"}
        s.locked_nodes = ["helper"]
        rows = interventions.status(s)
        by = {r["node"]: r for r in rows}
        self.assertEqual(by["helper"]["source"], "human")
        self.assertTrue(by["helper"]["locked"])
        self.assertEqual(by["main"]["status"], "unattempted")


class TestCli(unittest.TestCase):
    def test_cli_set_proof_and_status(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "ckpt.json"
            _state().save(ckpt)
            proof_file = Path(d) / "p.txt"
            proof_file.write_text(":= by exact trivial")
            r = subprocess.run(
                [sys.executable, str(ROOT / "eval" / "edit_checkpoint.py"),
                 "set-proof", str(ckpt), "helper", "--file", str(proof_file),
                 "--reason", "cli test"],
                capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("human_written", r.stdout)
            r = subprocess.run(
                [sys.executable, str(ROOT / "eval" / "edit_checkpoint.py"),
                 "status", str(ckpt)],
                capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("helper", r.stdout)
            self.assertNotIn("LOCKED", r.stdout)  # nothing locked yet
            self.assertIn("interventions: 1", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
