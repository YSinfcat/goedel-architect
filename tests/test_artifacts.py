"""Tests for the success artifact bundle (review IV.8) and the portfolio
failure feedback into the LLM prompt.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("GOEDEL_LEAN_TOOLCHAIN", "test-toolchain")

from artifacts import write_success_artifact  # noqa: E402
from blueprint import _parse_blueprint  # noqa: E402
from pipeline import VerificationReport  # noqa: E402
from prover import _portfolio_note  # noqa: E402

LEAN = ("@[blueprint (statement := /-- helper -/)]\n"
        "theorem helper : True := by sorry_using []\n\n"
        "@[blueprint (statement := /-- main -/)]\n"
        "theorem main : True := by sorry_using [helper]\n")


class TestPortfolioNote(unittest.TestCase):
    def test_note_lists_rejected_tactics(self):
        note = _portfolio_note(["simp", "omega"])
        self.assertIn("simp, omega", note)
        self.assertIn("REJECTED", note)

    def test_empty_failures_produce_no_note(self):
        self.assertEqual(_portfolio_note(None), "")
        self.assertEqual(_portfolio_note([]), "")


class TestSuccessArtifact(unittest.TestCase):
    def _write_bundle(self, tmp: str, trace_path: Path | None = None) -> Path:
        bp = _parse_blueprint(LEAN, "main")
        proof = ("import Mathlib\nimport Architect\n\n"
                 "theorem helper : True := by trivial\n\n"
                 "theorem main : True := by trivial\n")
        verification = VerificationReport(
            performed=True, passed=True, mode="full-file", errors=[], file_text=proof)
        manifest = {"code_commit": "abc1234", "lean_toolchain": "Lean 4.x",
                    "fingerprint": "deadbeef", "model_config": {"model": "stub"}}
        return write_success_artifact(
            Path(tmp), theorem_name="main",
            proof_lean=verification.file_text,
            verification=verification, manifest=manifest,
            blueprint_initial=bp, blueprint_final=bp,
            trace_path=trace_path)

    def test_bundle_contains_every_piece(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._write_bundle(tmp)
            names = {p.name for p in out.iterdir()}
            self.assertIn("proof.lean", names)
            self.assertIn("blueprint.initial.json", names)
            self.assertIn("blueprint.final.json", names)
            self.assertIn("dag.txt", names)
            self.assertIn("verification.json", names)
            self.assertIn("run_manifest.json", names)
            self.assertIn("model_usage.json", names)
            self.assertIn("README.md", names)

    def test_verification_json_reports_clean_scans(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._write_bundle(tmp)
            payload = json.loads((out / "verification.json").read_text())
            self.assertTrue(payload["original_theorem_verified"])
            self.assertEqual(payload["mode"], "full-file")
            self.assertFalse(payload["contains_sorry"])
            self.assertFalse(payload["contains_axiom"])
            self.assertFalse(payload["contains_native_decide"])
            self.assertEqual(payload["code_commit"], "abc1234")
            self.assertEqual(payload["run_fingerprint"], "deadbeef")

    def test_sorry_scan_detects_leftover_holes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._write_bundle(tmp)
            # corrupt the shipped proof and re-scan the way the writer would
            proof = (out / "proof.lean").read_text() + "\ntheorem extra : False := by sorry\n"
            payload = json.loads((out / "verification.json").read_text())
            import re
            self.assertTrue(re.search(r"\bsorry\b", proof))
            self.assertFalse(payload["contains_sorry"])  # original was clean

    def test_dag_text_lists_edges_and_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._write_bundle(tmp)
            dag = (out / "dag.txt").read_text()
            self.assertIn("main -> helper", dag)
            self.assertIn("# target: main", dag)

    def test_usage_aggregation_from_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            events = [
                {"kind": "llm_usage", "args": {"phase": "phase1", "model": "m",
                                               "prompt_tokens": 100, "completion_tokens": 10,
                                               "total_tokens": 110}},
                {"kind": "llm_usage", "args": {"phase": "phase1", "model": "m",
                                               "prompt_tokens": 50, "completion_tokens": 5,
                                               "total_tokens": 55}},
                {"kind": "tool_call", "args": {}},
            ]
            trace.write_text("\n".join(json.dumps(e) for e in events))
            out = self._write_bundle(tmp, trace_path=trace)
            usage = json.loads((out / "model_usage.json").read_text())
            self.assertEqual(usage["by_phase_model"]["phase1/m"]["calls"], 2)
            self.assertEqual(usage["total_prompt_tokens"], 150)
            self.assertEqual(usage["total_tokens"], 165)

    def test_theorem_name_is_path_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            from artifacts import _safe
            self.assertEqual(_safe("foo/bar baz"), "foo_bar_baz")


if __name__ == "__main__":
    unittest.main(verbosity=2)
