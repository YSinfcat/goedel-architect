"""Regression tests for the reproducibility layer (review phase 3):

  - run fingerprints: same config -> same fingerprint; any experiment-
    defining change (model, cascade, prompts, flags) changes it.
  - checkpoint schema: fingerprint recorded/round-tripped; legacy
    schema-0 checkpoints and mismatched fingerprints refuse to resume;
    the GOEDEL_SKIP_FINGERPRINT escape works and is recorded.
  - checkpoint forward-compatibility: unknown keys are ignored on load.
  - metrics: empty JSONL no longer divides by zero.
  - graph_viz: model-controlled `</script>` cannot break out of the
    embedded DATA blob.

Pure Python - no Lean, no network, no LLM.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

# Keep fingerprints hermetic and fast: no lake probing in tests.
os.environ.setdefault("GOEDEL_LEAN_TOOLCHAIN", "test-toolchain")

from checkpoint import CheckpointFingerprintError, CheckpointState  # noqa: E402
from run_fingerprint import fingerprint, run_manifest  # noqa: E402


def setUpModule():
    # run_fingerprint caches probe results (lru_cache) - freeze them once.
    import run_fingerprint
    run_fingerprint.code_commit.cache_clear()
    run_fingerprint.lean_toolchain.cache_clear()


class TestRunFingerprint(unittest.TestCase):
    def test_same_config_same_fingerprint(self):
        a = run_manifest(model="m1", cascade_model=None, max_iterations=8)
        b = run_manifest(model="m1", cascade_model=None, max_iterations=8)
        self.assertEqual(fingerprint(a), fingerprint(b))

    def test_model_change_changes_fingerprint(self):
        a = run_manifest(model="m1")
        b = run_manifest(model="m2")
        self.assertNotEqual(fingerprint(a), fingerprint(b))

    def test_pipeline_config_change_changes_fingerprint(self):
        a = run_manifest(model="m1", enable_negation_probe=False)
        b = run_manifest(model="m1", enable_negation_probe=True)
        self.assertNotEqual(fingerprint(a), fingerprint(b))

    def test_manifest_carries_provenance(self):
        m = run_manifest(model="m1")
        for key in ("manifest_schema", "code_commit", "prompt_hashes",
                    "lean_toolchain", "model_config", "pipeline_config"):
            self.assertIn(key, m)
        # prompt hashes cover every prompt file
        prompts = {p.name for p in (ROOT / "prompts").iterdir() if p.is_file()}
        self.assertEqual(set(m["prompt_hashes"].keys()), prompts)

    def test_fingerprint_ignores_key_order(self):
        a = run_manifest(model="m1", cascade_model="c1")
        b = dict(reversed(list(a.items())))
        self.assertEqual(fingerprint(a), fingerprint(b))


class TestCheckpointFingerprint(unittest.TestCase):
    def _state_with(self, manifest: dict) -> CheckpointState:
        state = CheckpointState(theorem_stmt="theorem t : True := sorry", model="m1")
        state.run_fingerprint = {**manifest, "fingerprint": fingerprint(manifest)}
        return state

    def test_round_trip_preserves_fingerprint(self):
        manifest = run_manifest(model="m1")
        state = self._state_with(manifest)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ckpt.json"
            state.save(path)
            reloaded = CheckpointState.load(path)
            self.assertEqual(reloaded.run_fingerprint["fingerprint"],
                             fingerprint(manifest))
            self.assertEqual(reloaded.schema_version, 1)

    def test_matching_fingerprint_passes(self):
        manifest = run_manifest(model="m1")
        state = self._state_with(manifest)
        state.require_fingerprint(manifest, fingerprint(manifest))  # no raise

    def test_mismatched_fingerprint_refuses(self):
        state = self._state_with(run_manifest(model="m1"))
        other = run_manifest(model="different-model")
        with self.assertRaises(CheckpointFingerprintError):
            state.require_fingerprint(other, fingerprint(other))

    def test_legacy_checkpoint_without_fingerprint_refuses(self):
        state = CheckpointState(theorem_stmt="theorem t : True := sorry", model="m1")
        state.run_fingerprint = {}
        manifest = run_manifest(model="m1")
        with self.assertRaises(CheckpointFingerprintError):
            state.require_fingerprint(manifest, fingerprint(manifest))

    def test_skip_env_escapes_the_check(self):
        old = os.environ.get("GOEDEL_SKIP_FINGERPRINT")
        try:
            import run_fingerprint
            run_fingerprint._skip_cache = True  # simulate the env being set
            state = self._state_with(run_manifest(model="m1"))
            other = run_manifest(model="different-model")
            state.require_fingerprint(other, fingerprint(other))  # no raise
        finally:
            run_fingerprint._skip_cache = False
            if old is None:
                os.environ.pop("GOEDEL_SKIP_FINGERPRINT", None)
            else:
                os.environ["GOEDEL_SKIP_FINGERPRINT"] = old

    def test_load_ignores_unknown_keys(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ckpt.json"
            payload = {
                "theorem_stmt": "theorem t : True := sorry",
                "model": "m1",
                "schema_version": 99,          # newer schema's version
                "some_future_field": {"x": 1},  # unknown key
            }
            path.write_text(json.dumps(payload))
            state = CheckpointState.load(path)
            self.assertEqual(state.model, "m1")
            # unknown field dropped, not crashed on
            self.assertFalse(hasattr(state, "some_future_field"))


class TestMetricsZeroDivision(unittest.TestCase):
    def test_empty_jsonl_does_not_crash(self):
        import metrics
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "empty.jsonl"
            path.write_text("")
            buf = io.StringIO()
            with redirect_stdout(buf):
                metrics.summarize(str(path))
            self.assertIn("no records", buf.getvalue())


class TestGraphVizEscaping(unittest.TestCase):
    def test_script_breakout_is_escaped(self):
        import graph_viz
        events = [
            {"kind": "theorem_start", "thm_name": "evil</script><b>",
             "args": {"thm_stmt": "theorem evil : True := sorry", "lean_root": "r"}},
            {"kind": "final_verify", "thm_name": "evil</script><b>",
             "ok": True, "args": {"wall_time_s": 1.0, "proof": "by trivial"}},
        ]
        with tempfile.TemporaryDirectory() as d:
            trace = Path(d) / "trace.jsonl"
            trace.write_text("\n".join(json.dumps(e) for e in events))
            out = graph_viz.generate(trace)
            html = out.read_text()
            # the DATA blob must contain the escaped form, never raw markup
            self.assertIn("\\u003c/script\\u003e", html)
            # and the raw breakout sequence must not appear inside a JSON
            # string value (the template's own </script> tags are fine -
            # they sit outside const DATA = ...;)
            data_line = next(l for l in html.splitlines() if l.startswith("const DATA"))
            self.assertNotIn("</script>", data_line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
