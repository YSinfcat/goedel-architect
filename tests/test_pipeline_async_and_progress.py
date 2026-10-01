"""End-to-end pipeline tests for the async split (review IV.5), the
no-progress stop (review III.2), and the eval/repo_retrieval shim.

Both pipeline tests stub generate_blueprint / refine_blueprint / the
compiler, so the FULL prove_theorem loop runs with zero LLM and zero Lean:
checkpointing, phase gating, final verification, refinement rounds, and
the oscillation stop are all exercised for real.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("GOEDEL_LEAN_TOOLCHAIN", "test-toolchain")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("FIREWORKS_API_KEY", "test-key")
os.environ.setdefault("MISTRAL_API_KEY", "test-key")

import pipeline  # noqa: E402
from blueprint import _parse_blueprint  # noqa: E402
from checkpoint import CheckpointState  # noqa: E402
from lean_compiler import CompilerResult  # noqa: E402
from pipeline import prove_theorem, prove_theorem_async  # noqa: E402
from run_fingerprint import fingerprint, run_manifest  # noqa: E402

LEAN = ("@[blueprint (statement := /-- s -/)]\n"
        "theorem main : True := by sorry_using []\n")


def _seeded_checkpoint(path: Path, *, max_iterations: int) -> None:
    """A checkpoint whose root is already proved (Phase 2 becomes a no-op,
    so the loop runs with zero LLM) and whose run fingerprint matches the
    exact prove_theorem arguments the tests use."""
    bp = _parse_blueprint(LEAN, "main")
    bp.fully_validated = True
    state = CheckpointState(theorem_stmt="theorem main : True := sorry", model="stub")
    state.set_blueprint(bp)
    state.proved_cache = {"main": "by trivial"}
    manifest = run_manifest(model="stub", cascade_model=None,
                            max_iterations=max_iterations,
                            enable_negation_probe=False,
                            allow_unvalidated_blueprint=False)
    state.run_fingerprint = {**manifest, "fingerprint": fingerprint(manifest)}
    state.save(path)


class PassingCompiler:
    def check(self, code, **_):
        return CompilerResult(success=True)

    def check_blueprint(self, code, target):
        return CompilerResult(success=True)


class FailingVerifyCompiler(PassingCompiler):
    """check_blueprint passes (blueprint gate OK) but check fails - the
    final verification against the original theorem never succeeds."""

    def check(self, code, **_):
        return CompilerResult(success=False, errors=["stub: does not check"])


class TestAsyncSplit(unittest.TestCase):
    def setUp(self):
        self._orig_gen = pipeline.generate_blueprint
        bp = _parse_blueprint(LEAN, "main")
        bp.fully_validated = True
        pipeline.generate_blueprint = lambda **kwargs: bp

    def tearDown(self):
        pipeline.generate_blueprint = self._orig_gen

    def test_async_variant_proves_theorem(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "ckpt.json"
            _seeded_checkpoint(ckpt, max_iterations=2)

            async def run():
                return await prove_theorem_async(
                    theorem_stmt="theorem main : True := sorry",
                    model="stub", compiler=PassingCompiler(), max_iterations=2,
                    checkpoint_path=ckpt,
                )
            result = asyncio.run(run())
            self.assertTrue(result.success)
            self.assertTrue(result.final_verification.passed)

    def test_sync_wrapper_still_works(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "ckpt.json"
            _seeded_checkpoint(ckpt, max_iterations=2)
            result = prove_theorem(
                theorem_stmt="theorem main : True := sorry",
                model="stub", compiler=PassingCompiler(), max_iterations=2,
                checkpoint_path=ckpt,
            )
            self.assertTrue(result.success)
            reloaded = CheckpointState.load(ckpt)
            self.assertTrue(reloaded.success)


class TestNoProgressStop(unittest.TestCase):
    def setUp(self):
        self._orig_gen = pipeline.generate_blueprint
        self._orig_refine = pipeline.refine_blueprint
        bp = _parse_blueprint(LEAN, "main")
        bp.fully_validated = True
        pipeline.generate_blueprint = lambda **kwargs: bp
        # Refinement "succeeds" but echoes the same graph back every round -
        # the split/merge oscillation the review describes.
        pipeline.refine_blueprint = lambda **kwargs: _parse_blueprint(LEAN, "main")

    def tearDown(self):
        pipeline.generate_blueprint = self._orig_gen
        pipeline.refine_blueprint = self._orig_refine

    def test_oscillating_refinement_stops_early(self):
        # Seed a checkpoint with the root already proved so Phase 2 is a
        # no-op (no LLM): every round then fails final verification, the
        # refinement echoes the same structure, and the no-progress stop
        # must fire well before max_iterations.
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "ckpt.json"
            _seeded_checkpoint(ckpt, max_iterations=8)

            result = prove_theorem(
                theorem_stmt="theorem main : True := sorry",
                model="stub", compiler=FailingVerifyCompiler(),
                max_iterations=8, checkpoint_path=ckpt,
            )
            self.assertFalse(result.success)
            self.assertFalse(result.final_verification.passed)
            # stopped by the no-progress rule, not by exhausting the cap:
            # rounds 1 and 2 were both stagnant, so the stop fires at round 2
            self.assertLess(result.iterations, 8)
            self.assertGreaterEqual(result.iterations, 2)


class TestEvalShim(unittest.TestCase):
    def test_eval_retrieval_shim_resolves_to_canonical(self):
        # Import fresh with eval/ ahead of src/ on sys.path - the shadowing
        # order that used to bind the drifted duplicate.
        import importlib
        import repo_retrieval as already
        if already.__file__.endswith("src/repo_retrieval.py"):
            # loaded from src; force the shim file explicitly instead
            spec = importlib.util.spec_from_file_location(
                "shim_check", ROOT / "eval" / "repo_retrieval.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        else:
            mod = already
        self.assertTrue(hasattr(mod.RepoRetrieval, "_content_hash"))
        self.assertTrue(hasattr(mod.RepoRetrieval, "_cache_key"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
