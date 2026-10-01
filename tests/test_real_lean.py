"""Real-Lean regression tests - the verification chain under the actual
kernel (Lean 4 + Mathlib + GoedelArch oleans), not stub compilers.

These run ONLY on machines with a working `lake` and a built goedel_lean
workspace (the author's machine; CI skips them - see the guard below).
Each test is a real elaboration (seconds to tens of seconds each), so the
count is deliberately small and each one pins a load-bearing promise:

  - check_blueprint validates a real @[blueprint]/sorry_using skeleton
  - the final verification PASSES an honest proof of the original theorem
  - the final verification CATCHES a weakened root (review P0-1 live)
  - a false statement never compiles; forbidden constructs are rejected
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("GOEDEL_LEAN_TOOLCHAIN", "test-toolchain")


def _lean_available() -> str | None:
    if shutil.which("lake") is None:
        return "lake not on PATH"
    olean = (ROOT / "goedel_lean" / ".lake" / "packages" / "LeanArchitect"
             / ".lake" / "build" / "lib" / "lean" / "Architect.olean")
    if not olean.exists():
        return f"Architect olean missing ({olean})"
    return None


_LEAN_SKIP = _lean_available()

ORIGINAL = "theorem main : ∃ n : ℕ, n + n = 4 := sorry"
BLUEPRINT = """import Mathlib
import Architect

@[blueprint (statement := /-- helper -/)]
theorem helper : (1 + 1 = 2) := by sorry_using []

@[blueprint (statement := /-- main -/)]
theorem main : ∃ n : ℕ, n + n = 4 := by sorry_using [helper]
"""


@unittest.skipIf(_LEAN_SKIP, f"real Lean unavailable: {_LEAN_SKIP}")
class TestRealLeanVerificationChain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lean_compiler import LeanCompiler
        cls.compiler = LeanCompiler()

    def test_check_blueprint_validates_real_skeleton(self):
        from blueprint import _parse_blueprint
        bp = _parse_blueprint(BLUEPRINT, "main")
        self.assertEqual(len(bp.nodes), 2)
        r = self.compiler.check_blueprint(BLUEPRINT, "main")
        self.assertTrue(r.success, r.errors[:2])
        self.assertTrue(r.validated)

    def test_final_verification_passes_honest_proof(self):
        from blueprint import _parse_blueprint
        from pipeline import _final_verification
        bp = _parse_blueprint(BLUEPRINT, "main")
        r = _final_verification(
            ORIGINAL, bp,
            {"helper": "by norm_num", "main": "by exact ⟨2, by norm_num⟩"},
            self.compiler)
        self.assertTrue(r.passed, r.errors[:2])
        self.assertEqual(r.mode, "full-file")
        # the shipped file_text is the exact compiled artifact
        self.assertIn("theorem main : ∃ n : ℕ, n + n = 4", r.file_text)
        self.assertNotIn("sorry", r.file_text)

    def test_final_verification_catches_weakened_root(self):
        # Review P0-1, live under the real kernel: the blueprint quietly
        # weakened main's signature to `True`; every node "solved"; the
        # root proof compiles fine against the WEAKENED goal - but the
        # independent check against the ORIGINAL statement must fail.
        from blueprint import _parse_blueprint
        from pipeline import _final_verification
        weak = BLUEPRINT.replace("theorem main : ∃ n : ℕ, n + n = 4",
                                 "theorem main : True")
        bpw = _parse_blueprint(weak, "main")
        r = _final_verification(
            ORIGINAL, bpw,
            {"helper": "by norm_num", "main": "by trivial"},
            self.compiler)
        self.assertFalse(r.passed)
        self.assertTrue(any("∃" in e or "n + n = 4" in e for e in r.errors),
                        r.errors[:3])
        # and no file_text ships for a failed verification
        self.assertEqual(r.file_text, "")

    def test_false_statement_never_compiles(self):
        r = self.compiler.check(
            "by norm_num", node_decl="theorem bad : (1 + 1 = 3) := by sorry_using []")
        self.assertFalse(r.success)

    def test_forbidden_construct_rejected_before_lean(self):
        r = self.compiler.check(
            "by native_decide",
            node_decl="theorem x : (1 + 1 = 2) := by sorry_using []")
        self.assertFalse(r.success)
        self.assertIn("Safeguard", r.raw_output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
