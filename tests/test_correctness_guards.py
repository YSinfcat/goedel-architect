"""Regression tests for the correctness chain added in the review fixes.

Each test pins a historical bug so it cannot silently return:

  1. `:= := by ...` double-assignment assembly (proof bodies now have ONE
     canonical internal form; splice sites add the single `:=` themselves).
  2. Text-based STATEMENT_WRONG verdicts (downgraded to the advisory
     MODEL_SUSPECTS_WRONG; legacy checkpoint values still deserialize).
  3. Unvalidated blueprints entering Phase 2 (hard gate + explicit escape).
  4. Structurally broken graphs (duplicate names / unknown deps / cycles /
     dead nodes / missing target) — central validator.
  5. Success without verifying the ORIGINAL theorem statement (the final
     verification assembly splices the root proof into the immutable
     theorem_stmt, never the refined blueprint root).
  6. Malformed tool-call JSON crashing the prover loop.

Pure Python - no Lean, no network, no LLM.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from blueprint import (  # noqa: E402
    Blueprint,
    BlueprintNode,
    BlueprintValidationError,
    validate_blueprint,
)
from lean_compiler import (  # noqa: E402
    MalformedProofBody,
    _assemble_node_attempt,
    format_proof_assign,
    normalize_proof_body,
)
from pipeline import (  # noqa: E402
    _assemble_original_theorem_file,
    _gate_blueprint,
    _substitute_proof,
)
from prover import ProofSignal, _classify_failure, _safe_tool_args  # noqa: E402


def _node(name: str, deps: list[str] | None = None) -> BlueprintNode:
    return BlueprintNode(
        name=name, kind="lemma", statement=f"nl {name}", proof_sketch="sketch",
        dependencies=list(deps or []),
        lean_declaration=f"theorem {name} : True := by sorry_using [{', '.join(deps or [])}]",
    )


def _blueprint(nodes: list[BlueprintNode], target: str = "main") -> Blueprint:
    return Blueprint(nodes=nodes, lean_file="", target_theorem=target)


class TestProofBodyNormalization(unittest.TestCase):
    def test_canonical_by_form_passes_through(self):
        self.assertEqual(normalize_proof_body("by simp"), "by simp")

    def test_legacy_assign_by_is_normalized(self):
        self.assertEqual(normalize_proof_body(":= by simp"), "by simp")
        self.assertEqual(normalize_proof_body(":=\nby simp"), "by simp")

    def test_double_assign_is_rejected(self):
        with self.assertRaises(MalformedProofBody):
            normalize_proof_body(":= := by simp")

    def test_format_adds_exactly_one_assign(self):
        self.assertEqual(format_proof_assign("by simp"), ":= by simp")
        self.assertEqual(format_proof_assign(":= by simp"), ":= by simp")

    def test_assemble_node_attempt_never_double_assigns(self):
        decl = "@[blueprint (statement := /-- x -/)]\ntheorem main : True := by sorry_using []"
        # The model-facing tool schema historically REQUIRED a ':= by' body;
        # that used to re-assemble as ':= := by ...' and could never compile.
        assembled = _assemble_node_attempt(decl, "", ":= by simp")
        self.assertIn(":= by simp", assembled)
        self.assertNotIn(":= := by", assembled)

    def test_substitute_proof_never_double_assigns(self):
        lean = "theorem main : True := by sorry_using []"
        out = _substitute_proof(lean, "main", ":= by simp")
        self.assertIn(":= by simp", out)
        self.assertNotIn(":= := by", out)

    def test_substitute_proof_escapes_backslashes(self):
        # A template-string replacement would interpret '\g' / '\1' in the
        # proof as regex group references; the function-based replacement
        # must keep them literal.
        lean = "theorem main : True := by sorry_using []"
        out = _substitute_proof(lean, "main", "by exact ⟨1, 2⟩ \\ sorry_free")
        self.assertIn("\\ sorry_free", out)


class TestPrecompileSorryGuard(unittest.TestCase):
    """Sorry/admit submissions must be rejected BEFORE Lean runs - the
    smoke trace showed each one previously burning a full elaboration."""

    def _compiler_that_must_not_run_lean(self):
        from lean_compiler import LeanCompiler

        class Guarded(LeanCompiler):
            def _run_lean(self, code):
                raise AssertionError("Lean must not run for sorry submissions")

        return Guarded()

    def test_sorry_rejected_without_compilation(self):
        c = self._compiler_that_must_not_run_lean()
        r = c.check("by sorry", node_decl="theorem t : True := by sorry_using []")
        self.assertFalse(r.success)
        self.assertIn("WITHOUT compiling", r.errors[0])

    def test_admit_and_hidden_sorry_rejected(self):
        c = self._compiler_that_must_not_run_lean()
        for body in ("by admit",
                     "by rw [h]; sorry",
                     "by sorry_using []"):  # raw skeleton resubmission
            r = c.check(body, node_decl="theorem t : True := by sorry_using []")
            self.assertFalse(r.success, body)
            self.assertIn("Safeguard", r.raw_output)

    def test_sorry_in_aux_lemmas_also_rejected(self):
        c = self._compiler_that_must_not_run_lean()
        r = c.check("by trivial",
                    aux_lemmas="theorem bad : False := by sorry",
                    node_decl="theorem t : True := by sorry_using []")
        self.assertFalse(r.success)

    def test_clean_proof_still_compiles(self):
        from lean_compiler import LeanCompiler, CompilerResult
        # real invocation is fine for a clean proof - stub _run_lean to
        # success to keep the test hermetic
        class Ok(LeanCompiler):
            def _run_lean(self, code):
                return CompilerResult(success=True)
        r = Ok().check("by trivial", node_decl="theorem t : True := by sorry_using []")
        self.assertTrue(r.success)


class TestSignals(unittest.TestCase):
    def test_text_false_claims_are_advisory(self):
        self.assertEqual(_classify_failure("this is false, see my counterexample"),
                         ProofSignal.MODEL_SUSPECTS_WRONG)

    def test_type_mismatch_is_not_evidence(self):
        self.assertEqual(_classify_failure("type mismatch, expected Nat"),
                         ProofSignal.PROOF_TOO_HARD)

    def test_legacy_checkpoint_signal_still_deserializes(self):
        self.assertEqual(ProofSignal("statement_wrong"), ProofSignal.STATEMENT_WRONG)
        self.assertTrue(ProofSignal("statement_wrong").advisory_statement_suspect)
        self.assertTrue(ProofSignal.MODEL_SUSPECTS_WRONG.advisory_statement_suspect)
        self.assertFalse(ProofSignal.PROOF_TOO_HARD.advisory_statement_suspect)

    def test_malformed_tool_json_is_contained(self):
        class TC:
            class function:
                name = "lean_compile"
                arguments = "{not valid json"
        args = _safe_tool_args(TC())
        self.assertIn("__parse_error__", args)


class TestBlueprintValidator(unittest.TestCase):
    def test_clean_graph_passes(self):
        bp = _blueprint([_node("helper"), _node("main", ["helper"])])
        self.assertEqual(validate_blueprint(bp), [])

    def test_duplicate_names_rejected(self):
        bp = _blueprint([_node("a"), _node("a"), _node("main", ["a"])])
        errs = validate_blueprint(bp)
        self.assertTrue(any("Duplicate" in e for e in errs))

    def test_unknown_dependency_rejected(self):
        bp = _blueprint([_node("main", ["ghost"])])
        errs = validate_blueprint(bp)
        self.assertTrue(any("undeclared" in e for e in errs))

    def test_cycle_rejected(self):
        bp = _blueprint([_node("a", ["b"]), _node("b", ["a"]), _node("main", ["a"])])
        errs = validate_blueprint(bp)
        self.assertTrue(any("cycle" in e.lower() for e in errs))

    def test_self_loop_rejected(self):
        bp = _blueprint([_node("a", ["a"]), _node("main", ["a"])])
        errs = validate_blueprint(bp)
        self.assertTrue(any("cycle" in e.lower() for e in errs))

    def test_dead_node_rejected(self):
        bp = _blueprint([_node("helper"), _node("main", []), _node("orphan", ["helper"])])
        errs = validate_blueprint(bp)
        self.assertTrue(any("Dead node 'orphan'" in e for e in errs))

    def test_missing_target_rejected(self):
        bp = _blueprint([_node("helper")], target="nonexistent")
        errs = validate_blueprint(bp)
        self.assertTrue(any("Target theorem" in e for e in errs))


class TestBlueprintGate(unittest.TestCase):
    def test_structural_failure_always_raises(self):
        bp = _blueprint([_node("main", ["ghost"])])
        bp.fully_validated = True
        with self.assertRaises(BlueprintValidationError):
            _gate_blueprint(bp, allow_unvalidated=False, had_compiler=True)
        # even the debug escape cannot skip structural validation
        with self.assertRaises(BlueprintValidationError):
            _gate_blueprint(bp, allow_unvalidated=True, had_compiler=True)

    def test_unvalidated_with_compiler_raises(self):
        bp = _blueprint([_node("helper"), _node("main", ["helper"])])
        bp.fully_validated = False
        with self.assertRaises(BlueprintValidationError):
            _gate_blueprint(bp, allow_unvalidated=False, had_compiler=True)

    def test_unvalidated_escape_only_with_flag(self):
        bp = _blueprint([_node("helper"), _node("main", ["helper"])])
        bp.fully_validated = False
        _gate_blueprint(bp, allow_unvalidated=True, had_compiler=True)  # no raise

    def test_unvalidated_without_compiler_passes(self):
        # The compiler_factory-only VSB path cannot compile-validate the
        # blueprint at pipeline level (per-node checks happen inside the
        # repo environment instead) - the gate must not brick that path.
        bp = _blueprint([_node("helper"), _node("main", ["helper"])])
        bp.fully_validated = False
        _gate_blueprint(bp, allow_unvalidated=False, had_compiler=False)  # no raise

    def test_validated_blueprint_passes(self):
        bp = _blueprint([_node("helper"), _node("main", ["helper"])])
        bp.fully_validated = True
        _gate_blueprint(bp, allow_unvalidated=False, had_compiler=True)  # no raise


class TestOriginalTheoremVerification(unittest.TestCase):
    STMT = (
        "theorem putnam_1962_a1\n(S : Set (ℝ × ℝ))\n(hS : S.ncard = 5)\n"
        ": ∃ T ⊆ S, T.ncard = 4 :=\nsorry"
    )

    def test_sorry_tail_is_replaced_with_root_proof(self):
        out = _assemble_original_theorem_file(self.STMT, "", "by exact ⟨1, 2⟩")
        self.assertNotIn("sorry", out.replace("sorry_using", ""))
        self.assertIn(":= by exact ⟨1, 2⟩", out)
        # the ORIGINAL binders survive verbatim (a weakened signature would
        # be exactly the false-success mode this assembly exists to catch)
        self.assertIn("(hS : S.ncard = 5)", out)
        self.assertIn(": ∃ T ⊆ S, T.ncard = 4", out)

    def test_by_sorry_tail_is_replaced(self):
        stmt = "theorem t (n : Nat) : n = n := by\n  sorry"
        out = _assemble_original_theorem_file(stmt, "", "by rfl")
        self.assertNotIn("sorry", out)
        self.assertIn(":= by rfl", out)

    def test_aux_lemmas_precede_the_original_statement(self):
        out = _assemble_original_theorem_file(self.STMT, "theorem helper : True := by trivial", "by exact ⟨1, 2⟩")
        self.assertLess(out.index("theorem helper"), out.index("theorem putnam_1962_a1"))

    def test_no_tail_statement_gets_proof_appended(self):
        stmt = "theorem t (n : Nat) : n = n"
        out = _assemble_original_theorem_file(stmt, "", "by rfl")
        self.assertIn(":= by rfl", out)

    def test_root_proof_from_legacy_cache_form(self):
        # Old checkpoints stored ':= by ...' bodies; the assembly must still
        # produce a single assignment.
        out = _assemble_original_theorem_file(self.STMT, "", ":= by exact ⟨1, 2⟩")
        self.assertNotIn(":= := by", out)


class TestRunPhase2SuccessGate(unittest.TestCase):
    """Zero-LLM/zero-Lean integration of the success criterion: an
    all-proved run whose assembled proof does NOT check against the
    original theorem must not record success in the checkpoint."""

    def _verified_state(self) -> CheckpointState:
        from checkpoint import CheckpointState
        lean = ("@[blueprint (statement := /-- s -/)]\n"
                "theorem main : True := by sorry_using []")
        bp = Blueprint(nodes=[_node("main")], lean_file=lean, target_theorem="main")
        bp.fully_validated = True
        state = CheckpointState(theorem_stmt="theorem main : True := sorry", model="stub")
        state.set_blueprint(bp)
        state.proved_cache = {"main": "by trivial"}
        return state

    def test_all_proved_without_verification_is_not_success(self):
        import tempfile
        from checkpoint import CheckpointState
        from lean_compiler import CompilerResult
        from pipeline import run_phase2

        class FailingCompiler:
            def check(self, code, **_):
                return CompilerResult(success=False, errors=["stub: proof does not check"])

            def check_blueprint(self, code, target):
                return CompilerResult(success=True)

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ckpt.json"
            self._verified_state().save(path)
            result = run_phase2(path, compiler=FailingCompiler())
            self.assertTrue(result.all_proved())  # every node reported solved...
            reloaded = CheckpointState.load(path)
            self.assertTrue(reloaded.done)
            self.assertFalse(reloaded.success)  # ...yet success is withheld

    def test_all_proved_with_passing_verification_records_success(self):
        import tempfile
        from checkpoint import CheckpointState
        from lean_compiler import CompilerResult
        from pipeline import run_phase2

        class PassingCompiler:
            def check(self, code, **_):
                self.seen = code
                return CompilerResult(success=True)

            def check_blueprint(self, code, target):
                return CompilerResult(success=True)

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ckpt.json"
            self._verified_state().save(path)
            compiler = PassingCompiler()
            result = run_phase2(path, compiler=compiler)
            self.assertTrue(result.all_proved())
            reloaded = CheckpointState.load(path)
            self.assertTrue(reloaded.success)
            # the verification file must contain the ORIGINAL statement...
            self.assertIn("theorem main : True", compiler.seen)
            # ...with the sorry tail replaced by the cached root proof
            self.assertIn(":= by trivial", compiler.seen)


if __name__ == "__main__":
    unittest.main(verbosity=2)
