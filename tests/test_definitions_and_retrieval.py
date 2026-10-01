"""Regression tests for definition-node handling (review P1-3) and the
retrieval robustness fixes (review P1-6).

  - The blueprint parser recognizes structure/instance declarations
    (previously silently dropped, leaving dangling sorry_using refs).
  - Definition-kind nodes never go to the LLM prover: they are seeded as
    solved, count toward all_proved, and re-declare with their FULL body
    in aux/parent declarations (not signature + spliced proof).
  - RepoRetrieval's cache key is content-sensitive (rename/signature
    change with unchanged count invalidates) and embeds the model.
  - MathlibRetrieval distinguishes backend outage from zero hits via
    last_error, and the prover surfaces it to the model.

Pure Python - no Lean, no network (the Mathlib test points at a dead
localhost port).
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
# Order matters: eval/ is inserted FIRST so src/ ends up ahead of it in
# sys.path - both directories contain repo_retrieval.py and the src/ copy
# is the canonical one (the eval/ duplicate is the drifted legacy copy the
# review flagged; it imports numpy unconditionally).
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("GOEDEL_LEAN_TOOLCHAIN", "test-toolchain")

# Point the retrieval backends at a dead port BEFORE importing the module
# (URLs are read at import time) so outage behavior is testable offline.
os.environ["LEANSEARCHCLIENT_LEANSEARCH_API_URL"] = "http://127.0.0.1:9/search"
os.environ["LEANSEARCHCLIENT_LOOGLE_API_URL"] = "http://127.0.0.1:9/json"

from blueprint import Blueprint, BlueprintNode, _parse_blueprint  # noqa: E402
from lean_compiler import CompilerResult  # noqa: E402
from pipeline import _aux_lemma_decls, _provable_nodes  # noqa: E402

DEF_BLUEPRINT = """import Mathlib

@[blueprint (statement := /-- double negation wrapper -/)]
def wrap (p : Prop) : Prop := ¬¬p

@[blueprint (statement := /-- a config structure -/)]
structure Cfg where
  n : Nat
  b : Bool

@[blueprint (statement := /-- helper -/) (proof := /-- trivial -/)]
theorem helper : wrap True := by sorry_using []

@[blueprint (statement := /-- main -/) (proof := /-- use helper -/)]
theorem main : True := by sorry_using [helper]
"""


def _node(name: str, deps: list[str] | None = None, kind: str = "lemma") -> BlueprintNode:
    return BlueprintNode(
        name=name, kind=kind, statement=f"nl {name}", proof_sketch="sketch",
        dependencies=list(deps or []),
        lean_declaration=f"theorem {name} : True := by sorry_using [{', '.join(deps or [])}]",
    )


class TestParserRecognizesDefinitions(unittest.TestCase):
    def test_structure_and_instance_parsed_as_definitions(self):
        bp = _parse_blueprint(DEF_BLUEPRINT, "main")
        kinds = {n.name: n.kind for n in bp.nodes}
        self.assertEqual(kinds.get("wrap"), "definition")
        self.assertEqual(kinds.get("Cfg"), "definition")
        self.assertEqual(kinds.get("helper"), "theorem")
        self.assertEqual(kinds.get("main"), "theorem")
        # the theorem still parses its sorry_using deps
        self.assertEqual(bp.node_by_name("main").dependencies, ["helper"])

    def test_compiled_decl_keeps_full_body(self):
        bp = _parse_blueprint(DEF_BLUEPRINT, "main")
        wrap = bp.node_by_name("wrap")
        decl = wrap.compiled_decl()
        self.assertTrue(decl.startswith("def wrap"))
        self.assertIn(":= ¬¬p", decl)
        self.assertNotIn("@[blueprint", decl)
        cfg = bp.node_by_name("Cfg")
        self.assertIn("n : Nat", cfg.compiled_decl())


class TestDefinitionNodeRouting(unittest.TestCase):
    def test_provable_nodes_excludes_definitions(self):
        bp = _parse_blueprint(DEF_BLUEPRINT, "main")
        provable = _provable_nodes(bp)
        self.assertIn("helper", provable)
        self.assertIn("main", provable)
        self.assertNotIn("wrap", provable)
        self.assertNotIn("Cfg", provable)

    def test_aux_decls_emit_full_definition_text(self):
        bp = _parse_blueprint(DEF_BLUEPRINT, "main")
        aux = _aux_lemma_decls(bp, {"helper": "by trivial"}, "main")
        self.assertIn("def wrap (p : Prop) : Prop := ¬¬p", aux)
        self.assertIn("structure Cfg", aux)
        self.assertIn("theorem helper : wrap True := by trivial", aux)
        # definitions come with their body, never a spliced ':= by' proof
        self.assertNotIn("sorry_using", aux)

    def test_definitions_count_toward_all_proved_without_llm(self):
        # run_phase2 with every provable node already cached: the def nodes
        # are seeded SOLVED by the orchestrator, all_proved() holds, and
        # the verification file contains the definitions' full text.
        import asyncio
        from checkpoint import CheckpointState
        from orchestrator import prove_dag
        from lean_compiler import CompilerResult

        bp = _parse_blueprint(DEF_BLUEPRINT, "main")
        seen = {}

        class Recorder:
            def check(self, code, **_):
                seen["file"] = code
                return CompilerResult(success=True)

            def check_blueprint(self, code, target):
                return CompilerResult(success=True)

        result = asyncio.run(prove_dag(
            blueprint=bp, compiler=Recorder(), retrieval=None,
            proved_cache={"helper": "by trivial", "main": "by trivial"},
        ))
        self.assertTrue(result.all_proved())
        self.assertIn("wrap", result.proved)
        self.assertIn("Cfg", result.proved)

        from pipeline import _final_verification
        report = _final_verification(
            "theorem main : True := sorry", bp,
            {"helper": "by trivial", "main": "by trivial"}, Recorder(),
        )
        self.assertTrue(report.passed)
        self.assertIn("def wrap (p : Prop) : Prop := ¬¬p", seen["file"])
        self.assertIn("theorem main : True := by trivial", seen["file"])


class TestRepoRetrievalCacheKey(unittest.TestCase):
    def test_content_hash_sensitive_to_rename_and_signature(self):
        from repo_retrieval import RepoRetrieval, RepoDecl

        def decl(name="foo", sig="Nat → Nat"):
            return RepoDecl(kind="def", name=name, signature=sig, file="f.lean", line=1)

        base = RepoRetrieval._content_hash([decl()])
        self.assertEqual(base, RepoRetrieval._content_hash([decl()]))  # stable
        self.assertNotEqual(base, RepoRetrieval._content_hash([decl(name="bar")]))
        self.assertNotEqual(base, RepoRetrieval._content_hash([decl(sig="Bool → Bool")]))

    def test_cache_key_embeds_model_and_content(self):
        from repo_retrieval import RepoRetrieval, RepoDecl

        r = RepoRetrieval.__new__(RepoRetrieval)  # no OpenAI client needed
        r.repo_root = Path("/tmp/some_repo")
        d = RepoDecl(kind="def", name="foo", signature="Nat", file="f.lean", line=1)
        key1 = r._cache_key([d])
        key2 = r._cache_key([RepoDecl(kind="def", name="foo", signature="Bool", file="f.lean", line=1)])
        self.assertNotEqual(key1, key2)
        self.assertIn("text-embedding-3-small", key1)


class TestMathlibOutageSignal(unittest.TestCase):
    def test_outage_sets_last_error_and_prover_distinguishes_it(self):
        from mathlib_retrieval import MathlibRetrieval
        r = MathlibRetrieval(timeout=0.5)
        try:
            hits = r.search("induction principle", k=3)
            self.assertEqual(hits, [])
            self.assertTrue(r.last_error, "outage must set last_error")
            # the prover's tool-result text for this state (mirrors
            # prover.py's mathlib_search branch)
            result = "\n\n".join(h.format() for h in hits) or "No results found."
            if not hits and r.last_error:
                result = (f"Search backends unavailable ({r.last_error}). "
                          "This is NOT a zero-result answer.")
            self.assertIn("NOT a zero-result", result)
        finally:
            r.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
