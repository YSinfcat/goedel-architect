"""Tests for the dependency-DAG visualizer (review IV.1).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("GOEDEL_LEAN_TOOLCHAIN", "test-toolchain")

from blueprint import _parse_blueprint  # noqa: E402
import dag_viz  # noqa: E402
from dag_viz import (assign_positions, blocker_scores, build_viz_nodes,  # noqa: E402
                     critical_path, diff_graphs, generate_html)

LEAN = ("@[blueprint (statement := /-- base -/)]\n"
        "theorem base : True := by sorry_using []\n\n"
        "@[blueprint (statement := /-- mid -/)]\n"
        "theorem mid : True := by sorry_using [base]\n\n"
        "@[blueprint (statement := /-- main -/)]\n"
        "theorem main : True := by sorry_using [mid, base]\n")


def _viz(lean: str = LEAN, statuses=None):
    bp = _parse_blueprint(lean, "main")
    nodes = build_viz_nodes(bp, statuses or {}, [], {})
    return bp, nodes


class TestLayout(unittest.TestCase):
    def test_levels_follow_dependencies(self):
        _, nodes = _viz()
        assign_positions(nodes)
        by = {n.name: n for n in nodes}
        self.assertEqual(by["base"].level, 1)
        self.assertEqual(by["mid"].level, 2)
        self.assertEqual(by["main"].level, 3)
        self.assertLess(by["base"].x, by["main"].x)

    def test_critical_path_picks_longest_chain(self):
        _, nodes = _viz()
        # main -> mid -> base is longer than main -> base
        crit = critical_path(nodes, "main")
        self.assertEqual(crit, {"main", "mid", "base"})

    def test_blocker_scores_count_transitive_dependents(self):
        _, nodes = _viz()
        blocker_scores(nodes)
        by = {n.name: n for n in nodes}
        self.assertEqual(by["base"].blockers, 2)   # mid and main depend on it
        self.assertEqual(by["mid"].blockers, 1)    # only main
        self.assertEqual(by["main"].blockers, 0)

    def test_statuses_from_checkpoint_kinds(self):
        _, nodes = _viz(statuses={
            "base": ("proved", "by trivial"),
            "mid": ("infra", ""),
            "main": ("suspect", ""),
        })
        by = {n.name: n for n in nodes}
        self.assertEqual(by["base"].status, "proved")
        self.assertEqual(by["mid"].status, "infra")
        self.assertEqual(by["main"].status, "suspect")


class TestDiff(unittest.TestCase):
    def test_diff_marks_added_changed_removed(self):
        initial = _parse_blueprint(LEAN, "main")
        edited = LEAN.replace("theorem mid : True", "theorem mid : True and True")
        final = _parse_blueprint(edited + "\n@[blueprint (statement := /-- x -/)]\n"
                                 "theorem extra : True := by sorry_using [base]\n",
                                 "main")
        diff_graphs(initial, final)
        by = {n.name: n for n in final.nodes}
        self.assertEqual(by["mid"].diff, "changed")
        self.assertEqual(by["extra"].diff, "added")
        removed = [n for n in final.nodes if "(removed)" in n.kind]
        # nothing removed in this scenario (base/main survive)
        self.assertEqual(removed, [])


class TestHtml(unittest.TestCase):
    def test_checkpoint_mode_generates_self_contained_html(self):
        from checkpoint import CheckpointState
        bp = _parse_blueprint(LEAN, "main")
        bp.fully_validated = True
        state = CheckpointState(theorem_stmt="theorem main : True := sorry",
                                 model="stub")
        state.set_blueprint(bp)
        state.proved_cache = {"base": "by trivial"}
        state.node_results = {
            "mid": {"signal": "infra_error", "proof_body": "", "analysis": "",
                    "suggested_fix": "", "lean_errors": []},
            "main": {"signal": "model_suspects_wrong", "proof_body": "",
                     "analysis": "", "suggested_fix": "", "lean_errors": []},
        }
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "ckpt.json"
            state.save(ckpt)
            out = dag_viz.build(ckpt, None, None)
            html = out.read_text()
            self.assertEqual(out.name, "ckpt.dag.html")
            for name in ("base", "mid", "main"):
                self.assertIn(name, html)
            # statuses made it into the embedded data
            self.assertIn("proved", html)
            self.assertIn("infra", html)
            self.assertIn("suspect", html)
            # no CDN dependency and no raw breakout inside the DATA line
            self.assertNotIn("https://", html)
            data_line = next(l for l in html.splitlines()
                             if l.lstrip().startswith("const DATA"))
            self.assertNotIn("</script>", data_line)
            # critical path computed and embedded
            self.assertIn("on_critical_path", html)

    def test_bundle_mode_reads_json_files(self):
        with tempfile.TemporaryDirectory() as d:
            bundle = Path(d)
            (bundle / "blueprint.final.json").write_text(json.dumps({
                "target_theorem": "main",
                "nodes": [
                    {"name": "base", "kind": "lemma", "statement": "s",
                     "proof_sketch": "", "dependencies": [],
                     "lean_declaration": "theorem base : True := by sorry_using []"},
                    {"name": "main", "kind": "theorem", "statement": "s",
                     "proof_sketch": "", "dependencies": ["base"],
                     "lean_declaration": "theorem main : True := by sorry_using [base]"},
                ]}))
            out = dag_viz.build(bundle, None, None)
            html = out.read_text()
            self.assertIn("success bundle", html)
            self.assertIn('"status": "proved"', html)


class TestTraceStats(unittest.TestCase):
    def test_per_node_stats_from_trace(self):
        with tempfile.TemporaryDirectory() as d:
            trace = Path(d) / "t.jsonl"
            events = [
                {"kind": "tool_call", "thm_name": "mid", "tool_name": "lean_compile"},
                {"kind": "tool_call", "thm_name": "mid", "tool_name": "lean_compile"},
                {"kind": "tool_call", "thm_name": "mid", "tool_name": "repo_search"},
                {"kind": "final_verify", "thm_name": "mid", "ok": True,
                 "args": {"wall_time_s": 3.5}},
                {"kind": "llm_usage", "thm_name": "mid", "turn": 1,
                 "args": {"total_tokens": 700}},
                {"kind": "llm_usage", "thm_name": "mid", "turn": 2,
                 "args": {"total_tokens": 300}},
            ]
            trace.write_text("\n".join(json.dumps(e) for e in events))
            stats = dag_viz._node_stats(trace)
            self.assertEqual(stats["mid"]["compile_calls"], 2)
            self.assertEqual(stats["mid"]["wall_time_s"], 3.5)
            self.assertEqual(stats["mid"]["tokens"], 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
