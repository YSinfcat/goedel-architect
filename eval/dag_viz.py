"""Dependency-DAG visualizer (review IV.1).

The existing graph_viz.py renders the tool-call timeline; this one renders
the MATHEMATICAL dependency graph. Inputs:

  - an artifact bundle directory (produced by --artifacts): renders the
    final DAG plus an initial-vs-final structural diff;
  - or a checkpoint JSON (in-progress / failed runs): renders statuses
    straight from proved_cache and node_results - the view where
    proved/failed/infra/suspected-false coloring actually matters.

Everything computational happens in Python (layout levels, critical path,
blocker scores, diff classification, per-node stats from the trace); the
embedded JS only renders SVG and handles clicks/toggles. No CDN - the
output is a single self-contained HTML file.

Usage:
    python eval/dag_viz.py results/artifacts/main/            # bundle
    python eval/dag_viz.py checkpoints/foo.json --trace t.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from blueprint import Blueprint, BlueprintNode, _parse_blueprint  # noqa: E402
from checkpoint import CheckpointState  # noqa: E402
from lineage import statement_hash  # noqa: E402

SIGNAL_CLASS = {
    "solved": "proved",
    "proof_too_hard": "failed",
    "model_suspects_wrong": "suspect",
    "statement_wrong": "suspect",
    "formally_negated": "negated",
    "infra_error": "infra",
}

COL_WIDTH = 190
ROW_HEIGHT = 92
NODE_W = 150
NODE_H = 54


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

@dataclass
class VizNode:
    name: str
    kind: str
    status: str                  # proved | failed | suspect | negated | infra | definition | unattempted
    deps: list[str]
    statement: str = ""
    proof_sketch: str = ""
    lean_declaration: str = ""
    proof: str = ""
    lineage_op: str = ""
    revision: int = 0
    compile_calls: int = 0
    wall_time_s: float | None = None
    tokens: int = 0
    diff: str = ""               # added | removed | changed | unchanged | ""
    x: int = 0
    y: int = 0
    level: int = 0
    on_critical_path: bool = False
    blockers: int = 0            # transitive dependents count

    def to_json(self) -> dict:
        return {
            "name": self.name, "kind": self.kind, "status": self.status,
            "deps": self.deps, "statement": self.statement,
            "proof_sketch": self.proof_sketch,
            "lean_declaration": self.lean_declaration, "proof": self.proof,
            "lineage_op": self.lineage_op, "revision": self.revision,
            "compile_calls": self.compile_calls,
            "wall_time_s": self.wall_time_s, "tokens": self.tokens,
            "diff": self.diff, "x": self.x, "y": self.y,
            "level": self.level, "on_critical_path": self.on_critical_path,
            "blockers": self.blockers,
        }


def _node_stats(trace_path: Path | None) -> dict[str, dict]:
    """Per-node (thm_name-keyed) compile counts, wall time, tokens."""
    stats: dict[str, dict] = {}
    if trace_path is None or not Path(trace_path).exists():
        return stats
    usage_by_turn: dict[tuple, int] = {}
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            thm = ev.get("thm_name", "")
            bucket = stats.setdefault(thm, {"compile_calls": 0,
                                            "wall_time_s": None, "tokens": 0})
            kind = ev.get("kind")
            if kind == "tool_call" and ev.get("tool_name") == "lean_compile":
                bucket["compile_calls"] += 1
            elif kind == "final_verify" and ev.get("ok") is not None:
                wall = (ev.get("args") or {}).get("wall_time_s")
                if wall is not None:
                    bucket["wall_time_s"] = wall
            elif kind == "llm_usage":
                usage_by_turn[(thm, ev.get("turn"))] = \
                    (ev.get("args") or {}).get("total_tokens", 0) or 0
    for (thm, _), tokens in usage_by_turn.items():
        if thm in stats:
            stats[thm]["tokens"] += tokens
    return stats


def load_bundle(bundle_dir: Path) -> tuple[Blueprint, Blueprint | None, list, Path | None]:
    final_json = bundle_dir / "blueprint.final.json"
    initial_json = bundle_dir / "blueprint.initial.json"
    if not final_json.exists():
        raise SystemExit(f"No blueprint.final.json in {bundle_dir} - is this an artifact bundle?")
    final = _bp_from_json(json.loads(final_json.read_text()))
    initial = (_bp_from_json(json.loads(initial_json.read_text()))
               if initial_json.exists() else None)
    lineage = json.loads((bundle_dir / "lineage.json").read_text()) \
        if (bundle_dir / "lineage.json").exists() else []
    trace = bundle_dir / "trace.jsonl"
    return final, initial, lineage, (trace if trace.exists() else None)


def load_checkpoint(path: Path) -> tuple[Blueprint, Blueprint | None, list, Path | None, CheckpointState]:
    state = CheckpointState.load(path)
    bp = state.get_blueprint()
    if bp is None:
        raise SystemExit(f"Checkpoint {path} has no blueprint.")
    return bp, None, state.lineage_history or [], None, state


def _bp_from_json(payload: dict) -> Blueprint:
    nodes = [
        BlueprintNode(
            name=n["name"], kind=n["kind"], statement=n.get("statement", ""),
            proof_sketch=n.get("proof_sketch", ""),
            dependencies=n.get("dependencies", []),
            lean_declaration=n.get("lean_declaration", ""),
        )
        for n in payload.get("nodes", [])
    ]
    return Blueprint(nodes=nodes, lean_file="",
                     target_theorem=payload.get("target_theorem", ""))


# ---------------------------------------------------------------------------
# Graph computations (pure, testable)
# ---------------------------------------------------------------------------

def assign_levels(nodes: list[VizNode]) -> None:
    by_name = {n.name: n for n in nodes}
    level: dict[str, int] = {}

    def lvl(name: str, seen: frozenset = frozenset()) -> int:
        if name in level:
            return level[name]
        if name in seen:      # cycle guard; validator should have caught it
            return 0
        node = by_name.get(name)
        deps = node.deps if node else []
        level[name] = 1 + max((lvl(d, seen | {name}) for d in deps), default=0)
        return level[name]

    for n in nodes:
        n.level = lvl(n.name)


def assign_positions(nodes: list[VizNode]) -> None:
    assign_levels(nodes)
    by_level: dict[int, list[VizNode]] = {}
    for n in nodes:
        by_level.setdefault(n.level, []).append(n)
    max_count = max((len(v) for v in by_level.values()), default=1)
    for lvl, group in by_level.items():
        group.sort(key=lambda n: n.name)
        for i, n in enumerate(group):
            n.x = (lvl - 1) * COL_WIDTH + 30
            n.y = i * ROW_HEIGHT + 30


def critical_path(nodes: list[VizNode], target: str) -> set[str]:
    """Longest dependency chain (by node count) ending at the target."""
    by_name = {n.name: n for n in nodes}
    best: dict[str, tuple[int, list[str]]] = {}

    def walk(name: str, seen: frozenset) -> tuple[int, list[str]]:
        if name in best:
            return best[name]
        if name in seen:
            return (0, [])
        node = by_name.get(name)
        deps = node.deps if node else []
        if not deps:
            best[name] = (1, [name])
            return best[name]
        length, chain = max((walk(d, seen | {name}) for d in deps),
                            key=lambda t: t[0])
        best[name] = (length + 1, chain + [name])
        return best[name]

    _, chain = walk(target, frozenset())
    return set(chain)


def blocker_scores(nodes: list[VizNode]) -> None:
    """For each node: how many OTHER nodes transitively depend on it."""
    dependents: dict[str, set[str]] = {n.name: set() for n in nodes}
    for n in nodes:
        for dep in n.deps:
            dependents.setdefault(dep, set()).add(n.name)

    def transitive(name: str, seen: frozenset) -> set[str]:
        out: set[str] = set()
        for d in dependents.get(name, ()):
            if d in seen:
                continue
            out.add(d)
            out |= transitive(d, seen | {d})
        return out

    for n in nodes:
        n.blockers = len(transitive(n.name, frozenset()))


def diff_graphs(initial: Blueprint | None, final: Blueprint) -> None:
    """Mark final-view diff status via statement hashes."""
    if initial is None:
        return
    init_hashes = {statement_hash(n): n.name for n in initial.nodes}
    init_names = {n.name for n in initial.nodes}
    final_hashes = {statement_hash(n) for n in final.nodes}
    for n in final.nodes:
        h = statement_hash(n)
        if h in init_hashes:
            n_diff = "unchanged" if init_hashes[h] == n.name else "renamed"
        elif n.name in init_names:
            n_diff = "changed"
        else:
            n_diff = "added"
        n.diff = n_diff
    # removed nodes: present initially, absent finally (by hash and name)
    for n in initial.nodes:
        h = statement_hash(n)
        if h not in final_hashes and n.name not in {fn.name for fn in final.nodes}:
            # surface removed nodes as data-only entries
            final.nodes.append(BlueprintNode(
                name=n.name, kind=n.kind + " (removed)", statement=n.statement,
                proof_sketch=n.proof_sketch, dependencies=n.dependencies,
                lean_declaration=n.lean_declaration))


# ---------------------------------------------------------------------------
# VizNode construction
# ---------------------------------------------------------------------------

def build_viz_nodes(
    blueprint: Blueprint,
    statuses: dict[str, tuple[str, str]],   # name -> (status, proof)
    lineage_rounds: list,
    stats: dict[str, dict],
) -> list[VizNode]:
    last_lineage = {}
    for rnd in lineage_rounds:
        for entry in rnd:
            last_lineage[entry["name"]] = entry
    out: list[VizNode] = []
    for node in blueprint.nodes:
        status, proof = statuses.get(node.name, ("unattempted", ""))
        if node.kind == "definition":
            status = "definition"
        entry = last_lineage.get(node.name, {})
        st = stats.get(node.name, {})
        out.append(VizNode(
            name=node.name, kind=node.kind, status=status,
            deps=list(node.dependencies), statement=node.statement,
            proof_sketch=node.proof_sketch,
            lean_declaration=node.lean_declaration, proof=proof,
            lineage_op=entry.get("operation", ""), revision=entry.get("revision", 0),
            compile_calls=st.get("compile_calls", 0),
            wall_time_s=st.get("wall_time_s"),
            tokens=st.get("tokens", 0),
        ))
    return out


def statuses_from_checkpoint(state: CheckpointState) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for name, body in state.proved_cache.items():
        out[name] = ("proved", body)
    for name, d in state.node_results.items():
        signal = d.get("signal", "")
        if signal == "solved" and name not in out:
            out[name] = ("proved", d.get("proof_body", ""))
        elif signal != "solved":
            out[name] = (SIGNAL_CLASS.get(signal, "failed"), d.get("proof_body", ""))
    return out


def statuses_from_bundle(bundle_final: Blueprint) -> dict[str, tuple[str, str]]:
    # A success bundle's final graph is all-proved by construction.
    return {n.name: ("proved", "") for n in bundle_final.nodes
            if n.kind != "definition"}


# ---------------------------------------------------------------------------
# HTML emission
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DAG — __TITLE__</title>
<style>
  body { font-family: -apple-system, "Helvetica Neue", sans-serif; margin: 0;
         background: #faf8f4; color: #222; }
  header { padding: 10px 16px; background: #2b3a8f; color: #fff; }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; }
  header .meta { font-size: 11px; opacity: .8; margin-top: 2px; }
  #wrap { display: flex; height: calc(100vh - 44px); }
  #svgwrap { flex: 1; overflow: auto; }
  #detail { width: 360px; border-left: 1px solid #ddd; padding: 14px;
            overflow-y: auto; font-size: 12px; background: #fff; }
  .node rect { stroke-width: 1.5; cursor: pointer; }
  .node text { font-size: 10px; pointer-events: none; }
  .node.proved rect { fill: #d8eed8; stroke: #2e7d32; }
  .node.failed rect { fill: #f8d7d7; stroke: #b02a2a; }
  .node.suspect rect { fill: #fdf1d0; stroke: #b8860b; }
  .node.negated rect { fill: #f0d0f0; stroke: #8e24aa; }
  .node.infra rect { fill: #e3e7ef; stroke: #5c6bc0; stroke-dasharray: 4 2; }
  .node.definition rect { fill: #e8e8e8; stroke: #777; }
  .node.unattempted rect { fill: #fff; stroke: #999; }
  .node.critical rect { stroke: #b02a2a; stroke-width: 3; }
  .node.removed rect { fill: #f0f0f0; stroke: #ccc; stroke-dasharray: 3 3; }
  .edge { stroke: #889; stroke-width: 1.2; fill: none; }
  .edge.critical { stroke: #b02a2a; stroke-width: 2.6; }
  .badge { font-size: 8px; fill: #555; }
  .controls { padding: 8px 16px; background: #f0ede6; font-size: 12px;
              display: flex; gap: 18px; align-items: center; }
  .controls label { cursor: pointer; }
  pre { background: #f6f4ef; padding: 8px; overflow-x: auto;
        white-space: pre-wrap; font-size: 11px; }
  h2 { font-size: 13px; margin: 4px 0; }
  .pill { display: inline-block; padding: 1px 7px; border-radius: 9px;
          font-size: 10px; background: #eee; margin-right: 4px; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="meta">__META__</div>
</header>
<div class="controls">
  <label><input type="checkbox" id="anc"> only root ancestors (hide dead nodes)</label>
  <label><input type="checkbox" id="diff" __DIFFCHECK__> highlight initial→final diff</label>
  <span style="color:#666" id="legend"></span>
</div>
<div id="wrap">
  <div id="svgwrap"></div>
  <div id="detail"><p style="color:#888">click a node</p></div>
</div>
<script>
const DATA = __DATA__;
const EDGES = __EDGES__;
const svgNS = "http://www.w3.org/2000/svg";
const NODE_W = __NODE_W__, NODE_H = __NODE_H__;

const svg = document.createElementNS(svgNS, "svg");
svg.setAttribute("width", W); svg.setAttribute("height", H);
document.getElementById("svgwrap").appendChild(svg);

const nodeMap = {}; DATA.forEach(n => nodeMap[n.name] = n);

function visible(n) {
  if (!document.getElementById("anc").checked) return true;
  return n.ancestor_of_target || n.name === DATA.target;
}

function draw() {
  svg.textContent = "";
  const showDiff = document.getElementById("diff").checked;
  const W2 = NODE_W / 2, H2 = NODE_H / 2;
  EDGES.forEach(e => {
    const a = nodeMap[e.from], b = nodeMap[e.to];
    if (!a || !b || !visible(a) || !visible(b)) return;
    const p = document.createElementNS(svgNS, "path");
    const x1 = a.x + W2, y1 = a.y + H2, x2 = b.x + W2, y2 = b.y + H2;
    const mx = (x1 + x2) / 2;
    p.setAttribute("d", `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`);
    p.setAttribute("class", "edge" + (e.critical ? " critical" : ""));
    svg.appendChild(p);
  });
  DATA.forEach(n => {
    if (!visible(n)) return;
    const g = document.createElementNS(svgNS, "g");
    g.setAttribute("class", "node " + n.status
      + (n.on_critical_path ? " critical" : "")
      + (n.diff === "removed" || n.kind.endsWith("(removed)") ? " removed" : ""));
    const r = document.createElementNS(svgNS, "rect");
    r.setAttribute("x", n.x); r.setAttribute("y", n.y);
    r.setAttribute("width", NODE_W); r.setAttribute("height", NODE_H);
    r.setAttribute("rx", 7);
    g.appendChild(r);
    const label = document.createElementNS(svgNS, "text");
    label.setAttribute("x", n.x + NODE_W / 2); label.setAttribute("y", n.y + 24);
    label.setAttribute("text-anchor", "middle");
    label.textContent = n.name.length > 22 ? n.name.slice(0, 21) + "…" : n.name;
    g.appendChild(label);
    const sub = document.createElementNS(svgNS, "text");
    sub.setAttribute("x", n.x + NODE_W / 2); sub.setAttribute("y", n.y + 40);
    sub.setAttribute("text-anchor", "middle");
    sub.setAttribute("class", "badge");
    const bits = [n.status];
    if (showDiff && n.diff) bits.push(n.diff);
    if (n.lineage_op && n.lineage_op !== "unchanged") bits.push(n.lineage_op + " r" + n.revision);
    if (n.blockers > 0) bits.push("▾" + n.blockers);
    sub.textContent = bits.join(" · ");
    g.appendChild(sub);
    g.addEventListener("click", () => showDetail(n));
    svg.appendChild(g);
  });
}

function showDetail(n) {
  const d = document.getElementById("detail");
  d.textContent = "";
  const title = document.createElement("h2");
  title.textContent = n.name + "  (" + n.kind + ")";
  d.appendChild(title);
  const pills = document.createElement("div");
  [["status", n.status], ["diff", n.diff], ["lineage", n.lineage_op || "—"],
   ["revision", n.revision || "—"], ["compiles", n.compile_calls],
   ["wall_s", n.wall_time_s == null ? "—" : n.wall_time_s],
   ["tokens", n.tokens], ["blocks ↓", n.blockers]].forEach(([k, v]) => {
    const s = document.createElement("span");
    s.className = "pill";
    s.textContent = k + ": " + v;
    pills.appendChild(s);
  });
  d.appendChild(pills);
  const add = (heading, text) => {
    if (!text) return;
    const h = document.createElement("h2"); h.textContent = heading;
    d.appendChild(h);
    const pre = document.createElement("pre");
    pre.textContent = text;      // textContent only - no HTML injection
    d.appendChild(pre);
  };
  add("NL statement", n.statement);
  add("Proof sketch", n.proof_sketch);
  add("Formal declaration", n.lean_declaration);
  add("Proof", n.proof);
  add("Dependencies", n.deps.join(", "));
}

// ancestors-of-target marking for the dead-node filter
(function () {
  const depsOf = {}; DATA.forEach(n => depsOf[n.name] = n.deps);
  const target = DATA.target;
  const anc = new Set([target]);
  let grew = true;
  while (grew) {
    grew = false;
    DATA.forEach(n => {
      if (anc.has(n.name)) return;
      if (n.deps.some(d => anc.has(d))) { anc.add(n.name); grew = true; }
    });
  }
  DATA.forEach(n => n.ancestor_of_target = anc.has(n.name));
})();

document.getElementById("anc").addEventListener("change", draw);
document.getElementById("diff").addEventListener("change", draw);
document.getElementById("legend").textContent =
  "proved ▪ failed ▪ suspect ▪ infra ▪ definition";
draw();
</script>
</body>
</html>
"""


def generate_html(
    nodes: list[VizNode],
    edges: list[dict],
    target: str,
    title: str,
    meta: str,
    has_diff: bool,
) -> str:
    width = max((n.x for n in nodes), default=0) + NODE_W + 60
    height = max((n.y for n in nodes), default=0) + NODE_H + 60
    payload = [n.to_json() for n in nodes]
    data_json = json.dumps({"target": target, "nodes": payload},
                           ensure_ascii=False, indent=None)
    edges_json = json.dumps(edges, ensure_ascii=False)
    # JSON-in-<script> discipline (see graph_viz.generate)
    data_json = (data_json.replace("<", "\\u003c").replace(">", "\\u003e")
                 .replace("&", "\\u0026")
                 .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))
    edges_json = (edges_json.replace("<", "\\u003c").replace(">", "\\u003e")
                  .replace("&", "\\u0026")
                  .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))
    return (HTML_TEMPLATE
            .replace("__TITLE__", title)
            .replace("__META__", meta)
            .replace("__DIFFCHECK__", "checked" if has_diff else "")
            .replace("__DATA__", data_json)
            .replace("__EDGES__", edges_json)
            .replace("__WIDTH__", str(width))
            .replace("__HEIGHT__", str(height))
            .replace("__NODE_W__", str(NODE_W))
            .replace("__NODE_H__", str(NODE_H)))


def build(source: Path, trace_override: Path | None, out: Path | None) -> Path:
    if (source / "blueprint.final.json").exists():
        blueprint, initial, lineage, trace = load_bundle(source)
        statuses = statuses_from_bundle(blueprint)
        title = f"{blueprint.target_theorem} (success bundle)"
        meta = (f"{len(blueprint.nodes)} nodes · bundle {source.name}"
                + (f" · trace {'bundled' if trace else 'absent'}"))
        has_diff = initial is not None
    elif source.suffix == ".json":
        blueprint, initial, lineage, trace, state = load_checkpoint(source)
        statuses = statuses_from_checkpoint(state)
        title = f"{blueprint.target_theorem} (checkpoint, success={state.success})"
        meta = (f"iteration {state.iteration} · {len(state.proved_cache)} proved"
                f" · {len(lineage)} lineage rounds")
        has_diff = False
    else:
        raise SystemExit(f"{source} is neither an artifact bundle nor a checkpoint JSON.")
    trace = trace_override or trace
    stats = _node_stats(trace)

    viz_nodes = build_viz_nodes(blueprint, statuses, lineage, stats)
    diff_graphs(initial, blueprint)
    # re-sync viz nodes after diff may have appended removed markers
    viz_nodes = build_viz_nodes(blueprint, statuses, lineage, stats)
    assign_positions(viz_nodes)
    blocker_scores(viz_nodes)
    crit = critical_path(viz_nodes, blueprint.target_theorem)
    for n in viz_nodes:
        n.on_critical_path = n.name in crit
    edges = [{"from": d, "to": n.name, "critical": n.name in crit and d in crit}
             for n in viz_nodes for d in n.deps]
    html = generate_html(viz_nodes, edges, blueprint.target_theorem,
                         title, meta, has_diff)
    out = out or (source / "dag.html" if source.is_dir()
                  else source.with_suffix(".dag.html"))
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Dependency-DAG visualizer")
    ap.add_argument("source", help="artifact bundle dir or checkpoint .json")
    ap.add_argument("--trace", type=Path, default=None,
                    help="trace.jsonl for per-node stats (defaults to bundled trace)")
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()
    out = build(Path(args.source), args.trace, args.output)
    print(f"DAG report written to {out}")


if __name__ == "__main__":
    main()
