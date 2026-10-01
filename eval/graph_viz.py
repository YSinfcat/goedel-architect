"""
Generate a self-contained interactive HTML visualization from a JSONL trace file.

Layout:
  Left sidebar  — list of all theorems (PASS/FAIL badges)
  Center panel  — vis.js step graph for selected theorem
  Right panel   — detail view when a node is clicked

Usage:
    python eval/graph_viz.py trace.jsonl          # writes trace.html
    python eval/graph_viz.py trace.jsonl out.html  # custom output path
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trace(path: Path) -> dict[str, list[dict]]:
    """Read the JSONL trace and group events by thm_name."""
    groups: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            thm = event.get("thm_name", "unknown")
            groups.setdefault(thm, []).append(event)
    return groups


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def _esc(s: str) -> str:
    """Minimal HTML escape for detail panel content."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_theorem_data(thm_name: str, events: list[dict]) -> dict:
    """
    Convert trace events for one theorem into vis.js graph data + summary.
    Returns a dict with keys: thm_name, lean_root, thm_stmt, ok, wall_time_s,
    tool_calls_used, graph {nodes, edges, nodeDetails}.
    """
    thm_start = next((e for e in events if e["kind"] == "theorem_start"), {})
    start_args = thm_start.get("args") or {}
    lean_root = start_args.get("lean_root", "")
    thm_stmt = start_args.get("thm_stmt", "")
    rel_path = start_args.get("rel_path", "")

    final = next((e for e in events if e["kind"] == "final_verify"), None)
    final_args = (final or {}).get("args") or {}
    ok = bool(final.get("ok")) if final else False
    wall_time_s = final_args.get("wall_time_s", 0)
    final_proof = final_args.get("proof", "")
    final_error = final_args.get("error", "")

    # ---- collect calls and results ----
    calls: dict[str, dict] = {}      # call_id -> tool_call event
    results: dict[str, dict] = {}    # call_id -> tool_result event
    for e in events:
        if e["kind"] == "tool_call" and e.get("call_id"):
            calls[e["call_id"]] = e
        elif e["kind"] == "tool_result" and e.get("call_id"):
            results[e["call_id"]] = e

    # Count from events (more reliable than what the prover returns)
    tool_calls_used = len(calls)

    # Group call_ids by turn number (preserving order within turn)
    calls_by_turn: dict[int, list[str]] = {}
    for call_id, ev in calls.items():
        t = ev.get("turn", 0)
        calls_by_turn.setdefault(t, []).append(call_id)

    # ---- build vis.js nodes / edges ----
    nodes: list[dict] = []
    edges: list[dict] = []
    node_details: dict[str, dict] = {}  # id -> {title, content (HTML)}

    # Root theorem node
    stmt_html = _esc(thm_stmt)
    nodes.append({
        "id": "thm",
        "label": thm_name.split(".")[-1] + f"\n[{lean_root}]",
        "shape": "box",
        "color": {"background": "#90CAF9", "border": "#1565C0"},
        "font": {"color": "#000", "size": 12},
        "level": 0,
    })
    node_details["thm"] = {
        "title": f"Theorem: {thm_name}",
        "content": (
            f"<b>Repo:</b> {_esc(lean_root)}<br>"
            f"<b>File:</b> {_esc(rel_path)}<br><br>"
            f"<b>Statement:</b><pre>{stmt_html}</pre>"
        ),
    }

    # Turn-by-turn nodes
    lean_count = 0
    search_count = 0
    prev_result_ids: list[str] = ["thm"]

    for turn_num in sorted(calls_by_turn):
        call_ids = calls_by_turn[turn_num]
        cur_result_ids: list[str] = []
        level_call = turn_num * 2 + 1
        level_result = turn_num * 2 + 2

        for call_id in call_ids:
            call_ev = calls[call_id]
            result_ev = results.get(call_id, {})
            tool = call_ev.get("tool_name", "unknown")
            call_args = call_ev.get("args") or {}
            result_text = result_ev.get("result") or ""
            is_ok = bool(result_ev.get("ok"))

            # ---- call node ----
            call_node_id = f"call:{call_id}"
            if tool == "lean_compile":
                lean_count += 1
                proof = call_args.get("proof_body", "")
                short_proof = proof.strip().replace("\n", " ")[:35]
                if len(proof.strip()) > 35:
                    short_proof += "…"
                label = f"lean_compile #{lean_count}\n{short_proof}"
                if is_ok:
                    bg, border = "#C8E6C9", "#2E7D32"
                else:
                    bg, border = "#FFCDD2", "#C62828"
                detail_content = (
                    f"<b>lean_compile #{lean_count}</b><br>"
                    f"<b>Status:</b> {'✓ SUCCESSFUL' if is_ok else '✗ FAILED'}<br><br>"
                    f"<b>Proof body:</b><pre>{_esc(proof)}</pre>"
                )
                if call_args.get("aux_lemmas"):
                    detail_content += f"<b>Aux lemmas:</b><pre>{_esc(call_args['aux_lemmas'])}</pre>"
            elif tool == "mathlib_search":
                search_count += 1
                query = call_args.get("query", "")
                short_q = query[:30] + ("…" if len(query) > 30 else "")
                label = f"search #{search_count}\n\"{short_q}\""
                bg, border = "#FFF9C4", "#F57F17"
                detail_content = (
                    f"<b>mathlib_search #{search_count}</b><br>"
                    f"<b>Query:</b> {_esc(query)}<br>"
                    f"<b>k:</b> {call_args.get('k', 10)}"
                )
            else:
                label = tool
                bg, border = "#E0E0E0", "#757575"
                detail_content = f"<b>{_esc(tool)}</b>"

            nodes.append({
                "id": call_node_id,
                "label": label,
                "shape": "box",
                "color": {"background": bg, "border": border},
                "font": {"face": "monospace", "size": 11},
                "level": level_call,
            })
            node_details[call_node_id] = {
                "title": f"Turn {turn_num} — {tool}",
                "content": detail_content,
            }

            # ---- result node ----
            result_node_id = f"result:{call_id}"
            if tool == "lean_compile":
                if is_ok:
                    r_label = "✓ OK"
                    r_bg, r_border = "#81C784", "#2E7D32"
                else:
                    first_line = result_text.split("\n")[0].strip()[:40]
                    r_label = f"✗ {first_line}"
                    r_bg, r_border = "#EF9A9A", "#C62828"
                r_detail = (
                    f"<b>Lean result:</b><br>"
                    f"<pre>{_esc(result_text[:3000])}</pre>"
                )
            elif tool == "mathlib_search":
                n_hits = result_text.count("\n\n") + 1 if result_text and result_text != "No results found." else 0
                r_label = f"{n_hits} hit{'s' if n_hits != 1 else ''}"
                r_bg, r_border = "#FFF59D", "#F9A825"
                r_detail = f"<b>Search results:</b><pre>{_esc(result_text[:3000])}</pre>"
            else:
                r_label = "result"
                r_bg, r_border = "#E0E0E0", "#757575"
                r_detail = f"<pre>{_esc(result_text[:2000])}</pre>"

            nodes.append({
                "id": result_node_id,
                "label": r_label,
                "shape": "ellipse",
                "color": {"background": r_bg, "border": r_border},
                "font": {"face": "monospace", "size": 11},
                "level": level_result,
            })
            node_details[result_node_id] = {
                "title": f"Turn {turn_num} result",
                "content": r_detail,
            }

            # call → result edge
            edges.append({"from": call_node_id, "to": result_node_id, "arrows": "to"})
            cur_result_ids.append(result_node_id)

        # previous results → this turn's calls
        for prev_id in prev_result_ids:
            for call_id in call_ids:
                edges.append({"from": prev_id, "to": f"call:{call_id}", "arrows": "to"})

        prev_result_ids = cur_result_ids

    # Final verify node
    if final is not None:
        v_bg = "#4CAF50" if ok else "#F44336"
        v_border = "#1B5E20" if ok else "#B71C1C"
        nodes.append({
            "id": "verify",
            "label": f"{'✓ PASS' if ok else '✗ FAIL'}\n{tool_calls_used} calls · {wall_time_s:.1f}s",
            "shape": "box",
            "color": {"background": v_bg, "border": v_border},
            "font": {"color": "#fff", "bold": True, "size": 13},
            "level": len(calls_by_turn) * 2 + 2,
        })
        v_content = (
            f"<b>Final verification: {'PASS ✓' if ok else 'FAIL ✗'}</b><br>"
            f"<b>Tool calls used:</b> {tool_calls_used}<br>"
            f"<b>Wall time:</b> {wall_time_s:.1f}s<br>"
        )
        if final_proof:
            v_content += f"<br><b>Final proof:</b><pre>{_esc(final_proof)}</pre>"
        if final_error:
            v_content += f"<br><b>Error:</b><pre>{_esc(final_error[:1500])}</pre>"
        node_details["verify"] = {"title": "Final verification", "content": v_content}
        for prev_id in prev_result_ids:
            edges.append({"from": prev_id, "to": "verify", "arrows": "to"})

    return {
        "thm_name": thm_name,
        "lean_root": lean_root,
        "thm_stmt": thm_stmt,
        "ok": ok,
        "wall_time_s": wall_time_s,
        "tool_calls_used": tool_calls_used,
        "graph": {"nodes": nodes, "edges": edges, "nodeDetails": node_details},
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Goedel-Architect — VeriSoftBench Trace</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }
#header { background: #16213e; padding: 12px 20px; border-bottom: 1px solid #0f3460; display: flex; align-items: center; gap: 20px; flex-shrink: 0; }
#header h1 { font-size: 16px; font-weight: 600; color: #90CAF9; }
#header .stat { font-size: 13px; color: #aaa; }
#header .stat b { color: #e0e0e0; }
#main { display: flex; flex: 1; overflow: hidden; }

/* Left sidebar */
#sidebar { width: 240px; background: #16213e; border-right: 1px solid #0f3460; overflow-y: auto; flex-shrink: 0; }
#sidebar-header { padding: 10px 12px; font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid #0f3460; }
.thm-item { padding: 8px 12px; border-bottom: 1px solid #0f3460; cursor: pointer; display: flex; align-items: center; gap: 8px; transition: background 0.15s; }
.thm-item:hover { background: #1a2a4a; }
.thm-item.active { background: #0f3460; }
.badge { font-size: 10px; font-weight: bold; padding: 2px 5px; border-radius: 3px; flex-shrink: 0; }
.badge.pass { background: #2E7D32; color: #fff; }
.badge.fail { background: #C62828; color: #fff; }
.thm-label { font-size: 12px; color: #ccc; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.thm-meta { font-size: 10px; color: #666; margin-top: 2px; }

/* Center graph */
#graph-wrap { flex: 1; position: relative; overflow: hidden; }
#graph-container { width: 100%; height: 100%; }
#empty-msg { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; color: #555; pointer-events: none; }
#empty-msg p { font-size: 16px; margin-bottom: 8px; }
#empty-msg small { font-size: 12px; color: #444; }

/* Right detail panel */
#detail { width: 340px; background: #16213e; border-left: 1px solid #0f3460; overflow-y: auto; flex-shrink: 0; display: flex; flex-direction: column; }
#detail-header { padding: 12px 14px; border-bottom: 1px solid #0f3460; font-size: 13px; font-weight: 600; color: #90CAF9; }
#detail-body { padding: 14px; flex: 1; font-size: 12px; line-height: 1.6; color: #ccc; }
#detail-body pre { background: #0a0a1a; border: 1px solid #0f3460; border-radius: 4px; padding: 10px; font-size: 11px; overflow-x: auto; white-space: pre-wrap; word-break: break-all; margin-top: 8px; color: #a8d8a8; }
#detail-body b { color: #90CAF9; }
#detail-placeholder { color: #444; font-size: 13px; text-align: center; padding-top: 40px; }
</style>
</head>
<body>
<div id="header">
  <h1>Goedel-Architect — VeriSoftBench Trace</h1>
  <span class="stat"><b id="stat-pass">0</b> passed</span>
  <span class="stat"><b id="stat-fail">0</b> failed</span>
  <span class="stat">avg <b id="stat-calls">—</b> tool calls</span>
  <span class="stat">avg <b id="stat-time">—</b>s</span>
</div>
<div id="main">
  <div id="sidebar">
    <div id="sidebar-header">Theorems</div>
    <div id="thm-list"></div>
  </div>
  <div id="graph-wrap">
    <div id="graph-container"></div>
    <div id="empty-msg">
      <p>Select a theorem from the left</p>
      <small>Click any node in the graph to see details</small>
    </div>
  </div>
  <div id="detail">
    <div id="detail-header">Node Detail</div>
    <div id="detail-body"><div id="detail-placeholder">Click a node in the graph</div></div>
  </div>
</div>

<script>
const DATA = __DATA__;

// ── Stats ────────────────────────────────────────────────────────────────────
const passed = DATA.filter(t => t.ok).length;
const failed = DATA.length - passed;
const avgCalls = DATA.length ? (DATA.reduce((s, t) => s + t.tool_calls_used, 0) / DATA.length).toFixed(1) : '—';
const avgTime  = DATA.length ? (DATA.reduce((s, t) => s + t.wall_time_s, 0) / DATA.length).toFixed(1) : '—';
document.getElementById('stat-pass').textContent  = passed;
document.getElementById('stat-fail').textContent  = failed;
document.getElementById('stat-calls').textContent = avgCalls;
document.getElementById('stat-time').textContent  = avgTime;

// ── Sidebar ──────────────────────────────────────────────────────────────────
const thmList = document.getElementById('thm-list');
DATA.forEach((thm, i) => {
  const div = document.createElement('div');
  div.className = 'thm-item';
  div.dataset.idx = i;
  // Built with textContent assignments, not innerHTML: thm_name/lean_root
  // come from trace data ultimately controlled by model output, and an
  // innerHTML template would let a crafted name inject markup.
  const badge = document.createElement('span');
  badge.className = 'badge ' + (thm.ok ? 'pass' : 'fail');
  badge.textContent = thm.ok ? 'PASS' : 'FAIL';
  const label = document.createElement('div');
  label.className = 'thm-label';
  label.title = thm.thm_name;
  label.textContent = thm.thm_name.split('.').pop();
  const meta = document.createElement('div');
  meta.className = 'thm-meta';
  meta.textContent = thm.lean_root + ' · ' + thm.tool_calls_used + ' calls';
  const wrap = document.createElement('div');
  wrap.appendChild(label);
  wrap.appendChild(meta);
  div.appendChild(badge);
  div.appendChild(wrap);
  div.addEventListener('click', () => loadThm(i, div));
  thmList.appendChild(div);
});

// ── Graph ────────────────────────────────────────────────────────────────────
const container = document.getElementById('graph-container');
const emptyMsg  = document.getElementById('empty-msg');
const detailBody = document.getElementById('detail-body');
const detailHeader = document.getElementById('detail-header');

let network = null;
let activeItem = null;

const NETWORK_OPTIONS = {
  layout: {
    hierarchical: {
      enabled: true,
      direction: 'UD',
      sortMethod: 'directed',
      levelSeparation: 110,
      nodeSpacing: 160,
      treeSpacing: 200,
    }
  },
  physics: { enabled: false },
  edges: {
    color: { color: '#555', highlight: '#90CAF9' },
    smooth: { type: 'cubicBezier', forceDirection: 'vertical', roundness: 0.4 },
  },
  nodes: {
    borderWidth: 2,
    shadow: false,
  },
  interaction: {
    hover: true,
    tooltipDelay: 200,
  }
};

function loadThm(idx, listItem) {
  // Update sidebar highlight
  if (activeItem) activeItem.classList.remove('active');
  listItem.classList.add('active');
  activeItem = listItem;

  const thm = DATA[idx];
  emptyMsg.style.display = 'none';

  const nodes = new vis.DataSet(thm.graph.nodes);
  const edges = new vis.DataSet(thm.graph.edges);

  if (network) {
    network.destroy();
  }
  network = new vis.Network(container, { nodes, edges }, NETWORK_OPTIONS);

  network.on('click', function(params) {
    if (params.nodes.length > 0) {
      const nodeId = params.nodes[0];
      const detail = thm.graph.nodeDetails[nodeId];
      if (detail) {
        detailHeader.textContent = detail.title;
        detailBody.innerHTML = detail.content;
      }
    }
  });

  // Show theorem detail by default on the right
  const thmDetail = thm.graph.nodeDetails['thm'];
  if (thmDetail) {
    detailHeader.textContent = thmDetail.title;
    detailBody.innerHTML = thmDetail.content;
  }
}

// Auto-load first theorem if only one exists
if (DATA.length === 1) {
  loadThm(0, thmList.firstChild);
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate(trace_path: Path | str, output_path: Path | str | None = None) -> Path:
    """Read trace_path JSONL and write a self-contained HTML to output_path."""
    trace_path = Path(trace_path)
    if output_path is None:
        output_path = trace_path.with_suffix(".html")
    output_path = Path(output_path)

    groups = load_trace(trace_path)
    theorem_data = [
        build_theorem_data(thm_name, events)
        for thm_name, events in groups.items()
    ]

    # Sort: PASS first, then by thm_name
    theorem_data.sort(key=lambda t: (not t["ok"], t["thm_name"]))

    data_json = json.dumps(theorem_data, ensure_ascii=False, indent=None)
    # JSON-in-<script> embedding: a model-generated string containing
    # `</script>` (or U+2028/29, illegal in JS string literals) would
    # otherwise break out of the script tag or the parse. Escape exactly
    # the characters that matter, after JSON encoding.
    data_json = (data_json
                 .replace("<", "\\u003c")
                 .replace(">", "\\u003e")
                 .replace("&", "\\u0026")
                 .replace("\u2028", "\\u2028")
                 .replace("\u2029", "\\u2029"))
    html = HTML_TEMPLATE.replace("__DATA__", data_json)
    output_path.write_text(html, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python graph_viz.py <trace.jsonl> [output.html]")
        sys.exit(1)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    out = generate(src, dst)
    print(f"Report written to: {out}")
