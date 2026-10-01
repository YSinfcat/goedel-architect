"""Success artifact bundle (review IV.8): a self-contained, independently
auditable directory per successful proof.

    proof.lean               the exact bytes the verification compiled
    blueprint.initial.json   the Phase-1 dependency graph
    blueprint.final.json     the graph that actually closed the theorem
    dag.txt                  edge list + topological order (SVG later)
    verification.json        what was checked, and what was scanned
    run_manifest.json        the experiment fingerprint manifest
    model_usage.json         token totals aggregated from the trace
    README.md                how to re-verify by hand

Everything needed to re-check a claim without the pipeline's internal
state. Written only on success (failures aren't citable artifacts); any
write error is reported and swallowed - a broken artifact directory must
not turn a verified proof into a reported failure.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from blueprint import Blueprint

_FORBIDDEN_SCAN = {
    "contains_sorry": re.compile(r"\bsorry\b|\badmit\b"),
    "contains_axiom": re.compile(r"\baxiom\b"),
    "contains_native_decide": re.compile(r"\bnative_decide\b"),
}


def _blueprint_json(blueprint: Blueprint) -> dict:
    return {
        "target_theorem": blueprint.target_theorem,
        "fully_validated": blueprint.fully_validated,
        "nodes": [
            {
                "name": n.name,
                "kind": n.kind,
                "statement": n.statement,
                "proof_sketch": n.proof_sketch,
                "dependencies": n.dependencies,
                "lean_declaration": n.lean_declaration,
            }
            for n in blueprint.nodes
        ],
    }


def _dag_text(blueprint: Blueprint) -> str:
    lines = ["# dependency edges (dependent -> dependency)"]
    for n in blueprint.nodes:
        for dep in n.dependencies:
            lines.append(f"{n.name} -> {dep}")
    lines.append("")
    lines.append("# topological order (dependencies first)")
    lines.append(", ".join(n.name for n in blueprint.dependency_order()))
    lines.append("")
    lines.append(f"# target: {blueprint.target_theorem}")
    return "\n".join(lines) + "\n"


def _aggregate_usage(trace_path: Path | None) -> dict:
    """Sum llm_usage events from a JSONL trace: per (phase, model) token
    totals plus a grand total. Empty summary when no trace is available."""
    summary: dict = {"by_phase_model": {}, "total_prompt_tokens": 0,
                     "total_completion_tokens": 0, "total_tokens": 0}
    if trace_path is None or not Path(trace_path).exists():
        summary["note"] = "no trace file available"
        return summary
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("kind") != "llm_usage":
                continue
            args = event.get("args") or {}
            key = f"{args.get('phase', '?')}/{args.get('model', '?')}"
            bucket = summary["by_phase_model"].setdefault(
                key, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
            bucket["prompt_tokens"] += args.get("prompt_tokens", 0) or 0
            bucket["completion_tokens"] += args.get("completion_tokens", 0) or 0
            bucket["calls"] += 1
            summary["total_prompt_tokens"] += args.get("prompt_tokens", 0) or 0
            summary["total_completion_tokens"] += args.get("completion_tokens", 0) or 0
            summary["total_tokens"] += args.get("total_tokens", 0) or 0
    return summary


def write_success_artifact(
    out_dir: Path | str,
    *,
    theorem_name: str,
    proof_lean: str,
    verification,
    manifest: dict,
    blueprint_initial: Blueprint | None = None,
    blueprint_final: Blueprint | None = None,
    trace_path: Path | None = None,
    lineage_history: list | None = None,
) -> Path:
    """Write the bundle; returns the directory. See module docstring."""
    out = Path(out_dir) / _safe(theorem_name)
    (out / "proof.lean").parent.mkdir(parents=True, exist_ok=True)

    (out / "proof.lean").write_text(proof_lean, encoding="utf-8")
    if blueprint_initial is not None:
        (out / "blueprint.initial.json").write_text(
            json.dumps(_blueprint_json(blueprint_initial), indent=2, ensure_ascii=False),
            encoding="utf-8")
    if blueprint_final is not None:
        (out / "blueprint.final.json").write_text(
            json.dumps(_blueprint_json(blueprint_final), indent=2, ensure_ascii=False),
            encoding="utf-8")
        (out / "dag.txt").write_text(_dag_text(blueprint_final), encoding="utf-8")

    verification_payload = {
        "original_theorem_verified": bool(verification.passed),
        "mode": verification.mode,
        "performed": verification.performed,
        "errors": verification.errors,
        "lean_toolchain": manifest.get("lean_toolchain", "unknown"),
        "code_commit": manifest.get("code_commit", "unknown"),
        "run_fingerprint": manifest.get("fingerprint", ""),
        # honesty scans over the shipped proof bytes themselves
        **{name: bool(rx.search(proof_lean)) for name, rx in _FORBIDDEN_SCAN.items()},
    }
    (out / "verification.json").write_text(
        json.dumps(verification_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "model_usage.json").write_text(
        json.dumps(_aggregate_usage(trace_path), indent=2, ensure_ascii=False),
        encoding="utf-8")
    if lineage_history:
        (out / "lineage.json").write_text(
            json.dumps(lineage_history, indent=2, ensure_ascii=False),
            encoding="utf-8")
    if trace_path is not None and Path(trace_path).exists():
        shutil.copyfile(trace_path, out / "trace.jsonl")

    readme = f"""# Proof artifact: {theorem_name}

Verified against the ORIGINAL theorem statement by an independent Lean
compilation ({verification.mode} mode) before success was recorded.

Re-verify by hand (from the goedel_lean project root):

    lake env lean <this directory>/proof.lean

Contents:
- proof.lean               exactly the file the verification compiled
- blueprint.initial.json   Phase-1 dependency graph
- blueprint.final.json     the graph that closed the theorem
- dag.txt                  edge list + topological order
- verification.json        what was checked (and sorry/axiom scans)
- run_manifest.json        code commit, prompt hashes, model config
- model_usage.json         token totals from the run's trace
- lineage.json             per-round node lineage (ids, edits, splits)
- trace.jsonl              the run's event trace (when available)
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    return out


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name) or "theorem"
