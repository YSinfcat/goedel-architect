"""Formalized human intervention on checkpoints (review IV.3).

Experiment logs already recorded hand-injected proved_cache entries; this
module makes that practice a first-class, auditable operation instead of
undocumented JSON surgery:

  set-proof   a human-written proof enters proved_cache, source-marked
              ("human") and journaled with operator/reason/before-after
  lock        a node is protected: stale-proof invalidation will not drop
              it when refinement reshapes the graph (final verification
              still gates success - a lock can never smuggle a proof past
              the original-theorem check)
  retry       a node's cached result is cleared so the next run re-attempts
  status      read-only view of nodes, statuses, sources, locks

Every mutation appends to CheckpointState.interventions:
    {type, operator, node, before, after, reason, ts}

Autonomy classification (see pipeline._autonomy_label): results with no
interventions are fully_autonomous; interventions without human proofs
are human_guided; any human-written proof makes the result human_written.
Human-written proofs pass through the SAME independent final verification
as model proofs - the trust chain is unchanged.
"""
from __future__ import annotations

import getpass
import time
from pathlib import Path

from checkpoint import CheckpointState
from lean_compiler import format_proof_assign, normalize_proof_body

SOURCE_MODEL = "model"
SOURCE_HUMAN = "human"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _operator() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def _record(state: CheckpointState, type_: str, node: str,
            before, after, reason: str) -> None:
    state.interventions.append({
        "type": type_, "operator": _operator(), "node": node,
        "before": before, "after": after,
        "reason": reason or "", "ts": _now(),
    })


def status(state: CheckpointState) -> list[dict]:
    rows = []
    blueprint = state.get_blueprint()
    names = [n.name for n in blueprint.nodes] if blueprint else []
    for name in names:
        rows.append({
            "node": name,
            "status": ("proved" if name in state.proved_cache
                       else state.node_results.get(name, {}).get("signal", "unattempted")),
            "source": state.proof_sources.get(name, SOURCE_MODEL
                                              if name in state.proved_cache else ""),
            "locked": name in state.locked_nodes,
        })
    return rows


def set_proof(state: CheckpointState, node: str, proof_body: str,
              reason: str = "") -> None:
    """Inject a human-written proof for `node` (canonical `by ...` form
    accepted with or without a leading ':=')."""
    canonical = normalize_proof_body(proof_body)
    before = state.proved_cache.get(node, "")
    state.proved_cache[node] = canonical
    state.proof_sources[node] = SOURCE_HUMAN
    # A human-set proof is also authoritative: drop any stale failure
    # verdict so refinement doesn't keep trying to "fix" it.
    state.node_results.pop(node, None)
    _record(state, "set-proof", node, before, canonical, reason)


def lock(state: CheckpointState, node: str, reason: str = "") -> None:
    if node not in state.locked_nodes:
        state.locked_nodes.append(node)
        _record(state, "lock", node, False, True, reason)


def unlock(state: CheckpointState, node: str, reason: str = "") -> None:
    if node in state.locked_nodes:
        state.locked_nodes = [n for n in state.locked_nodes if n != node]
        _record(state, "unlock", node, True, False, reason)


def retry(state: CheckpointState, node: str, reason: str = "") -> None:
    """Clear a node's cached proof/verdict so the next run re-attempts it."""
    before = state.proved_cache.get(node, "")
    state.proved_cache.pop(node, None)
    state.proof_cache_keys.pop(node, None)
    state.node_results.pop(node, None)
    state.proof_sources.pop(node, None)
    if not state.done:
        pass  # mid-run checkpoint: nothing else to reset
    else:
        # A finished checkpoint would immediately return the cached
        # verdict again - reopen it so prove_theorem(retry_failed=...)
        # style continuation is possible. (Fingerprint still gates reuse.)
        state.done = False
        state.success = False
    _record(state, "retry", node, before, "", reason)


def autonomy_label(state: CheckpointState) -> str:
    """fully_autonomous | human_guided | human_written (see module doc)."""
    if any(s == SOURCE_HUMAN for s in state.proof_sources.values()):
        return "human_written"
    if state.interventions:
        return "human_guided"
    return "fully_autonomous"


def load_for_edit(path: Path) -> CheckpointState:
    return CheckpointState.load(path)


def save_after_edit(state: CheckpointState, path: Path) -> None:
    state.save(path)
