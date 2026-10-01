"""Node lineage across refinement rounds (review IV.2).

Refinement freely renames, splits, and merges nodes; until now the only
way to tell whether a new round's `foo_v2` was the same mathematical
commitment as last round's `foo` was to read the prompts. Lineage makes
that mechanical:

  statement_hash   sha256 of a node's signature - name-INDEPENDENT, so a
                   pure rename is recognizable as the same statement
  node_id          stable UUID carried across rounds while the statement
                   survives (by hash), regardless of renames
  revision         bumped when the same node_id's statement changes (edit)
  operation        how this node came to be, relative to the previous
                   graph: unchanged | edit | rename | split | merge | new
  parents_in_previous_graph   names of the previous-round nodes this one
                   was derived from (exact for split/merge, else itself)

The durable per-round record lives on CheckpointState.lineage_history
and ships in the success artifact; the in-memory BlueprintNode fields
are transient (a checkpoint round-trips nodes from lean_file text).

Split/merge detection is heuristic (dependency-overlap between
disappeared and appeared nodes) - deliberately conservative: when the
evidence is ambiguous the entry is recorded as `new`, never a confident
wrong `split`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable

from blueprint import Blueprint, BlueprintNode

UNCHANGED = "unchanged"
EDIT = "edit"
RENAME = "rename"
SPLIT = "split"
MERGE = "merge"
NEW = "new"


def statement_hash(node: BlueprintNode) -> str:
    """sha256 of the node's signature - the mathematical commitment,
    independent of the node's current name."""
    return hashlib.sha256(node.signature().encode()).hexdigest()[:16]


@dataclass
class LineageEntry:
    name: str                     # name in the NEW graph
    node_id: str
    revision: int
    operation: str
    statement_hash: str
    parents_in_previous_graph: list[str] = field(default_factory=list)


def _by_statement_hash(nodes: Iterable[BlueprintNode]) -> dict[str, BlueprintNode]:
    out: dict[str, BlueprintNode] = {}
    for n in nodes:
        out.setdefault(statement_hash(n), n)
    return out


def compute_lineage(
    previous: list[BlueprintNode],
    current: list[BlueprintNode],
    previous_ids: dict[str, str] | None = None,
    previous_revisions: dict[str, int] | None = None,
) -> tuple[dict[str, LineageEntry], dict[str, str]]:
    """Match `current` nodes against `previous` and assign lineage.

    previous_ids / previous_revisions: name -> node_id / revision for the
    PREVIOUS graph (from its own lineage pass); omitted on the first
    round, where every node is `new`.

    Returns (entries_by_current_name, ids_by_current_name) - the second
    map feeds the next round's call.
    """
    previous_ids = previous_ids or {}
    previous_revisions = previous_revisions or {}
    prev_by_name = {n.name: n for n in previous}
    prev_by_hash = _by_statement_hash(previous)
    prev_hashes = set(prev_by_hash)

    cur_hashes = {statement_hash(n) for n in current}
    disappeared = [n for n in previous if statement_hash(n) not in cur_hashes]
    appeared = [n for n in current if statement_hash(n) not in prev_hashes]

    # Heuristic parent matching for disappeared -> appeared: an appeared
    # node is a split-child of a disappeared node when its dependency set
    # overlaps the disappeared node's dependencies; a merge when two or
    # more disappeared nodes' dependency sets are both contained in one
    # appeared node's. Conservative: overlap below the threshold leaves
    # the appeared node as plain `new`.
    parents_of: dict[str, list[str]] = {}
    for app in appeared:
        app_deps = set(app.dependencies)
        candidates = [
            d for d in disappeared
            if d.dependencies and set(d.dependencies) & app_deps
        ]
        if len(candidates) == 1:
            parents_of[app.name] = [candidates[0].name]
        elif len(candidates) >= 2:
            parents_of[app.name] = sorted(c.name for c in candidates)

    entries: dict[str, LineageEntry] = {}
    ids: dict[str, str] = {}
    for node in current:
        h = statement_hash(node)
        # 1. exact statement carried over (by hash), possibly renamed
        if h in prev_hashes:
            prev_node = prev_by_hash[h]
            prev_id = previous_ids.get(prev_node.name) or f"n-{h[:12]}"
            ids[node.name] = prev_id
            op = UNCHANGED if prev_node.name == node.name else RENAME
            entries[node.name] = LineageEntry(
                name=node.name, node_id=prev_id,
                revision=previous_revisions.get(prev_node.name, 1),
                operation=op, statement_hash=h,
                parents_in_previous_graph=[prev_node.name],
            )
            continue
        # 2. split / merge child of disappeared node(s)
        if node.name in parents_of:
            parents = parents_of[node.name]
            op = SPLIT if len(parents) == 1 else MERGE
            # inherit the primary parent's id with a bumped revision
            primary = prev_by_name[parents[0]]
            prev_id = previous_ids.get(primary.name) or f"n-{statement_hash(primary)[:12]}"
            ids[node.name] = prev_id
            entries[node.name] = LineageEntry(
                name=node.name, node_id=prev_id,
                revision=previous_revisions.get(primary.name, 1) + 1,
                operation=op, statement_hash=h,
                parents_in_previous_graph=parents,
            )
            continue
        # 3. same name, changed statement: an edit of the previous node
        if node.name in prev_by_name:
            prev_node = prev_by_name[node.name]
            prev_id = previous_ids.get(node.name) or f"n-{statement_hash(prev_node)[:12]}"
            ids[node.name] = prev_id
            entries[node.name] = LineageEntry(
                name=node.name, node_id=prev_id,
                revision=previous_revisions.get(node.name, 1) + 1,
                operation=EDIT, statement_hash=h,
                parents_in_previous_graph=[node.name],
            )
            continue
        # 4. genuinely new
        fresh_id = f"n-{h[:12]}"
        ids[node.name] = fresh_id
        entries[node.name] = LineageEntry(
            name=node.name, node_id=fresh_id, revision=1,
            operation=NEW, statement_hash=h,
        )
    return entries, ids


def lineage_snapshot(entries: dict[str, LineageEntry]) -> list[dict]:
    """JSON-serializable round snapshot for the checkpoint / artifact."""
    return [
        {
            "name": e.name,
            "node_id": e.node_id,
            "revision": e.revision,
            "operation": e.operation,
            "statement_hash": e.statement_hash,
            "parents_in_previous_graph": e.parents_in_previous_graph,
        }
        for e in entries.values()
    ]
