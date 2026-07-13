"""Per-theorem checkpoint state for resumable Phase 1/2/3 runs.

Persists exactly enough for each phase to be invoked standalone, without
re-running the phases before it:
  - Phase 1 (blueprint generation) writes `blueprint`.
  - Phase 2 (parallel proving) reads `blueprint`, writes `node_results` +
    `proved_cache`.
  - Phase 3 (refinement) reads `blueprint` + `node_results` (needs Phase 2's
    diagnostics to know what to fix), writes a new `blueprint` and bumps
    `iteration`.

A `Blueprint` is fully reconstructible from its raw `lean_file` text (see
`blueprint._parse_blueprint`), so only that string is stored rather than the
parsed node list. One JSON file per theorem, rewritten atomically (tmp file +
os.replace) after every phase so a run can be killed and resumed from
wherever it left off.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from blueprint import Blueprint, _parse_blueprint
from orchestrator import OrchestratorResult
from prover import ProofSignal, ProverResult


def _invalidate_stale_proofs(
    new_blueprint: Blueprint,
    proved_cache: dict[str, str],
    proof_cache_keys: dict[str, str],
) -> dict[str, str]:
    """Drop cached proofs that no longer match the node they were compiled against.

    proved_cache tracks nodes by NAME only. Refinement (Phase 3) can reuse a
    node's name while restructuring its signature or dependency list (e.g.
    splitting a hypothesis, changing its goal shape, adding a new
    sorry_using [...] dependency) - the paper's rule only promises
    SOLVED/FORMALLY_NEGATED nodes carry forward byte-identical, but nothing
    enforces that, and a name collision with a differently-shaped node would
    otherwise leave a proof compiled against the OLD shape marked "already
    solved" forever, never recompiled against the new one.

    Rather than diffing the immediately-previous blueprint (which misses a
    node deleted at round N and reintroduced with a different shape at round
    N+2 - neither adjacent diff N->N+1 or N+1->N+2 ever sees both shapes at
    once), this compares against `proof_cache_keys[name]`: the exact
    BlueprintNode.cache_key() recorded at the moment the proof was accepted.
    A name missing from `proof_cache_keys` (e.g. an older checkpoint written
    before this field existed) is treated as stale and re-checked once.
    """
    pruned = dict(proved_cache)
    for name in list(pruned):
        new_node = new_blueprint.node_by_name(name)
        if new_node is None or proof_cache_keys.get(name) != new_node.cache_key():
            del pruned[name]
    return pruned


@dataclass
class CheckpointState:
    theorem_stmt: str
    model: str = "gpt-5.5"
    repo_context: str = ""
    iteration: int = 0
    blueprint_lean_file: str = ""
    blueprint_target: str = ""
    blueprint_fully_validated: bool = False
    proved_cache: dict[str, str] = field(default_factory=dict)
    # name -> BlueprintNode.cache_key() recorded at the moment the proof was
    # accepted into proved_cache. Lets a later refinement round tell whether
    # a cached proof still matches the node it was compiled against (see
    # _invalidate_stale_proofs below). Missing entries (e.g. a checkpoint
    # written before this field existed) are treated as stale on first use -
    # a safe, one-time re-check rather than trusting an unrecorded cache.
    proof_cache_keys: dict[str, str] = field(default_factory=dict)
    # name -> serialized ProverResult (signal/proof_body/analysis/suggested_fix/lean_errors)
    node_results: dict[str, dict] = field(default_factory=dict)
    refinement_history: list[str] = field(default_factory=list)
    done: bool = False
    success: bool = False

    # -- Phase transitions --------------------------------------------------
    #
    # These are the only way to mutate phase-relevant state - an illegal
    # field combination (e.g. node_results left stale against a just-refined
    # blueprint) is then impossible to express instead of merely discouraged
    # by comment.

    def complete_phase1(self, blueprint: Blueprint) -> None:
        self.blueprint_lean_file = blueprint.lean_file
        self.blueprint_target = blueprint.target_theorem
        self.blueprint_fully_validated = blueprint.fully_validated

    def complete_phase2(self, orch_result: OrchestratorResult, blueprint: Blueprint) -> None:
        """Does not touch `iteration` - only Phase 3 (refinement) advances
        the round counter; Phase 2 may be re-run in place (e.g. standalone
        `--phase 2` invoked again before any refinement) without that
        counting as a new round."""
        for name, nr in orch_result.node_results.items():
            if nr.result.signal == ProofSignal.SOLVED:
                self.proved_cache[name] = nr.result.proof_body
                node = blueprint.node_by_name(name)
                if node:
                    self.proof_cache_keys[name] = node.cache_key()
        self.node_results = {
            name: {
                "signal": nr.result.signal.value,
                "proof_body": nr.result.proof_body,
                "analysis": nr.result.analysis,
                "suggested_fix": nr.result.suggested_fix,
                "lean_errors": nr.result.lean_errors,
            }
            for name, nr in orch_result.node_results.items()
        }
        self.done = orch_result.all_proved()
        self.success = self.done

    def complete_phase3(self, new_blueprint: Blueprint, refinement_history: list[str]) -> None:
        self.complete_phase1(new_blueprint)
        pruned = _invalidate_stale_proofs(new_blueprint, self.proved_cache, self.proof_cache_keys)
        self.proved_cache = pruned
        self.proof_cache_keys = {name: key for name, key in self.proof_cache_keys.items() if name in pruned}
        self.refinement_history = list(refinement_history)
        self.iteration += 1
        self.node_results = {}  # stale against the new blueprint
        self.done = False
        self.success = False

    # -- Blueprint (de)serialization -------------------------------------

    def get_blueprint(self) -> Blueprint | None:
        if not self.blueprint_lean_file:
            return None
        bp = _parse_blueprint(self.blueprint_lean_file, self.blueprint_target)
        bp.fully_validated = self.blueprint_fully_validated
        return bp

    # -- Node results (de)serialization ----------------------------------

    def get_prover_results(self) -> dict[str, ProverResult]:
        return {
            name: ProverResult(
                signal=ProofSignal(d["signal"]),
                proof_body=d.get("proof_body", ""),
                analysis=d.get("analysis", ""),
                suggested_fix=d.get("suggested_fix", ""),
                lean_errors=d.get("lean_errors", []),
            )
            for name, d in self.node_results.items()
        }

    # -- Persistence -------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp_ckpt_")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(asdict(self), f, indent=2)
            os.replace(tmp_path, path)  # atomic on POSIX
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Path) -> "CheckpointState":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def load_or_none(cls, path: Path | None) -> "CheckpointState | None":
        if path is None or not path.exists():
            return None
        return cls.load(path)


def path_for_theorem(checkpoint_dir: Path, thm_name: str) -> Path:
    safe_name = thm_name.replace("/", "_").replace("\\", "_")
    return checkpoint_dir / f"{safe_name}.json"
