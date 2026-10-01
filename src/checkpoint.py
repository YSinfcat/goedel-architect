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
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from blueprint import Blueprint, _parse_blueprint
from prover import ProofSignal, ProverResult


class CheckpointFingerprintError(ValueError):
    """Resume refused: the checkpoint was produced by a different
    code/prompt/config experiment (run fingerprint mismatch, or a legacy
    checkpoint with no recorded fingerprint at all)."""


@dataclass
class CheckpointState:
    # Schema 0 = checkpoints written before fingerprints existed (any file
    # without this key on load). Schema 1 = records run_fingerprint and is
    # subject to the resume fingerprint check.
    schema_version: int = 1
    theorem_stmt: str = ""
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
    # pipeline._invalidate_stale_proofs). Missing entries (e.g. a checkpoint
    # written before this field existed) are treated as stale on first use -
    # a safe, one-time re-check rather than trusting an unrecorded cache.
    proof_cache_keys: dict[str, str] = field(default_factory=dict)
    # name -> serialized ProverResult (signal/proof_body/analysis/suggested_fix/lean_errors)
    node_results: dict[str, dict] = field(default_factory=dict)
    refinement_history: list[str] = field(default_factory=list)
    done: bool = False
    success: bool = False
    # Provenance manifest (run_fingerprint.run_manifest) + its hash under
    # "fingerprint". Empty dict = legacy schema-0 checkpoint.
    run_fingerprint: dict = field(default_factory=dict)
    # Per-refinement-round node lineage snapshots (see src/lineage.py):
    # list of round snapshots, each a list of entry dicts. Additive -
    # checkpoints written before this field exist simply have [].
    lineage_history: list = field(default_factory=list)

    # -- Blueprint (de)serialization -------------------------------------

    def set_blueprint(self, blueprint: Blueprint) -> None:
        self.blueprint_lean_file = blueprint.lean_file
        self.blueprint_target = blueprint.target_theorem
        self.blueprint_fully_validated = blueprint.fully_validated

    def get_blueprint(self) -> Blueprint | None:
        if not self.blueprint_lean_file:
            return None
        bp = _parse_blueprint(self.blueprint_lean_file, self.blueprint_target)
        bp.fully_validated = self.blueprint_fully_validated
        return bp

    # -- Node results (de)serialization ----------------------------------

    def set_node_results(self, node_results: dict) -> None:
        """node_results: dict[str, NodeResult] as produced by orchestrator.prove_dag."""
        self.node_results = {
            name: {
                "signal": nr.result.signal.value,
                "proof_body": nr.result.proof_body,
                "analysis": nr.result.analysis,
                "suggested_fix": nr.result.suggested_fix,
                "lean_errors": nr.result.lean_errors,
            }
            for name, nr in node_results.items()
        }

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
        # Tolerate unknown keys: a checkpoint written by a newer schema (or
        # a hand-edited one) must not crash an older reader - forward
        # compatibility beats strict round-tripping here.
        known = {fld.name for fld in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def load_or_none(cls, path: Path | None) -> "CheckpointState | None":
        if path is None or not path.exists():
            return None
        return cls.load(path)

    def require_fingerprint(self, manifest: dict, fingerprint: str) -> None:
        """Refuse to reuse this checkpoint for a different experiment.

        A legacy schema-0 checkpoint has no recorded fingerprint - its
        provenance is unknowable, so it is treated as a mismatch rather
        than silently trusted. Remediation is the caller's choice: delete
        the checkpoint to recompute, or set GOEDEL_SKIP_FINGERPRINT=1 for
        a local debugging resume (recorded in the manifest).
        """
        recorded = (self.run_fingerprint or {}).get("fingerprint", "")
        if recorded == fingerprint:
            return
        from run_fingerprint import skip_requested
        if skip_requested():
            print("[Resume] fingerprint check skipped (GOEDEL_SKIP_FINGERPRINT=1)",
                  flush=True)
            return
        if not recorded:
            raise CheckpointFingerprintError(
                f"Checkpoint has no run fingerprint (schema-0, written before "
                f"provenance tracking) - refusing to resume it. Delete the "
                f"checkpoint to recompute, or set GOEDEL_SKIP_FINGERPRINT=1 "
                f"for a debugging resume."
            )
        recorded_manifest = self.run_fingerprint
        diffs = [
            f"{key}: checkpoint={recorded_manifest.get(key)!r} != current={manifest.get(key)!r}"
            for key in sorted(set(manifest) | set(recorded_manifest) - {"fingerprint"})
            if key != "fingerprint" and recorded_manifest.get(key) != manifest.get(key)
        ]
        raise CheckpointFingerprintError(
            "Checkpoint was produced by a different experiment configuration "
            f"(run fingerprint mismatch).\n  checkpoint: {recorded}\n  current:   {fingerprint}\n"
            + ("\n".join(f"  {d}" for d in diffs[:6]))
            + "\nDelete the checkpoint to recompute, or set GOEDEL_SKIP_FINGERPRINT=1 "
              "for a debugging resume."
        )


def path_for_theorem(checkpoint_dir: Path, thm_name: str) -> Path:
    safe_name = thm_name.replace("/", "_").replace("\\", "_")
    return checkpoint_dir / f"{safe_name}.json"
