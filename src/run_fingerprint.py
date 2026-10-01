"""Run fingerprint: identity of the code+prompts+config that produced a
checkpoint, so --resume cannot silently mix results across experiments.

The review's P1-5: checkpoints previously stored theorem/model/blueprint
only, so a run could resume an old result computed by different code, a
different prompt, or a different cascade policy and report it as if it
belonged to the current experiment. Every checkpoint now records a
manifest (this module's `run_manifest`) and its SHA-256 fingerprint;
resume refuses a mismatch.

Everything here is best-effort and offline: git/lake are probed once per
process, failures degrade to "unknown", and `GOEDEL_SKIP_FINGERPRINT=1`
disables the resume check for local debugging (the escape is recorded in
the manifest so a debug-resumed checkpoint stays identifiable).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
PROMPTS_DIR = REPO_ROOT / "prompts"

# Bump when the manifest's semantics change such that old fingerprints are
# no longer comparable (added/renamed fields, changed hashing rules).
MANIFEST_SCHEMA = 1

_skip_cache: bool | None = None


def skip_requested() -> bool:
    global _skip_cache
    if _skip_cache is None:
        _skip_cache = os.environ.get("GOEDEL_SKIP_FINGERPRINT", "") == "1"
    return _skip_cache


@lru_cache(maxsize=1)
def code_commit() -> str:
    """Short commit hash of the goedel-arch checkout, +dirty marker."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if not commit:
            return "unknown"
        return f"{commit}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


@lru_cache(maxsize=1)
def prompt_hashes() -> dict[str, str]:
    """sha256[:16] of every prompt file - prompt edits are experiment-
    defining even when the code commit is unchanged."""
    hashes: dict[str, str] = {}
    if not PROMPTS_DIR.is_dir():
        return {"_prompts_dir": "missing"}
    for p in sorted(PROMPTS_DIR.iterdir()):
        if p.is_file():
            hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return hashes or {"_prompts_dir": "empty"}


@lru_cache(maxsize=1)
def lean_toolchain() -> str:
    """First line of `lean --version` in the goedel_lean environment.
    GOEDEL_LEAN_TOOLCHAIN overrides (for sandboxes without lake)."""
    override = os.environ.get("GOEDEL_LEAN_TOOLCHAIN")
    if override:
        return override
    try:
        out = subprocess.run(
            ["lake", "env", "lean", "--version"],
            cwd=REPO_ROOT / "goedel_lean",
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        return out.splitlines()[0] if out else "unknown"
    except Exception:
        return "unknown"


def run_manifest(
    model: str,
    cascade_model: str | None = None,
    max_iterations: int | None = None,
    enable_negation_probe: bool = False,
    allow_unvalidated_blueprint: bool = False,
    tactic_portfolio: list[str] | None = None,
    provider_base_urls: dict[str, str] | None = None,
) -> dict:
    """The full provenance manifest for one prove_theorem configuration."""
    return {
        "manifest_schema": MANIFEST_SCHEMA,
        "code_commit": code_commit(),
        "prompt_hashes": prompt_hashes(),
        "lean_toolchain": lean_toolchain(),
        "model_config": {
            "model": model,
            "cascade_model": cascade_model,
        },
        "pipeline_config": {
            "max_iterations": max_iterations,
            "enable_negation_probe": enable_negation_probe,
            "allow_unvalidated_blueprint": allow_unvalidated_blueprint,
            "tactic_portfolio": tactic_portfolio,
        },
        "provider_base_urls": provider_base_urls or {},
        "fingerprint_skip": skip_requested(),
    }


def fingerprint(manifest: dict) -> str:
    """Stable SHA-256 of the manifest (sorted keys, so field order in the
    JSON file never matters)."""
    canonical = json.dumps(manifest, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()
