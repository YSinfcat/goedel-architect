"""Unified evaluation record schema (review IV.7).

run_minif2f, run_putnam, and (via its own dict fields) run_verisoftbench
used to emit overlapping-but-different key sets, so metrics.py could not
aggregate across them. Every runner now writes records through
`evaluation_record` with one stable schema:

    name            problem identifier
    status          SOLVED | FAILED | ERROR | TIMEOUT
    autonomy        fully_autonomous | human_guided | human_written
    iterations      refinement rounds actually run
    proved_nodes / failed_nodes
    final_verified  True only if the ORIGINAL theorem passed independent
                    verification (False on failure, None if never reached)
    verify_mode     full-file | vsb-root-recheck | unavailable
    stopped_reason  budget stop text ONLY, "" when the run ran to completion
    error           exception text for ERROR records, None otherwise
    elapsed_s
    ts              ISO timestamp
"""
from __future__ import annotations

import time


def evaluation_record(name: str, result, elapsed_s: float,
                      status: str | None = None,
                      error: str | None = None) -> dict:
    """Build one JSONL record from a pipeline ProofResult (or an error)."""
    if result is None:
        return {
            "name": name,
            "status": status or "ERROR",
            "autonomy": None,
            "iterations": None,
            "proved_nodes": [],
            "failed_nodes": [],
            "final_verified": None,
            "verify_mode": None,
            "stopped_reason": "",
            "error": error,
            "elapsed_s": round(elapsed_s, 2),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    verification = result.final_verification
    return {
        "name": name,
        "status": status or ("SOLVED" if result.success else "FAILED"),
        "autonomy": result.autonomy,
        "iterations": result.iterations,
        "proved_nodes": result.proved_nodes,
        "failed_nodes": result.failed_nodes,
        "final_verified": (verification.passed if verification is not None
                           else (True if result.success else None)),
        "verify_mode": verification.mode if verification is not None else None,
        "stopped_reason": getattr(result, "stopped_reason", ""),
        "error": None,
        "elapsed_s": round(elapsed_s, 2),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
