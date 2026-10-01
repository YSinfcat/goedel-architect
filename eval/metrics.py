"""Aggregate results from JSONL output files and print a summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter


def summarize(path: str) -> None:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    total = len(records)
    status_counts = Counter(r["status"] for r in records)
    solved = status_counts.get("SOLVED", 0)

    print(f"Results from: {path}")
    if total == 0:
        print("  (no records - empty or all-blank JSONL)")
        return
    print(f"  Total:  {total}")
    print(f"  Solved: {solved} ({100*solved/total:.1f}%)")
    print(f"  Failed: {status_counts.get('FAILED', 0)}")
    print(f"  Errors: {status_counts.get('ERROR', 0)}")

    # Unified-schema fields (eval_records.evaluation_record); each section
    # silently degrades when older records lack the keys.
    autonomies = Counter(r["autonomy"] for r in records if r.get("autonomy"))
    if autonomies:
        print("  Autonomy: " + ", ".join(f"{k}={v}" for k, v in autonomies.most_common()))
    verified = [r for r in records if r.get("final_verified") is not None]
    failed_verify = sum(1 for r in verified if r["final_verified"] is False)
    if verified:
        print(f"  Final verification: {len(verified)-failed_verify} passed, "
              f"{failed_verify} rejected (weakened/corrupted proofs caught)")
    budget_stopped = sum(1 for r in records if r.get("stopped_reason"))
    if budget_stopped:
        print(f"  Budget-stopped: {budget_stopped}")

    solved_records = [r for r in records if r["status"] == "SOLVED"]
    if solved_records:
        avg_t = sum(r["elapsed_s"] for r in solved_records) / len(solved_records)
        avg_iters = sum(r.get("iterations") or 0 for r in solved_records) / len(solved_records)
        print(f"  Avg time (solved): {avg_t:.1f}s")
        print(f"  Avg iterations (solved): {avg_iters:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize evaluation results")
    parser.add_argument("paths", nargs="+", help="JSONL result files")
    args = parser.parse_args()
    for path in args.paths:
        summarize(path)
        print()


if __name__ == "__main__":
    main()
