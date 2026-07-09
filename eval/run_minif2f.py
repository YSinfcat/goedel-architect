"""Evaluate the full pipeline on MiniF2F-test (244 problems, Lean 4 split)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import prove_theorem, ProofResult

MINIF2F_DIR = Path(__file__).parent.parent / "data" / "minif2f"


def load_minif2f(split: str = "test") -> list[dict]:
    """Load MiniF2F problems from JSONL file."""
    path = MINIF2F_DIR / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"MiniF2F data not found at {path}. "
            "Run: git clone https://github.com/openai/miniF2F data/minif2f"
        )
    problems = []
    with open(path) as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on MiniF2F-test")
    parser.add_argument("--model", default="accounts/fireworks/models/deepseek-v4-flash", help="OpenAI or Fireworks model name")
    parser.add_argument("--split", default="test", choices=["test", "valid"])
    parser.add_argument("--limit", type=int, default=None, help="Max problems to run")
    parser.add_argument("--output", default="results/minif2f_results.jsonl")
    parser.add_argument("--max-iterations", type=int, default=8)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    problems = load_minif2f(args.split)
    if args.limit:
        problems = problems[: args.limit]

    solved = 0
    with open(args.output, "w") as out_f:
        for i, problem in enumerate(problems):
            name = problem.get("name", f"problem_{i}")
            stmt = problem.get("formal_statement", problem.get("statement", ""))
            nl_proof = problem.get("informal_proof")

            print(f"[{i+1}/{len(problems)}] {name} ...", end=" ", flush=True)
            t0 = time.time()
            try:
                result = prove_theorem(
                    theorem_stmt=stmt,
                    nl_proof=nl_proof,
                    model=args.model,
                    max_iterations=args.max_iterations,
                )
                elapsed = time.time() - t0
                status = "SOLVED" if result.success else "FAILED"
                if result.success:
                    solved += 1
            except Exception as e:
                elapsed = time.time() - t0
                status = "ERROR"
                result = None

            print(f"{status} ({elapsed:.1f}s)")
            record = {
                "name": name,
                "status": status,
                "elapsed_s": round(elapsed, 2),
                "iterations": result.iterations if result else None,
                "proved_nodes": result.proved_nodes if result else [],
                "failed_nodes": result.failed_nodes if result else [],
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    total = len(problems)
    print(f"\nResults: {solved}/{total} solved ({100*solved/total:.1f}%)")
    print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()
