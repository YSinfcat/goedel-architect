"""Evaluate the full pipeline on MiniF2F-test (244 problems, Lean 4 split)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import prove_theorem, ProofResult
from tactic_portfolio import DEFAULT_TACTIC_PORTFOLIO

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
    parser.add_argument("--model", default="labs-leanstral-1-5", help="OpenAI, Fireworks, or Mistral model name")
    parser.add_argument("--split", default="test", choices=["test", "valid"])
    parser.add_argument("--limit", type=int, default=None, help="Max problems to run")
    parser.add_argument("--output", default="results/minif2f_results.jsonl")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--allow-unvalidated-blueprint", action="store_true",
                        help="Debugging escape: let blueprints that never passed a real Lean "
                             "compile into Phase 2. Success still requires independent final "
                             "verification, so this cannot manufacture a fake success.")
    parser.add_argument("--enable-negation-probe", action="store_true",
                        help="Enable the experimental FORMALLY_NEGATED probe (known flaw: it "
                             "never compiles a real negated goal - treat its output as advisory).")
    parser.add_argument("--tactic-portfolio", action="store_true",
                        help="Try a deterministic tactic list (simp/aesop/omega/...) "
                             "on each node before any model call - a hit costs zero "
                             "LLM tokens. Recorded in the run fingerprint.")
    parser.add_argument("--artifacts", metavar="DIR", default=None,
                        help="On success, write an auditable artifact bundle "
                             "(proof.lean, blueprints, verification.json, ...) per theorem.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Continue past a checkpointed terminal failure (done=True, "
                             "success=False) instead of returning the cached verdict forever.")
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
                    allow_unvalidated_blueprint=args.allow_unvalidated_blueprint,
                    enable_negation_probe=args.enable_negation_probe,
                    retry_failed=args.retry_failed,
                    tactic_portfolio=(list(DEFAULT_TACTIC_PORTFOLIO)
                                      if args.tactic_portfolio else None),
                    artifact_dir=Path(args.artifacts) if args.artifacts else None,
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
