"""Evaluate the full pipeline on PutnamBench (672 problems, Lean 4)."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from blueprint import _reasoning_kwargs
from budget import Budget
from eval_records import evaluation_record
from llm_client import make_client
from tactic_portfolio import DEFAULT_TACTIC_PORTFOLIO

PUTNAM_DIR = Path(__file__).parent.parent / "data" / "putnam"


def _prove_in_subprocess(theorem_stmt, nl_proof, model, max_iterations, trace_path, queue,
                         allow_unvalidated_blueprint=False, enable_negation_probe=False,
                         retry_failed=False, tactic_portfolio=None,
                         artifact_dir=None, budget_kwargs=None):
    """Runs in a forked child process so a timeout can SIGKILL real work,
    not just abandon a thread that keeps burning API calls in the background.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from pipeline import prove_theorem
    from tactic_portfolio import DEFAULT_TACTIC_PORTFOLIO
    from tracer import JsonlTracer, NullTracer
    tracer = JsonlTracer(trace_path) if trace_path else NullTracer()
    result = prove_theorem(
        theorem_stmt=theorem_stmt,
        nl_proof=nl_proof,
        model=model,
        max_iterations=max_iterations,
        tracer=tracer,
        allow_unvalidated_blueprint=allow_unvalidated_blueprint,
        enable_negation_probe=enable_negation_probe,
        retry_failed=retry_failed,
        tactic_portfolio=list(tactic_portfolio) if tactic_portfolio else None,
        artifact_dir=Path(artifact_dir) if artifact_dir else None,
        budget=Budget(**(budget_kwargs or {})),
    )
    queue.put(result)

# PutnamBench ships informal *statements*, not proofs (informal_solution is
# only a one-line answer for "find X" problems, and is "None." for the rest).
# The paper's own "+NL" mode generates a proof sketch with a separate model
# call before blueprint generation -- this isn't a verbatim Appendix C prompt
# since the paper doesn't publish one for this auxiliary step.
NL_SKETCH_SYSTEM_PROMPT = (
    "You are a mathematician. Write a rigorous but concise natural-language proof "
    "sketch for the given competition problem. Plain mathematical prose only -- no "
    "Lean or other formal notation. State the key claims and the justification for "
    "each; skip routine algebra, but do not skip the actual mathematical ideas."
)


def generate_nl_proof_sketch(informal_statement: str, model: str) -> str:
    if not informal_statement:
        return ""
    client = make_client(model)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": NL_SKETCH_SYSTEM_PROMPT},
            {"role": "user", "content": informal_statement},
        ],
        max_completion_tokens=4096,
        **_reasoning_kwargs(model),
    )
    return response.choices[0].message.content or ""


def load_putnam() -> list[dict]:
    path = PUTNAM_DIR / "putnam_bench.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"PutnamBench data not found at {path}. "
            "Run: git clone https://github.com/trishullab/PutnamBench data/putnam"
        )
    problems = []
    with open(path) as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on PutnamBench")
    parser.add_argument("--model", default="labs-leanstral-1-5")
    parser.add_argument("--limit", type=int, default=50, help="Problems to run (default 50 for quick eval)")
    parser.add_argument("--output", default="results/putnam_results.jsonl")
    parser.add_argument("--max-iterations", type=int, default=16,
                        help="Refinement iterations (paper uses 16 for PutnamBench, 8 for MiniF2F)")
    parser.add_argument("--trace", metavar="PATH", nargs="?", const="",
                        help="Write JSONL trace (default: results/putnam/trace.jsonl)")
    parser.add_argument("--nl", action="store_true",
                        help="Generate a natural-language proof sketch to seed blueprint "
                             "generation (paper's '+NL' mode). Off by default, matching the paper.")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-problem wall-clock timeout in seconds (default 600 = 10 min). "
                             "The paper doesn't report wall-clock time at all, only token/dollar "
                             "cost, so this has no paper-derived value -- it's purely to stop one "
                             "stuck problem from eating the whole batch's time budget.")
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
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-compile-calls", type=int, default=None)
    parser.add_argument("--max-wall-time", type=float, default=None)
    args = parser.parse_args()
    budget_kwargs = {"max_total_tokens": args.max_tokens,
                     "max_compile_calls": args.max_compile_calls,
                     "max_wall_time_s": args.max_wall_time}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    problems = load_putnam()
    if args.limit:
        problems = problems[: args.limit]

    if args.trace is None:
        trace_base = None
    elif args.trace == "":
        trace_base = Path("results/putnam/trace.jsonl")
    else:
        trace_base = Path(args.trace)
    if trace_base:
        # Each problem gets its own trace file (trace_<problem>.jsonl) so the
        # live viewer never overlays nodes from two different problems into
        # one graph. Clear any trace files left over from a previous run.
        trace_base.parent.mkdir(parents=True, exist_ok=True)
        trace_base.unlink(missing_ok=True)
        for old in trace_base.parent.glob(f"{trace_base.stem}_*.jsonl"):
            old.unlink()
        print(f"Tracing per-problem to: {trace_base.parent}/{trace_base.stem}_<problem>.jsonl")

    solved = 0
    ctx = mp.get_context("fork")
    with open(args.output, "w") as out_f:
        for i, problem in enumerate(problems):
            name = problem.get("name", f"putnam_{i}")
            stmt = problem.get("formal_statement", problem.get("statement", ""))

            print(f"[{i+1}/{len(problems)}] {name} ...", end=" ", flush=True)
            t0 = time.time()
            try:
                nl_proof = None
                if args.nl:
                    nl_proof = generate_nl_proof_sketch(problem.get("informal_statement", ""), args.model)

                problem_trace_path = None
                if trace_base:
                    problem_trace_path = trace_base.parent / f"{trace_base.stem}_{name}.jsonl"
                    problem_trace_path.unlink(missing_ok=True)

                queue = ctx.Queue()
                proc = ctx.Process(
                    target=_prove_in_subprocess,
                    args=(stmt, nl_proof, args.model, args.max_iterations, problem_trace_path, queue,
                          args.allow_unvalidated_blueprint, args.enable_negation_probe,
                          args.retry_failed,
                          list(DEFAULT_TACTIC_PORTFOLIO) if args.tactic_portfolio else None,
                          args.artifacts, budget_kwargs),
                )
                proc.start()
                proc.join(timeout=args.timeout)
                if proc.is_alive():
                    # SIGTERM first, give it a moment, then SIGKILL -- this
                    # actually stops the API calls/spend, unlike abandoning
                    # a thread (which keeps running unbounded in the
                    # background even after the main loop moves on).
                    proc.terminate()
                    proc.join(5)
                    if proc.is_alive():
                        proc.kill()
                        proc.join()
                    elapsed = time.time() - t0
                    status = f"TIMEOUT (>{args.timeout}s)"
                    result = None
                else:
                    elapsed = time.time() - t0
                    result = queue.get() if not queue.empty() else None
                    if result is None:
                        status = "ERROR: subprocess exited without a result"
                    else:
                        status = "SOLVED" if result.success else "FAILED"
                        if result.success:
                            solved += 1
            except Exception as e:
                elapsed = time.time() - t0
                status = f"ERROR: {e}"
                result = None

            print(f"{status} ({elapsed:.1f}s)")
            record = evaluation_record(name, result, elapsed,
                                       status=("SOLVED" if result and result.success
                                               else "FAILED" if result else "ERROR"),
                                       error=(status if result is None and status.startswith("ERROR") else None))
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    total = len(problems)
    print(f"\nResults: {solved}/{total} solved ({100*solved/total:.1f}%)")
    print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()
