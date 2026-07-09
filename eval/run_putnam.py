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
from llm_client import make_client

PUTNAM_DIR = Path(__file__).parent.parent / "data" / "putnam"


def _prove_in_subprocess(theorem_stmt, nl_proof, model, max_iterations, trace_path, queue):
    """Runs in a forked child process so a timeout can SIGKILL real work,
    not just abandon a thread that keeps burning API calls in the background.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from pipeline import prove_theorem
    from tracer import JsonlTracer, NullTracer
    tracer = JsonlTracer(trace_path) if trace_path else NullTracer()
    result = prove_theorem(
        theorem_stmt=theorem_stmt,
        nl_proof=nl_proof,
        model=model,
        max_iterations=max_iterations,
        tracer=tracer,
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
    parser.add_argument("--model", default="accounts/fireworks/models/deepseek-v4-flash")
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
    args = parser.parse_args()

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
                    args=(stmt, nl_proof, args.model, args.max_iterations, problem_trace_path, queue),
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
