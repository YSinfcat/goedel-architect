"""
Evaluate Goedel-Architect (full 3-phase pipeline) on VeriSoftBench.

This script is a thin adapter: it loads VeriSoftBench problems, builds the
per-theorem compiler and repo-search instances, delegates to the goedel-arch
src/pipeline.py, then verifies results with VeriSoftBench's LeanREPL.

Usage:
    python eval/run_verisoftbench.py --repo VCV-io --limit 5 --trace
    python eval/run_verisoftbench.py --repo VCV-io --limit 20 --blueprint
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
GOEDEL_ARCH_ROOT = Path(__file__).parent.parent
GOEDEL_ARCH_SRC  = GOEDEL_ARCH_ROOT / "src"
VSB_ROOT         = GOEDEL_ARCH_ROOT.parent / "VeriSoftBench"

# Insert VSB first, then src/ — src/ wins on name conflicts (e.g. prompts.py)
sys.path.insert(0, str(VSB_ROOT))
sys.path.insert(0, str(GOEDEL_ARCH_SRC))

from core.evaluator import _clean_thm_stmt
from core.lean_interface import LeanREPL
from prompts.prompt_builder import PromptBuilder
import utils.utils as utils

from checkpoint import CheckpointState, path_for_theorem
from lean_compiler import AbstractLeanCompiler
from mathlib_retrieval import MathlibRetrieval
from pipeline import prove_theorem, ProofResult, run_phase1, run_phase2, run_phase3
from prover import GoedelProver
from repo_retrieval import RepoRetrieval
from tracer import JsonlTracer, NullTracer, TraceEvent
import graph_viz
from vsb_lean_compiler import VSBLeanCompiler

VSB_DATA      = VSB_ROOT / "data" / "verisoftbench.jsonl"
VSB_LEAN_SRC  = VSB_ROOT / "data" / "lean_repos"
VSB_PROMPTS   = VSB_ROOT / "prompts" / "templates"
CACHE_DIR     = GOEDEL_ARCH_ROOT / "results" / "repo_index_cache"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(
    limit: int | None = None,
    repo_filter: str | None = None,
    thm_filter: str | None = None,
    subset: int | None = None,
    seed: int = 0,
    exclude_repos: set[str] | None = None,
    aristotle_only: bool = False,
) -> list[dict]:
    """
    subset: random sample of `subset` tasks spanning the whole (filtered)
        dataset, rather than `limit`'s first-N-in-file-order (which clusters
        entirely on one repo — the jsonl is grouped by lean_root, e.g. the
        first 60 rows are all ArkLib). Use subset for a repo-diverse pilot.

    aristotle_only: restrict to the 100-task VeriSoftBench-Aristotle subset
        (dataset field `subset_aristotle`) — the specific task population
        Aristotle's reported 69% and Gemini-3-Pro's reported 65% (Table 3,
        VeriSoftBench-Aristotle row) were measured against. Comparing to
        those numbers on any other task selection isn't a like-for-like
        comparison.
    """
    problems = []
    with open(VSB_DATA) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if repo_filter and entry.get("lean_root") != repo_filter:
                    continue
                if exclude_repos and entry.get("lean_root") in exclude_repos:
                    continue
                if thm_filter and thm_filter not in entry.get("thm_name", ""):
                    continue
                if aristotle_only and not entry.get("subset_aristotle"):
                    continue
                problems.append(entry)
                if subset is None and limit and len(problems) >= limit:
                    break
    if subset is not None:
        rng = random.Random(seed)
        problems = rng.sample(problems, min(subset, len(problems)))
        problems.sort(key=lambda e: e.get("id", 0))
    return problems


# ---------------------------------------------------------------------------
# Per-theorem evaluation
# ---------------------------------------------------------------------------

def evaluate_theorem(
    theorem_entry: dict,
    lean_repl: LeanREPL,
    prompt_builder: PromptBuilder,
    retrieval: MathlibRetrieval,
    repo_retrieval: RepoRetrieval | None,
    mode: str,
    use_blueprint: bool,
    model: str,
    tracer,
    verbose: bool,
    checkpoint_dir: Path | None = None,
    aristotle_mode: bool = False,
    cascade_model: str | None = None,
    cascade_timeout_s: float | None = None,
    escalation_max_tool_calls: int | None = 1,
) -> dict:
    thm_name  = theorem_entry["thm_name"]
    lean_root = theorem_entry["lean_root"]
    rel_path  = theorem_entry["rel_path"]
    imports   = theorem_entry.get("imports", [])
    suffix    = theorem_entry.get("suffix", "")
    local_ctx = theorem_entry.get("local_ctx", "")

    thm_stmt = _clean_thm_stmt(
        theorem_entry["thm_stmt"],
        gt_proof=theorem_entry.get("ground_truth_proof", ""),
    )
    theorem_entry["thm_stmt"] = thm_stmt

    result = {
        "thm_name": thm_name, "lean_root": lean_root, "rel_path": rel_path,
        "thm_stmt": thm_stmt, "success": False,
        "proof": None, "aux_lemmas": None, "error": None,
        "wall_time_s": 0.0, "blueprint_used": use_blueprint,
        "model": model, "mode": mode,
    }

    t0 = time.monotonic()
    usage_names = {thm_name}

    try:
        # Build verification context
        verif_ctx = _build_verif_context(lean_repl, theorem_entry, lean_root,
                                          rel_path, imports, local_ctx, thm_stmt, thm_name,
                                          aristotle_mode=aristotle_mode)
        theorem_entry["verif_local_ctxs"] = verif_ctx

        # VeriSoftBench prompts
        sys_prompt  = prompt_builder.retrive_sys_prompt()
        user_prompt = prompt_builder.build_user_prompt(theorem_entry, mode=mode)

        # Per-theorem compiler (tracks call count for proof_id)
        def make_compiler() -> AbstractLeanCompiler:
            return VSBLeanCompiler(lean_repl, theorem_entry, call_prefix="ga")

        if use_blueprint:
            # Full 3-phase pipeline. checkpoint_path lets a killed/restarted
            # run resume mid-theorem (skip Phase 1, skip already-proved
            # nodes) instead of starting over — see src/checkpoint.py.
            checkpoint_path = path_for_theorem(checkpoint_dir, thm_name) if checkpoint_dir else None
            proof_result: ProofResult = prove_theorem(
                theorem_stmt=thm_stmt,
                model=model,
                compiler_factory=make_compiler,
                retrieval=retrieval,
                repo_retrieval=repo_retrieval,
                tracer=tracer,
                repo_context=verif_ctx,
                checkpoint_path=checkpoint_path,
                thm_name=thm_name,
                cascade_model=cascade_model,
                cascade_timeout_s=cascade_timeout_s,
                escalation_max_tool_calls=escalation_max_tool_calls,
            )
            # Use only the root-node proof body — final_lean_file is a full Lean
            # file with import statements, not a proof body VSBLeanCompiler can verify.
            raw_output = proof_result.proof_body
            # proof_body is JUST the root node's own tactic block; anything it
            # references by name (a proved sibling node) only exists as prompt
            # text unless re-declared as a real lemma here.
            aux_lemmas = proof_result.aux_lemma_decls
            # Phase 2's llm_usage events are tagged with each blueprint NODE's
            # own name, not the overall theorem name - widen the usage lookup
            # (Phase 1/3 events are already tagged with thm_name directly).
            usage_names |= set(proof_result.proved_nodes) | set(proof_result.failed_nodes)
        else:
            # Phase 2 only (direct per-theorem proving, no blueprint)
            prover = GoedelProver(model_id=model, retrieval=retrieval, tracer=tracer)
            result_obj = prover.prove_node(
                compiler=make_compiler(),
                node_name=thm_name,
                node_stmt=thm_stmt,
                sys_prompt=sys_prompt,
                user_prompt=user_prompt,
                repo_retrieval=repo_retrieval,
            )
            raw_output = result_obj.proof_body
            aux_lemmas = utils.get_lemmas_from_llm_output(raw_output)

        proof = utils.get_proof_from_llm_output(raw_output) or raw_output.strip()

        if not proof:
            result["error"] = "No proof found in prover output."
            result["wall_time_s"] = round(time.monotonic() - t0, 2)
            return result

        # thm_stmt (via _clean_thm_stmt) always ends in a bare ':=' expecting a
        # proof with no separator of its own, but GoedelProver's prover always
        # emits proof bodies starting with ':=' (see SYSTEM_SUFFIX in
        # prover.py) - concatenating both left a literal "... :=\n:= by ..."
        # in the compiled file, a hard parse error unrelated to proof
        # correctness. Drop the redundant leading ':=' before it's appended.
        if thm_stmt.rstrip().endswith(":=") and proof.lstrip().startswith(":="):
            proof = proof.lstrip()[2:].lstrip()

        name_mapping = utils.find_conflicting_names_from_local_context(verif_ctx, aux_lemmas)
        aux_lemmas, proof = utils.apply_name_replacements(aux_lemmas, proof, name_mapping)
        proof, aux_lemmas = utils.clean_leaked_identifiers(theorem_entry, proof, aux_lemmas)

        if verbose:
            print(f"[{thm_name}] Verifying final proof ...")

        success, error_msg = lean_repl.verify_proof(
            thm_name=thm_name,
            repo_name=lean_root,
            rel_path=rel_path,
            local_context=verif_ctx,
            theorem_stmt=thm_stmt,
            theorem_proof=proof,
            proof_id="ga_final",
            aux_lemmas=aux_lemmas or "",
            suffix=suffix,
        )

        result.update(proof=proof, aux_lemmas=aux_lemmas, success=success,
                      error=None if success else error_msg)

    except Exception as exc:
        result["error"] = f"Exception: {exc}"

    result["wall_time_s"] = round(time.monotonic() - t0, 2)

    tracer.emit(TraceEvent(
        kind="final_verify",
        thm_name=theorem_entry.get("thm_name", ""),
        ok=result["success"],
        args={
            "proof": result.get("proof", ""),
            "error": result.get("error", ""),
            "wall_time_s": result["wall_time_s"],
        },
    ))

    usage = _sum_usage_from_trace(tracer, usage_names)
    if usage:
        result["token_usage"] = usage
        print(f"  [{thm_name}] tokens used: {usage['total_tokens']:,} "
              f"({usage['prompt_tokens']:,} prompt + {usage['completion_tokens']:,} "
              f"completion, {usage['calls']} call(s))", flush=True)

    return result


def _build_verif_context(lean_repl, entry, lean_root, rel_path,
                          imports, local_ctx, thm_stmt, thm_name,
                          aristotle_mode: bool = False) -> str:
    """Build the repo_context string shown to Phase 1/3 (and used as the
    compilation-time local context for candidate proofs).

    Uses the dataset's own pre-curated `local_ctx` field (matching
    VeriSoftBench's "base context": local-file content excluding lemma
    proofs, per Section 3.2 of the paper), rather than reading the raw
    source file from disk. The raw-file read previously left preceding
    same-file lemma/theorem PROOF BODIES fully intact, handing the model
    complete worked proofs of nearby lemmas verbatim - a real, uncontrolled
    advantage no VeriSoftBench baseline condition gets (their curated_context
    and full_context both explicitly elide proof bodies of everything they
    provide). Tried eliding proofs from the raw-file read instead of
    switching to local_ctx (via VSB's own `utils.elide_proofs`), but that
    helper is only exercised elsewhere one declaration at a time - fed a
    whole multi-declaration file blob, it mis-tracks bracket/proof
    boundaries across declarations and silently swallows unrelated later
    content (confirmed concretely on StateT.set_get: truncated 6113 chars
    down to 1494 and dropped several lemmas entirely rather than eliding
    just their proofs). local_ctx is what VSB's own PromptBuilder already
    relies on for their reported results, so it's a safer source of truth.
    Falls back to the raw-file read only if local_ctx is empty.

    aristotle_mode additionally appends `used_local_lemmas` statements
    (already proof-free) to approximate VeriSoftBench-Aristotle's guarantee
    that every lemma statement the ground-truth proof used is visible to the
    prover, rather than left to repo_search to find.
    """
    if lean_root == "iris-lean":
        imports = [imp.replace("import src.", "import ") for imp in imports]
    ctx = "\n".join(imports) + "\n" + local_ctx

    if not local_ctx.strip():
        try:
            src_path = lean_repl.lean_src_dir / lean_root / rel_path
            if src_path.exists():
                full = src_path.read_text(encoding="utf-8")
                raw_ctx = utils.get_content_before_theorem(full, thm_stmt, thm_name=thm_name)
                if raw_ctx is not None:
                    ctx = raw_ctx
        except Exception:
            pass

    if aristotle_mode:
        lemma_texts = []
        for item in entry.get("used_local_lemmas", []):
            text = item.get("content", "").strip()
            if not text:
                continue
            # `used_local_lemmas` content is statement-only (no `:=`) - a bare
            # `theorem name : type` with nothing after it is a Lean parse
            # error, not just an incomplete proof, and breaks compilation of
            # everything that follows it in the same file. Give it a stub
            # proof so it's a syntactically valid (if unproven) declaration,
            # same convention as the paper's own admit-elided lemmas.
            if ":=" not in text:
                text += " := by sorry"
            lemma_texts.append(text)
        if lemma_texts:
            ctx += ("\n\n-- Lemma statements used by the ground-truth proof --\n"
                    + "\n\n".join(lemma_texts))

    return ctx


def evaluate_theorem_phase(
    theorem_entry: dict,
    lean_repl: LeanREPL,
    retrieval: MathlibRetrieval,
    repo_retrieval: RepoRetrieval | None,
    phase: int,
    checkpoint_dir: Path,
    model: str,
    tracer,
    aristotle_mode: bool = False,
    cascade_model: str | None = None,
    cascade_timeout_s: float | None = None,
    escalation_max_tool_calls: int | None = 1,
) -> dict:
    """Run exactly one of Phase 1/2/3 against this theorem's checkpoint file.

    Phase 2 requires a checkpoint already holding a blueprint (i.e. Phase 1
    ran at some point, in this process or an earlier one). Phase 3 requires
    that checkpoint to also hold node_results (i.e. Phase 2 ran). Neither
    phase re-runs what came before it — see src/checkpoint.py.
    """
    thm_name  = theorem_entry["thm_name"]
    lean_root = theorem_entry["lean_root"]
    rel_path  = theorem_entry["rel_path"]
    imports   = theorem_entry.get("imports", [])
    local_ctx = theorem_entry.get("local_ctx", "")

    thm_stmt = _clean_thm_stmt(
        theorem_entry["thm_stmt"],
        gt_proof=theorem_entry.get("ground_truth_proof", ""),
    )
    theorem_entry["thm_stmt"] = thm_stmt

    checkpoint_path = path_for_theorem(checkpoint_dir, thm_name)

    result = {
        "thm_name": thm_name, "lean_root": lean_root, "rel_path": rel_path,
        "phase": phase, "checkpoint": str(checkpoint_path), "ok": False,
    }

    def make_compiler() -> AbstractLeanCompiler:
        return VSBLeanCompiler(lean_repl, theorem_entry, call_prefix="ga")

    t0 = time.monotonic()
    try:
        verif_ctx = _build_verif_context(lean_repl, theorem_entry, lean_root,
                                          rel_path, imports, local_ctx, thm_stmt, thm_name,
                                          aristotle_mode=aristotle_mode)
        theorem_entry["verif_local_ctxs"] = verif_ctx

        if phase == 1:
            blueprint = run_phase1(
                theorem_stmt=thm_stmt, model=model, compiler=make_compiler(),
                repo_context=verif_ctx, checkpoint_path=checkpoint_path,
                repo_retrieval=repo_retrieval, tracer=tracer, thm_name=thm_name,
            )
            result.update(
                ok=True, validated=blueprint.fully_validated,
                nodes=[n.name for n in blueprint.nodes],
            )

        elif phase == 2:
            orch_result = run_phase2(
                checkpoint_path=checkpoint_path, compiler_factory=make_compiler,
                retrieval=retrieval, repo_retrieval=repo_retrieval, tracer=tracer,
                cascade_model=cascade_model, cascade_timeout_s=cascade_timeout_s,
                escalation_max_tool_calls=escalation_max_tool_calls,
            )
            result.update(
                ok=True,
                validated=orch_result.all_proved(),
                all_proved=orch_result.all_proved(),
                proved=sorted(orch_result.proved),
                failed=sorted(orch_result.failed.keys()),
            )
            # Phase 2's llm_usage events are tagged with each blueprint NODE's
            # own name (matching theorem_start's existing convention, which
            # graph_viz.py depends on), not the overall theorem name - widen
            # the usage lookup to the node names this call actually attempted.
            usage_names = set(orch_result.node_results.keys()) | {thm_name}

        elif phase == 3:
            blueprint = run_phase3(
                checkpoint_path=checkpoint_path, compiler=make_compiler(),
                model=model, repo_context=verif_ctx,
                repo_retrieval=repo_retrieval, tracer=tracer, thm_name=thm_name,
            )
            result.update(
                ok=True, validated=blueprint.fully_validated,
                nodes=[n.name for n in blueprint.nodes],
            )

        else:
            raise ValueError(f"Unknown phase {phase}")

    except Exception as exc:
        result["error"] = str(exc)

    result["wall_time_s"] = round(time.monotonic() - t0, 2)
    usage = _sum_usage_from_trace(tracer, usage_names if phase == 2 else {thm_name})
    if usage:
        result["token_usage"] = usage
        print(f"  [{thm_name}] tokens used this phase: "
              f"{usage['total_tokens']:,} ({usage['prompt_tokens']:,} prompt + "
              f"{usage['completion_tokens']:,} completion, {usage['calls']} call(s))",
              flush=True)
    return result


def _sum_usage_from_trace(tracer, names: set[str]) -> dict | None:
    """Sum `llm_usage` trace events for a set of names. Phase 1/3 events are
    tagged with the overall theorem name; Phase 2 events are tagged with
    each blueprint NODE's own name (matching theorem_start's existing
    convention, which graph_viz.py depends on) - callers must pass the right
    set of names for whichever phase just ran. Only works when tracing is
    enabled (NullTracer has no file to read back from, so this silently
    returns None rather than requiring --trace everywhere)."""
    path = getattr(tracer, "path", None)
    if path is None or not path.exists():
        return None
    prompt = completion = total = calls = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("kind") != "llm_usage" or event.get("thm_name") not in names:
                continue
            args = event.get("args") or {}
            prompt += args.get("prompt_tokens", 0)
            completion += args.get("completion_tokens", 0)
            total += args.get("total_tokens", 0)
            calls += 1
    if calls == 0:
        return None
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": total, "calls": calls}


def summarise(results: list[dict]) -> dict:
    total  = len(results)
    proved = sum(1 for r in results if r["success"])
    return {
        "total": total,
        "proved": proved,
        "pass_rate": round(proved / total, 4) if total else 0.0,
        "avg_wall_time_s": round(sum(r["wall_time_s"] for r in results) / total, 2) if total else 0.0,
    }


def load_done_results(results_file: Path, model: str, mode: str, blueprint: bool) -> list[dict]:
    """Results already recorded in results_file under the same
    model/mode/blueprint config — used by --resume to skip re-running
    theorems a prior (possibly interrupted) run already finished, and to
    fold their outcome back into this run's final summary."""
    done: list[dict] = []
    if not results_file.exists():
        return done
    with open(results_file) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if (r.get("model") == model and r.get("mode") == mode
                    and r.get("blueprint_used") == blueprint):
                done.append(r)
    return done


def _run_phase_only(
    problems: list[dict],
    lean_repl: LeanREPL,
    retrieval: MathlibRetrieval,
    repo_retrievals: dict[str, RepoRetrieval | None],
    args: argparse.Namespace,
    output_dir: Path,
    checkpoint_dir: Path,
    tracer,
) -> None:
    """Run --phase N over `problems`, one theorem at a time per repo (same
    lake-safety constraint as the full pipeline — see run_repo_queue)."""
    phase_results_file = output_dir / f"phase{args.phase}_results.jsonl"

    by_repo: dict[str, list[dict]] = defaultdict(list)
    for entry in problems:
        by_repo[entry.get("lean_root", "")].append(entry)

    num_workers = max(1, min(args.workers, len(by_repo))) if by_repo else 0
    total = len(problems)
    completed = 0
    results_lock = threading.Lock()
    all_results: list[dict] = []

    def run_repo_queue(repo_problems: list[dict]) -> None:
        nonlocal completed
        for entry in repo_problems:
            thm_name = entry["thm_name"]
            res = evaluate_theorem_phase(
                theorem_entry=entry,
                lean_repl=lean_repl,
                retrieval=retrieval,
                repo_retrieval=repo_retrievals.get(entry.get("lean_root", "")),
                phase=args.phase,
                checkpoint_dir=checkpoint_dir,
                model=args.model,
                tracer=tracer,
                aristotle_mode=args.aristotle_subset,
                cascade_model=args.cascade_model,
                cascade_timeout_s=args.cascade_timeout,
                escalation_max_tool_calls=args.escalation_max_tool_calls,
            )
            with results_lock:
                all_results.append(res)
                completed += 1
                i = completed
                with open(phase_results_file, "a") as f:
                    f.write(json.dumps(res) + "\n")
            # "validated" (real compile/proof signal) is the ground truth for
            # OK/FAIL when present; "ok" (didn't raise) is only a fallback for
            # phases/paths that don't yet set it, so a give-up/best-effort
            # result is never reported as a pass.
            validated = res.get("validated", res["ok"])
            status = "OK" if validated else "FAIL"
            print(f"  [{i}/{total}] {status} {thm_name} ({res.get('wall_time_s', 0):.1f}s)"
                  + (f" — {res['error']}" if not res["ok"] else ""))

    if by_repo:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(run_repo_queue, repo_problems)
                       for repo_problems in by_repo.values()]
            for fut in as_completed(futures):
                fut.result()

    ok_count = sum(1 for r in all_results if r.get("validated", r["ok"]))
    print(f"\nPhase {args.phase}: {ok_count}/{total} succeeded")
    print(f"Results saved to {phase_results_file}")
    print(f"Checkpoints saved to {checkpoint_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Goedel-Architect on VeriSoftBench")
    parser.add_argument("--model",    default="accounts/fireworks/models/deepseek-v4-flash")
    parser.add_argument("--cascade-model", default=None,
                        help="If set, Phase 2 first attempts every node with this "
                             "(cheaper) model; --model is only used to escalate a "
                             "node that doesn't come back solved. Skips --model's "
                             "rate entirely for nodes the cheap model can already "
                             "close (many blueprint nodes are trivial one-liners). "
                             "Default (unset): every node goes straight to --model, "
                             "unchanged from prior behavior.")
    parser.add_argument("--cascade-timeout", type=float, default=90.0,
                        help="Wall-clock bound (seconds) for the cheap cascade "
                             "attempt only, independent of the escalation attempt's "
                             "own timeout — bounds how long a stuck cheap model can "
                             "delay falling back to --model. Only used when "
                             "--cascade-model is set.")
    parser.add_argument("--escalation-max-tool-calls", type=int, default=1,
                        help="Caps the escalated (--model) attempt's own internal "
                             "fix-and-retry budget after a failed cascade attempt "
                             "(default 1: a single lean_compile call, no follow-up "
                             "correction round). The cheap cascade attempt keeps its "
                             "full default budget regardless. Only applies when "
                             "--cascade-model is set; a direct (non-cascaded) call "
                             "to --model is unaffected.")
    parser.add_argument("--limit",   type=int, default=None)
    parser.add_argument("--subset",  type=int, default=None,
                        help="Random sample of N tasks spanning all repos "
                             "(unlike --limit, which takes the first N in "
                             "file order and clusters on one repo)")
    parser.add_argument("--seed",    type=int, default=0, help="--subset sample seed")
    parser.add_argument("--aristotle-subset", action="store_true",
                        help="Restrict to the 100-task VeriSoftBench-Aristotle "
                             "subset and use the matching context variant "
                             "(same-file preceding proofs elided, plus "
                             "ground-truth-used lemma statements guaranteed "
                             "visible) — for comparing against Table 3's "
                             "VeriSoftBench-Aristotle row (Aristotle 69%, "
                             "Gemini-3-Pro 65%)")
    parser.add_argument("--repo",    default=None)
    parser.add_argument("--exclude-repo", default=None,
                        help="Comma-separated repo names to skip (e.g. known-broken builds)")
    parser.add_argument("--thm",     default=None, help="Run a single theorem by name")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers, sharded by repo (theorems from the "
                             "same repo always run sequentially within one worker — "
                             "`lake build`/compilation is not safe to parallelize on "
                             "a shared repo; different repos may run concurrently)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip theorems already recorded in the output "
                             "results.jsonl under the same model/mode/blueprint config")
    parser.add_argument("--output",  default="results/verisoftbench/")
    parser.add_argument("--mode",    default="filtered_context",
                        choices=["filtered_context", "full_context"])
    parser.add_argument("--quick",   action="store_true", help="10 problems, 2 workers")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--blueprint", action="store_true",
                        help="Use full 3-phase pipeline (Phase 1 + 2 + 3)")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=None,
                        help="Run exactly one phase per theorem instead of the "
                             "full pipeline, persisting/reading state from "
                             "--checkpoint-dir so a later invocation can pick "
                             "up where this one left off. Phase 2 requires a "
                             "checkpoint with a blueprint (run --phase 1 first, "
                             "any time before); Phase 3 requires a checkpoint "
                             "with node results (run --phase 2 first). Neither "
                             "re-runs the phase(s) before it.")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Directory for per-theorem checkpoint files used "
                             "by --phase (default: <output>/checkpoints/)")
    parser.add_argument("--trace", metavar="PATH", nargs="?", const="",
                        help="Write JSONL trace (default: results/verisoftbench/trace.jsonl)")
    args = parser.parse_args()

    if args.quick:
        args.limit, args.workers = 10, 2

    output_dir = GOEDEL_ARCH_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.trace is None:
        trace_path = None
    elif args.trace == "":
        trace_path = output_dir / "trace.jsonl"
    else:
        trace_path = Path(args.trace)
    if trace_path:
        trace_path.unlink(missing_ok=True)
    tracer = JsonlTracer(trace_path) if trace_path else NullTracer()

    exclude_repos = set(r.strip() for r in args.exclude_repo.split(",") if r.strip()) if args.exclude_repo else None

    print("Loading VeriSoftBench dataset ...")
    problems = load_dataset(limit=args.limit, repo_filter=args.repo, thm_filter=args.thm,
                             subset=args.subset, seed=args.seed, exclude_repos=exclude_repos,
                             aristotle_only=args.aristotle_subset)
    print(f"  {len(problems)} problems"
          + (f" (repo={args.repo})" if args.repo else "")
          + (f" (exclude_repo={sorted(exclude_repos)})" if exclude_repos else "")
          + (f" (thm={args.thm})" if args.thm else "")
          + (f" (limit={args.limit})" if args.limit else "")
          + (f" (subset={args.subset}, seed={args.seed})" if args.subset else "")
          + (" (aristotle_subset=100)" if args.aristotle_subset else ""))
    if trace_path:
        print(f"  Tracing to: {trace_path}")
    if args.blueprint:
        print("  Mode: full 3-phase pipeline (Phase 1 + Phase 2 + Phase 3)")
    else:
        print("  Mode: Phase 2 only (direct per-theorem proving)")

    retrieval    = MathlibRetrieval()
    lean_repl    = LeanREPL(lean_src_dir=VSB_LEAN_SRC)
    prompt_builder = PromptBuilder(VSB_PROMPTS, mode=args.mode)

    # Build repo retrieval index (shared across all theorems in the same repo)
    repo_retrievals: dict[str, RepoRetrieval | None] = {}
    for entry in problems:
        repo = entry.get("lean_root", "")
        if repo not in repo_retrievals:
            repo_root = VSB_LEAN_SRC / repo
            if repo_root.exists():
                repo_retrievals[repo] = RepoRetrieval(repo_root, cache_dir=CACHE_DIR)
            else:
                repo_retrievals[repo] = None

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.phase is not None:
        print(f"  Phase-only mode: Phase {args.phase}, checkpoints in {checkpoint_dir}")
        _run_phase_only(problems, lean_repl, retrieval, repo_retrievals, args, output_dir, checkpoint_dir, tracer)
        return

    if args.blueprint:
        print(f"  Checkpoints in {checkpoint_dir} — a killed/restarted run resumes "
              f"mid-theorem automatically (skips Phase 1, skips already-proved nodes)")

    results_file = output_dir / "results.jsonl"

    # Resume: skip theorems already recorded under the same run config, and
    # keep prior results (from this or earlier interrupted runs) in the file
    # and in the final summary.
    all_results: list[dict] = []
    if args.resume:
        prior = load_done_results(results_file, args.model, args.mode, args.blueprint)
        if prior:
            done_keys = {(r.get("thm_name", ""), r.get("lean_root", "")) for r in prior}
            before = len(problems)
            problems = [p for p in problems
                        if (p["thm_name"], p.get("lean_root", "")) not in done_keys]
            all_results.extend(prior)
            print(f"  --resume: skipping {before - len(problems)} already-done theorem(s)")
    else:
        results_file.unlink(missing_ok=True)  # fresh run overwrites prior results

    # Group by repo: theorems from the same repo must run sequentially (a
    # shared `lake`/compiler working directory is not safe to touch from two
    # threads at once), but different repos can run fully in parallel.
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for entry in problems:
        by_repo[entry.get("lean_root", "")].append(entry)

    num_workers = max(1, min(args.workers, len(by_repo))) if by_repo else 0
    print(f"Running with model={args.model}, workers={num_workers} "
          f"(sharded across {len(by_repo)} repo(s)) ...")

    results_lock = threading.Lock()
    total = len(problems)  # theorems being run this invocation (excludes --resume skips)
    completed = 0

    def run_one(entry: dict) -> dict:
        return evaluate_theorem(
            theorem_entry=entry,
            lean_repl=lean_repl,
            prompt_builder=prompt_builder,
            retrieval=retrieval,
            repo_retrieval=repo_retrievals.get(entry.get("lean_root", "")),
            mode=args.mode,
            use_blueprint=args.blueprint,
            model=args.model,
            tracer=tracer,
            verbose=args.verbose,
            checkpoint_dir=checkpoint_dir if args.blueprint else None,
            aristotle_mode=args.aristotle_subset,
            cascade_model=args.cascade_model,
            cascade_timeout_s=args.cascade_timeout,
            escalation_max_tool_calls=args.escalation_max_tool_calls,
        )

    def run_repo_queue(repo_problems: list[dict]) -> None:
        nonlocal completed
        for entry in repo_problems:
            thm_name = entry["thm_name"]
            try:
                res = run_one(entry)
            except Exception as exc:
                res = {"thm_name": thm_name, "lean_root": entry.get("lean_root", ""),
                       "success": False, "error": str(exc), "wall_time_s": 0.0,
                       "model": args.model, "mode": args.mode, "blueprint_used": args.blueprint}
            with results_lock:
                all_results.append(res)
                completed += 1
                i = completed
                with open(results_file, "a") as f:
                    f.write(json.dumps(res) + "\n")
            status = "PASS" if res["success"] else "FAIL"
            print(f"  [{i}/{total}] {status} {thm_name} ({res.get('wall_time_s',0):.1f}s)")

    if by_repo:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(run_repo_queue, repo_problems)
                       for repo_problems in by_repo.values()]
            for fut in as_completed(futures):
                fut.result()  # surface any unexpected exception from run_repo_queue itself

    summary = summarise(all_results)
    print(f"\nResults: {summary['proved']}/{summary['total']} proved "
          f"({summary['pass_rate']*100:.1f}%), avg {summary['avg_wall_time_s']:.1f}s/theorem")

    with open(output_dir / "summary.json", "w") as f:
        json.dump({**summary, "model": args.model, "mode": args.mode,
                   "blueprint": args.blueprint}, f, indent=2)

    print(f"Results saved to {output_dir}")

    if trace_path and trace_path.exists():
        graph_viz.generate(trace_path)
        print(f"\nView graph:  python eval/serve_viz.py {trace_path}")


if __name__ == "__main__":
    main()
