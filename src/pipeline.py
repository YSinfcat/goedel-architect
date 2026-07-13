"""Top-level pipeline: Blueprint → Parallel Proving → Refinement loop.

Wires all three phases together and runs up to 8 refinement iterations
(matching Appendix A of the paper).

The `compiler` parameter is injectable so the same pipeline works with:
  - LeanCompiler (standalone Lean projects via `lake env lean`)
  - VSBLeanCompiler (VeriSoftBench repos via LeanREPL)

Each phase's real work (`_execute_phase1/2/3`) is shared between this
module's all-in-one `prove_theorem` loop and the standalone `run_phase1/2/3`
entry points below - both drive the exact same phase bodies against a
`CheckpointState`, so a fix to one no longer risks leaving the other
unfixed. `run_phase1/2/3`'s public signatures (`checkpoint_path: Path`) are
unchanged from before - `eval/run_verisoftbench.py` depends on running each
phase as a separate process against a checkpoint file on disk.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from blueprint import Blueprint, generate_blueprint
from checkpoint import CheckpointState
from lean_compiler import AbstractLeanCompiler, LeanCompiler
from mathlib_retrieval import MathlibRetrieval
from orchestrator import NodeResult, OrchestratorResult, prove_dag
from refinement import refine_blueprint
from tracer import NullTracer

# From Appendix A
MAX_REFINEMENT_ITERATIONS = 8


@dataclass
class ProofResult:
    success: bool
    theorem_name: str
    proof_body: str = ""        # proof body of the root node if solved
    final_lean_file: str = ""   # full assembled Lean file
    aux_lemma_decls: str = ""   # every other proved node, re-declared as a real
                                # lemma/theorem, so a caller compiling just
                                # proof_body against the root's bare signature
                                # (e.g. VeriSoftBench's own verify_proof) can
                                # still resolve the root proof's references to
                                # its dependencies by name
    iterations: int = 0
    proved_nodes: list[str] = field(default_factory=list)
    failed_nodes: list[str] = field(default_factory=list)


def _aux_lemma_decls(blueprint: Blueprint, proved_cache: dict[str, str], root_name: str) -> str:
    return "\n\n".join(
        f"{node.signature()} {proved_cache[node.name]}"
        for node in blueprint.dependency_order()
        if node.name != root_name and node.name in proved_cache
    )


# ---------------------------------------------------------------------------
# Phase bodies - shared by prove_theorem's loop and the standalone
# run_phase1/2/3 entry points below.
# ---------------------------------------------------------------------------

def _execute_phase1(
    state: CheckpointState,
    theorem_stmt: str,
    nl_proof: str | None,
    model: str,
    compiler: AbstractLeanCompiler | None,
    repo_context: str | None,
    repo_retrieval,
    tracer,
    thm_name: str,
) -> Blueprint:
    blueprint = generate_blueprint(
        theorem_stmt=theorem_stmt,
        nl_proof=nl_proof,
        model=model,
        compiler=compiler,
        repo_context=repo_context,
        repo_retrieval=repo_retrieval,
        tracer=tracer,
        thm_name=thm_name,
    )
    state.complete_phase1(blueprint)
    return blueprint


def _execute_phase2(
    state: CheckpointState,
    blueprint: Blueprint,
    compiler: AbstractLeanCompiler | None,
    compiler_factory: Callable[[], AbstractLeanCompiler] | None,
    retrieval: MathlibRetrieval,
    repo_retrieval,
    model: str,
    tracer,
    node_timeout_s: float | None,
    cascade_model: str | None,
    cascade_timeout_s: float | None,
    escalation_max_tool_calls: int | None,
    thm_name: str = "",
) -> OrchestratorResult:
    nodes_to_try = set(blueprint.nodes_by_name()) - set(state.proved_cache)
    orch_result = asyncio.run(
        prove_dag(
            blueprint=blueprint,
            compiler=compiler,
            compiler_factory=compiler_factory,
            retrieval=retrieval,
            repo_retrieval=repo_retrieval,
            model=model,
            proved_cache=dict(state.proved_cache),
            nodes_to_retry=nodes_to_try,
            tracer=tracer,
            node_timeout_s=node_timeout_s,
            cascade_model=cascade_model,
            cascade_timeout_s=cascade_timeout_s,
            escalation_max_tool_calls=escalation_max_tool_calls,
            thm_name=thm_name,
        )
    )
    state.complete_phase2(orch_result, blueprint)
    return orch_result


def _execute_phase3(
    state: CheckpointState,
    blueprint: Blueprint,
    orch_result: OrchestratorResult,
    compiler: AbstractLeanCompiler,
    model: str,
    repo_context: str | None,
    iteration: int,
    max_iterations: int,
    repo_retrieval,
    tracer,
    thm_name: str,
) -> Blueprint:
    new_blueprint = refine_blueprint(
        blueprint=blueprint,
        orch_result=orch_result,
        compiler=compiler,
        model=model,
        repo_context=repo_context,
        history=state.refinement_history,  # mutated in place, even on failure
        iteration=iteration,
        max_iterations=max_iterations,
        repo_retrieval=repo_retrieval,
        tracer=tracer,
        thm_name=thm_name,
    )
    state.complete_phase3(new_blueprint, state.refinement_history)
    return new_blueprint


def prove_theorem(
    theorem_stmt: str,
    nl_proof: str | None = None,
    model: str = "gpt-5.5",
    phase1_model: str | None = None,
    phase3_model: str | None = None,
    compiler: AbstractLeanCompiler | None = None,
    compiler_factory: Callable[[], AbstractLeanCompiler] | None = None,
    retrieval: MathlibRetrieval | None = None,
    repo_retrieval=None,
    max_iterations: int = MAX_REFINEMENT_ITERATIONS,
    tracer=None,
    project_root: Path | None = None,
    repo_context: str | None = None,
    node_timeout_s: float | None = 300.0,
    checkpoint_path: Path | None = None,
    thm_name: str = "",
    cascade_model: str | None = None,
    cascade_timeout_s: float | None = None,
    escalation_max_tool_calls: int | None = 1,
) -> ProofResult:
    """
    Full Goedel-Architect pipeline for a single theorem.

    1. Generate @[blueprint]-annotated dependency graph (Phase 1)
    2. Prove each node in parallel (Phase 2)
    3. Refine blueprint on failures and repeat (Phase 3)
    Up to `max_iterations` refinement loops (default 8, per Appendix A).

    Args:
        model: Phase 2 (per-node proving) model. Also the Phase 1/3 model
            whenever phase1_model/phase3_model are left unset.
        phase1_model: Overrides `model` for Phase 1 (blueprint generation) only.
        phase3_model: Overrides `model` for Phase 3 (refinement) only.
        compiler: Shared compiler instance (used for all nodes).
        compiler_factory: Called once per node to get a fresh compiler.
            Use this for VSBLeanCompiler which tracks call state per-theorem.
            If both are provided, compiler_factory takes precedence.
        repo_retrieval: Optional RepoRetrieval for repo_search tool.
        tracer: Optional tracer for emitting events.
        node_timeout_s: Per-node wall-clock bound in Phase 2 (see
            orchestrator.prove_dag). None disables the bound.
        checkpoint_path: If given, state is saved after every phase and, if
            the file already exists, resumed from wherever it left off
            (skipping Phase 1 and any already-proved nodes). See checkpoint.py
            and run_phase1/run_phase2/run_phase3 below for running phases
            standalone instead of through this all-in-one loop.
    """
    tracer = tracer or NullTracer()

    if compiler is None and compiler_factory is None:
        root = project_root or Path(__file__).parent.parent / "goedel_lean"
        compiler = LeanCompiler(root)

    retrieval = retrieval or MathlibRetrieval()

    state = CheckpointState.load_or_none(checkpoint_path)

    if state and state.theorem_stmt and state.theorem_stmt != theorem_stmt:
        # checkpoint_path is normally keyed by theorem name (see
        # path_for_theorem), so this should never fire in normal use - a
        # mismatch means something unusual happened (a manually-overridden
        # checkpoint_path, a copy/paste error). Silently resuming or
        # returning a cached result for the WRONG theorem statement is worse
        # than refusing outright.
        raise ValueError(
            f"Checkpoint {checkpoint_path} was created for a different "
            f"theorem_stmt than requested - refusing to resume/reuse it.\n"
            f"  checkpoint theorem_stmt: {state.theorem_stmt!r}\n"
            f"  requested theorem_stmt:  {theorem_stmt!r}"
        )

    if state and state.done:
        # A prior run already finished this theorem (success or exhausted
        # all iterations) — reconstruct the result from the checkpoint
        # instead of re-running Phase 2/3 (which would burn API calls
        # re-deriving an answer that's already on disk).
        print(f"[Resume] checkpoint at {checkpoint_path} already done "
              f"(success={state.success}) — returning cached result", flush=True)
        return _proof_result_from_checkpoint(state)

    resumed_blueprint = state.get_blueprint() if state else None

    if resumed_blueprint is not None:
        blueprint = resumed_blueprint
        print(f"[Resume] loaded checkpoint at iteration {state.iteration + 1}, "
              f"{len(state.proved_cache)} node(s) already proved", flush=True)
    else:
        # Phase 1: Blueprint generation
        # Don't use compiler_factory for blueprint validation: factory compilers are
        # stateful (track call counts, write temp files) and Phase 1 would exhaust
        # retries on type-signature errors the LLM can't fix without repo context.
        # Pass only an explicitly-shared compiler (e.g. standalone LeanCompiler).
        blueprint_compiler = compiler  # None when only compiler_factory is provided
        state = CheckpointState(theorem_stmt=theorem_stmt, model=model, repo_context=repo_context or "")
        blueprint = _execute_phase1(
            state, theorem_stmt, nl_proof, phase1_model or model, blueprint_compiler,
            repo_context, repo_retrieval, tracer, thm_name,
        )
        if checkpoint_path:
            state.save(checkpoint_path)

    orch_result: OrchestratorResult | None = None

    for iteration in range(state.iteration, max_iterations):
        # Phase 2: Parallel proving
        nodes_to_try = set(blueprint.nodes_by_name()) - set(state.proved_cache)
        print(f"\n[Phase 2 iteration {iteration+1}] Proving {len(nodes_to_try)} nodes: {sorted(nodes_to_try)}", flush=True)

        orch_result = _execute_phase2(
            state, blueprint, compiler, compiler_factory, retrieval, repo_retrieval, model,
            tracer, node_timeout_s, cascade_model, cascade_timeout_s, escalation_max_tool_calls,
            thm_name=thm_name,
        )
        state.iteration = iteration

        for name, nr in orch_result.node_results.items():
            status = nr.result.signal.value
            proof_preview = repr(nr.result.proof_body[:60]) if nr.result.proof_body else ""
            print(f"  node '{name}': {status} {proof_preview}", flush=True)

        print(f"  proved so far: {sorted(state.proved_cache.keys())}", flush=True)

        if checkpoint_path:
            state.save(checkpoint_path)

        if orch_result.all_proved():
            root_name = blueprint.target_theorem
            root_proof = state.proved_cache.get(root_name, "")
            if checkpoint_path:
                state.done = True
                state.success = True
                state.save(checkpoint_path)
            return ProofResult(
                success=True,
                theorem_name=root_name,
                proof_body=root_proof,
                final_lean_file=_assemble_final_file(blueprint, orch_result),
                aux_lemma_decls=_aux_lemma_decls(blueprint, state.proved_cache, root_name),
                iterations=iteration + 1,
                proved_nodes=list(orch_result.proved),
                failed_nodes=[],
            )

        if iteration == max_iterations - 1:
            break

        # Phase 3: Refinement
        print(f"\n[Phase 3 iteration {iteration+1}] Refining blueprint ...", flush=True)
        failed = [n for n in orch_result.node_results if orch_result.node_results[n].result.signal.value != "solved"]
        print(f"  failed nodes: {failed}", flush=True)
        refinement_compiler = compiler or (compiler_factory() if compiler_factory else None)
        if refinement_compiler is None:
            break
        proved_before = set(state.proved_cache)
        try:
            blueprint = _execute_phase3(
                state, blueprint, orch_result, refinement_compiler, phase3_model or model,
                repo_context, iteration, max_iterations, repo_retrieval, tracer, thm_name,
            )
            print(f"  new blueprint has {len(blueprint.nodes)} nodes: {[n.name for n in blueprint.nodes]}", flush=True)
        except RuntimeError as e:
            print(f"  refinement failed: {e}", flush=True)
            # refine_blueprint mutates state.refinement_history in place
            # before its own retry loop, so this round's attempt is already
            # recorded even though refinement ultimately failed - persisting
            # here keeps the checkpoint's refinement_history from being
            # silently shorter than what was actually tried (matters for
            # post-mortem diagnosis).
            if checkpoint_path:
                state.save(checkpoint_path)
            break  # refinement failed, stop iterations

        stale = proved_before - set(state.proved_cache)
        if stale:
            print(f"  invalidated stale proof(s) (no longer match the current node shape): {sorted(stale)}", flush=True)

        if checkpoint_path:
            state.save(checkpoint_path)

    proved = list(orch_result.proved) if orch_result else []
    failed = list(orch_result.failed.keys()) if orch_result else []
    root_proof = state.proved_cache.get(blueprint.target_theorem, "")
    if checkpoint_path:
        state.done = True
        state.success = False
        state.save(checkpoint_path)
    return ProofResult(
        success=False,
        theorem_name=blueprint.target_theorem,
        proof_body=root_proof,
        final_lean_file=_assemble_partial_file(blueprint, orch_result, state.proved_cache),
        aux_lemma_decls=_aux_lemma_decls(blueprint, state.proved_cache, blueprint.target_theorem),
        iterations=max_iterations,
        proved_nodes=proved,
        failed_nodes=failed,
    )


# ---------------------------------------------------------------------------
# Standalone phase entry points
#
# Each function does exactly one phase against a checkpoint file on disk, so
# a caller can run e.g. Phase 2 without Phase 1 having just run in the same
# process (only having run at some point and left a checkpoint behind), and
# Phase 3 without re-running Phase 1 or Phase 2.
# ---------------------------------------------------------------------------

def run_phase1(
    theorem_stmt: str,
    nl_proof: str | None = None,
    model: str = "gpt-5.5",
    compiler: AbstractLeanCompiler | None = None,
    repo_context: str | None = None,
    checkpoint_path: Path | None = None,
    repo_retrieval=None,
    tracer=None,
    thm_name: str = "",
) -> Blueprint:
    """Run Phase 1 (blueprint generation) alone and checkpoint the result."""
    state = CheckpointState(theorem_stmt=theorem_stmt, model=model, repo_context=repo_context or "")
    blueprint = _execute_phase1(
        state, theorem_stmt, nl_proof, model, compiler, repo_context, repo_retrieval, tracer, thm_name,
    )
    if checkpoint_path:
        state.save(checkpoint_path)
    return blueprint


def run_phase2(
    checkpoint_path: Path,
    compiler: AbstractLeanCompiler | None = None,
    compiler_factory: Callable[[], AbstractLeanCompiler] | None = None,
    retrieval: MathlibRetrieval | None = None,
    repo_retrieval=None,
    model: str | None = None,
    tracer=None,
    node_timeout_s: float | None = 300.0,
    cascade_model: str | None = None,
    cascade_timeout_s: float | None = None,
    escalation_max_tool_calls: int | None = 1,
    thm_name: str = "",
) -> OrchestratorResult:
    """Run one Phase 2 (parallel proving) pass against a checkpointed blueprint.

    Requires Phase 1 to have already produced a checkpoint at `checkpoint_path`
    (raises if it's missing or has no blueprint). Only nodes not already in
    `proved_cache` are attempted; the checkpoint is updated with the new
    `proved_cache` and `node_results` (the latter needed by Phase 3).

    model: Overrides whatever model Phase 1 checkpointed (state.model) for
        this Phase 2 pass only - lets a caller run Phase 1 with one model
        (e.g. a strong blueprint-generation model) and Phase 2 with another
        (e.g. a cheaper per-node proving model) against the same checkpoint.
        Defaults to state.model when not given.
    thm_name: the overall theorem's qualified name (e.g. "Hidden.Nat.mul_assoc"),
        used only to detect a namespace-shadowed local type for each node's
        prover (see prover._looks_namespace_shadowed) - optional, has no effect
        on checkpointing or results structure.
    """
    state = CheckpointState.load(checkpoint_path)
    blueprint = state.get_blueprint()
    if blueprint is None:
        raise RuntimeError(f"No blueprint in checkpoint {checkpoint_path} — run Phase 1 first.")

    retrieval = retrieval or MathlibRetrieval()
    orch_result = _execute_phase2(
        state, blueprint, compiler, compiler_factory, retrieval, repo_retrieval,
        model or state.model, tracer, node_timeout_s, cascade_model, cascade_timeout_s,
        escalation_max_tool_calls, thm_name=thm_name,
    )
    state.save(checkpoint_path)
    return orch_result


def run_phase3(
    checkpoint_path: Path,
    compiler: AbstractLeanCompiler,
    model: str | None = None,
    repo_context: str | None = None,
    max_iterations: int = MAX_REFINEMENT_ITERATIONS,
    repo_retrieval=None,
    tracer=None,
    thm_name: str = "",
) -> Blueprint:
    """Run one Phase 3 (refinement) pass against a checkpointed blueprint.

    Requires Phase 2 to have already run against this checkpoint (i.e.
    `node_results` present with at least one failure) — refinement needs
    those diagnostics to know what to fix. Raises if the checkpoint has no
    blueprint, no node results, or every node already solved.
    """
    state = CheckpointState.load(checkpoint_path)
    blueprint = state.get_blueprint()
    if blueprint is None:
        raise RuntimeError(f"No blueprint in checkpoint {checkpoint_path} — run Phase 1 first.")
    if not state.node_results:
        raise RuntimeError(f"No node results in checkpoint {checkpoint_path} — run Phase 2 first.")
    if state.iteration >= max_iterations:
        # prove_theorem's own all-in-one loop already self-limits at
        # max_iterations; standalone Phase 3 calls (this function) had no
        # equivalent check, so a caller driving Phase 1/2/3 by hand via
        # checkpoints could refine past the paper's bound indefinitely.
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is already at iteration "
            f"{state.iteration} >= max_iterations={max_iterations} — "
            "refusing another refinement round."
        )

    orch_result = _orch_result_from_checkpoint(state, blueprint)
    if orch_result.all_proved():
        raise RuntimeError(f"All nodes already proved in checkpoint {checkpoint_path} — nothing to refine.")

    new_blueprint = _execute_phase3(
        state, blueprint, orch_result, compiler, model or state.model,
        repo_context if repo_context is not None else state.repo_context,
        state.iteration, max_iterations, repo_retrieval, tracer, thm_name,
    )
    state.save(checkpoint_path)
    return new_blueprint


def _orch_result_from_checkpoint(state: CheckpointState, blueprint: Blueprint) -> OrchestratorResult:
    prover_results = state.get_prover_results()
    return OrchestratorResult(node_results={
        name: NodeResult(node=blueprint.node_by_name(name), result=pr)
        for name, pr in prover_results.items()
        if blueprint.node_by_name(name) is not None
    })


def _proof_result_from_checkpoint(state: CheckpointState) -> ProofResult:
    blueprint = state.get_blueprint()
    proved_cache = dict(state.proved_cache)
    orch_result = _orch_result_from_checkpoint(state, blueprint)
    root_name = blueprint.target_theorem
    root_proof = proved_cache.get(root_name, "")
    if state.success:
        return ProofResult(
            success=True,
            theorem_name=root_name,
            proof_body=root_proof,
            final_lean_file=_assemble_final_file(blueprint, orch_result),
            aux_lemma_decls=_aux_lemma_decls(blueprint, proved_cache, root_name),
            iterations=state.iteration + 1,
            proved_nodes=list(orch_result.proved),
            failed_nodes=[],
        )
    return ProofResult(
        success=False,
        theorem_name=root_name,
        proof_body=root_proof,
        final_lean_file=_assemble_partial_file(blueprint, orch_result, proved_cache),
        aux_lemma_decls=_aux_lemma_decls(blueprint, proved_cache, root_name),
        iterations=state.iteration + 1,
        proved_nodes=list(orch_result.proved),
        failed_nodes=list(orch_result.failed.keys()),
    )


def _substitute_proof(lean: str, name: str, proof_body: str) -> str:
    """Replace `name`'s `:= by sorry_using [...]` tail with a real proof body.

    Tolerant of whitespace/newlines between `by` and `sorry_using` (the model
    doesn't always keep them on one line), and uses a replacement function
    (not a template string) so backslashes in `proof_body` aren't
    misinterpreted as regex group references.
    """
    pattern = rf"(theorem|lemma)\s+{re.escape(name)}(.*?):=\s*by\s*sorry_using\s*\[.*?\]"
    return re.sub(
        pattern,
        lambda m: f"{m.group(1)} {name}{m.group(2)}:= {proof_body}",
        lean,
        flags=re.DOTALL,
    )


def _assemble_final_file(blueprint: Blueprint, orch_result: OrchestratorResult) -> str:
    lean = blueprint.lean_file
    for name, nr in orch_result.node_results.items():
        if nr.result.proof_body:
            lean = _substitute_proof(lean, name, nr.result.proof_body)
    return lean


def _assemble_partial_file(
    blueprint: Blueprint,
    orch_result: OrchestratorResult | None,
    proved_cache: dict[str, str],
) -> str:
    if orch_result is None:
        return blueprint.lean_file
    lean = blueprint.lean_file
    for name, body in proved_cache.items():
        lean = _substitute_proof(lean, name, body)
    return lean
