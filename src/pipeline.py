"""Top-level pipeline: Blueprint → Parallel Proving → Refinement loop.

Wires all three phases together and runs up to 8 refinement iterations
(matching Appendix A of the paper).

The `compiler` parameter is injectable so the same pipeline works with:
  - LeanCompiler (standalone Lean projects via `lake env lean`)
  - VSBLeanCompiler (VeriSoftBench repos via LeanREPL)
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from blueprint import Blueprint, BlueprintValidationError, generate_blueprint, validate_blueprint
from checkpoint import CheckpointState
from run_fingerprint import fingerprint, run_manifest, skip_requested
from lean_compiler import (
    AbstractLeanCompiler,
    LeanCompiler,
    MATHLIB_HEADER,
    format_proof_assign,
    normalize_proof_body,
)
from mathlib_retrieval import MathlibRetrieval
from orchestrator import NodeResult, OrchestratorResult, prove_dag
from prover import ProofSignal, ProverResult
from refinement import refine_blueprint
from tracer import NullTracer, TraceEvent

# From Appendix A
MAX_REFINEMENT_ITERATIONS = 8


@dataclass
class VerificationReport:
    """Outcome of the independent final verification against the ORIGINAL
    theorem statement (the immutable `theorem_stmt` input, never the refined
    blueprint's root). A pipeline result may claim success=True only when
    `passed` is True - everything else (all nodes reporting solved, a
    weakened-but-provable root signature, an assembled final file) is
    subordinate to this check."""
    performed: bool
    passed: bool = False
    mode: str = ""       # "full-file" | "vsb-root-recheck" | "unavailable"
    errors: list[str] = field(default_factory=list)


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
    final_verification: VerificationReport | None = None


def _invalidate_stale_proofs(
    new_blueprint: Blueprint,
    proved_cache: dict[str, str],
    proof_cache_keys: dict[str, str],
) -> dict[str, str]:
    """Drop cached proofs that no longer match the node they were compiled against.

    proved_cache tracks nodes by NAME only. Refinement (Phase 3) can reuse a
    node's name while restructuring its signature or dependency list (e.g.
    splitting a hypothesis, changing its goal shape, adding a new
    sorry_using [...] dependency) - the paper's rule only promises
    SOLVED/FORMALLY_NEGATED nodes carry forward byte-identical, but nothing
    enforces that, and a name collision with a differently-shaped node would
    otherwise leave a proof compiled against the OLD shape marked "already
    solved" forever, never recompiled against the new one.

    Rather than diffing the immediately-previous blueprint (which misses a
    node deleted at round N and reintroduced with a different shape at round
    N+2 - neither adjacent diff N->N+1 or N+1->N+2 ever sees both shapes at
    once), this compares against `proof_cache_keys[name]`: the exact
    BlueprintNode.cache_key() recorded at the moment the proof was accepted.
    A name missing from `proof_cache_keys` (e.g. an older checkpoint written
    before this field existed) is treated as stale and re-checked once.
    """
    pruned = dict(proved_cache)
    for name in list(pruned):
        new_node = new_blueprint.node_by_name(name)
        if new_node is None or proof_cache_keys.get(name) != new_node.cache_key():
            del pruned[name]
    return pruned


def _structure_fingerprint(blueprint: Blueprint) -> str:
    """sha256 over the sorted (name, cache_key) pairs - the graph's shape
    (node names + signatures + dependency sets), independent of the proofs
    currently cached against it. Two refinements that produce the same
    shape fingerprint would run Phase 2 on identical work."""
    parts = sorted(f"{n.name}\x00{n.cache_key()}" for n in blueprint.nodes)
    return hashlib.sha256("\x01".join(parts).encode()).hexdigest()


def _provable_nodes(blueprint: Blueprint) -> set[str]:
    """Nodes that still need an LLM proving attempt: everything except
    definition-kind nodes, which carry real Lean bodies and were verified
    by the blueprint's own compilation (the orchestrator seeds them as
    solved for bookkeeping)."""
    return {n.name for n in blueprint.nodes if n.kind != "definition"}


def _aux_lemma_decls(blueprint: Blueprint, proved_cache: dict[str, str], root_name: str) -> str:
    # Definitions emit their full declaration (real body/fields); theorems
    # re-declare signature + single-assignment proof. Both orders respect
    # the dependency order so Lean never sees a forward reference.
    parts: list[str] = []
    for node in blueprint.dependency_order():
        if node.name == root_name:
            continue
        if node.kind == "definition":
            parts.append(node.compiled_decl())
        elif node.name in proved_cache:
            parts.append(f"{node.signature()} {format_proof_assign(proved_cache[node.name])}")
    return "\n\n".join(parts)


def _gate_blueprint(blueprint: Blueprint, allow_unvalidated: bool, had_compiler: bool) -> None:
    """Hard gate between Phase 1/3 and Phase 2.

    Two layers:
      - structural validation (duplicate names, unknown dependencies,
        cycles, dead nodes, missing target) ALWAYS applies - it is pure
        Python and correctness-critical, with no legitimate debug escape;
      - compile validation (blueprint.fully_validated) applies whenever a
        blueprint compiler existed to validate it. Escapable only via the
        explicit allow_unvalidated flag (a debugging mode); the pipeline's
        final verification still has to pass before any success is
        reported, so the escape can never manufacture a fake success.

    `had_compiler` distinguishes "failed validation" (compiler present,
    fully_validated=False) from "validation was impossible" (the
    compiler_factory-only VSB path, where blueprint compilation happens
    per-node inside the repo environment instead).
    """
    errors = validate_blueprint(blueprint)
    if errors:
        raise BlueprintValidationError(
            "Blueprint failed structural validation:\n  - " + "\n  - ".join(errors)
        )
    if had_compiler and not blueprint.fully_validated and not allow_unvalidated:
        raise BlueprintValidationError(
            "Blueprint was never accepted by a real Lean compile "
            "(fully_validated=False) - refusing to enter Phase 2. Pass "
            "allow_unvalidated_blueprint=True (debugging only; success still "
            "requires independent final verification) to override."
        )


_SORRY_TAIL_RE = re.compile(r":=\s*(?:by\s+)?sorry\s*\Z", re.DOTALL)


def _assemble_original_theorem_file(
    theorem_stmt: str, aux_lemma_decls: str, root_proof: str,
) -> str:
    """Assemble the independent final-verification file from the IMMUTABLE
    original theorem statement - never the refined blueprint's root node.

    The root proof (canonical `by ...` form) is spliced into the original
    statement's `sorry` tail, with every proved auxiliary node re-declared
    ahead of it. If the blueprint weakened the root's signature, the root
    proof will not typecheck against the ORIGINAL statement here and the
    compile fails - exactly the false-success mode this file exists to
    catch. Fails closed: a statement with no recognizable `sorry` tail and
    no `:=` at all gets the proof appended; a statement that already has a
    non-sorry body is left as-is (its compile will fail rather than
    silently re-proving something else).
    """
    proof = format_proof_assign(root_proof)
    stmt = theorem_stmt.strip()
    if _SORRY_TAIL_RE.search(stmt):
        stmt = _SORRY_TAIL_RE.sub(lambda _: proof, stmt, count=1)
    elif ":=" not in stmt:
        stmt = f"{stmt} {proof}"
    parts = [MATHLIB_HEADER.rstrip("\n")]
    if aux_lemma_decls.strip():
        parts.append(aux_lemma_decls.strip())
    parts.append(stmt)
    return "\n\n".join(parts) + "\n"


def _final_verification(
    theorem_stmt: str,
    blueprint: Blueprint,
    proved_cache: dict[str, str],
    compiler: AbstractLeanCompiler | None,
    compiler_factory=None,
) -> VerificationReport:
    """Independently verify the ORIGINAL theorem with the assembled proofs.

    Shared-compiler (standalone Lean) mode compiles the full file built by
    _assemble_original_theorem_file. Factory-only (VSB) mode re-checks the
    root proof against the ORIGINAL root statement by calling a fresh
    factory compiler WITHOUT node_decl - VSBLeanCompiler.check falls back
    to the entry's thm_stmt (the original proposition) exactly when
    node_decl is absent, so this re-check closes the same hole for VSB
    (belt to run_verisoftbench's own external verifyProof braces).
    """
    root_name = blueprint.target_theorem
    root_proof = proved_cache.get(root_name, "")
    if not root_proof.strip():
        return VerificationReport(performed=False, mode="unavailable",
                                  errors=["root node has no cached proof"])
    aux = _aux_lemma_decls(blueprint, proved_cache, root_name)
    if compiler is not None:
        file = _assemble_original_theorem_file(theorem_stmt, aux, root_proof)
        result = compiler.check(file)
        return VerificationReport(
            performed=True, passed=result.success, mode="full-file",
            errors=result.errors,
        )
    if compiler_factory is not None:
        try:
            fresh = compiler_factory()
        except Exception as exc:  # factory failure is infra, not a pass
            return VerificationReport(performed=False, mode="unavailable",
                                      errors=[f"compiler_factory raised {exc!r}"])
        # ':= by ...' form - the VSB harness splices this after the bare
        # original signature (format_generated_lean contract).
        result = fresh.check(format_proof_assign(root_proof), aux_lemmas=aux)
        return VerificationReport(
            performed=True, passed=result.success, mode="vsb-root-recheck",
            errors=result.errors,
        )
    return VerificationReport(performed=False, mode="unavailable",
                              errors=["no compiler available for final verification"])


async def prove_theorem_async(
    theorem_stmt: str,
    nl_proof: str | None = None,
    model: str = "labs-leanstral-1-5",
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
    allow_unvalidated_blueprint: bool = False,
    enable_negation_probe: bool = False,
    retry_failed: bool = False,
    tactic_portfolio: list[str] | None = None,
) -> ProofResult:
    """
    Full Goedel-Architect pipeline for a single theorem.

    1. Generate @[blueprint]-annotated dependency graph (Phase 1)
    2. Prove each node in parallel (Phase 2)
    3. Refine blueprint on failures and repeat (Phase 3)
    Up to `max_iterations` refinement loops (default 8, per Appendix A).

    Args:
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

    # Provenance: every checkpoint records the experiment identity (code
    # commit, prompt hashes, toolchain, model/cascade config). Resume must
    # not silently mix results across experiments - a mismatch refuses the
    # checkpoint outright (GOEDEL_SKIP_FINGERPRINT=1 is the debug escape).
    manifest = run_manifest(
        model=model,
        cascade_model=cascade_model,
        max_iterations=max_iterations,
        enable_negation_probe=enable_negation_probe,
        allow_unvalidated_blueprint=allow_unvalidated_blueprint,
        tactic_portfolio=list(tactic_portfolio) if tactic_portfolio else None,
    )
    current_fingerprint = fingerprint(manifest)
    if state is not None:
        state.require_fingerprint(manifest, current_fingerprint)

    if state and state.done and not state.success and retry_failed:
        # A cached failure used to be terminal forever (done=True,
        # success=False) - retry_failed lets a new model/config continue
        # the same checkpoint instead of returning the stale verdict.
        print("[Resume] retry_failed=True: ignoring cached failure, "
              "continuing the run", flush=True)
        state.done = False
        state.success = False

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
        cached = _proof_result_from_checkpoint(state, compiler=compiler, compiler_factory=compiler_factory)
        if cached.success and (cached.final_verification is None or not cached.final_verification.passed):
            # Checkpoints written before independent final verification
            # existed (or by the allow_unvalidated escape) claim success
            # without a verified original theorem. Re-verify now - it's one
            # local compile - and downgrade to a failure if it doesn't hold.
            verification = _final_verification(
                theorem_stmt, state.get_blueprint() or Blueprint(nodes=[], lean_file="", target_theorem=""),
                state.proved_cache, compiler, compiler_factory=compiler_factory,
            )
            if not verification.passed:
                print(f"[Resume] cached success FAILED re-verification ({verification.mode}): "
                      f"{verification.errors[:3]} — downgrading to failure", flush=True)
                return ProofResult(
                    success=False,
                    theorem_name=cached.theorem_name,
                    proof_body=cached.proof_body,
                    final_lean_file=cached.final_lean_file,
                    aux_lemma_decls=cached.aux_lemma_decls,
                    iterations=cached.iterations,
                    proved_nodes=cached.proved_nodes,
                    failed_nodes=cached.failed_nodes,
                    final_verification=verification,
                )
            cached.final_verification = verification
        return cached

    resumed_blueprint = state.get_blueprint() if state else None

    if resumed_blueprint is not None:
        blueprint = resumed_blueprint
        # Normalize cached bodies to the canonical `by ...` form - older
        # checkpoints may predate the single-representation rule and hold
        # `:= by ...` bodies that would double-assign on re-assembly.
        proved_cache: dict[str, str] = {
            name: normalize_proof_body(body) for name, body in state.proved_cache.items()
        }
        proof_cache_keys: dict[str, str] = dict(state.proof_cache_keys)
        refinement_history: list[str] = list(state.refinement_history)
        start_iteration = state.iteration
        print(f"[Resume] loaded checkpoint at iteration {start_iteration + 1}, "
              f"{len(proved_cache)} node(s) already proved", flush=True)
        # Same gate a fresh Phase 1 blueprint must pass: a resumed graph
        # that is structurally broken (or was never compile-validated by a
        # real Lean run) must not quietly continue into Phase 2.
        _gate_blueprint(blueprint, allow_unvalidated=allow_unvalidated_blueprint,
                        had_compiler=compiler is not None)
    else:
        # Phase 1: Blueprint generation
        # Don't use compiler_factory for blueprint validation: factory compilers are
        # stateful (track call counts, write temp files) and Phase 1 would exhaust
        # retries on type-signature errors the LLM can't fix without repo context.
        # Pass only an explicitly-shared compiler (e.g. standalone LeanCompiler).
        blueprint_compiler = compiler  # None when only compiler_factory is provided
        blueprint = generate_blueprint(
            theorem_stmt=theorem_stmt,
            nl_proof=nl_proof,
            model=model,
            compiler=blueprint_compiler,
            repo_context=repo_context,
            repo_retrieval=repo_retrieval,
            tracer=tracer,
            thm_name=thm_name,
        )
        proved_cache = {}
        proof_cache_keys = {}
        refinement_history = []
        start_iteration = 0
        # Hard gate: a blueprint that failed (or was never given) a real
        # compile must not enter Phase 2 - previously the give-up path after
        # MAX_RETRIES returned fully_validated=False and the pipeline
        # happily burned model calls proving nodes of an unvalidated graph.
        _gate_blueprint(blueprint, allow_unvalidated=allow_unvalidated_blueprint,
                        had_compiler=blueprint_compiler is not None)
        state = CheckpointState(theorem_stmt=theorem_stmt, model=model, repo_context=repo_context or "")
        state.run_fingerprint = {**manifest, "fingerprint": current_fingerprint}
        state.set_blueprint(blueprint)
        if checkpoint_path:
            state.save(checkpoint_path)

    orch_result: OrchestratorResult | None = None
    last_verification: VerificationReport | None = None
    # structure fingerprint -> solved count at the moment that exact graph
    # shape was worked on. Used by the no-progress stop (review III.2):
    # refinement that oscillates (split, merge, re-split the same nodes)
    # burns a full Phase-2 proving budget per round for nothing.
    seen_structures: dict[str, int] = {
        _structure_fingerprint(blueprint): len(proved_cache)
    }
    stagnation_rounds = 0

    for iteration in range(start_iteration, max_iterations):
        # Phase 2: Parallel proving
        nodes_to_try = _provable_nodes(blueprint) - set(proved_cache)
        print(f"\n[Phase 2 iteration {iteration+1}] Proving {len(nodes_to_try)} nodes: {sorted(nodes_to_try)}", flush=True)

        orch_result = await prove_dag(
                blueprint=blueprint,
                compiler=compiler,
                compiler_factory=compiler_factory,
                retrieval=retrieval,
                repo_retrieval=repo_retrieval,
                model=model,
                proved_cache=proved_cache,
                nodes_to_retry=nodes_to_try,
                tracer=tracer,
                node_timeout_s=node_timeout_s,
                cascade_model=cascade_model,
                cascade_timeout_s=cascade_timeout_s,
                escalation_max_tool_calls=escalation_max_tool_calls,
                enable_negation_probe=enable_negation_probe,
                tactic_portfolio=tactic_portfolio,
            )

        for name, nr in orch_result.node_results.items():
            status = nr.result.signal.value
            proof_preview = repr(nr.result.proof_body[:60]) if nr.result.proof_body else ""
            print(f"  node '{name}': {status} {proof_preview}", flush=True)
            if status == "solved" and nr.result.proof_body:
                # (definition-kind nodes are seeded SOLVED with an empty
                # body - they live in the blueprint file, not the cache)
                proved_cache[name] = normalize_proof_body(nr.result.proof_body)
                node = blueprint.node_by_name(name)
                if node:
                    proof_cache_keys[name] = node.cache_key()

        print(f"  proved so far: {sorted(proved_cache.keys())}", flush=True)

        if checkpoint_path:
            state.iteration = iteration
            state.proved_cache = dict(proved_cache)
            state.proof_cache_keys = dict(proof_cache_keys)
            state.set_node_results(orch_result.node_results)

        if orch_result.all_proved():
            root_name = blueprint.target_theorem
            root_proof = proved_cache.get(root_name, "")
            aux_decls = _aux_lemma_decls(blueprint, proved_cache, root_name)
            final_file = _assemble_final_file(blueprint, orch_result)
            # THE success criterion: every node reporting solved proves
            # nothing about the ORIGINAL theorem (a weakened-but-provable
            # root signature would also get here). Independently re-verify
            # the immutable theorem_stmt with the assembled proofs; only a
            # pass here may set success - this is also what a resumed
            # checkpoint's success is re-checked against.
            verification = _final_verification(
                theorem_stmt, blueprint, proved_cache, compiler,
                compiler_factory=compiler_factory,
            )
            tracer.emit(TraceEvent(
                kind="final_verify", thm_name=thm_name or root_name,
                ok=verification.passed,
                iteration=iteration + 1,
                args={"mode": verification.mode, "errors": verification.errors[:5]},
            ))
            if not verification.passed:
                print(f"  [final verification] FAILED ({verification.mode}): "
                      f"{verification.errors[:3]} - treating this iteration as "
                      f"unsuccessful and continuing to refinement", flush=True)
                last_verification = verification
                # Mark the root node as needing rework so Phase 3 gets a
                # real diagnostic instead of an all-PROVED graph it can
                # only echo back unchanged.
                orch_result.node_results[root_name] = NodeResult(
                    node=blueprint.node_by_name(root_name),
                    result=ProverResult(
                        signal=ProofSignal.PROOF_TOO_HARD,
                        analysis=(
                            "Assembled proofs FAILED independent verification "
                            f"against the ORIGINAL theorem statement ({verification.mode}): "
                            + "; ".join(verification.errors[:3])
                        ),
                    ),
                )
            else:
                print("  [final verification] passed against the original theorem statement", flush=True)
                if checkpoint_path:
                    state.done = True
                    state.success = True
                    state.save(checkpoint_path)
                return ProofResult(
                    success=True,
                    theorem_name=root_name,
                    proof_body=root_proof,
                    final_lean_file=final_file,
                    aux_lemma_decls=aux_decls,
                    iterations=iteration + 1,
                    proved_nodes=list(orch_result.proved),
                    failed_nodes=[],
                    final_verification=verification,
                )

        if checkpoint_path:
            state.save(checkpoint_path)

        if iteration == max_iterations - 1:
            break

        # Phase 3: Refinement
        print(f"\n[Phase 3 iteration {iteration+1}] Refining blueprint ...", flush=True)
        failed = [n for n in orch_result.node_results if orch_result.node_results[n].result.signal.value != "solved"]
        print(f"  failed nodes: {failed}", flush=True)
        refinement_compiler = compiler or (compiler_factory() if compiler_factory else None)
        if refinement_compiler is None:
            break
        try:
            blueprint = refine_blueprint(
                blueprint=blueprint,
                orch_result=orch_result,
                compiler=refinement_compiler,
                model=model,
                repo_context=repo_context,
                history=refinement_history,
                iteration=iteration,
                max_iterations=max_iterations,
                repo_retrieval=repo_retrieval,
                tracer=tracer,
                thm_name=thm_name,
            )
            print(f"  new blueprint has {len(blueprint.nodes)} nodes: {[n.name for n in blueprint.nodes]}", flush=True)
            # A refined graph must still be structurally sound (the compile
            # gate already ran inside refine_blueprint's check_blueprint;
            # this catches shape regressions the Lean check can't see, like
            # a dead node or a lost target).
            structural_errors = validate_blueprint(blueprint)
            if structural_errors:
                raise BlueprintValidationError(
                    "Refined blueprint failed structural validation:\n  - "
                    + "\n  - ".join(structural_errors)
                )
        except RuntimeError as e:
            print(f"  refinement failed: {e}", flush=True)
            # refine_blueprint mutates `history` in place before its own
            # retry loop, so this round's attempt is already in memory even
            # though refinement ultimately failed - persist it so the
            # checkpoint's refinement_history isn't silently shorter than
            # what was actually tried (matters for post-mortem diagnosis).
            if checkpoint_path:
                state.refinement_history = list(refinement_history)
                state.save(checkpoint_path)
            break  # refinement failed, stop iterations

        stale = set(proved_cache) - set(_invalidate_stale_proofs(blueprint, proved_cache, proof_cache_keys))
        if stale:
            print(f"  invalidated stale proof(s) (no longer match the current node shape): {sorted(stale)}", flush=True)
        proved_cache = _invalidate_stale_proofs(blueprint, proved_cache, proof_cache_keys)
        proof_cache_keys = {name: key for name, key in proof_cache_keys.items() if name in proved_cache}

        # No-progress stop (review III.2): refinement sometimes oscillates -
        # split a node, merge it back, re-split it - and each round re-buys a
        # full Phase-2 proving budget for a graph that is structurally the
        # one already attempted. Stop when the NEW graph shape has been seen
        # before AND it solved no more nodes back then than we have now
        # (i.e. two consecutive rounds without a solved-count increase and a
        # repeated structure fingerprint).
        new_fp = _structure_fingerprint(blueprint)
        if len(proved_cache) <= seen_structures.get(new_fp, -1):
            stagnation_rounds += 1
        else:
            stagnation_rounds = 0
        if new_fp in seen_structures and stagnation_rounds >= 2:
            print(f"  [no-progress] refined graph repeats an earlier structure "
                  f"(solved {len(proved_cache)} unchanged for {stagnation_rounds} "
                  f"refinement rounds) - stopping early instead of re-proving it",
                  flush=True)
            if checkpoint_path:
                state.set_blueprint(blueprint)
                state.refinement_history = list(refinement_history)
                state.iteration = iteration + 1
                state.proved_cache = dict(proved_cache)
                state.proof_cache_keys = dict(proof_cache_keys)
                state.node_results = {}
                state.save(checkpoint_path)
            break
        seen_structures[new_fp] = len(proved_cache)

        if checkpoint_path:
            state.set_blueprint(blueprint)
            state.refinement_history = list(refinement_history)
            state.iteration = iteration + 1
            state.proved_cache = dict(proved_cache)
            state.proof_cache_keys = dict(proof_cache_keys)
            state.node_results = {}  # stale against the new blueprint
            state.save(checkpoint_path)

    proved = list(orch_result.proved) if orch_result else []
    failed = list(orch_result.failed.keys()) if orch_result else []
    root_proof = proved_cache.get(blueprint.target_theorem, "")
    if checkpoint_path:
        state.done = True
        state.success = False
        state.save(checkpoint_path)
    return ProofResult(
        success=False,
        theorem_name=blueprint.target_theorem,
        proof_body=root_proof,
        final_lean_file=_assemble_partial_file(blueprint, orch_result, proved_cache),
        aux_lemma_decls=_aux_lemma_decls(blueprint, proved_cache, blueprint.target_theorem),
        # actual rounds run, not the cap: refinement failure and the
        # no-progress stop both end the loop before max_iterations
        iterations=(iteration + 1) if orch_result is not None else 0,
        proved_nodes=proved,
        failed_nodes=failed,
        final_verification=last_verification,
    )


def prove_theorem(
    theorem_stmt: str,
    nl_proof: str | None = None,
    model: str = "labs-leanstral-1-5",
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
    allow_unvalidated_blueprint: bool = False,
    enable_negation_probe: bool = False,
    retry_failed: bool = False,
    tactic_portfolio: list[str] | None = None,
) -> ProofResult:
    """Synchronous wrapper around prove_theorem_async (review IV.5): the
    library no longer calls asyncio.run() deep inside a sync API, which
    blew up under Jupyter/FastAPI/any already-running event loop. Use the
    async variant natively inside one; this wrapper starts a fresh loop
    exactly like the old behavior.
    """
    return asyncio.run(prove_theorem_async(
        theorem_stmt=theorem_stmt,
        nl_proof=nl_proof,
        model=model,
        compiler=compiler,
        compiler_factory=compiler_factory,
        retrieval=retrieval,
        repo_retrieval=repo_retrieval,
        max_iterations=max_iterations,
        tracer=tracer,
        project_root=project_root,
        repo_context=repo_context,
        node_timeout_s=node_timeout_s,
        checkpoint_path=checkpoint_path,
        thm_name=thm_name,
        cascade_model=cascade_model,
        cascade_timeout_s=cascade_timeout_s,
        escalation_max_tool_calls=escalation_max_tool_calls,
        allow_unvalidated_blueprint=allow_unvalidated_blueprint,
        enable_negation_probe=enable_negation_probe,
        retry_failed=retry_failed,
        tactic_portfolio=tactic_portfolio,
    ))


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
    model: str = "labs-leanstral-1-5",
    compiler: AbstractLeanCompiler | None = None,
    repo_context: str | None = None,
    checkpoint_path: Path | None = None,
    repo_retrieval=None,
    tracer=None,
    thm_name: str = "",
    run_config: dict | None = None,
) -> Blueprint:
    """Run Phase 1 (blueprint generation) alone and checkpoint the result.

    run_config: the caller's pipeline configuration (cascade model,
    max_iterations, negation-probe/blueprint-gate flags), recorded into
    the checkpoint's run fingerprint so a later resume under the same
    experiment matches. When omitted, a Phase-1-only manifest is recorded;
    pass the SAME config you intend to resume with (see
    run_verisoftbench's phase CLI) to avoid a fingerprint mismatch.
    """
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
    if checkpoint_path:
        manifest = run_config if run_config is not None else run_manifest(
            model=model, cascade_model=None, max_iterations=None,
            enable_negation_probe=False, allow_unvalidated_blueprint=False,
        )
        state = CheckpointState(theorem_stmt=theorem_stmt, model=model, repo_context=repo_context or "")
        state.run_fingerprint = {**manifest, "fingerprint": fingerprint(manifest)}
        state.set_blueprint(blueprint)
        state.save(checkpoint_path)
    return blueprint


async def run_phase2_async(
    checkpoint_path: Path,
    compiler: AbstractLeanCompiler | None = None,
    compiler_factory: Callable[[], AbstractLeanCompiler] | None = None,
    retrieval: MathlibRetrieval | None = None,
    repo_retrieval=None,
    tracer=None,
    node_timeout_s: float | None = 300.0,
    model: str | None = None,
    cascade_model: str | None = None,
    cascade_timeout_s: float | None = None,
    escalation_max_tool_calls: int | None = 1,
    tactic_portfolio: list[str] | None = None,
) -> OrchestratorResult:
    """Run one Phase 2 (parallel proving) pass against a checkpointed blueprint.

    Requires Phase 1 to have already produced a checkpoint at `checkpoint_path`
    (raises if it's missing or has no blueprint). Only nodes not already in
    `proved_cache` are attempted; the checkpoint is updated with the new
    `proved_cache` and `node_results` (the latter needed by Phase 3).
    """
    state = CheckpointState.load(checkpoint_path)
    blueprint = state.get_blueprint()
    if blueprint is None:
        raise RuntimeError(f"No blueprint in checkpoint {checkpoint_path} — run Phase 1 first.")
    # Standalone Phase 2 entries get the same structural gate the all-in-one
    # loop applies - a checkpointed graph that is structurally broken must
    # not quietly continue into proving.
    structural_errors = validate_blueprint(blueprint)
    if structural_errors:
        raise BlueprintValidationError(
            f"Blueprint in {checkpoint_path} failed structural validation:\n  - "
            + "\n  - ".join(structural_errors)
        )

    retrieval = retrieval or MathlibRetrieval()
    proved_cache = {name: normalize_proof_body(body) for name, body in state.proved_cache.items()}
    proof_cache_keys = dict(state.proof_cache_keys)
    nodes_to_try = _provable_nodes(blueprint) - set(proved_cache)

    orch_result = await prove_dag(
            blueprint=blueprint,
            compiler=compiler,
            compiler_factory=compiler_factory,
            retrieval=retrieval,
            repo_retrieval=repo_retrieval,
            model=model or state.model,
            proved_cache=proved_cache,
            nodes_to_retry=nodes_to_try,
            tracer=tracer,
            node_timeout_s=node_timeout_s,
            cascade_model=cascade_model,
            cascade_timeout_s=cascade_timeout_s,
            escalation_max_tool_calls=escalation_max_tool_calls,
            tactic_portfolio=tactic_portfolio,
        )

    for name, nr in orch_result.node_results.items():
        if nr.result.signal.value == "solved" and nr.result.proof_body:
            proved_cache[name] = normalize_proof_body(nr.result.proof_body)
            node = blueprint.node_by_name(name)
            if node:
                proof_cache_keys[name] = node.cache_key()

    state.proved_cache = proved_cache
    state.proof_cache_keys = proof_cache_keys
    state.set_node_results(orch_result.node_results)
    if orch_result.all_proved():
        # Same success criterion as the all-in-one loop: all nodes solved
        # must still survive independent verification against the original
        # theorem before the checkpoint may record success.
        verification = _final_verification(
            state.theorem_stmt, blueprint, proved_cache, compiler,
            compiler_factory=compiler_factory,
        )
        state.done = True
        state.success = verification.passed
        if not verification.passed:
            print(f"[Phase 2 standalone] final verification FAILED "
                  f"({verification.mode}): {verification.errors[:3]}", flush=True)
    else:
        state.done = False
        state.success = False
    state.save(checkpoint_path)
    return orch_result


def run_phase2(
    checkpoint_path: Path,
    compiler: AbstractLeanCompiler | None = None,
    compiler_factory: Callable[[], AbstractLeanCompiler] | None = None,
    retrieval: MathlibRetrieval | None = None,
    repo_retrieval=None,
    tracer=None,
    node_timeout_s: float | None = 300.0,
    model: str | None = None,
    cascade_model: str | None = None,
    cascade_timeout_s: float | None = None,
    escalation_max_tool_calls: int | None = 1,
    tactic_portfolio: list[str] | None = None,
) -> OrchestratorResult:
    """Synchronous wrapper around run_phase2_async (see prove_theorem)."""
    return asyncio.run(run_phase2_async(
        checkpoint_path=checkpoint_path,
        compiler=compiler,
        compiler_factory=compiler_factory,
        retrieval=retrieval,
        repo_retrieval=repo_retrieval,
        tracer=tracer,
        node_timeout_s=node_timeout_s,
        model=model,
        cascade_model=cascade_model,
        cascade_timeout_s=cascade_timeout_s,
        escalation_max_tool_calls=escalation_max_tool_calls,
        tactic_portfolio=tactic_portfolio,
    ))


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

    refinement_history = list(state.refinement_history)
    new_blueprint = refine_blueprint(
        blueprint=blueprint,
        orch_result=orch_result,
        compiler=compiler,
        model=model or state.model,
        repo_context=repo_context if repo_context is not None else state.repo_context,
        history=refinement_history,
        iteration=state.iteration,
        max_iterations=max_iterations,
        repo_retrieval=repo_retrieval,
        tracer=tracer,
        thm_name=thm_name,
    )

    new_proved_cache = _invalidate_stale_proofs(new_blueprint, state.proved_cache, state.proof_cache_keys)
    state.proved_cache = new_proved_cache
    state.proof_cache_keys = {
        name: key for name, key in state.proof_cache_keys.items() if name in new_proved_cache
    }
    state.set_blueprint(new_blueprint)
    state.refinement_history = refinement_history
    state.iteration += 1
    state.node_results = {}  # stale against the new blueprint
    state.done = False
    state.success = False
    state.save(checkpoint_path)
    return new_blueprint


def _orch_result_from_checkpoint(state: CheckpointState, blueprint: Blueprint) -> OrchestratorResult:
    prover_results = state.get_prover_results()
    return OrchestratorResult(node_results={
        name: NodeResult(node=blueprint.node_by_name(name), result=pr)
        for name, pr in prover_results.items()
        if blueprint.node_by_name(name) is not None
    })


def _proof_result_from_checkpoint(
    state: CheckpointState,
    compiler: AbstractLeanCompiler | None = None,
    compiler_factory=None,
) -> ProofResult:
    blueprint = state.get_blueprint()
    proved_cache = {name: normalize_proof_body(body) for name, body in state.proved_cache.items()}
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
            # final_verification is left None here on purpose: the caller
            # re-verifies pre-verification-era successes and either attaches
            # the passing report or downgrades the result.
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
    misinterpreted as regex group references. The body is normalized first -
    this used to prepend its own `:= ` to a `:= by ...` body and emit
    `:= := by ...`, which can never compile.
    """
    pattern = rf"(theorem|lemma)\s+{re.escape(name)}(.*?):=\s*by\s*sorry_using\s*\[.*?\]"
    return re.sub(
        pattern,
        lambda m: f"{m.group(1)} {name}{m.group(2)}{format_proof_assign(proof_body)}",
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
