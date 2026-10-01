# goedel-arch

> **Research preview.** This is a research prototype, not a verified theorem
> prover. A `success=True` result now requires an independent final Lean
> compilation of the assembled proofs against the **original** theorem
> statement, but known limitations remain — see [Known limitations](#known-limitations).
> Do not cite benchmark numbers from commits predating this verification
> chain without re-running them.

An implementation of the Goedel-Architect pipeline for Lean 4 theorem proving:
decompose a theorem into a dependency graph of sub-lemmas (Phase 1), prove each
node in parallel (Phase 2), and refine the decomposition when nodes fail
(Phase 3), repeating up to a fixed number of iterations.

## Verification chain

Results are only reported as successful after passing, in order:

1. **Blueprint gate** — the Phase 1/3 skeleton must compile in Lean *and*
   pass structural validation (unique names, known dependencies, acyclic,
   no dead nodes, target present). A blueprint that never passed a real
   compile is refused entry to Phase 2 unless `allow_unvalidated_blueprint`
   is explicitly set (debugging only).
2. **Per-node proofs** — every node's proof must compile on its own, with
   `axiom`/`native_decide` rejected outright.
3. **Independent final verification** — the root proof plus all auxiliary
   lemmas are spliced into the *original, immutable* theorem statement and
   compiled once more. A blueprint that quietly weakened the main theorem's
   signature fails here. Only after this pass may `success=True` be set, on
   results and checkpoints alike; resumed checkpoints claiming pre-fix
   successes are re-verified on load and downgraded if they no longer hold.

## Known limitations

- The negation probe (`FORMALLY_NEGATED`) is **disabled by default**: it
  never constructed a real `¬P` goal, so a positive proof of a true
  statement could pass it and be mislabeled as a refutation. Enable it only
  for experiments that treat its output as advisory (`MODEL_SUSPECTS_WRONG`
  is the advisory form used for text-based falsity claims).
- Model-generated Lean runs via `lake env lean` on this machine with only a
  regex-level filter for forbidden constructs — there is no OS-level
  sandboxing (container, seccomp, resource limits) yet.
- Checkpoints do not yet carry a run fingerprint (code commit, prompt
  hashes, toolchain); `--resume` compares theorem/model/blueprint only.

## Requirements

- Python 3.10+
- [elan](https://github.com/leanprover/elan) (Lean version manager, provides `lake`)
- An OpenAI API key

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENAI_API_KEY
```

Fetch and build the Lean side (`goedel_lean/`) — this pulls in the real
[LeanArchitect](https://github.com/hanwenzhu/LeanArchitect) package, which
provides the `@[blueprint]` attribute and `sorry_using` syntax the pipeline
relies on:

```bash
cd goedel_lean
lake exe cache get   # downloads prebuilt Mathlib .olean files
lake build
cd ..
```

This build is toolchain-independent from whatever project you're actually
proving theorems in: Phase 1's blueprint validation strips `@[blueprint ...]`
and `sorry_using` annotations before compiling against the target project's
own toolchain, so `goedel_lean/`'s pinned Lean version (see
`goedel_lean/lean-toolchain`) never has to match.

## Repo layout

```
src/            Phase 1/2/3 pipeline (blueprint.py, orchestrator.py, prover.py, refinement.py, pipeline.py)
goedel_lean/    Small Lean package depending on the real LeanArchitect, plus
                #validate_blueprint / #list_blueprint helper commands
prompts/        System/user prompt templates (Appendix C of the paper)
eval/           Per-benchmark driver scripts (see below)
data/           Benchmark data used by the eval scripts
```



## Running against a benchmark

Each benchmark has its own thin adapter script under `eval/` that wires
`src/pipeline.py`'s `prove_theorem()` to that benchmark's data format and
verification method.

### VeriSoftBench

Requires a local checkout of [VeriSoftBench](https://github.com/utopia-group/VeriSoftBench)
as a sibling directory (`../VeriSoftBench`), with its Lean repos built per
its own README (`./scripts/setup_repos.sh --clone --config-dir build_config --output-dir data/lean_repos`, or `lake exe cache get && lake build` per repo
if a repo's build is incomplete).

```bash
# Smoke test: a single theorem
python eval/run_verisoftbench.py --thm add_comm --repo lean-formal-reasoning-program --blueprint

# A repo-diverse pilot: 20 random tasks spanning many repos, sharded across 8 workers
python eval/run_verisoftbench.py --model gpt-5.5 --blueprint --subset 20 --workers 8 --resume --output results/pilot/

# Full benchmark
python eval/run_verisoftbench.py --model gpt-5.5 --blueprint --workers 20 --resume --output results/full/

# VeriSoftBench-Aristotle (100-task subset used for Aristotle / Gemini-3-Pro Table 3 numbers)
python eval/run_verisoftbench.py --model gpt-5.5 --blueprint --aristotle-subset --workers 8 --resume --output results/aristotle/
```



#### VeriSoftBench-Aristotle Repositories

`--aristotle-subset` restricts to the 100 tasks marked `subset_aristotle` in
`verisoftbench.jsonl` (11 of VeriSoftBench's 23 repos). Task counts:


| Repo                              | Tasks |
| --------------------------------- | ----- |
| `clean`                           | 47    |
| `VCV-io`                          | 17    |
| `lean-formal-reasoning-program`   | 7     |
| `ArkLib`                          | 5     |
| `juvix-lean`                      | 5     |
| `loom`                            | 5     |
| `veil`                            | 5     |
| `LeroyCompilerVerificationCourse` | 4     |
| `LeanExprEvaluator`               | 3     |
| `iris-lean`                       | 1     |
| `pcf-lean`                        | 1     |


This flag also enables Aristotle-mode context (same-file local lemmas appended
to approximate VeriSoftBench-Aristotle's evaluation setting).

Key flags:


| Flag                 | Purpose                                                                                                                                                                                                              |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--blueprint`        | Use the full 3-phase pipeline (omit for Phase-2-only direct proving)                                                                                                                                                 |
| `--aristotle-subset` | Restrict to the 100-task VeriSoftBench-Aristotle subset (see table above)                                                                                                                                            |
| `--subset N`         | Random sample of N tasks spanning all repos (the dataset is grouped by repo, so `--limit N` alone clusters on whichever repo sorts first)                                                                            |
| `--workers N`        | Parallel workers, **sharded by repo** — theorems from the same repo always run sequentially (a shared `lake`/compiler working directory isn't safe for concurrent access), but different repos run fully in parallel |
| `--resume`           | Skip theorems already recorded in the output dir's `results.jsonl` under the same model/mode/blueprint config, and append new results incrementally instead of overwriting                                           |
| `--repo` / `--thm`   | Filter to one repo or theorem name, for smoke testing                                                                                                                                                                |
| `--trace`            | Write a JSONL trace of every tool call/compile attempt, viewable via `eval/serve_viz.py`                                                                                                                             |
| `--phase {1,2,3}`    | Run exactly one phase per theorem instead of the full loop (see below)                                                                                                                                               |
| `--checkpoint-dir`   | Where per-theorem checkpoint files live (default `<output>/checkpoints/`)                                                                                                                                            |


Results land in `<output>/results.jsonl` (one line per theorem) and
`<output>/summary.json` (pass rate, average wall time).

#### Checkpointing and resuming mid-theorem

With `--blueprint`, every theorem gets a checkpoint file at
`<output>/checkpoints/<thm_name>.json`, rewritten after each phase. It holds
the current blueprint (as raw Lean text — cheaply re-parsed, so nothing
fancier is stored), which nodes are already proved, that iteration's failure
diagnostics, and the refinement history.

This makes `--blueprint` runs resumable for free: if a run is killed and you
rerun the *same command*, each theorem picks back up from its last completed
phase — no re-generating a blueprint that already compiled, no re-proving a
node that already succeeded, no re-paying for API calls on a theorem that had
already finished (success or exhausted retries) in a prior invocation.
`--resume` (skipping whole theorems via `results.jsonl`) and checkpointing
(resuming *within* a theorem) are complementary — use both for a long run.

For finer control, `--phase N` runs a single phase across the selected
theorems and stops, so you can drive Phase 1/2/3 as separate invocations
instead of one continuous loop:

```bash
# Phase 1 alone: generate and checkpoint a blueprint for one theorem
python eval/run_verisoftbench.py --thm add_comm --repo lean-formal-reasoning-program --phase 1

# Phase 2 alone: prove nodes against that checkpoint (requires Phase 1 to have run at some point)
python eval/run_verisoftbench.py --thm add_comm --repo lean-formal-reasoning-program --phase 2

# Phase 3 alone: refine the blueprint from Phase 2's failure diagnostics (requires Phase 2 to have run)
python eval/run_verisoftbench.py --thm add_comm --repo lean-formal-reasoning-program --phase 3
```

Each `--phase` invocation is a separate process reading/writing the same
checkpoint file — Phase 2 refuses to run without a blueprint already in the
checkpoint, and Phase 3 refuses to run without Phase 2's node results (both
raise a clear error rather than silently doing nothing). Results land in
`<output>/phase{N}_results.jsonl`; the mutated checkpoint itself is the real
output. This is mainly useful for debugging one theorem's decomposition or
manually driving refinement rounds — for normal runs, plain `--blueprint`
already checkpoints and resumes automatically.

### PutnamBench / miniF2F

```bash
python eval/run_putnam.py --model gpt-5.5 --limit 20
python eval/run_minif2f.py --model gpt-5.5 --limit 20
```



## Notes

- `MAX_REFINEMENT_ITERATIONS = 8` and a 300s per-node timeout
(`src/pipeline.py`) match the paper's Appendix A; both are overridable
per-call if you're running a cost-constrained pilot.
- Each node gets up to 8 tool calls (`lean_compile`, `repo_search`,
`mathlib_search`) before being marked `proof_too_hard`.
- A node's outcome is one of `SOLVED`, `PROOF_TOO_HARD`, `STATEMENT_WRONG`, or
`FORMALLY_NEGATED` (the last is machine-checked: the pipeline actually
proves the *negation* of the stated sub-lemma, meaning the decomposition
asserted something false — see `_probe_negation` in `src/prover.py`).

