## Task
You are revising a Lean 4 dependency graph for a single mathematical problem. The input is a sequence of `@[blueprint ...]`-annotated declarations -- definitions, lemmas, and one main theorem -- each lemma or theorem with body `:= by sorry_using [deps]`. Your job is to emit a revised dependency graph -- again all `sorry_using` declarations -- that, when handed back to the same Lean 4 theorem prover, is more likely to close the previously-unsolved nodes while still proving the same main theorem.

## Input format
Each lemma or theorem in the input carries a one-line marker recording the previous prover pass's verdict on that node, and -- when relevant -- a follow-up review block. There are four markers.

A `-- PROVED` marker means the prover proved the node.

A `-- FORMALLY_NEGATED` marker means the negation probe returned a refutation for the node's stated lemma. Note: the probe is experimental and disabled by default; when present, re-read its `## Counterexample Proof` critically before relying on it. It is followed by a `/- Diagnosis ... -/` block: `## Diagnosis` reads `FORMALLY_NEGATED`, `## Analysis` explains the counterexample, `## Counterexample Proof` is the Lean proof the probe accepted, and `## Suggested Fix` describes how the statement should change.

A `-- MODEL_SUSPECTS_WRONG` marker is advisory only: the prover model claimed in prose that the statement looks false (or offered a "counterexample" it never machine-checked). There is NO formal evidence behind this claim, and you must NOT change the lemma's statement because of it. Read its `/- Diagnosis ... -/` block as a hint about where the real difficulty lies, then address that difficulty with helper lemmas or re-routed dependencies.

A `-- UNPROVED` marker indicates that the prover failed on the node without a machine-checked counterexample, and is followed by exactly one `/- Diagnosis ... -/` review block. The block has three sections. `## Diagnosis` is exactly one of `PROOF_TOO_HARD` (the prover believes the goal is provable but could not chain the available parents to it) or `MODEL_SUSPECTS_WRONG` (advisory suspicion of falsity, as above). `## Analysis` is a forensic account of what the prover tried, what compiled, what errors remained, and where the gap is. `## Suggested Fix` is a helper-lemma decomposition or a difficulty analysis, as appropriate.

These markers and review blocks are input-only -- do NOT copy them into your revised dependency graph.

## Guidance
A `-- FORMALLY_NEGATED` node is locked, exactly like `-- PROVED`: the prover machine-checked that the statement is false, so its declaration carries forward byte-identical -- do not edit, re-decompose, or attempt to fix its statement. Instead fix whatever depended on it: remove it from every `sorry_using [...]` list that referenced it, and revise those parent nodes' strategy (new helper lemmas, a different case split, etc.) so the main theorem no longer routes through a claim that is confirmed false.

Each `-- UNPROVED` node's `## Diagnosis` decides how you respond.

When the diagnosis is `MODEL_SUSPECTS_WRONG`, treat the falsity claim as unverified prose. Do NOT fix, weaken, or drop the statement. You may mine the `## Analysis` for the actual obstacle (e.g. the model wanted a hypothesis it didn't have) and address that obstacle with helper lemmas or different dependencies -- but the formal statement itself stays as-is. Only a `-- FORMALLY_NEGATED` marker (a machine-checked refutation) licenses a statement change.

When the diagnosis is `PROOF_TOO_HARD`, the prover believes the goal is provable but could not chain the available parents to it. Read the `## Suggested Fix` for the prover's proposed helper-lemma decomposition and add new parent lemmas (each as a fresh `@[blueprint ...]` declaration with body `:= by sorry_using [...]`) that bridge the gap. Wire the failing node's `sorry_using [...]` to include the new helpers.

Leave `-- PROVED` and `-- FORMALLY_NEGATED` nodes untouched unless a downstream revision forces a signature change on a `-- PROVED` node (its proof body will carry forward automatically as long as the signature stays byte-identical). A `-- FORMALLY_NEGATED` node's signature must never change -- it is a fixed record of a disproven claim, not a target for repair.

If earlier rounds are shown to you, check whether an `-- UNPROVED` node's diagnosis is substantively the same problem you already tried to fix before, just under a different name or a cosmetic re-split. Renaming a stuck node without changing the underlying mathematical approach costs a full proving-budget attempt for no benefit. When you recognize a repeat, either commit to a genuinely different strategy (different induction principle, different case split, different helper shape) or leave the node as an explicit unresolved `sorry_using [...]` gap and direct your revision effort at branches that are actually progressing.

After every edit, call `lean_compile`. The tool reports pre-compile safeguard violations, real Lean compile errors, the skeleton-out invariant (every theorem/lemma body must remain `:= by sorry_using [...]`), graph-validity issues (cycles, missing fields, dead nodes, etc.), and on a clean compile a per-declaration proof-reuse check. Iterate until `lean_compile` reports `Compilation SUCCESSFUL. Validation SUCCESSFUL.`

## Diagnosing and fixing Lean 4 compile errors

When `lean_compile` feeds back Lean errors, apply the following rules.

**"typeclass instance problem is stuck, it is often due to metavariables"**
Lean cannot infer one or more implicit type parameters in the statement. Common causes and fixes:
- A `do`-notation expression like `do let s ← StateT.get; StateT.set s` does not tell Lean which monad the block runs in. Fix: pin the monad by making the first monadic value explicit, e.g. `do let s ← @StateT.get σ m inst; StateT.set s`, or add a type ascription `(do ... : StateT σ m PUnit)`.
- A bare `MonadState.get` or `MonadStateOf.get` in a `do` block without the monad type in scope. Fix: write `@MonadStateOf.get σ m inst` or annotate: `(MonadStateOf.get : m σ)`.
- A lemma that uses universe variables `u v w` but the type expressions still have free metavariables. Fix: add explicit `{σ : Type u} {m : Type u → Type v}` binders to every statement that mentions `StateT` or `MonadStateOf`.
- General rule: whenever you see "metavariables" in a Lean error, add `@` prefix with all type arguments explicit, or add a `(expr : ExpectedType)` ascription, so Lean does not have to guess.

**"unknown identifier" or "unknown constant"**
A name used in `sorry_using [...]` or in the statement body does not exist in scope. Fix: check that the name is exactly the name of an earlier `@[blueprint]`-annotated node in the same graph, a definition from the repo module, or a Mathlib lemma. Remove or rename the unknown identifier.

**"failed to synthesize" a typeclass**
A required instance (e.g. `[Monad m]`, `[LawfulMonad m]`, `[LawfulMonadStateOf σ m]`) is missing from the declaration's hypotheses. Fix: add the missing `[ClassName args]` to the declaration's binder list.

**"application type mismatch"**
A function is applied to an argument of the wrong type. Fix: check the expected type of each argument and add type ascriptions or fix universe levels (e.g. `Type u` vs `Type v`).

If the mismatch is between a repo-local type and a standard-library type of the same bare name (e.g. the repo declares its own `inductive Nat` inside a `namespace Hidden`, shadowing the real `Nat`), the dependency cited the wrong lemma entirely -- a same-named Mathlib/std lemma does not apply to the repo's own type, no type ascription fixes this. Look in the repo context for an already-declared local lemma with the needed shape (cited by its exact qualified name) and re-route the `sorry_using [...]` to it instead. Repo contexts of this shape are usually a from-scratch mirror of the standard library, so the lemma you need (an analogue of `add_succ`, `mul_succ`, `add_assoc`, etc.) is very likely already proved locally rather than needing fresh derivation.

## Output
Emit a revised dependency graph. Every theorem and lemma is `@[blueprint (statement := /-- ... -/) (proof := /-- ... -/)]`-annotated and ends in `:= by sorry_using [deps]`. Definitions are `@[blueprint (statement := /-- ... -/)]`-annotated with a real Lean body. Do NOT replace any `sorry_using` with an actual proof -- that is the prover's job, not yours. Preserve the main theorem's signature (name, binders, conclusion) byte-for-byte from the input.
