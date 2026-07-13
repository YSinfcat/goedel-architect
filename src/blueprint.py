"""Phase 1: Blueprint generation.

Calls the LLM with the verbatim system prompt from the paper (prompts/blueprint_system.md)
and validates the resulting @[blueprint]-annotated Lean file via LeanArchitect.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from lean_compiler import AbstractLeanCompiler, LeanCompiler
from llm_client import make_client
from model_backend import AUTO, ChatCompletionsBackend, ToolSpec, default_reasoning_effort
from tool_dispatcher import ToolDispatcher, repo_search_handler
from goedel_prompts import load, render
from tracer import TraceEvent

BLUEPRINT_SYSTEM_PROMPT = load("blueprint_system")
BLUEPRINT_USER_TEMPLATE = load("blueprint_user")

# Appendix A specifies 262,144 (matches DeepSeek-V4-Flash's completion budget).
# OpenAI's chat.completions API hard-caps max_completion_tokens at 128,000
# regardless of model, and this is capped further to 64,000 to control cost.
MAX_TOKENS = 64_000
MAX_RETRIES = 8

# `repo_context` is built from only the target file's own preceding content
# (see eval/run_verisoftbench.py's _build_verif_context) - it never follows
# `import` statements, so a theorem needing a type/def declared in a merely
# imported sibling file gets no information about it and fabricates a
# placeholder. repo_search (same tool Phase 2 already has) lets Phase 1/3
# look up cross-file declarations on demand instead. Optional: passing
# repo_retrieval=None (the default) reproduces the old no-tools behavior
# exactly, so existing callers are unaffected.
REPO_SEARCH_TOOL = ToolSpec(
    name="repo_search",
    description=(
        "Semantic search over the target repository's .lean files, "
        "including files merely imported by (not textually preceding) "
        "the theorem's own file."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language or identifier fragment."},
            "k": {"type": "integer", "description": "Number of results (default 10).", "default": 10},
        },
        "required": ["query"],
    },
)

# Bounds the extra repo_search back-and-forth before a call must produce a
# final text response, separate from MAX_RETRIES (whole-attempt retries after
# a failed compile). Kept small - this is a targeted lookup for missing
# cross-file context, not an open-ended exploration loop.
MAX_SEARCH_TURNS = 4

REPO_SEARCH_SUFFIX = """

## repo_search tool

You also have a `repo_search` tool: semantic search over the target
repository's .lean files, including files merely `import`-ed by (not
textually preceding) the theorem's own file. The repo context above only
shows the target file's own preceding content - if the theorem's statement
needs a type or definition not visible there (e.g. it lives in an imported
sibling file), call repo_search for it before inventing a placeholder
definition.
"""


def _emit_usage(tracer, thm_name: str, phase: str, model: str, usage) -> None:
    if tracer is None or usage is None:
        return
    tracer.emit(TraceEvent(
        kind="llm_usage",
        thm_name=thm_name,
        args={
            "phase": phase, "model": model,
            "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
    ))


def _call_with_repo_search(
    backend: ChatCompletionsBackend,
    messages: list[dict],
    repo_retrieval,
    reasoning_effort: str | None,
    max_tokens: int,
    tracer=None,
    thm_name: str = "",
    phase: str = "",
) -> str:
    """Drive one attempt through chat.completions, transparently handling
    repo_search tool calls.

    `messages` is mutated in place (tool round-trips appended, and the
    backend appends its own assistant replies as it goes) so the caller's
    own subsequent messages (e.g. compile-error feedback) continue to append
    correctly after this exchange, matching the existing retry loops in
    generate_blueprint/refine_blueprint. Returns the final text response
    (the caller no longer needs to append it - the backend already recorded
    it in `messages`).
    """
    tools = [REPO_SEARCH_TOOL] if repo_retrieval is not None else []
    dispatcher = ToolDispatcher({"repo_search": repo_search_handler(repo_retrieval)}) if repo_retrieval is not None else None

    for _ in range(MAX_SEARCH_TURNS):
        turn = backend.call_raw(messages, tools, AUTO, max_tokens, reasoning_effort)
        _emit_usage(tracer, thm_name, phase, backend.model_id, turn.usage)
        if not turn.tool_calls:
            return turn.text
        # Every tool_call in this turn must get a matching reply before the
        # next call.raw - a chat.completions request whose history contains
        # an assistant message with an unanswered tool_call is rejected
        # outright ("An assistant message with 'tool_calls' must be followed
        # by tool messages responding to each 'tool_call_id'").
        for tc in turn.tool_calls:
            output = dispatcher.dispatch(tc.name, tc.args) if dispatcher else f"Tool unavailable: {tc.name}"
            messages.append({"role": "tool", "tool_call_id": tc.call_id, "content": output})
    # Exhausted search turns without a final text response - one last call
    # with tools withheld forces the model to commit to an answer.
    turn = backend.call_raw(messages, [], AUTO, max_tokens, reasoning_effort)
    _emit_usage(tracer, thm_name, phase, backend.model_id, turn.usage)
    return turn.text


# Shared text-surgery helpers for the `@[blueprint]` grammar. Previously
# reimplemented independently (with subtly different regexes) in
# eval/vsb_lean_compiler.py's `_node_signature()`/`_build_blueprint_file()` -
# consolidated here as the single canonical version so a fix in one place
# can't silently leave a copy elsewhere unfixed.
_BLUEPRINT_ATTR_RE = re.compile(r"@\[blueprint\b[^\]]*\]", re.DOTALL)
_LEMMA_KW_RE = re.compile(r"(?m)^(\s*)lemma\b")


def strip_blueprint_attr(text: str) -> str:
    return _BLUEPRINT_ATTR_RE.sub("", text)


def lemma_to_theorem(text: str) -> str:
    """`lemma` needs Mathlib/Batteries; `theorem` works in every environment."""
    return _LEMMA_KW_RE.sub(r"\1theorem", text)


@dataclass
class BlueprintNode:
    name: str
    kind: str  # "definition" | "lemma" | "theorem"
    statement: str
    proof_sketch: str
    dependencies: list[str] = field(default_factory=list)
    lean_declaration: str = ""

    def signature(self) -> str:
        """Strip the @[blueprint ...] attribute and sorry_using proof body,
        returning just the declaration up to (not including) ':='.

        Used to re-declare an already-proved dependency as a real, standalone
        lemma (signature + its actual proof) so sibling nodes can reference it
        by name instead of hitting "unknown identifier" - proven dependencies
        are otherwise only ever shown to the model as prompt text, never
        actually compiled into scope.
        """
        text = lemma_to_theorem(strip_blueprint_attr(self.lean_declaration))
        return text.split(":=", 1)[0].strip()

    def cache_key(self) -> str:
        """Signature plus dependency set: a cached proof is only valid for
        the exact (signature, dependencies) shape it was compiled against.
        `signature()` alone only covers the text before `:=`, so a node
        whose sorry_using [...] dependency list changes (text AFTER `:=`)
        while its exposed statement stays byte-identical would otherwise be
        invisible to staleness checks, even though its cached proof was
        spliced together with the OLD set of sibling declarations in scope.
        """
        return self.signature() + "\x00deps:" + ",".join(sorted(self.dependencies))


@dataclass
class Blueprint:
    nodes: list[BlueprintNode]
    lean_file: str  # full compilable @[blueprint]-annotated Lean file
    target_theorem: str
    # True only when this exact lean_file was confirmed to compile by a real
    # Lean invocation (not a structural-only fallback, and not a give-up
    # after MAX_RETRIES). Defaults to False so any code path that forgets to
    # set it explicitly fails safe rather than silently claiming validation.
    fully_validated: bool = False

    def node_by_name(self, name: str) -> BlueprintNode | None:
        return next((n for n in self.nodes if n.name == name), None)

    def nodes_by_name(self) -> dict[str, BlueprintNode]:
        return {n.name: n for n in self.nodes}

    def dependency_order(self) -> list[BlueprintNode]:
        """Topological order (definitions first, theorem last)."""
        ordered: list[BlueprintNode] = []
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            node = self.node_by_name(name)
            if node:
                for dep in node.dependencies:
                    visit(dep)
                ordered.append(node)

        for node in self.nodes:
            visit(node.name)
        return ordered


def generate_blueprint(
    theorem_stmt: str,
    nl_proof: str | None = None,
    model: str = "gpt-5.5",
    compiler: AbstractLeanCompiler | None = None,
    repo_context: str | None = None,
    repo_retrieval=None,
    tracer=None,
    thm_name: str = "",
) -> Blueprint:
    """
    Generate a @[blueprint]-annotated Lean dependency graph for `theorem_stmt`.

    Uses the verbatim system prompt from Appendix C.1 of the paper.
    Validates via lean_compile after each LLM attempt (up to MAX_RETRIES).

    repo_retrieval: optional RepoRetrieval, giving the model a repo_search
        tool for cross-file lookups repo_context itself can't provide (see
        REPO_SEARCH_TOOL). Omit for the old no-tools behavior.
    """
    backend = ChatCompletionsBackend(make_client(model), model)
    reasoning_effort = default_reasoning_effort(model)

    system_content = BLUEPRINT_SYSTEM_PROMPT
    if repo_retrieval is not None:
        system_content = system_content.strip() + "\n" + REPO_SEARCH_SUFFIX
    user_content = _build_user_prompt(theorem_stmt, nl_proof, repo_context)
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    last_lean_code = None
    for attempt in range(MAX_RETRIES):
        content = _call_with_repo_search(
            backend, messages, repo_retrieval, reasoning_effort, MAX_TOKENS,
            tracer=tracer, thm_name=thm_name, phase="phase1",
        )
        lean_code = _extract_lean_code(content)
        last_lean_code = lean_code

        if compiler is not None:
            target = _extract_target_name(lean_code, theorem_stmt)
            result = compiler.check_blueprint(lean_code, target)
            if result.success:
                parsed = _parse_blueprint(lean_code, target)
                if parsed.nodes:
                    cycle = find_blueprint_cycle(parsed)
                    if cycle:
                        messages.append({
                            "role": "user",
                            "content": (
                                f"The blueprint has an invalid dependency structure "
                                f"(attempt {attempt + 1}/{MAX_RETRIES}): {cycle}\n\n"
                                "Re-emit the blueprint with an acyclic dependency structure "
                                "where every node's sorry_using [...] list only cites "
                                "already-established facts, never the target theorem itself."
                            ),
                        })
                        continue
                    parsed.fully_validated = result.validated
                    return parsed
                # Compiles, but has zero @[blueprint]-annotated declarations —
                # e.g. the model wrote a plain (already-complete or sorry-free)
                # theorem with no blueprint/sorry_using annotations at all.
                # Downstream, an empty node set makes all_proved() vacuously
                # true with no actual proof recorded, so this must be retried
                # rather than accepted as a usable blueprint.
                messages.append({
                    "role": "user",
                    "content": (
                        f"The file compiled, but contains no `@[blueprint ...]`-annotated "
                        f"declarations (attempt {attempt + 1}/{MAX_RETRIES}). You must "
                        "annotate the target theorem (and any helper lemmas) with "
                        "`@[blueprint ...]` and give each a `sorry_using [...]` proof body. "
                        "Re-emit the blueprint with proper annotations."
                    ),
                })
                continue
            # Feed errors back to the model for the next attempt
            error_feedback = "\n".join(result.errors) or result.raw_output[-2000:]
            messages.append({
                "role": "user",
                "content": (
                    f"lean_compile reported errors (attempt {attempt + 1}/{MAX_RETRIES}):\n\n"
                    f"{error_feedback}\n\n"
                    "Fix the issues and call lean_compile again."
                ),
            })
        else:
            target = _extract_target_name(lean_code, theorem_stmt)
            parsed = _parse_blueprint(lean_code, target)
            if parsed.nodes:
                return parsed
            # No compiler here to validate against (see comment above this
            # branch), but a response with zero @[blueprint]-annotated
            # declarations - a refusal, an apology, a plain sorry-free proof -
            # is pure text-parsing to detect and needs no compiler at all.
            # Silently accepting it as a "blueprint" would make all_proved()
            # vacuously true downstream with no actual proof recorded, so
            # this must be retried the same way the compiler branch already
            # retries its own zero-node case above.
            messages.append({
                "role": "user",
                "content": (
                    f"Your response contains no `@[blueprint ...]`-annotated "
                    f"declarations (attempt {attempt + 1}/{MAX_RETRIES}). You must "
                    "annotate the target theorem (and any helper lemmas) with "
                    "`@[blueprint ...]` and give each a `sorry_using [...]` proof body. "
                    "Re-emit the blueprint with proper annotations."
                ),
            })

    # All attempts failed compilation — use the last generated blueprint anyway
    # if it has real nodes (Phase 2/3 will encounter and surface type errors
    # during node proving). But an empty node set is never usable: it makes
    # all_proved() vacuously true downstream with no actual proof recorded,
    # so that must be a hard failure rather than a silent fake success.
    # `parsed.fully_validated` is deliberately left at its default False here -
    # this blueprint was never actually accepted by a real compile.
    if last_lean_code:
        target = _extract_target_name(last_lean_code, theorem_stmt)
        parsed = _parse_blueprint(last_lean_code, target)
        if parsed.nodes:
            return parsed
    raise RuntimeError(
        f"Blueprint generation failed after {MAX_RETRIES} attempts "
        "(no attempt produced any @[blueprint]-annotated nodes)"
    )


def _build_user_prompt(theorem_stmt: str, nl_proof: str | None, repo_context: str | None = None) -> str:
    return render(BLUEPRINT_USER_TEMPLATE, theorem_stmt=theorem_stmt, nl_proof=nl_proof or "", repo_context=repo_context or "")


# Matches the first line that looks like real Lean source, used to strip a
# leaked non-Lean preamble (e.g. a model hallucinating a tool-call-style tag
# like `<lean_compile>` instead of a code fence) when there's no fence to
# delimit the code block.
_LEAN_START_RE = re.compile(
    r"^\s*(?:import\b|@\[blueprint\b|theorem\b|lemma\b|noncomputable\s+def\b|def\b|abbrev\b)",
    re.MULTILINE,
)


def _extract_lean_code(content: str) -> str:
    """Extract the Lean code block from the LLM response."""
    match = re.search(r"```(?:lean)?\n(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    # No fence - the model may still have prefixed its response with
    # non-Lean text (a leaked tag, an apology, etc.). Start at the first
    # line that looks like real Lean rather than treating the raw response
    # as Lean verbatim.
    start_match = _LEAN_START_RE.search(content)
    if start_match:
        return content[start_match.start():].strip()
    return content.strip()


def _parse_blueprint(lean_code: str, target_theorem: str) -> Blueprint:
    """
    Parse @[blueprint]-annotated Lean code into a Blueprint datastructure.

    Extracts node names, kinds, statements, proof sketches, and sorry_using deps.
    """
    nodes: list[BlueprintNode] = []
    # Match @[blueprint ...] blocks followed by a declaration
    pattern = re.compile(
        r'@\[blueprint\s*(.*?)\]\s*\n\s*(def|lemma|theorem|noncomputable def|abbrev)\s+(\w+)(.*?)(?=@\[blueprint|\Z)',
        re.DOTALL,
    )
    for m in pattern.finditer(lean_code):
        attrs_block = m.group(1)
        kind_kw = m.group(2).strip()
        name = m.group(3)
        rest = m.group(4)

        kind = "definition" if kind_kw in ("def", "noncomputable def", "abbrev") else kind_kw

        statement = _extract_attr(attrs_block, "statement")
        proof_sketch = _extract_attr(attrs_block, "proof")

        # Extract sorry_using [...] dependencies
        dep_match = re.search(r"sorry_using\s*\[([^\]]*)\]", rest)
        deps = [d.strip() for d in dep_match.group(1).split(",") if d.strip()] if dep_match else []

        nodes.append(BlueprintNode(
            name=name,
            kind=kind,
            statement=statement,
            proof_sketch=proof_sketch,
            dependencies=deps,
            lean_declaration=m.group(0),
        ))

    return Blueprint(nodes=nodes, lean_file=lean_code, target_theorem=target_theorem)


def find_blueprint_cycle(blueprint: Blueprint) -> str | None:
    """Return a description of a dependency-cycle defect, or None if clean.

    Caught concretely on TM.progress: a refinement round had gpt-5.5 add a
    helper node whose `sorry_using [progress]` cited the target theorem
    itself as an "already available" fact (rationalized as reusing the
    ground-truth repo's own real proof of `progress` via its import), while
    `progress` in turn depended on that helper - a 2-cycle that crashed
    orchestrator.py's `nx.topological_generations` with an unhandled
    NetworkXUnfeasible. A node citing the target before it's proved is
    circular reasoning regardless of whether it closes a graph cycle through
    other nodes, so that's checked directly and not just via cycle-search.
    """
    names = {n.name for n in blueprint.nodes}
    graph = {n.name: [d for d in n.dependencies if d in names] for n in blueprint.nodes}

    for name, deps in graph.items():
        if name != blueprint.target_theorem and blueprint.target_theorem in deps:
            return (
                f"'{name}' lists the target theorem '{blueprint.target_theorem}' "
                "itself in sorry_using [...] - the target isn't proved yet, so no "
                "other node may cite it as an already-available fact (this includes "
                "citing an existing proof of it from an imported file)."
            )

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {name: WHITE for name in graph}
    path: list[str] = []

    def visit(name: str) -> str | None:
        color[name] = GRAY
        path.append(name)
        for dep in graph.get(name, []):
            if color.get(dep) == GRAY:
                return " -> ".join(path[path.index(dep):] + [dep])
            if color.get(dep) == WHITE:
                found = visit(dep)
                if found:
                    return found
        path.pop()
        color[name] = BLACK
        return None

    for name in graph:
        if color[name] == WHITE:
            found = visit(name)
            if found:
                return f"dependency cycle: {found}"
    return None


def _extract_attr(attrs: str, key: str) -> str:
    match = re.search(rf"\({key}\s*:=\s*/--\s*(.*?)\s*-/\)", attrs, re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_target_name(lean_code: str, fallback: str) -> str:
    """Extract the main theorem name from the blueprint Lean code.

    The main theorem is the last `theorem` declaration (which must equal the
    targeted identifier per the blueprint system prompt).
    """
    matches = re.findall(r"\btheorem\s+(\w+)", lean_code)
    if matches:
        return matches[-1]
    # Fallback: extract identifier from the theorem statement
    m = re.search(r"\btheorem\s+(\w+)", fallback)
    return m.group(1) if m else "main_theorem"
