"""Phase 1: Blueprint generation.

Calls the LLM with the verbatim system prompt from the paper (prompts/blueprint_system.md)
and validates the resulting @[blueprint]-annotated Lean file via LeanArchitect.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

from lean_compiler import AbstractLeanCompiler, LeanCompiler, CompilerResult
from llm_client import make_client
from goedel_prompts import load, render
from tracer import TraceEvent


def _reasoning_kwargs(model: str) -> dict:
    """Return reasoning_effort kwarg for models that support it (gpt-5.x series)."""
    if model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4"):
        return {"reasoning_effort": "low"}
    return {}

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
REPO_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "repo_search",
        "description": (
            "Semantic search over the target repository's .lean files, "
            "including files merely imported by (not textually preceding) "
            "the theorem's own file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language or identifier fragment."},
                "k": {"type": "integer", "description": "Number of results (default 10).", "default": 10},
            },
            "required": ["query"],
        },
    },
}

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


def _emit_usage(tracer, thm_name: str, phase: str, model: str, response) -> None:
    """Log token usage from a chat.completions/responses API response, if a
    tracer was given. `response.usage` is present on both APIs but with
    different field names, so normalize to prompt/completion/total."""
    if tracer is None:
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if prompt is None:
        prompt = getattr(usage, "input_tokens", 0)
    if completion is None:
        completion = getattr(usage, "output_tokens", 0)
    total = getattr(usage, "total_tokens", None) or (prompt + completion)
    tracer.emit(TraceEvent(
        kind="llm_usage",
        thm_name=thm_name,
        args={
            "phase": phase, "model": model,
            "prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": total,
        },
    ))


def _call_with_repo_search(
    client: OpenAI,
    model: str,
    messages: list[dict],
    repo_retrieval,
    reasoning_kwargs: dict,
    max_tokens: int,
    tracer=None,
    thm_name: str = "",
    phase: str = "",
):
    """chat.completions.create, transparently handling repo_search tool calls.

    `messages` is mutated in place (tool round-trips appended) so the
    caller's own subsequent messages (e.g. compile-error feedback) continue
    to append correctly after this exchange, matching the existing retry
    loops in generate_blueprint/refine_blueprint. Returns the first response
    that isn't a tool-calls-only response.
    """
    tools = [REPO_SEARCH_TOOL] if repo_retrieval is not None else None
    for _ in range(MAX_SEARCH_TURNS):
        # chat.completions rejects function tools + reasoning_effort together
        # for gpt-5.x ("Function tools with reasoning_effort are not
        # supported ... Please use /v1/responses instead") - drop
        # reasoning_effort on tool-enabled turns; it still applies to the
        # no-tools finalization call below (and to every call when
        # repo_retrieval is None, unchanged from before this tool existed).
        call_kwargs = {"tools": tools} if tools else dict(reasoning_kwargs)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
            **call_kwargs,
        )
        _emit_usage(tracer, thm_name, phase, model, response)
        msg = response.choices[0].message
        if not msg.tool_calls:
            return response
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })
        for tc in msg.tool_calls:
            if tc.function.name == "repo_search":
                args = json.loads(tc.function.arguments)
                hits = repo_retrieval.search(args.get("query", ""), args.get("k", 10))
                result = "\n\n".join(h.format() for h in hits) or "No results in repo."
            else:
                result = f"Tool unavailable: {tc.function.name}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    # Exhausted search turns without a final text response - one last call
    # with tools withheld forces the model to commit to an answer.
    response = client.chat.completions.create(
        model=model, messages=messages, max_completion_tokens=max_tokens, **reasoning_kwargs,
    )
    _emit_usage(tracer, thm_name, phase, model, response)
    return response


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
    client = make_client(model)

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
        response = _call_with_repo_search(
            client, model, messages, repo_retrieval, _reasoning_kwargs(model), MAX_TOKENS,
            tracer=tracer, thm_name=thm_name, phase="phase1",
        )
        lean_code = _extract_lean_code(response.choices[0].message.content)
        last_lean_code = lean_code

        if compiler is not None:
            target = _extract_target_name(lean_code, theorem_stmt)
            result = compiler.check_blueprint(lean_code, target)
            if result.success:
                parsed = _parse_blueprint(lean_code, target)
                if parsed.nodes:
                    parsed.fully_validated = result.validated
                    return parsed
                # Compiles, but has zero @[blueprint]-annotated declarations —
                # e.g. the model wrote a plain (already-complete or sorry-free)
                # theorem with no blueprint/sorry_using annotations at all.
                # Downstream, an empty node set makes all_proved() vacuously
                # true with no actual proof recorded, so this must be retried
                # rather than accepted as a usable blueprint.
                messages.append({"role": "assistant", "content": response.choices[0].message.content})
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
            messages.append({"role": "assistant", "content": response.choices[0].message.content})
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
            messages.append({"role": "assistant", "content": response.choices[0].message.content})
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
