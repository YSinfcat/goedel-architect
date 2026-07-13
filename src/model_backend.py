"""Backend-agnostic seam for driving one turn of a tool-calling conversation.

GoedelProver's Phase 2 tool loop used to be written directly against OpenAI's
Responses API wire shape (response.output items, previous_response_id
chaining), and Phase 1/3's retry loop directly against chat.completions'
message-list shape. Swapping either backend's wire format previously meant
rewriting the loop itself (see git history: the Leanstral integration
attempt rewrote ~400 lines of prover.py just to add a chat.completions-only
model). ModelBackend hides exactly one turn's wire-format differences
(tool-schema shape, tool_choice encoding, multi-turn continuation, usage
field names) behind a single interface; the multi-turn loop stays shared,
backend-agnostic code that calls `start`/`continue_with` repeatedly.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from llm_client import make_client


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict


@dataclass(frozen=True)
class ToolChoice:
    """auto: model may call 0+ tools. none: model must reply with text.
    required: model must call some tool. force: model must call `tool_name`
    specifically (each backend absorbs however its own API expresses that -
    see ResponsesBackend/ChatCompletionsBackend)."""
    mode: str  # "auto" | "none" | "required" | "force"
    tool_name: str | None = None

    @classmethod
    def force(cls, name: str) -> "ToolChoice":
        return cls(mode="force", tool_name=name)


AUTO = ToolChoice(mode="auto")
NONE = ToolChoice(mode="none")
REQUIRED = ToolChoice(mode="required")


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    args: dict


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    output: str


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage | None = None
    # Opaque continuation token - a Responses API response.id, or a
    # ChatCompletions messages list. Callers pass it back into
    # continue_with() unexamined; only the backend that produced it knows
    # what to do with it.
    handle: Any = None


def default_reasoning_effort(model_id: str) -> str | None:
    """The gpt-5/o1/o3/o4 reasoning-effort policy, shared by every backend.

    Override via GOEDEL_REASONING_EFFORT (none|low|medium|high|xhigh|max).
    Default remains low to match prior cost/latency behavior. Returning the
    string "none" (vs Python None) pins effort explicitly — needed on
    gpt-5.6-sol tool turns where omitting the kwarg defaults to medium and
    chat.completions then rejects tools + non-none effort.
    """
    import os
    if model_id.startswith("gpt-5") or model_id.startswith(("o1", "o3", "o4")):
        return os.environ.get("GOEDEL_REASONING_EFFORT", "low")
    return None


class ModelBackend(ABC):
    """One concrete backend's wire format for a single request/response turn."""

    def __init__(self, client, model_id: str):
        self.client = client
        self.model_id = model_id

    @abstractmethod
    def start(
        self,
        system: str,
        user: str,
        tools: list[ToolSpec] = (),
        tool_choice: ToolChoice = AUTO,
        max_tokens: int = 64_000,
        reasoning_effort: str | None = None,
    ) -> Turn:
        """Begin a fresh conversation (system + user message)."""

    @abstractmethod
    def continue_with(
        self,
        turn: Turn,
        *,
        results: list[ToolResult] | None = None,
        text: str | None = None,
        tools: list[ToolSpec] = (),
        tool_choice: ToolChoice = AUTO,
        max_tokens: int = 64_000,
        reasoning_effort: str | None = None,
    ) -> Turn:
        """Continue a conversation, submitting either tool results (the
        normal case) or a plain-text nudge (e.g. "output your best proof").
        Exactly one of `results`/`text` should be given."""


# ---------------------------------------------------------------------------
# ResponsesBackend - OpenAI's Responses API (also what Fireworks-hosted
# models speak, per llm_client.make_client's routing)
# ---------------------------------------------------------------------------

def _to_responses_tool(spec: ToolSpec) -> dict:
    return {"type": "function", "name": spec.name, "description": spec.description,
            "parameters": spec.parameters}


def _responses_tool_kwargs(tools: list[ToolSpec], tool_choice: ToolChoice) -> dict:
    if not tools:
        return {}
    if tool_choice.mode == "force":
        # Not OpenAI's named tool_choice shape ({"type": "function", "name":
        # ...}) - Fireworks' Responses API rejects that with a 400, accepting
        # only the bare "required"/"auto"/"none"/"any" literals, which force
        # *some* tool call but can't pin down which one. Restricting the
        # visible `tools` list to just the forced tool makes "required"
        # unambiguous on every provider, and also stops a model from
        # hallucinating a nonexistent tool.
        forced = next((t for t in tools if t.name == tool_choice.tool_name), None)
        wire_tools = [_to_responses_tool(forced)] if forced else [_to_responses_tool(t) for t in tools]
        return {"tools": wire_tools, "tool_choice": "required"}
    return {"tools": [_to_responses_tool(t) for t in tools], "tool_choice": tool_choice.mode}


def _responses_reasoning_kwargs(effort: str | None) -> dict:
    return {"reasoning": {"effort": effort}} if effort else {}


def _usage_from_responses(response) -> Usage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    prompt = getattr(usage, "input_tokens", 0)
    completion = getattr(usage, "output_tokens", 0)
    total = getattr(usage, "total_tokens", None) or (prompt + completion)
    return Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


def _turn_from_responses(response) -> Turn:
    text = ""
    tool_calls: list[ToolCall] = []
    for item in response.output:
        if item.type == "function_call":
            tool_calls.append(ToolCall(call_id=item.call_id, name=item.name,
                                        args=json.loads(item.arguments)))
        elif item.type == "message":
            for block in item.content:
                block_text = getattr(block, "text", None)
                if block_text:
                    text = block_text
    return Turn(text=text, tool_calls=tool_calls, usage=_usage_from_responses(response),
                handle=response.id)


class ResponsesBackend(ModelBackend):
    def start(self, system, user, tools=(), tool_choice=AUTO, max_tokens=64_000,
              reasoning_effort=None) -> Turn:
        response = self.client.responses.create(
            model=self.model_id,
            instructions=system,
            input=user,
            max_output_tokens=max_tokens,
            **_responses_tool_kwargs(list(tools), tool_choice),
            **_responses_reasoning_kwargs(reasoning_effort),
        )
        return _turn_from_responses(response)

    def continue_with(self, turn, *, results=None, text=None, tools=(), tool_choice=AUTO,
                       max_tokens=64_000, reasoning_effort=None) -> Turn:
        if results is not None:
            input_ = [{"type": "function_call_output", "call_id": r.call_id, "output": r.output}
                      for r in results]
        else:
            input_ = text
        response = self.client.responses.create(
            model=self.model_id,
            previous_response_id=turn.handle,
            input=input_,
            max_output_tokens=max_tokens,
            **_responses_tool_kwargs(list(tools), tool_choice),
            **_responses_reasoning_kwargs(reasoning_effort),
        )
        return _turn_from_responses(response)


# ---------------------------------------------------------------------------
# ChatCompletionsBackend - chat.completions, for backends that don't expose
# the Responses API (today: Phase 1/3, pinned to this regardless of model -
# see blueprint.py/refinement.py). A future chat.completions-only model
# (e.g. re-adding Mistral's Leanstral) would only need make_backend() to
# route it here - not a tool-loop rewrite.
# ---------------------------------------------------------------------------

def _to_cc_tool(spec: ToolSpec) -> dict:
    return {"type": "function", "function": {
        "name": spec.name, "description": spec.description, "parameters": spec.parameters,
    }}


def _cc_tool_kwargs(tools: list[ToolSpec], tool_choice: ToolChoice) -> dict:
    if not tools:
        return {}
    wire_tools = [_to_cc_tool(t) for t in tools]
    if tool_choice.mode == "force":
        return {"tools": wire_tools,
                "tool_choice": {"type": "function", "function": {"name": tool_choice.tool_name}}}
    return {"tools": wire_tools, "tool_choice": tool_choice.mode}


def _cc_reasoning_kwargs(effort: str | None) -> dict:
    return {"reasoning_effort": effort} if effort else {}


def _usage_from_cc(response) -> Usage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    prompt = getattr(usage, "prompt_tokens", 0)
    completion = getattr(usage, "completion_tokens", 0)
    total = getattr(usage, "total_tokens", None) or (prompt + completion)
    return Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


class ChatCompletionsBackend(ModelBackend):
    def start(self, system, user, tools=(), tool_choice=AUTO, max_tokens=64_000,
              reasoning_effort=None) -> Turn:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.call_raw(messages, list(tools), tool_choice, max_tokens, reasoning_effort)

    def continue_with(self, turn, *, results=None, text=None, tools=(), tool_choice=AUTO,
                       max_tokens=64_000, reasoning_effort=None) -> Turn:
        messages = turn.handle  # mutated in place turn-over-turn
        if results is not None:
            for r in results:
                messages.append({"role": "tool", "tool_call_id": r.call_id, "content": r.output})
        else:
            messages.append({"role": "user", "content": text})
        return self.call_raw(messages, list(tools), tool_choice, max_tokens, reasoning_effort)

    def call_raw(self, messages: list[dict], tools: list[ToolSpec], tool_choice: ToolChoice,
                 max_tokens: int, reasoning_effort: str | None = None) -> Turn:
        """Escape hatch for a caller that manages its own `messages` list
        directly (e.g. blueprint.py's retry loop, which appends its own
        user-feedback turns between attempts) - makes exactly one call
        against the given list, mutating it in place with the assistant's
        reply, and returns the resulting Turn. `start`/`continue_with` are
        both thin wrappers around this."""
        kwargs = _cc_tool_kwargs(tools, tool_choice)
        # chat.completions rejects function tools + non-none reasoning_effort
        # for gpt-5.x. gpt-5.6-sol defaults to medium when the kwarg is
        # omitted, so tool turns must explicitly pin effort to "none".
        if tools:
            if self.model_id.startswith("gpt-5") or self.model_id.startswith(("o1", "o3", "o4")):
                kwargs["reasoning_effort"] = "none"
        else:
            kwargs.update(_cc_reasoning_kwargs(reasoning_effort))
        response = self.client.chat.completions.create(
            model=self.model_id, messages=messages, max_completion_tokens=max_tokens, **kwargs,
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            messages.append({
                "role": "assistant", "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })
        else:
            messages.append({"role": "assistant", "content": msg.content})
        tool_calls = [
            ToolCall(call_id=tc.id, name=tc.function.name, args=json.loads(tc.function.arguments))
            for tc in (msg.tool_calls or [])
        ]
        return Turn(text=msg.content or "", tool_calls=tool_calls,
                    usage=_usage_from_cc(response), handle=messages)


def make_backend(model_id: str, timeout_s: float | None = None) -> ModelBackend:
    """Routes model_id to the right backend adapter.

    Every model in production use today speaks the Responses API (OpenAI
    directly, or Fireworks-hosted models, which are Responses-API-compatible
    per llm_client.make_client's routing) - this always returns
    ResponsesBackend. Phase 1/3 don't call this at all; they explicitly
    construct a ChatCompletionsBackend regardless of model_id (see
    blueprint.py), matching their existing chat.completions-only behavior.
    """
    client = make_client(model_id, timeout=timeout_s)
    return ResponsesBackend(client, model_id)
