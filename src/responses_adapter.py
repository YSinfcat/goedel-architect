"""chat.completions -> Responses API adapter.

Some gateways expose certain models ONLY through the Responses API
(e.g. OpenCode GO's gpt-6-luna rejects chat/completions with
ModelProtocolUnsupported), while this codebase's nine LLM call sites all
speak chat.completions (messages with tool results, tool_calls on the
reply, usage fields). Rather than rewriting the call sites, the custom
provider route can hand out an adapter client whose .chat.completions
surface translates transparently:

    chat.completions.create(model, messages, tools, tool_choice,
                            max_completion_tokens, reasoning_effort)
        -> responses.create(model, input, tools, tool_choice,
                            max_output_tokens, reasoning)

Everything the prover/blueprint loops read from a response is preserved:
choices[0].message.{content, tool_calls[{id, function.{name,arguments}}]},
usage.{prompt_tokens, completion_tokens, total_tokens}. Multi-choice
requests are not used by this codebase and surface as a single choice.
"""
from __future__ import annotations

from typing import Any


def _translate_messages(messages: list[dict]) -> list[dict]:
    """chat messages -> responses input items. Tool results become
    function_call_output items referencing the call_id."""
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            out.append({
                "type": "function_call_output",
                "call_id": m.get("tool_call_id", ""),
                "output": m.get("content") or "",
            })
        elif role == "assistant":
            item: dict[str, Any] = {"role": "assistant",
                                    "content": m.get("content") or ""}
            tool_calls = m.get("tool_calls") or []
            out.append(item)
            for tc in tool_calls:
                fn = tc.get("function", {})
                out.append({
                    "type": "function_call",
                    "call_id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", "{}"),
                })
        else:  # system / user / developer
            out.append({"role": role if role != "system" else "system",
                        "content": m.get("content") or ""})
    return out


def _translate_tools(tools: list[dict] | None) -> list[dict] | None:
    """chat tool definitions -> responses tool definitions."""
    if not tools:
        return None
    out = []
    for t in tools:
        fn = t.get("function", {})
        out.append({
            "type": "function",
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return out


def _translate_tool_choice(tool_choice) -> Any:
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return {"type": "function", "name": tool_choice["function"]["name"]}
    return tool_choice  # "auto" / "none" pass through


class _Function:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.type = "function"
        self.function = _Function(name, arguments)


class _Message:
    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls or None
        self.role = "assistant"


class _Choice:
    def __init__(self, message, finish_reason):
        self.message = message
        self.finish_reason = finish_reason or "stop"
        self.index = 0


class _Usage:
    def __init__(self, prompt: int, completion: int):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion


class _Response:
    def __init__(self, raw):
        self._raw = raw
        content = ""
        tool_calls: list[_ToolCall] = []
        finish = None
        for item in (getattr(raw, "output", None) or []):
            itype = getattr(item, "type", "")
            if itype == "message":
                for part in (getattr(item, "content", None) or []):
                    text = getattr(part, "text", None)
                    if text:
                        content += text
            elif itype == "function_call":
                tool_calls.append(_ToolCall(
                    getattr(item, "call_id", "") or getattr(item, "id", ""),
                    getattr(item, "name", ""),
                    getattr(item, "arguments", "{}")))
                finish = "tool_calls"
        self.choices = [_Choice(_Message(content, tool_calls), finish)]
        usage = getattr(raw, "usage", None)
        self.usage = _Usage(
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0) if usage else _Usage(0, 0)

    def model_dump(self):
        # compatibility with code paths that serialize the assistant turn
        # back into chat history (blueprint._append_assistant_turn)
        msg = {"role": "assistant", "content": self.choices[0].message.content}
        if self.choices[0].message.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in self.choices[0].message.tool_calls
            ]
        return msg


class _Completions:
    def __init__(self, responses_client, model_default_reasoning: str | None):
        self._responses = responses_client
        self._model_default_reasoning = model_default_reasoning

    def create(self, *, model: str, messages: list[dict],
               tools: list[dict] | None = None, tool_choice=None,
               max_completion_tokens: int | None = None,
               reasoning_effort: str | None = None, **extra):
        kwargs: dict[str, Any] = {
            "model": model,
            "input": _translate_messages(messages),
        }
        translated_tools = _translate_tools(tools)
        if translated_tools:
            kwargs["tools"] = translated_tools
        if tool_choice is not None and tool_choice != "auto":
            kwargs["tool_choice"] = _translate_tool_choice(tool_choice)
        if max_completion_tokens is not None:
            kwargs["max_output_tokens"] = max_completion_tokens
        effort = reasoning_effort or self._model_default_reasoning
        if effort:
            kwargs["reasoning"] = {"effort": effort}
        raw = self._responses.create(**kwargs)
        return _Response(raw)


class _Chat:
    def __init__(self, responses_client, model_default_reasoning):
        self.completions = _Completions(responses_client, model_default_reasoning)


class ResponsesAdapterClient:
    """Duck-typed stand-in for openai.OpenAI: .chat.completions.create
    translates to the underlying client's .responses.create."""

    def __init__(self, inner, default_reasoning: str | None = "low"):
        # inner.responses is the SDK's Responses resource (.create lives
        # there, not on the client itself).
        self.chat = _Chat(inner.responses, default_reasoning)
        self.base_url = getattr(inner, "base_url", None)
