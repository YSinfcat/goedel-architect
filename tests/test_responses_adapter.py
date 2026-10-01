"""Offline unit tests for the chat->responses translation layer (no
network): message/tool/choice mapping and the response-shape adapter.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("GOEDEL_LEAN_TOOLCHAIN", "test-toolchain")

from responses_adapter import (  # noqa: E402
    _Response, _translate_messages, _translate_tools)


class TestTranslation(unittest.TestCase):
    def test_system_user_pass_through(self):
        items = _translate_messages([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ])
        self.assertEqual(items[0], {"role": "system", "content": "sys"})
        self.assertEqual(items[1], {"role": "user", "content": "hello"})

    def test_assistant_tool_calls_split_into_function_call_items(self):
        items = _translate_messages([
            {"role": "assistant", "content": "thinking",
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "lean_compile",
                                          "arguments": '{"proof_body": "by simp"}'}}]},
        ])
        self.assertEqual(items[0], {"role": "assistant", "content": "thinking"})
        self.assertEqual(items[1]["type"], "function_call")
        self.assertEqual(items[1]["call_id"], "call_1")
        self.assertEqual(items[1]["name"], "lean_compile")

    def test_tool_result_becomes_function_call_output(self):
        items = _translate_messages([
            {"role": "tool", "tool_call_id": "call_1",
             "content": "Compilation SUCCESSFUL."},
        ])
        self.assertEqual(items[0]["type"], "function_call_output")
        self.assertEqual(items[0]["call_id"], "call_1")
        self.assertEqual(items[0]["output"], "Compilation SUCCESSFUL.")

    def test_tools_definition_shape(self):
        tools = _translate_tools([
            {"type": "function", "function": {
                "name": "repo_search",
                "description": "search",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}],
        )
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["name"], "repo_search")
        self.assertIn("q", tools[0]["parameters"]["properties"])


class TestResponseShape(unittest.TestCase):
    def _raw(self, output_items, usage=None):
        return SimpleNamespace(output=output_items,
                               usage=usage or SimpleNamespace(
                                   input_tokens=7, output_tokens=3))

    def test_text_and_usage(self):
        raw = self._raw([SimpleNamespace(type="message", content=[
            SimpleNamespace(text="hello ") , SimpleNamespace(text="world")])])
        r = _Response(raw)
        self.assertEqual(r.choices[0].message.content, "hello world")
        self.assertEqual(r.usage.prompt_tokens, 7)
        self.assertEqual(r.usage.total_tokens, 10)

    def test_function_call_maps_to_tool_calls(self):
        raw = self._raw([SimpleNamespace(
            type="function_call", call_id="call_9",
            name="lean_compile", arguments='{"proof_body":"by rfl"}')])
        r = _Response(raw)
        tc = r.choices[0].message.tool_calls[0]
        self.assertEqual(tc.id, "call_9")
        self.assertEqual(tc.function.name, "lean_compile")
        self.assertEqual(tc.function.arguments, '{"proof_body":"by rfl"}')
        self.assertEqual(r.choices[0].finish_reason, "tool_calls")

    def test_model_dump_for_history_replay(self):
        raw = self._raw([SimpleNamespace(
            type="function_call", call_id="c1",
            name="f", arguments="{}")])
        d = _Response(raw).model_dump()
        self.assertEqual(d["tool_calls"][0]["id"], "c1")
        self.assertEqual(d["tool_calls"][0]["function"]["name"], "f")


if __name__ == "__main__":
    unittest.main(verbosity=2)
