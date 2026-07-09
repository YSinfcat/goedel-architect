"""Shared OpenAI-client construction, routing Fireworks-hosted models to
Fireworks' base_url instead of OpenAI's.

Fireworks addresses its models as "accounts/<org>/models/<name>" (e.g.
"accounts/fireworks/models/deepseek-v4-flash"), and its API is
OpenAI-compatible for both chat.completions and the Responses API, so a
model_id in that shape is routed there while everything else keeps hitting
OpenAI as before.
"""
from __future__ import annotations

import os

from openai import OpenAI


def make_client(model_id: str, timeout: float | None = None) -> OpenAI:
    if model_id.startswith("accounts/"):
        return OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.environ["FIREWORKS_API_KEY"],
            timeout=timeout,
        )
    return OpenAI(timeout=timeout)
