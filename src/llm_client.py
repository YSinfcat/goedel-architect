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


DEFAULT_TIMEOUT_S = 300.0


def make_client(model_id: str, timeout: float | None = None) -> OpenAI:
    """timeout=None (the default, used by every caller that doesn't pass one
    explicitly - blueprint.py/refinement.py) must NOT reach OpenAI(timeout=None):
    the SDK treats an explicit None as "disable the timeout entirely" rather
    than "use my own default", so a hung request (observed: a Phase 3 call
    stuck for 70+ minutes with no way to abort) would block forever."""
    timeout = timeout if timeout is not None else DEFAULT_TIMEOUT_S
    if model_id.startswith("accounts/"):
        return OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.environ["FIREWORKS_API_KEY"],
            timeout=timeout,
        )
    return OpenAI(timeout=timeout)
