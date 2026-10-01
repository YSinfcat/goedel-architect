"""Shared OpenAI-client construction, routing Fireworks- and Mistral-hosted
models to their own base_url instead of OpenAI's.

Fireworks addresses its models as "accounts/<org>/models/<name>" (e.g.
"accounts/fireworks/models/deepseek-v4-flash"), and its API is
OpenAI-compatible for both chat.completions and the Responses API, so a
model_id in that shape is routed there.

Mistral's "Labs" models (e.g. "labs-leanstral-1-5") are addressed with a
"labs-" prefix; Mistral's API is OpenAI-compatible for chat.completions
only — it has no Responses API equivalent, so callers must use
chat.completions.create with this client, not client.responses.create.

Everything else keeps hitting OpenAI as before.
"""
from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI


def _load_env_once() -> None:
    """Load the repo-root .env once per process, if present.

    README instructs `cp .env.example .env` and requirements.txt ships
    python-dotenv, but nothing ever loaded the file - keys had to be
    exported manually. load_dotenv does not override variables that are
    already set in the environment, so real env vars keep precedence.
    Every entry point funnels through make_client, so this is the single
    choke point.
    """
    if getattr(_load_env_once, "_done", False):
        return
    _load_env_once._done = True  # type: ignore[attr-defined]
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env")
    except ImportError:
        pass  # python-dotenv not installed: fall back to real env vars only


def make_client(model_id: str, timeout: float | None = None) -> OpenAI:
    _load_env_once()
    if model_id.startswith("accounts/"):
        return OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.environ["FIREWORKS_API_KEY"],
            timeout=timeout,
        )
    if model_id.startswith("labs-"):
        return OpenAI(
            base_url="https://api.mistral.ai/v1",
            api_key=os.environ["MISTRAL_API_KEY"],
            timeout=timeout,
        )
    return OpenAI(timeout=timeout)
