"""Shared OpenAI-client construction and retry policy.

Routing is table-driven (PROVIDER_REGISTRY) instead of scattered prefix
checks: a model id maps to a provider entry carrying its base_url, API-key
environment variable, and capabilities. Clients are cached per
(provider, timeout) - Phase 2 constructs one prover (and thus one client)
per node, and a long benchmark would otherwise accumulate hundreds of
live connection pools that are never closed.

create_with_retry adds bounded exponential backoff for transient provider
failures (429 / 5xx / connection errors), honoring Retry-After when the
provider sends it. Non-transient errors (4xx other than 429) surface
immediately.
"""
from __future__ import annotations

import os
import random
import threading
import time
from pathlib import Path

from openai import OpenAI

REPO_ROOT = Path(__file__).parent.parent

# provider id -> routing config. `chat_only` providers have no Responses
# API equivalent - callers must use chat.completions.create exclusively.
PROVIDER_REGISTRY: dict[str, dict] = {
    "fireworks": {
        "prefix": "accounts/",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
        "chat_only": False,
    },
    "mistral-labs": {
        "prefix": "labs-",
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
        "chat_only": True,
    },
    # everything else: plain OpenAI
    "openai": {"prefix": "", "base_url": None, "api_key_env": "OPENAI_API_KEY", "chat_only": False},
}

_CLIENT_CACHE: dict[tuple[str, float | None], OpenAI] = {}
_CLIENT_CACHE_LOCK = threading.Lock()


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
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass  # python-dotenv not installed: fall back to real env vars only


def provider_for(model_id: str) -> tuple[str, dict]:
    """(provider_id, config) for a model id - the explicit counterpart to
    the old prefix `if` chain (review IV.9)."""
    for name, cfg in PROVIDER_REGISTRY.items():
        if cfg["prefix"] and model_id.startswith(cfg["prefix"]):
            return name, cfg
    return "openai", PROVIDER_REGISTRY["openai"]


def make_client(model_id: str, timeout: float | None = None) -> OpenAI:
    _load_env_once()
    provider, cfg = provider_for(model_id)
    key = (provider, timeout)
    with _CLIENT_CACHE_LOCK:
        cached = _CLIENT_CACHE.get(key)
        if cached is not None:
            return cached
        kwargs: dict = {"timeout": timeout}
        if cfg["base_url"]:
            kwargs["base_url"] = cfg["base_url"]
            kwargs["api_key"] = os.environ[cfg["api_key_env"]]
        client = OpenAI(**kwargs)
        _CLIENT_CACHE[key] = client
        return client


# ---------------------------------------------------------------------------
# Bounded retry for transient provider failures (review P1-2)
# ---------------------------------------------------------------------------

MAX_CREATE_RETRIES = 4      # total attempts = 1 + MAX_CREATE_RETRIES
RETRY_BASE_DELAY_S = 1.0
RETRY_MAX_DELAY_S = 30.0


def _status_code(exc: Exception) -> int | None:
    return getattr(exc, "status_code", None)


def _is_retryable(exc: Exception) -> bool:
    status = _status_code(exc)
    if status is not None:
        return status == 429 or status >= 500
    # openai's APIConnectionError/APITimeoutError carry no status_code but
    # are transient by nature; match by name to avoid importing every
    # exception type (and to keep stub exceptions in tests working).
    return type(exc).__name__ in ("APIConnectionError", "APITimeoutError", "ReadTimeout", "ConnectError")


def _retry_after_s(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def create_with_retry(client, **kwargs):
    """client.chat.completions.create with bounded exponential backoff.

    Retries 429/5xx/connection errors up to MAX_CREATE_RETRIES times,
    honoring a numeric Retry-After header when present (capped at
    RETRY_MAX_DELAY_S). Any other exception propagates immediately - a 400
    (bad request) will not fix itself by waiting.
    """
    attempt = 0
    while True:
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            if not _is_retryable(exc) or attempt >= MAX_CREATE_RETRIES:
                raise
            delay = _retry_after_s(exc)
            if delay is None:
                delay = min(RETRY_MAX_DELAY_S, RETRY_BASE_DELAY_S * (2 ** attempt))
                delay *= 0.5 + random.random()  # jitter: 50%-150% of nominal
            delay = min(delay, RETRY_MAX_DELAY_S)
            print(f"[llm] transient {type(exc).__name__} (attempt {attempt + 1}/"
                  f"{MAX_CREATE_RETRIES + 1}), retrying in {delay:.1f}s", flush=True)
            time.sleep(delay)
            attempt += 1
