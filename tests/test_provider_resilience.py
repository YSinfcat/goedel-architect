"""Regression tests for the provider-resilience layer (review P1-2 partial
and IV.6/IV.9/IV.10):

  - create_with_retry: bounded backoff on 429/5xx/connection errors
    (Retry-After honored), immediate propagation of non-transient errors,
    exhaustion after a fixed attempt count.
  - make_client: per-(provider, timeout) client caching - one shared
    connection pool instead of one per node (IV.10).
  - provider_for: explicit routing registry instead of scattered prefix
    checks (IV.9).
  - TraceEvent.iteration: refinement-round attribution (IV.6).

Pure Python - no network (retry sleeps are monkeypatched, provider
clients are never called).
"""
from __future__ import annotations

import os
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("GOEDEL_LEAN_TOOLCHAIN", "test-toolchain")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("FIREWORKS_API_KEY", "test-key")
os.environ.setdefault("MISTRAL_API_KEY", "test-key")

import llm_client  # noqa: E402
from llm_client import create_with_retry, provider_for  # noqa: E402
from tracer import TraceEvent  # noqa: E402


class _HTTPError(Exception):
    def __init__(self, status_code, headers=None):
        super().__init__(f"status {status_code}")
        self.status_code = status_code
        if headers is not None:

            class _Resp:
                pass

            resp = _Resp()
            resp.headers = headers
            self.response = resp


class _StubClient:
    def __init__(self, outcomes):
        # outcomes: list of exceptions to raise, then a final return value
        self._outcomes = list(outcomes)
        self.calls = 0

        class _Completions:
            def create(_self, **kwargs):
                self.calls += 1
                if self._outcomes:
                    outcome = self._outcomes.pop(0)
                    if isinstance(outcome, Exception):
                        raise outcome
                    return outcome
                return "ok"

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


class RetrySleeper:
    """Replaces time.sleep inside llm_client, recording delays."""

    def __init__(self):
        self.delays = []

    def __call__(self, seconds):
        self.delays.append(seconds)


class TestCreateWithRetry(unittest.TestCase):
    def setUp(self):
        self.sleeper = RetrySleeper()
        self._orig_sleep = llm_client.time.sleep
        llm_client.time.sleep = self.sleeper

    def tearDown(self):
        llm_client.time.sleep = self._orig_sleep

    def test_transient_429_retried_then_succeeds(self):
        client = _StubClient([_HTTPError(429), _HTTPError(429)])
        result = create_with_retry(client, model="m", messages=[])
        self.assertEqual(result, "ok")
        self.assertEqual(client.calls, 3)

    def test_5xx_is_transient(self):
        client = _StubClient([_HTTPError(503)])
        self.assertEqual(create_with_retry(client, model="m", messages=[]), "ok")
        self.assertEqual(client.calls, 2)

    def test_connection_error_is_transient_by_name(self):
        exc = type("APIConnectionError", (Exception,), {})("boom")
        client = _StubClient([exc])
        self.assertEqual(create_with_retry(client, model="m", messages=[]), "ok")

    def test_non_transient_raises_immediately(self):
        client = _StubClient([_HTTPError(400)])
        with self.assertRaises(_HTTPError):
            create_with_retry(client, model="m", messages=[])
        self.assertEqual(client.calls, 1)
        self.assertEqual(self.sleeper.delays, [])

    def test_exhaustion_raises_after_bounded_attempts(self):
        client = _StubClient([_HTTPError(429)] * 100)
        with self.assertRaises(_HTTPError):
            create_with_retry(client, model="m", messages=[])
        self.assertEqual(client.calls, llm_client.MAX_CREATE_RETRIES + 1)

    def test_retry_after_header_honored_and_capped(self):
        client = _StubClient([_HTTPError(429, headers={"retry-after": "120"})])
        client2 = _StubClient([_HTTPError(429, headers={"retry-after": "2"})])
        create_with_retry(client, model="m", messages=[])
        create_with_retry(client2, model="m", messages=[])
        self.assertEqual(self.sleeper.delays[0], llm_client.RETRY_MAX_DELAY_S)
        self.assertAlmostEqual(self.sleeper.delays[1], 2.0)

    def test_backoff_is_increasing(self):
        client = _StubClient([_HTTPError(500)] * 3)
        create_with_retry(client, model="m", messages=[])
        self.assertEqual(len(self.sleeper.delays), 3)
        self.assertLess(self.sleeper.delays[0], self.sleeper.delays[1])
        self.assertLess(self.sleeper.delays[1], self.sleeper.delays[2])


class TestClientCache(unittest.TestCase):
    def setUp(self):
        llm_client._CLIENT_CACHE.clear()

    def test_same_model_and_timeout_share_a_client(self):
        a = llm_client.make_client("some-model", timeout=30.0)
        b = llm_client.make_client("some-model", timeout=30.0)
        self.assertIs(a, b)

    def test_different_providers_get_distinct_clients(self):
        a = llm_client.make_client("labs-leanstral-1-5")
        b = llm_client.make_client("accounts/fireworks/models/x")
        c = llm_client.make_client("gpt-5.5")
        self.assertIsNot(a, b)
        self.assertIsNot(b, c)
        self.assertIsNot(a, c)


class TestCustomEndpointRoute(unittest.TestCase):
    def setUp(self):
        llm_client._CLIENT_CACHE.clear()
        self._saved = {k: os.environ.pop(k, None)
                       for k in ("GOEDEL_BASE_URL", "GOEDEL_API_KEY")}

    def tearDown(self):
        llm_client._CLIENT_CACHE.clear()
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v

    def test_custom_base_url_routes_everything(self):
        os.environ["GOEDEL_BASE_URL"] = "https://api.deepseek.com/v1"
        os.environ["GOEDEL_API_KEY"] = "sk-test"
        client = llm_client.make_client("deepseek-chat")
        # the SDK normalizes a trailing slash onto the base URL
        self.assertTrue(str(client.base_url).startswith("https://api.deepseek.com/v1"))

    def test_without_custom_env_default_openai(self):
        client = llm_client.make_client("gpt-5.5")
        self.assertNotIn("deepseek", str(client.base_url))


class TestProviderRegistry(unittest.TestCase):
    def test_prefix_routing(self):
        self.assertEqual(provider_for("accounts/fireworks/models/dsv4")[0], "fireworks")
        self.assertEqual(provider_for("labs-leanstral-1-5")[0], "mistral-labs")
        self.assertEqual(provider_for("gpt-5.5")[0], "openai")
        _, mistral_cfg = provider_for("labs-leanstral-1-5")
        self.assertTrue(mistral_cfg["chat_only"])
        _, fireworks_cfg = provider_for("accounts/x")
        self.assertEqual(fireworks_cfg["api_key_env"], "FIREWORKS_API_KEY")


class TestTraceIteration(unittest.TestCase):
    def test_iteration_field_round_trips(self):
        event = TraceEvent(kind="final_verify", thm_name="t", ok=True, iteration=3)
        data = asdict(event)
        self.assertEqual(data["iteration"], 3)
        default = TraceEvent(kind="model_text", thm_name="t")
        self.assertIsNone(default.iteration)


if __name__ == "__main__":
    unittest.main(verbosity=2)
