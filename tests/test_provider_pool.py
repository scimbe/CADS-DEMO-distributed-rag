"""Unit tests for rag/provider_pool.py -- the local-litellm -> Groq fallback router.

No network calls: requests.post is mocked throughout. These tests prove the
ROUTING LOGIC (when it falls back, when it doesn't, when it refuses to
silently reach an unconfigured third party) is correct. They do NOT verify
that Groq's real API actually answers as expected -- see the accompanying
report for that honest verified-vs-unverified breakdown.

Run: .venv/bin/python -m pytest tests/test_provider_pool.py -v
  (or: .venv/bin/python -m unittest tests.test_provider_pool -v)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from rag import provider_pool as pp

LOCAL_ENV = {
    "LITELLM_BASE_URL": "https://llm-test.example.org/v1",
    "LITELLM_API_KEY": "sk-local-test",
    "LITELLM_DEFAULT_MODEL": "local-devstral-small2",
}
GROQ_ENV = {
    "GROQ_API_KEY": "gsk-test",
}
NO_BACKEND_ENV: dict[str, str] = {}

MESSAGES = [{"role": "user", "content": "say hi"}]


def _resp(status_code: int, json_body: dict | None = None, text: str = "") -> MagicMock:
    m = MagicMock()
    m.status_code = status_code
    m.text = text
    if json_body is not None:
        m.json.return_value = json_body
    if status_code >= 400:
        m.raise_for_status.side_effect = requests.HTTPError(f"{status_code} error", response=m)
    else:
        m.raise_for_status.side_effect = None
    return m


def _ok_body(content: str = "hello") -> dict:
    return {"choices": [{"message": {"content": content}}]}


class ProviderPoolTest(unittest.TestCase):
    def test_local_success_never_calls_groq(self):
        with patch.dict("os.environ", {**LOCAL_ENV, **GROQ_ENV}, clear=True), patch.object(pp.requests, "post") as post:
            post.return_value = _resp(200, _ok_body("local answer"))
            result = pp.chat(MESSAGES)
            self.assertEqual(result.backend, "local-litellm")
            self.assertEqual(result.content, "local answer")
            self.assertEqual(post.call_count, 1)
            self.assertIn("llm-test.example.org", post.call_args.args[0])

    def test_local_rate_limited_falls_back_to_groq(self):
        with patch.dict("os.environ", {**LOCAL_ENV, **GROQ_ENV}, clear=True), patch.object(pp.requests, "post") as post:
            post.side_effect = [
                _resp(429, text="rate limited"),
                _resp(200, _ok_body("groq answer")),
            ]
            result = pp.chat(MESSAGES)
            self.assertEqual(result.backend, "groq")
            self.assertEqual(result.content, "groq answer")
            self.assertEqual(post.call_count, 2)
            # second call must hit Groq's endpoint, not local's
            self.assertIn("api.groq.com", post.call_args.args[0])

    def test_local_connection_error_falls_back_to_groq(self):
        with patch.dict("os.environ", {**LOCAL_ENV, **GROQ_ENV}, clear=True), patch.object(pp.requests, "post") as post:
            post.side_effect = [
                requests.ConnectionError("connection refused"),
                _resp(200, _ok_body("groq answer")),
            ]
            result = pp.chat(MESSAGES)
            self.assertEqual(result.backend, "groq")

    def test_local_5xx_falls_back_to_groq(self):
        with patch.dict("os.environ", {**LOCAL_ENV, **GROQ_ENV}, clear=True), patch.object(pp.requests, "post") as post:
            post.side_effect = [_resp(503, text="down for maintenance"), _resp(200, _ok_body("groq answer"))]
            result = pp.chat(MESSAGES)
            self.assertEqual(result.backend, "groq")

    def test_local_failure_without_groq_configured_raises_clear_error(self):
        with patch.dict("os.environ", LOCAL_ENV, clear=True), patch.object(pp.requests, "post") as post:
            post.return_value = _resp(429, text="rate limited")
            with self.assertRaises(pp.ProviderPoolError) as ctx:
                pp.chat(MESSAGES)
            self.assertIn("GROQ_API_KEY", str(ctx.exception))
            # never silently reached out to an unconfigured third party
            self.assertEqual(post.call_count, 1)

    def test_non_retryable_local_failure_never_falls_back(self):
        """A 401 (bad key) is a real config error, not a capacity problem -- it must
        surface immediately, not be masked by quietly routing around it to Groq."""
        with patch.dict("os.environ", {**LOCAL_ENV, **GROQ_ENV}, clear=True), patch.object(pp.requests, "post") as post:
            post.return_value = _resp(401, text="unauthorized")
            with self.assertRaises(requests.HTTPError):
                pp.chat(MESSAGES)
            self.assertEqual(post.call_count, 1)

    def test_no_backend_configured_raises_before_any_network_call(self):
        with patch.dict("os.environ", NO_BACKEND_ENV, clear=True), patch.object(pp.requests, "post") as post:
            with self.assertRaises(pp.ProviderPoolError):
                pp.chat(MESSAGES)
            post.assert_not_called()

    def test_local_not_configured_goes_straight_to_groq(self):
        with patch.dict("os.environ", GROQ_ENV, clear=True), patch.object(pp.requests, "post") as post:
            post.return_value = _resp(200, _ok_body("groq only"))
            result = pp.chat(MESSAGES)
            self.assertEqual(result.backend, "groq")
            self.assertEqual(post.call_count, 1)

    def test_model_override_is_per_backend_not_cross_provider(self):
        """local_model must never leak into the Groq call, and vice versa -- model
        names aren't portable between providers."""
        with patch.dict("os.environ", {**LOCAL_ENV, **GROQ_ENV}, clear=True), patch.object(pp.requests, "post") as post:
            post.side_effect = [_resp(429, text="rate limited"), _resp(200, _ok_body("groq answer"))]
            result = pp.chat(MESSAGES, local_model="local-devstral-small2", groq_model="llama-3.3-70b-versatile")
            self.assertEqual(result.model, "llama-3.3-70b-versatile")
            first_call_model = post.call_args_list[0].kwargs["json"]["model"]
            second_call_model = post.call_args_list[1].kwargs["json"]["model"]
            self.assertEqual(first_call_model, "local-devstral-small2")
            self.assertEqual(second_call_model, "llama-3.3-70b-versatile")

    def test_configured_backends_reports_status(self):
        with patch.dict("os.environ", LOCAL_ENV, clear=True):
            self.assertEqual(pp.configured_backends(), {"local-litellm": True, "groq": False})
        with patch.dict("os.environ", NO_BACKEND_ENV, clear=True):
            self.assertEqual(pp.configured_backends(), {"local-litellm": False, "groq": False})


if __name__ == "__main__":
    unittest.main()
