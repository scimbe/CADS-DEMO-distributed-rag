"""Dynamic free-LLM-provider fallback pool.

Tries the team's local litellm-proxy first (the primary LLM path every other
CADS-DEMO-* repo already uses -- LITELLM_BASE_URL / LITELLM_API_KEY /
LITELLM_DEFAULT_MODEL, OpenAI-compatible /chat/completions). On a
rate-limit/connection failure it falls through to ONE external free provider
-- Groq -- using the exact same call shape (Groq's API is itself
OpenAI-compatible: same endpoint suffix, same Bearer auth, same request/
response JSON), so callers never need to know which backend actually
answered.

The external fallback is opt-in and never a hard dependency: if GROQ_API_KEY
isn't set, this module simply doesn't fall back -- a local outage surfaces as
a normal error instead of silently reaching out to a third party. This
matches the issue's "really free, no costs under any circumstances" +
"fallback, never a hard dependency" requirement (marketplace#33).

Why Groq: of the candidates in https://github.com/ShaikhWarsi/free-ai-tools
("Fully Free Providers", re-checked 2026-08-30 -- OpenRouter, Groq, Cerebras,
Cloudflare Workers AI), Groq has the most generous free daily quota for a
single always-on fallback (14,400 req/day on llama-3.1-8b-instant, vs.
OpenRouter's 50 req/day shared across all free models), genuinely requires
no credit card for free-tier signup (confirmed against multiple independent
2026 sources -- see the accompanying report; Groq's own console/docs pages
render client-side and returned no usable content via automated fetch), and
its API is a drop-in OpenAI-compatible endpoint
(https://api.groq.com/openai/v1/chat/completions) -- no response-shape
translation needed to slot in behind the local litellm-proxy.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent

# HTTP statuses that count as "the backend is rate-limited or temporarily
# unavailable" -- these are the ONLY failures that trigger a fallback to the
# external provider. Anything else (401 bad key, 400 bad request, ...) is a
# real configuration/programming error and is raised straight to the caller
# instead of being silently masked by quietly routing around it.
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_DEFAULT_TIMEOUT = 60.0

GROQ_BASE_URL_DEFAULT = "https://api.groq.com/openai/v1"
GROQ_MODEL_DEFAULT = "llama-3.1-8b-instant"
LOCAL_MODEL_DEFAULT = "local-devstral-small2"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no python-dotenv dependency) -- same pattern used by
    CADS-DEMO-contractcheck's src/summarize.py. Only fills in vars not already
    set in the real environment."""
    import os

    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(REPO_ROOT / ".env")


@dataclass(frozen=True)
class Backend:
    name: str
    base_url: str
    api_key: str
    default_model: str


def _local_backend() -> Backend | None:
    import os

    base_url = os.environ.get("LITELLM_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("LITELLM_API_KEY", "")
    if not base_url or not api_key:
        return None
    model = os.environ.get("LITELLM_DEFAULT_MODEL", "").strip() or LOCAL_MODEL_DEFAULT
    return Backend(name="local-litellm", base_url=base_url, api_key=api_key, default_model=model)


def _groq_backend() -> Backend | None:
    import os

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return None
    base_url = os.environ.get("GROQ_BASE_URL", "").rstrip("/") or GROQ_BASE_URL_DEFAULT
    model = os.environ.get("GROQ_MODEL", "").strip() or GROQ_MODEL_DEFAULT
    return Backend(name="groq", base_url=base_url, api_key=api_key, default_model=model)


def configured_backends() -> dict[str, bool]:
    """Which backends are currently usable, for status reporting/CLI output."""
    return {"local-litellm": _local_backend() is not None, "groq": _groq_backend() is not None}


class ProviderPoolError(RuntimeError):
    """No backend could answer the request (misconfiguration, or local failed
    and no fallback is configured)."""


class _RetryableBackendError(RuntimeError):
    """Internal signal: a backend call failed in a way that justifies falling
    through to the next backend (rate-limited / temporarily unavailable)."""


@dataclass
class ChatResult:
    content: str
    backend: str  # "local-litellm" or "groq" -- which backend actually answered
    model: str
    raw: dict


def _post_chat(backend: Backend, messages: list[dict], *, model: str | None, temperature: float, max_tokens: int, timeout: float) -> dict:
    resolved_model = model or backend.default_model
    try:
        resp = requests.post(
            f"{backend.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {backend.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": resolved_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise _RetryableBackendError(f"{backend.name} unreachable: {exc}") from exc

    if resp.status_code in _RETRYABLE_STATUS:
        raise _RetryableBackendError(f"{backend.name} returned HTTP {resp.status_code} (rate-limited/unavailable): {resp.text[:300]}")
    resp.raise_for_status()
    return resp.json()


def _extract_content(backend: Backend, data: dict) -> str:
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError) as exc:
        raise ProviderPoolError(f"{backend.name} returned an unexpected response shape: {data!r}") from exc


def chat(
    messages: list[dict],
    *,
    local_model: str | None = None,
    groq_model: str | None = None,
    temperature: float = 0,
    max_tokens: int = 800,
    timeout: float = _DEFAULT_TIMEOUT,
) -> ChatResult:
    """Chat-complete `messages` (standard OpenAI [{"role", "content"}, ...] shape).

    Tries the local litellm-proxy first. On a connection failure or a
    rate-limit/5xx response, falls through to Groq if GROQ_API_KEY is set --
    otherwise the local failure is raised as-is (never a silent, unconfigured
    reach-out to a third party). If the local backend isn't configured at
    all, goes straight to Groq (if configured).

    `local_model` / `groq_model` optionally override each backend's own
    default model (LITELLM_DEFAULT_MODEL / GROQ_MODEL respectively) for this
    call. There is deliberately no single cross-backend `model=` override --
    model names aren't portable between providers, so passing one backend's
    model string to the other would silently break instead of falling back
    correctly.

    Returns a ChatResult naming which backend actually answered -- callers
    that don't care can just read `.content`.
    """
    local = _local_backend()
    groq = _groq_backend()

    if local is None and groq is None:
        raise ProviderPoolError(
            "No LLM backend configured. Set LITELLM_BASE_URL + LITELLM_API_KEY (the local "
            "litellm-proxy) and/or GROQ_API_KEY (free external fallback) -- see .env.example."
        )

    if local is not None:
        try:
            data = _post_chat(local, messages, model=local_model, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
            return ChatResult(content=_extract_content(local, data), backend=local.name, model=local_model or local.default_model, raw=data)
        except _RetryableBackendError as exc:
            if groq is None:
                raise ProviderPoolError(
                    f"local backend ({local.base_url}) is rate-limited or unreachable ({exc}), and no "
                    "GROQ_API_KEY fallback is configured. Set GROQ_API_KEY to enable the free "
                    "external fallback -- this is never a hard dependency."
                ) from exc
            print(f"[provider_pool] local backend failed ({exc}) -- falling back to {groq.name}", file=sys.stderr)

    # Either local wasn't configured, or it just failed retryably above.
    assert groq is not None  # guaranteed by the checks above
    data = _post_chat(groq, messages, model=groq_model, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    return ChatResult(content=_extract_content(groq, data), backend=groq.name, model=groq_model or groq.default_model, raw=data)
