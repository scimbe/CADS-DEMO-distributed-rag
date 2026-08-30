#!/usr/bin/env python3
"""Live-verify the provider_pool router against whatever backends are actually
configured in .env. Prints which backends are configured, then sends one small
chat request and reports which backend answered.

Does NOT force a fallback -- it just reports the truth of what's reachable
right now. To specifically prove the Groq fallback path, set GROQ_API_KEY and
point LITELLM_BASE_URL at something unreachable (or unset it) before running.

Usage:
    .venv/bin/python scripts/verify_provider_pool.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.provider_pool import ProviderPoolError, chat, configured_backends


def main() -> None:
    status = configured_backends()
    print(f"[verify] configured backends: {status}")
    if not any(status.values()):
        print("[verify] nothing configured -- copy .env.example to .env and fill in at least one backend", file=sys.stderr)
        sys.exit(1)

    try:
        result = chat([{"role": "user", "content": "Reply with exactly one word: ok"}], max_tokens=10)
    except ProviderPoolError as exc:
        print(f"[verify] FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[verify] answered by backend={result.backend!r} model={result.model!r}")
    print(f"[verify] content: {result.content!r}")


if __name__ == "__main__":
    main()
