"""Gutendex connector: fetch a real public-domain book's plain text from Project
Gutenberg via the free, key-less Gutendex API (https://gutendex.com), and chunk it
into passages for embedding.

No API key, no auth, no cost -- Gutendex and gutenberg.org are both free public
services with no billing relationship of any kind.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import requests

GUTENDEX_BASE = "https://gutendex.com"
USER_AGENT = "CADS-DEMO-distributed-rag/0.1 (bootstrap; https://github.com/scimbe/CADS-DEMO-distributed-rag)"


@dataclass
class BookMeta:
    gutenberg_id: int
    title: str
    authors: list[str]


def fetch_book_meta(gutenberg_id: int) -> tuple[BookMeta, dict[str, str]]:
    """Look up a book's metadata (title, authors) and its available format URLs by its
    Project Gutenberg id."""
    resp = requests.get(f"{GUTENDEX_BASE}/books/{gutenberg_id}/", headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    meta = BookMeta(
        gutenberg_id=data["id"],
        title=data["title"],
        authors=[a["name"] for a in data.get("authors", [])],
    )
    return meta, data["formats"]


def fetch_book_text(gutenberg_id: int) -> tuple[BookMeta, str]:
    """Fetch a book's metadata and full plain-text body from Gutendex + gutenberg.org."""
    meta, formats = fetch_book_meta(gutenberg_id)
    text_url = formats.get("text/plain; charset=utf-8") or formats.get("text/plain; charset=us-ascii") or formats.get("text/plain")
    if not text_url:
        raise ValueError(f"book {gutenberg_id} ({meta.title!r}) has no plain-text format available: {list(formats)}")
    resp = requests.get(text_url, headers={"User-Agent": USER_AGENT}, timeout=60)
    resp.raise_for_status()
    return meta, resp.text


_GUTENBERG_START_RE = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
_GUTENBERG_END_RE = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.IGNORECASE)


def strip_gutenberg_boilerplate(full_text: str) -> str:
    """Drop the standard Project Gutenberg license header/footer, keeping the book body."""
    start_match = _GUTENBERG_START_RE.search(full_text)
    body = full_text[start_match.end():] if start_match else full_text
    end_match = _GUTENBERG_END_RE.search(body)
    if end_match:
        body = body[: end_match.start()]
    return body.strip()


def chunk_text(text: str, target_chars: int = 800, overlap_chars: int = 120) -> list[str]:
    """Split text into overlapping passage-sized chunks on paragraph boundaries where
    possible, falling back to a hard character cut for any single paragraph that's
    still too long. Simple and inspectable -- no external chunking library needed at
    this scale.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= target_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(para) <= target_chars:
            current = para
        else:
            # a single paragraph longer than target_chars: hard-cut with overlap
            start = 0
            while start < len(para):
                end = start + target_chars
                chunks.append(para[start:end])
                start = end - overlap_chars
            current = ""
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) > 40]  # drop near-empty scraps (stray headers etc.)
