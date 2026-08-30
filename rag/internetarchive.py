"""Internet Archive connector: search archive.org's texts collection and fetch a
real, freely downloadable full-text file via the free, key-less Internet Archive API
(https://archive.org/developers/ -- Advanced Search + Metadata Read/Download APIs).

No API key, no auth, no cost -- archive.org's search, metadata, and download
endpoints are free public services with no billing relationship of any kind.

Many archive.org items are lending-library scans (borrow-only, no downloadable full
text without a login + checkout). This connector only ever returns text from items
that are NOT access-restricted and that actually expose a plain-text file -- it
raises ValueError rather than silently falling back to a borrow-gated item, so every
document that reaches the embedding pipeline is real, freely-fetched text.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

from rag.gutendex import chunk_text, strip_gutenberg_boilerplate  # re-exported for callers/tests

ARCHIVE_SEARCH_URL = "https://archive.org/advancedsearch.php"
ARCHIVE_METADATA_URL = "https://archive.org/metadata"
ARCHIVE_DOWNLOAD_URL = "https://archive.org/download"
USER_AGENT = "CADS-DEMO-distributed-rag/0.1 (bootstrap; https://github.com/scimbe/CADS-DEMO-distributed-rag)"

# Preference order when an item exposes more than one plain-text file: a clean
# Gutenberg-style "Text" transcription over noisier DjVu OCR output.
_TEXT_FORMAT_PREFERENCE = ("Text", "DjVuTXT")

__all__ = ["ArchiveItemMeta", "search_texts", "fetch_item_text", "chunk_text", "strip_gutenberg_boilerplate"]


@dataclass
class ArchiveItemMeta:
    identifier: str
    title: str
    creators: list[str]


def search_texts(query: str, limit: int = 10) -> list[ArchiveItemMeta]:
    """Search archive.org's texts collection for a free-text query, returning basic
    item metadata (identifier, title, creator(s)). Not every result is guaranteed to
    have a freely downloadable text file -- call fetch_item_text() to confirm/fetch."""
    resp = requests.get(
        ARCHIVE_SEARCH_URL,
        params={
            "q": f"({query}) AND mediatype:texts",
            "fl[]": ["identifier", "title", "creator"],
            "rows": limit,
            "output": "json",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    docs = resp.json().get("response", {}).get("docs", [])
    results: list[ArchiveItemMeta] = []
    for d in docs:
        creator = d.get("creator", [])
        if isinstance(creator, str):
            creator = [creator]
        results.append(ArchiveItemMeta(identifier=d["identifier"], title=d.get("title", ""), creators=creator or []))
    return results


def _pick_text_file(files: list[dict]) -> dict | None:
    by_format = {f.get("format"): f for f in files if f.get("format") in _TEXT_FORMAT_PREFERENCE and f.get("name")}
    for fmt in _TEXT_FORMAT_PREFERENCE:
        if fmt in by_format:
            return by_format[fmt]
    return None


def fetch_item_text(identifier: str) -> tuple[ArchiveItemMeta, str]:
    """Fetch an archive.org item's metadata and its full plain-text body, for an item
    that has one freely downloadable (no lending/borrowing gate). Raises ValueError if
    the item is access-restricted or exposes no usable plain-text file."""
    resp = requests.get(f"{ARCHIVE_METADATA_URL}/{identifier}", headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("files"):
        raise ValueError(f"item {identifier!r} not found (empty metadata)")

    md = data.get("metadata", {})
    if str(md.get("access-restricted-item", "false")).lower() == "true":
        raise ValueError(f"item {identifier!r} is access-restricted (lending only); no free full text available")

    title = md.get("title", identifier)
    creator = md.get("creator", [])
    if isinstance(creator, str):
        creator = [creator]
    meta = ArchiveItemMeta(identifier=identifier, title=title, creators=creator or [])

    text_file = _pick_text_file(data["files"])
    if text_file is None:
        raise ValueError(f"item {identifier!r} ({title!r}) has no plain-text file available")

    text_resp = requests.get(
        f"{ARCHIVE_DOWNLOAD_URL}/{identifier}/{text_file['name']}",
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    text_resp.raise_for_status()
    body = text_resp.text
    if body.lstrip().lower().startswith("<html"):
        raise ValueError(f"item {identifier!r} text download returned an HTML page, not text (likely access-restricted)")
    return meta, body
