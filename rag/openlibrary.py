"""Open Library connector: search Open Library's catalog and fetch a work's real
bibliographic/descriptive text via the free, key-less Open Library API
(https://openlibrary.org/developers/api).

No API key, no auth, no cost -- openlibrary.org's search and works endpoints are free
public services with no billing relationship of any kind.

Open Library itself does not serve full book body text (that lives on scanned copies
over on Internet Archive -- see rag/internetarchive.py). What it does have, and what
this connector pulls, is real bibliographic/descriptive text straight from the API:
synopsis/description, a first-sentence excerpt (when catalogued), and subject tags.
That's genuine live data, not invented -- exactly the grounding text a RAG system
needs to answer "what is this book" / "what is it about" style questions, distinct
from (and complementary to) the full-text passages the Gutenberg and Internet Archive
connectors provide.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

OPENLIBRARY_BASE = "https://openlibrary.org"
USER_AGENT = "CADS-DEMO-distributed-rag/0.1 (bootstrap; https://github.com/scimbe/CADS-DEMO-distributed-rag)"


@dataclass
class WorkMeta:
    olid: str  # e.g. "OL66554W"
    title: str
    authors: list[str]


def search_works(query: str, limit: int = 5) -> list[WorkMeta]:
    """Search Open Library's catalog for a free-text query, returning basic work
    metadata (Open Library work id, title, author names) for candidate results."""
    resp = requests.get(
        f"{OPENLIBRARY_BASE}/search.json",
        params={"q": query, "limit": limit, "fields": "key,title,author_name"},
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    docs = resp.json().get("docs", [])
    results: list[WorkMeta] = []
    for d in docs:
        key = d.get("key", "")  # e.g. "/works/OL66554W"
        olid = key.rsplit("/", 1)[-1] if key else ""
        if not olid:
            continue
        results.append(WorkMeta(olid=olid, title=d.get("title", ""), authors=d.get("author_name") or []))
    return results


def _description_text(description: object) -> str:
    """Open Library's `description` field is inconsistently either a plain string or
    a dict of {'type': ..., 'value': ...} depending on how the record was edited."""
    if isinstance(description, dict):
        return str(description.get("value") or "")
    if isinstance(description, str):
        return description
    return ""


def _resolve_author_name(author_key: str) -> str:
    """Resolve an author key (e.g. '/authors/OL21594A') to its real display name."""
    resp = requests.get(f"{OPENLIBRARY_BASE}{author_key}.json", headers={"User-Agent": USER_AGENT}, timeout=15)
    resp.raise_for_status()
    return resp.json().get("name", author_key)


def fetch_work_document(olid: str) -> tuple[WorkMeta, str]:
    """Fetch an Open Library work's metadata and assemble its real descriptive text
    (description, first-sentence excerpt, subjects, named characters) into one
    document suitable for chunking + embedding. Raises ValueError if the work has
    neither a description nor an excerpt -- nothing worth embedding."""
    resp = requests.get(f"{OPENLIBRARY_BASE}/works/{olid}.json", headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    title = data.get("title", olid)
    authors: list[str] = []
    for a in data.get("authors", []):
        author_key = a.get("author", {}).get("key", "")
        if author_key:
            authors.append(_resolve_author_name(author_key))

    description = _description_text(data.get("description"))
    excerpts = [e.get("excerpt", "") for e in data.get("excerpts", []) or [] if e.get("excerpt")]
    subjects = data.get("subjects", []) or []
    subject_people = data.get("subject_people", []) or []

    if not description and not excerpts:
        raise ValueError(f"work {olid!r} ({title!r}) has no description or excerpt text available")

    parts = [f"Title: {title}"]
    if authors:
        parts.append("Author(s): " + ", ".join(authors))
    if description:
        parts.append(description)
    if excerpts:
        parts.append("Opening line(s): " + " ".join(excerpts))
    if subject_people:
        parts.append("Characters: " + ", ".join(subject_people))
    if subjects:
        parts.append("Subjects: " + ", ".join(subjects[:40]))  # some works carry hundreds of subject tags

    meta = WorkMeta(olid=olid, title=title, authors=authors)
    return meta, "\n\n".join(parts)
