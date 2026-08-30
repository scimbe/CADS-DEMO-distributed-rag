#!/usr/bin/env python3
"""Ingest one real document into the local brute-force cosine SQLite store, from any
of three free, key-less connectors: fetch -> (strip boilerplate) -> chunk -> embed
locally -> store.

Usage:
    .venv/bin/python scripts/ingest.py gutenberg <gutenberg_id> [--db data/rag.sqlite3]
    .venv/bin/python scripts/ingest.py openlibrary <olid> [--db data/rag.sqlite3]
    .venv/bin/python scripts/ingest.py archive <identifier> [--db data/rag.sqlite3]

Examples (real ids/identifiers proven to work end to end -- see README):
    .venv/bin/python scripts/ingest.py gutenberg 1661
    .venv/bin/python scripts/ingest.py openlibrary OL66554W
    .venv/bin/python scripts/ingest.py archive prideandprejudic42671gut
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.embedder import Embedder
from rag.gutendex import chunk_text, fetch_book_text, strip_gutenberg_boilerplate
from rag.internetarchive import fetch_item_text
from rag.openlibrary import fetch_work_document
from rag.store import Chunk, Store


def _fetch_gutenberg(gutenberg_id: str) -> tuple[str, str, list[str], str]:
    """Returns (source, title, authors, chunkable_text)."""
    meta, raw_text = fetch_book_text(int(gutenberg_id))
    print(f"[ingest] fetched {meta.title!r} by {', '.join(meta.authors) or 'unknown'} ({len(raw_text)} chars raw)")
    body = strip_gutenberg_boilerplate(raw_text)
    print(f"[ingest] stripped Gutenberg boilerplate -> {len(body)} chars body")
    return f"gutenberg:{gutenberg_id}", meta.title, meta.authors, body


def _fetch_openlibrary(olid: str) -> tuple[str, str, list[str], str]:
    meta, document = fetch_work_document(olid)
    print(f"[ingest] fetched Open Library work {olid!r}: {meta.title!r} by {', '.join(meta.authors) or 'unknown'} ({len(document)} chars descriptive text)")
    return f"openlibrary:{olid}", meta.title, meta.authors, document


def _fetch_archive(identifier: str) -> tuple[str, str, list[str], str]:
    meta, raw_text = fetch_item_text(identifier)
    print(f"[ingest] fetched archive.org item {identifier!r}: {meta.title!r} by {', '.join(meta.creators) or 'unknown'} ({len(raw_text)} chars raw)")
    # Many archive.org texts (e.g. its own Gutenberg mirror) still carry the standard
    # Project Gutenberg header/footer; this is a safe no-op when those markers aren't
    # present (see rag/gutendex.py:strip_gutenberg_boilerplate).
    body = strip_gutenberg_boilerplate(raw_text)
    if len(body) != len(raw_text):
        print(f"[ingest] stripped Gutenberg boilerplate found in archive.org text -> {len(body)} chars body")
    return f"archive:{identifier}", meta.title, meta.creators, body


_FETCHERS = {
    "gutenberg": _fetch_gutenberg,
    "openlibrary": _fetch_openlibrary,
    "archive": _fetch_archive,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", choices=sorted(_FETCHERS), help="which connector to fetch from")
    parser.add_argument("id", help="gutenberg: numeric Gutenberg id. openlibrary: work olid (e.g. OL66554W). archive: item identifier (e.g. prideandprejudic42671gut)")
    parser.add_argument("--db", default="data/rag.sqlite3", help="SQLite store path")
    args = parser.parse_args()

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)

    print(f"[ingest] fetching via {args.source} connector: {args.id!r} ...")
    source, title, authors, body = _FETCHERS[args.source](args.id)

    chunks = chunk_text(body)
    print(f"[ingest] split into {len(chunks)} passages")
    if not chunks:
        print("[ingest] no passages produced from fetched text -- nothing to store", file=sys.stderr)
        sys.exit(1)

    print("[ingest] loading local embedding model (BAAI/bge-small-en-v1.5, first run downloads + caches it) ...")
    t0 = time.time()
    embedder = Embedder()
    print(f"[ingest] model ready in {time.time() - t0:.1f}s")

    print(f"[ingest] embedding {len(chunks)} passages locally ...")
    t0 = time.time()
    vectors = embedder.embed(chunks)
    print(f"[ingest] embedded {len(vectors)} passages in {time.time() - t0:.1f}s")

    store = Store(args.db)
    records = [
        Chunk(source=source, title=title, passage_index=i, text=text, embedding=vec)
        for i, (text, vec) in enumerate(zip(chunks, vectors))
    ]
    store.insert_many(records)
    print(f"[ingest] stored {len(records)} passages in {args.db} (store now holds {store.count()} chunks total)")
    store.close()


if __name__ == "__main__":
    main()
