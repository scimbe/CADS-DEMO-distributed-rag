#!/usr/bin/env python3
"""Ingest one real Project Gutenberg book (via the free Gutendex API) into the local
brute-force cosine SQLite store: fetch -> strip boilerplate -> chunk -> embed locally
-> store.

Usage:
    .venv/bin/python scripts/ingest.py <gutenberg_id> [--db data/rag.sqlite3]

Example (the book used to prove this pipeline end to end):
    .venv/bin/python scripts/ingest.py 1661
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.embedder import Embedder
from rag.gutendex import chunk_text, fetch_book_text, strip_gutenberg_boilerplate
from rag.store import Chunk, Store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gutenberg_id", type=int, help="Project Gutenberg book id, e.g. 1661")
    parser.add_argument("--db", default="data/rag.sqlite3", help="SQLite store path")
    args = parser.parse_args()

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)

    print(f"[ingest] fetching Gutenberg book {args.gutenberg_id} via Gutendex ...")
    meta, raw_text = fetch_book_text(args.gutenberg_id)
    print(f"[ingest] fetched {meta.title!r} by {', '.join(meta.authors) or 'unknown'} ({len(raw_text)} chars raw)")

    body = strip_gutenberg_boilerplate(raw_text)
    print(f"[ingest] stripped boilerplate -> {len(body)} chars body")

    chunks = chunk_text(body)
    print(f"[ingest] split into {len(chunks)} passages")

    print("[ingest] loading local embedding model (BAAI/bge-small-en-v1.5, first run downloads + caches it) ...")
    t0 = time.time()
    embedder = Embedder()
    print(f"[ingest] model ready in {time.time() - t0:.1f}s")

    print(f"[ingest] embedding {len(chunks)} passages locally ...")
    t0 = time.time()
    vectors = embedder.embed(chunks)
    print(f"[ingest] embedded {len(vectors)} passages in {time.time() - t0:.1f}s")

    source = f"gutenberg:{args.gutenberg_id}"
    store = Store(args.db)
    records = [
        Chunk(source=source, title=meta.title, passage_index=i, text=text, embedding=vec)
        for i, (text, vec) in enumerate(zip(chunks, vectors))
    ]
    store.insert_many(records)
    print(f"[ingest] stored {len(records)} passages in {args.db} (store now holds {store.count()} chunks total)")
    store.close()


if __name__ == "__main__":
    main()
