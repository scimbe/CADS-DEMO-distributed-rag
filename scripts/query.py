#!/usr/bin/env python3
"""Embed a query locally and retrieve the most similar passages from the local store.

Usage:
    .venv/bin/python scripts/query.py "your question" [--db data/rag.sqlite3] [--top-k 3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.embedder import Embedder
from rag.store import Store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="natural-language query")
    parser.add_argument("--db", default="data/rag.sqlite3", help="SQLite store path")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    store = Store(args.db)
    if store.count() == 0:
        print(f"[query] store {args.db} is empty -- run scripts/ingest.py first", file=sys.stderr)
        sys.exit(1)

    embedder = Embedder()
    query_vec = embedder.embed_one(args.query)

    results = store.search(query_vec, top_k=args.top_k)
    print(f"[query] top {len(results)} of {store.count()} stored passages for: {args.query!r}\n")
    for rank, r in enumerate(results, start=1):
        print(f"--- #{rank}  similarity={r.similarity:.4f}  source={r.source} ({r.title})  passage={r.passage_index} ---")
        print(r.text.strip())
        print()
    store.close()


if __name__ == "__main__":
    main()
