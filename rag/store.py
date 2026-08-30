"""SQLite storage for embedded book passages + brute-force cosine-similarity search
over their vector column.

Mirrors the storage pattern already established in this project's
`crates/harness-memory/src/db.rs` (CADS-agent-marketplace): a plain SQLite BLOB column
holding a packed float vector, and brute-force cosine similarity over every row rather
than a dedicated vector-DB dependency (Qdrant/ChromaDB/pgvector). That crate's own
comment applies here too: at this scale (one ingested book == a few hundred to a few
thousand chunks) brute-force cosine is sub-millisecond to low-milliseconds -- this is a
demo corpus, not a web-scale product, so a vector-DB dependency would be unjustified
complexity. The distributed-RAG issue's own "Memory-layer" design (ChromaDB/LanceDB for
cross-session interaction memory, separate from this document corpus) is explicitly NOT
built yet -- see README.
"""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Chunk:
    source: str  # e.g. "gutenberg:1661"
    title: str
    passage_index: int
    text: str
    embedding: np.ndarray


@dataclass
class SearchResult:
    id: int
    source: str
    title: str
    passage_index: int
    text: str
    similarity: float
    created_at: int = field(repr=False, default=0)


def _embedding_to_blob(v: np.ndarray) -> bytes:
    v = np.asarray(v, dtype=np.float32)
    return struct.pack(f"<{len(v)}f", *v.tolist())


def _blob_to_embedding(b: bytes) -> np.ndarray:
    n = len(b) // 4
    return np.array(struct.unpack(f"<{n}f", b), dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class Store:
    def __init__(self, path: str) -> None:
        # check_same_thread=False + an explicit lock: app.py's FastAPI service holds
        # one Store singleton for the whole process, but FastAPI runs sync endpoint
        # functions in a threadpool -- a different worker thread per request, not
        # necessarily the thread that constructed this connection. sqlite3 forbids
        # cross-thread use of one connection by default; the lock then serializes
        # actual access, since a single sqlite3.Connection still isn't safe for truly
        # concurrent use across threads even with that flag. CLI callers
        # (scripts/ingest.py, scripts/query.py) are single-threaded, so this is a
        # no-op for them.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                passage_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at INTEGER NOT NULL
            )"""
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def insert_chunk(self, c: Chunk, now: int | None = None) -> int:
        if c.embedding.size == 0:
            raise ValueError("embedding must not be empty")
        now = now if now is not None else int(time.time())
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO chunks (source, title, passage_index, text, embedding, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (c.source, c.title, c.passage_index, c.text, _embedding_to_blob(c.embedding), now),
            )
            self._conn.commit()
            return cur.lastrowid

    def insert_many(self, chunks: list[Chunk]) -> int:
        now = int(time.time())
        for c in chunks:
            self.insert_chunk(c, now=now)
        return len(chunks)

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        """Brute-force cosine search over every stored chunk. Rows whose embedding
        dimensionality doesn't match the query are skipped rather than erroring, same
        discipline as harness-memory's db.rs -- lets the embedding model change over
        time without corrupting old rows or crashing a mixed-dimension search."""
        with self._lock:
            rows = self._conn.execute("SELECT id, source, title, passage_index, text, embedding, created_at FROM chunks").fetchall()
        scored: list[SearchResult] = []
        for row_id, source, title, passage_index, text, blob, created_at in rows:
            emb = _blob_to_embedding(blob)
            if emb.shape != query_embedding.shape:
                continue
            sim = _cosine_similarity(query_embedding, emb)
            scored.append(SearchResult(row_id, source, title, passage_index, text, sim, created_at))
        scored.sort(key=lambda r: r.similarity, reverse=True)
        return scored[:top_k]
