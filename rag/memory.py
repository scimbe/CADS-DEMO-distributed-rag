"""Long-term interaction/fact memory: past query/answer pairs retrieved by top-K
semantic similarity instead of a linear-growing context window.

Distinct from `rag/store.py`, which holds the *document corpus* (book passages
ingested from Gutendex). This module holds the *memory layer* the issue asks for
separately: what was asked and answered before, so a later turn can `recall()` a
handful of relevant prior interactions instead of replaying the whole conversation
history into the context window.

## Library choice: LanceDB, not ChromaDB

The issue names either ChromaDB or LanceDB as acceptable (both free/self-hosted-
forever, no paid tier required for this use case). Both were license-checked directly
against their current source on 2026-08-30, not assumed from memory:

- **LanceDB** ([PyPI](https://pypi.org/project/lancedb/), current version 0.37.1):
  **Apache License, Version 2.0** -- confirmed both on the PyPI project page and by
  fetching `LICENSE` directly from
  https://github.com/lancedb/lancedb/blob/main/LICENSE ("Apache License, Version 2.0,
  January 2004"). Its embedded/local mode (`lancedb.connect(<local path>)`) writes
  plain Lance/Arrow files to disk and needs no separate server process, no account, no
  API key.
- **ChromaDB** ([PyPI](https://pypi.org/project/chromadb/), current version 1.5.9):
  also genuinely **Apache 2.0** for the core package and its local/self-hosted modes.
  "Chroma Cloud" is an optional *paid* hosted tier layered on top, but is never
  required -- the in-memory and self-hosted (`chroma run`) modes are free. Not
  disqualifying on license/cost grounds.

**LanceDB was picked** because its embedded mode is the closer match to this repo's
existing footprint: `rag/store.py` is a single local SQLite file with no server
process, and `lancedb.connect(path)` gives the same shape -- a local directory of
files, no daemon to start or manage. ChromaDB's local story more commonly steers
towards either a heavier embedded client (with its own SQLite+HNSW machinery bundled
in) or an explicit `chroma run --path ...` server process to talk to over HTTP; LanceDB
needed neither. LanceDB also already depends only on `pyarrow`/`numpy`/`pydantic`-style
libraries (all themselves permissively licensed, no native compilation surprises
beyond what `fastembed`'s `onnxruntime` already requires), so the net new dependency
footprint stays small.

No network call, no API key, no billing relationship of any kind -- this is 100% local
file storage, embedded in-process, same "really free, no costs under any
circumstances" guarantee as the rest of this repo.

## What gets embedded

Each remembered interaction is embedded on its *query* text (the same local
`Embedder` from `rag/embedder.py`, BAAI/bge-small-en-v1.5 via fastembed) -- mirroring
`rag/store.py`, which embeds the text you'd actually search against (there: passage
text; here: the question that was asked). `recall()` embeds a new query the same way
and does a cosine-similarity top-K search over past queries, returning each match's
full stored (query, answer) pair.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import lancedb
import numpy as np

TABLE_NAME = "interactions"


@dataclass
class Interaction:
    """One past query/answer pair to remember."""

    query: str
    answer: str
    created_at: int = 0


@dataclass
class RecalledInteraction:
    id: int
    query: str
    answer: str
    similarity: float
    created_at: int = field(repr=False, default=0)


class MemoryStore:
    """Long-term interaction memory: `remember()` a query/answer pair, `recall()` the
    top-K most semantically similar past interactions for a new query.

    Backed by an embedded LanceDB table (local Lance/Arrow files under `path`, no
    server process) -- the memory-layer counterpart to `rag/store.py`'s SQLite
    document-corpus store.
    """

    def __init__(self, path: str, embedder) -> None:
        self._embedder = embedder
        self._db = lancedb.connect(path)

    def _table_exists(self) -> bool:
        return TABLE_NAME in self._db.list_tables().tables

    def _open_or_none(self):
        return self._db.open_table(TABLE_NAME) if self._table_exists() else None

    def remember(self, interaction: Interaction, now: int | None = None) -> int:
        """Embed `interaction.query` and store the (query, answer) pair. Returns the
        new row's id."""
        now = now if now is not None else int(time.time())
        vector = np.asarray(self._embedder.embed_one(interaction.query), dtype=np.float32)
        tbl = self._open_or_none()
        next_id = tbl.count_rows() + 1 if tbl is not None else 1
        row = {
            "id": next_id,
            "query": interaction.query,
            "answer": interaction.answer,
            "vector": vector.tolist(),
            "created_at": now,
        }
        if tbl is None:
            self._db.create_table(TABLE_NAME, data=[row], mode="overwrite")
        else:
            tbl.add([row])
        return next_id

    def count(self) -> int:
        tbl = self._open_or_none()
        return tbl.count_rows() if tbl is not None else 0

    def recall(
        self, query: str, top_k: int = 5, query_vec: np.ndarray | None = None
    ) -> list[RecalledInteraction]:
        """Return the top-K most similar past interactions, ranked by cosine
        similarity (highest first).

        `query_vec` lets a caller that already embedded `query` for its own purposes
        (e.g. the document-corpus search) pass that vector straight through instead of
        paying for a second, redundant embedding call on the same text."""
        tbl = self._open_or_none()
        if tbl is None:
            return []
        if query_vec is None:
            query_vec = np.asarray(self._embedder.embed_one(query), dtype=np.float32)
        rows = tbl.search(query_vec).metric("cosine").limit(top_k).to_list()
        return [
            RecalledInteraction(
                id=r["id"],
                query=r["query"],
                answer=r["answer"],
                similarity=1.0 - float(r["_distance"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]
