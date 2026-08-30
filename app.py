#!/usr/bin/env python3
"""HTTP service for CADS-DEMO-distributed-rag (marketplace#33).

Wires together every piece built for this demo into ONE real web service, matching
the pattern already established by `crates/harness-memory` in CADS-agent-marketplace
(a small HTTP service, not a CLI script) and by `manifests/litellm-proof/heartbeat-proxy`
(a single `app.py`, Dockerfile, requirements.txt at a service's root):

    rag/embedder.py        -- local BGE-small embeddings (fastembed/ONNX, no network at
                               inference time)
    rag/store.py            -- document-corpus SQLite store (Gutendex + Open Library +
                               Internet Archive passages)
    rag/memory.py            -- long-term interaction/fact memory (embedded LanceDB)
    rag/provider_pool.py      -- local litellm-proxy, with Groq as an opt-in free
                               fallback -- never a hard dependency

POST /query is the acceptance-criterion endpoint: retrieve relevant document-corpus
chunks for a real question, check + update the interaction memory layer, and ask the
LLM (via the provider pool) to synthesize a grounded answer that cites which real
source(s) backed each claim. The system prompt explicitly forbids adding any fact that
is not present in the retrieved chunks -- see `_SYSTEM_PROMPT` below.

Run directly:
    .venv/bin/python app.py

Run as a container:
    docker build -t distributed-rag .
    docker run -p 8080:8080 -v "$PWD/data:/app/data" --env-file .env distributed-rag

Env config (all optional, sensible defaults for local/dev use):
    RAG_HOST              -- bind host (default 0.0.0.0)
    RAG_PORT              -- bind port (default 8080)
    RAG_DB_PATH            -- document-corpus SQLite path (default data/rag.sqlite3)
    RAG_MEMORY_PATH         -- interaction-memory LanceDB path (default data/memory.lance)
    RAG_FASTEMBED_CACHE_DIR -- persistent cache dir for the one-time model download
                               (default: data/fastembed_cache)
    RAG_TOP_K               -- default number of corpus chunks retrieved per query (default 5)
    RAG_MEMORY_TOP_K         -- default number of past interactions recalled per query (default 3)
    LITELLM_BASE_URL / LITELLM_API_KEY / LITELLM_DEFAULT_MODEL / GROQ_API_KEY / ...
                             -- read by rag/provider_pool.py, see .env.example
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.embedder import Embedder
from rag.memory import Interaction, MemoryStore
from rag.provider_pool import ChatResult, ProviderPoolError, chat, configured_backends
from rag.store import SearchResult, Store

REPO_ROOT = Path(__file__).resolve().parent

DB_PATH = os.environ.get("RAG_DB_PATH", str(REPO_ROOT / "data" / "rag.sqlite3"))
MEMORY_PATH = os.environ.get("RAG_MEMORY_PATH", str(REPO_ROOT / "data" / "memory.lance"))
FASTEMBED_CACHE_DIR = os.environ.get("RAG_FASTEMBED_CACHE_DIR", str(REPO_ROOT / "data" / "fastembed_cache"))
DEFAULT_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))
DEFAULT_MEMORY_TOP_K = int(os.environ.get("RAG_MEMORY_TOP_K", "3"))
MAX_TOP_K = 20  # hard ceiling regardless of what a caller requests

_SYSTEM_PROMPT = (
    "You are a grounded question-answering assistant over a small real book/library corpus. "
    "You are given a numbered list of SOURCE PASSAGES retrieved from real, previously-ingested "
    "book/library text (Project Gutenberg, Open Library, or Internet Archive). "
    "Answer the user's QUESTION using ONLY facts that are explicitly stated in those passages. "
    "After every factual claim in your answer, cite the passage number(s) it came from in square "
    "brackets, e.g. [1] or [1][3]. "
    "Do not use any outside or background knowledge, even if you believe it to be true, and do not "
    "invent, assume, or infer any fact that is not explicitly present in the passages. "
    "If the passages do not contain enough information to answer the question, say so plainly "
    "instead of guessing or filling in gaps."
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="A real natural-language question.")
    top_k: Optional[int] = Field(None, ge=1, le=MAX_TOP_K, description="Corpus chunks to retrieve.")
    memory_top_k: Optional[int] = Field(None, ge=0, le=MAX_TOP_K, description="Past interactions to recall.")


class SourceOut(BaseModel):
    citation: int
    source: str
    title: str
    passage_index: int
    similarity: float
    text: str


class RecalledOut(BaseModel):
    query: str
    answer: str
    similarity: float


class MemoryOut(BaseModel):
    recalled: list[RecalledOut]
    remembered_id: Optional[int] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    backend: str
    model: str
    sources: list[SourceOut]
    memory: MemoryOut


class HealthResponse(BaseModel):
    status: str
    corpus_chunks: int
    memory_interactions: int
    llm_backends: dict[str, bool]


# --- singletons, loaded once at process startup (mirrors scripts/ingest.py and
# scripts/query.py's own "load the model once, reuse it" discipline) ---
_embedder: Embedder | None = None
_store: Store | None = None
_memory: MemoryStore | None = None


def _init_state() -> None:
    global _embedder, _store, _memory
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(MEMORY_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(FASTEMBED_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    _embedder = Embedder(cache_dir=FASTEMBED_CACHE_DIR)
    _store = Store(DB_PATH)
    _memory = MemoryStore(MEMORY_PATH, _embedder)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _init_state()
    yield


app = FastAPI(title="CADS-DEMO-distributed-rag", version="0.1.0", lifespan=_lifespan)


def _format_context(results: list[SearchResult]) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(
            f"[{i}] (source: {r.source}, title: {r.title!r}, passage: {r.passage_index}, "
            f"similarity: {r.similarity:.3f})\n{r.text.strip()}"
        )
    return "\n\n".join(blocks)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    assert _store is not None and _memory is not None
    return HealthResponse(
        status="ok",
        corpus_chunks=_store.count(),
        memory_interactions=_memory.count(),
        llm_backends=configured_backends(),
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    assert _embedder is not None and _store is not None and _memory is not None
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be blank")

    if _store.count() == 0:
        raise HTTPException(
            status_code=503,
            detail="document corpus is empty -- run scripts/ingest.py (gutenberg/openlibrary/archive) first",
        )

    top_k = req.top_k or DEFAULT_TOP_K
    memory_top_k = DEFAULT_MEMORY_TOP_K if req.memory_top_k is None else req.memory_top_k

    query_vec = _embedder.embed_one(question)
    results = _store.search(query_vec, top_k=top_k)

    # Check the interaction-memory layer (recall) before answering -- surfaced in the
    # response for observability, kept OUT of the LLM prompt itself so a prior answer
    # can never masquerade as a "fact present in the retrieved passages" (the grounding
    # requirement is specifically about the document corpus, not about memory).
    recalled = _memory.recall(question, top_k=memory_top_k) if memory_top_k > 0 else []

    if not results:
        answer = (
            "I don't have any retrieved passages that are relevant to this question, so I "
            "can't answer it from the corpus."
        )
        backend, model = "none", "none"
    else:
        context = _format_context(results)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"SOURCE PASSAGES:\n\n{context}\n\nQUESTION: {question}"},
        ]
        try:
            result: ChatResult = chat(messages)
        except ProviderPoolError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        answer, backend, model = result.content, result.backend, result.model

    remembered_id = _memory.remember(Interaction(query=question, answer=answer, created_at=int(time.time())))

    return QueryResponse(
        question=question,
        answer=answer,
        backend=backend,
        model=model,
        sources=[
            SourceOut(
                citation=i,
                source=r.source,
                title=r.title,
                passage_index=r.passage_index,
                similarity=r.similarity,
                text=r.text,
            )
            for i, r in enumerate(results, start=1)
        ],
        memory=MemoryOut(
            recalled=[RecalledOut(query=r.query, answer=r.answer, similarity=r.similarity) for r in recalled],
            remembered_id=remembered_id,
        ),
    )


if __name__ == "__main__":
    import uvicorn

    # _init_state() itself runs via FastAPI's "startup" event above -- this just
    # binds and serves; works identically whether launched directly (`python app.py`)
    # or via `uvicorn app:app` (e.g. from a container CMD).
    uvicorn.run(app, host=os.environ.get("RAG_HOST", "0.0.0.0"), port=int(os.environ.get("RAG_PORT", "8080")))
