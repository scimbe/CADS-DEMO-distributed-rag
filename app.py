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
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from rag.embedder import Embedder
from rag.memory import Interaction, MemoryStore
from rag.provider_pool import ChatResult, ProviderPoolError, chat, configured_backends
from rag.store import SearchResult, Store
from rag.verification import ClaimVerification, verify_answer

REPO_ROOT = Path(__file__).resolve().parent
STATIC_DIR = REPO_ROOT / "static"

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
    # Grounding-verification rollup (marketplace#40) for every claim in `answer`
    # that cited this passage: None when no claim cited it, otherwise the worst
    # case across those claims (any not-fully-supported claim makes this False;
    # confidence is the lowest confidence among them). See rag/verification.py.
    verified: Optional[bool] = None
    confidence: Optional[float] = None


class RecalledOut(BaseModel):
    query: str
    answer: str
    similarity: float


class MemoryOut(BaseModel):
    recalled: list[RecalledOut]
    remembered_id: Optional[int] = None


class ClaimVerificationOut(BaseModel):
    text: str
    citations: list[int]
    verdict: str  # "supported" | "partially_supported" | "not_supported" | "uncited"
    confidence: Optional[float] = None
    method: str  # "lexical" | "llm" | "lexical_fallback" | "uncited"


class VerificationOut(BaseModel):
    claims: list[ClaimVerificationOut]
    # True iff every CITED claim verdict is "supported" (vacuously True if there
    # are no cited claims at all). Uncited sentences don't count against this --
    # see rag/verification.py's module docstring for why that's out of scope here.
    all_supported: bool


class QueryResponse(BaseModel):
    question: str
    answer: str
    backend: str
    model: str
    sources: list[SourceOut]
    memory: MemoryOut
    verification: VerificationOut


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

# Runs memory.recall() concurrently with the LLM call in /query -- recall's result is
# deliberately excluded from the LLM prompt already (see the comment at its call site),
# so it has no real dependency on the chat() call and shouldn't block in front of it.
_memory_pool = ThreadPoolExecutor(max_workers=2)


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


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


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

    # Check the interaction-memory layer (recall) before answering -- surfaced in the
    # response for observability, kept OUT of the LLM prompt itself so a prior answer
    # can never masquerade as a "fact present in the retrieved passages" (the grounding
    # requirement is specifically about the document corpus, not about memory). Recall
    # has no dependency on the document search or the LLM call below, so it runs
    # concurrently with both instead of blocking in front of them; reuses `query_vec`
    # instead of re-embedding the same question a second time.
    recall_future = (
        _memory_pool.submit(_memory.recall, question, top_k=memory_top_k, query_vec=query_vec)
        if memory_top_k > 0
        else None
    )

    results = _store.search(query_vec, top_k=top_k)

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

    recalled = recall_future.result() if recall_future is not None else []
    remembered_id = _memory.remember(Interaction(query=question, answer=answer, created_at=int(time.time())))

    # Programmatic grounding verification (marketplace#40): independently check
    # each cited claim in `answer` against the passage(s) it cites -- see
    # rag/verification.py for the technique and why it's a separate mechanism
    # from the answer-generation call above.
    passages_by_citation = {i: r.text for i, r in enumerate(results, start=1)}
    claim_verifications: list[ClaimVerification] = verify_answer(answer, passages_by_citation)
    verification = VerificationOut(
        claims=[
            ClaimVerificationOut(
                text=cv.text, citations=cv.citations, verdict=cv.verdict, confidence=cv.confidence, method=cv.method
            )
            for cv in claim_verifications
        ],
        all_supported=all(cv.verdict == "supported" for cv in claim_verifications if cv.verdict != "uncited"),
    )

    def _source_rollup(citation: int) -> tuple[Optional[bool], Optional[float]]:
        citing = [cv for cv in claim_verifications if citation in cv.citations]
        if not citing:
            return None, None
        return (
            all(cv.verdict == "supported" for cv in citing),
            min(cv.confidence for cv in citing if cv.confidence is not None),
        )

    sources = []
    for i, r in enumerate(results, start=1):
        verified, confidence = _source_rollup(i)
        sources.append(
            SourceOut(
                citation=i,
                source=r.source,
                title=r.title,
                passage_index=r.passage_index,
                similarity=r.similarity,
                text=r.text,
                verified=verified,
                confidence=confidence,
            )
        )

    return QueryResponse(
        question=question,
        answer=answer,
        backend=backend,
        model=model,
        sources=sources,
        memory=MemoryOut(
            recalled=[RecalledOut(query=r.query, answer=r.answer, similarity=r.similarity) for r in recalled],
            remembered_id=remembered_id,
        ),
        verification=verification,
    )


if __name__ == "__main__":
    import uvicorn

    # _init_state() itself runs via FastAPI's "startup" event above -- this just
    # binds and serves; works identically whether launched directly (`python app.py`)
    # or via `uvicorn app:app` (e.g. from a container CMD).
    uvicorn.run(app, host=os.environ.get("RAG_HOST", "0.0.0.0"), port=int(os.environ.get("RAG_PORT", "8080")))
