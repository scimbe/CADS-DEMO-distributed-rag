# CADS-DEMO-distributed-rag

A retrieval-augmented question-answering service over a small real public-domain book
corpus. Ask a question, get an answer that cites exactly which retrieved passage backed
each claim — with a system prompt that structurally forbids the model from adding
anything not present in those passages. Part of the bunsenbrenner.org demo portfolio
([marketplace#33](https://github.com/scimbe/CADS-agent-marketplace/issues/33)).

**Status: working demo, running live on this host, not yet packaged for the
marketplace.** The `/query` and `/health` endpoints are real and containerized; there is
no signed marketplace manifest and no install/uninstall cycle through ct-agent's
installer-engine yet. See "What's not built yet" below.

## Why this exists

The point isn't "another chatbot wrapper" — it's demonstrating that the platform can
host a genuinely grounded RAG pipeline with **zero hard external dependency**: local
embeddings (no API key, no per-call billing), three free key-less connectors for real
book/library text, and a local vector store, all wired behind one containerized HTTP
service that only reaches out to an LLM at answer-synthesis time. Two things make an
answer trustworthy rather than just plausible: retrieval is grounded in real ingested
text (not the model's training data), and every claim carries a `[n]` citation back to
the passage that supports it — verifiable by a human reading the same passage.

## Architecture

```
Gutendex/gutenberg.org   Open Library   Internet Archive     (free, key-less connectors)
        │                     │                │
        └─────────────────────┴────────┬───────┘
                                        ▼
                          chunk_text() — ~800-char, paragraph-aligned
                                        │
                                        ▼
        rag/embedder.py — BAAI/bge-small-en-v1.5 via fastembed (local ONNX,
                           no network call at inference time)
                                        │  384-dim float32 vector
                                        ▼
        rag/store.py — SQLite, one row per passage, embedding as a packed
                        float32 BLOB, brute-force cosine search

        rag/memory.py — separate LanceDB store of past (query, answer) pairs,
                        recalled by similarity but never fed into the LLM prompt
                        (so a prior answer can never masquerade as a source fact)

question ──► embed once ──► POST /query (app.py)
                              1. retrieve top-k corpus chunks
                              2. recall related past interactions (memory, informational only)
                              3. build a grounded prompt from the chunks only
                              4. call the LLM via rag/provider_pool.py
                              5. remember (question, answer)
```

`rag/provider_pool.py` routes LLM calls to the team's local litellm-proxy first and
falls through to Groq only if `GROQ_API_KEY` is set — never a hard dependency; with no
Groq key configured, a local outage just surfaces as a normal error.

## Running it

Directly:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in LITELLM_* (or GROQ_API_KEY) — see Configuration below
.venv/bin/python scripts/ingest.py gutenberg 1661             # fetch + embed + store one real book
.venv/bin/python app.py                                        # binds 0.0.0.0:8080 (RAG_PORT to override)
curl -s localhost:8080/query -H 'content-type: application/json' \
  -d '{"question": "your question about ingested content"}'
```

As a container:

```bash
docker build -t distributed-rag .
docker run -p 8080:8080 -v "$PWD/data:/app/data" --env-file .env distributed-rag
# or:
docker compose up --build
docker compose exec rag python scripts/ingest.py gutenberg 1661
```

## Endpoints (verified live, 2026-09-01)

The container currently runs on this host as `cads-demo-distributed-rag-rag-1`, mapped
to `127.0.0.1:8081` (not 8080 — that host port is already taken by an unrelated
pre-existing container, `cadserv-cadvisor`, on `cads-lambda`; see the comment in
`docker-compose.yml`). Inside the container it's still `RAG_PORT=8080`.

- `GET /` — serves `static/index.html`, a single-page frontend (ask a question, see the
  answer with its cited sources and any recalled memory) with a DE/EN language toggle
  and the portfolio's standard dark-first styling and funding footer.
- `GET /health` — curled live just now: `{"status":"ok","corpus_chunks":2167,
  "memory_interactions":26,"llm_backends":{"local-litellm":true,"groq":false}}`.
  `memory_interactions` grows by one on every `/query` call; the number above is a
  snapshot, not a fixed fact.
- `POST /query` — body `{"question": "...", "top_k": 5, "memory_top_k": 3}` (`top_k`
  and `memory_top_k` optional). Curled live just now with a real question:

  ```json
  {
    "question": "What does Open Library say Pride and Prejudice is about?",
    "answer": "According to Open Library, *Pride and Prejudice* is an 1813 novel of manners written by Jane Austen that follows the character development of Elizabeth Bennet, the dynamic protagonist of the book who learns about the repercussions of hasty judgments and comes to appreciate the difference between superficial goodness and actual goodness [1].",
    "backend": "local-litellm",
    "model": "bunsenbrenner-default",
    "sources": [ { "citation": 1, "source": "openlibrary:OL66554W", "title": "Pride and Prejudice", "passage_index": 0, "similarity": 0.760, "text": "..." } ],
    "memory": { "recalled": [], "remembered_id": 26 }
  }
  ```

  Every `sources[]` entry carries the real `source` id, title, passage index,
  similarity, and the retrieved text itself, so an answer's grounding can be checked
  directly rather than trusted. If no relevant chunks are retrieved, the service returns
  a fixed "I don't have any retrieved passages relevant to this question" answer
  *without calling the LLM at all*. Returns `400` on a blank question and `503` if the
  corpus is empty.

## The corpus (verified live, 2026-09-01)

Queried the running container's SQLite store directly (`docker exec ... sqlite3`
against the `chunks` table, not just trusting `/health`'s count):

| source | title | chunks |
|---|---|---|
| `gutenberg:1661` | The Adventures of Sherlock Holmes | 962 |
| `archive:prideandprejudic42671gut` | Pride and Prejudice (Internet Archive full text) | 1202 |
| `openlibrary:OL66554W` | Pride and Prejudice (Open Library descriptive text) | 3 |
| **total** | | **2167** |

All three connectors are represented in the live corpus — this was previously a gap
(only Sherlock Holmes was ingested at one point) and is now fixed. Ingest another book
with `scripts/ingest.py <gutenberg|openlibrary|archive> <id>` (or the equivalent
`docker compose exec rag ...` form).

## Configuration

Copy `.env.example` to `.env` (gitignored) and fill in real values. Read by
`rag/provider_pool.py`:

| var | purpose |
|---|---|
| `LITELLM_BASE_URL` / `LITELLM_API_KEY` / `LITELLM_DEFAULT_MODEL` | the team's local litellm-proxy — primary LLM path |
| `GROQ_API_KEY` (+ optional `GROQ_MODEL`, `GROQ_BASE_URL`) | free external fallback, only used on a local rate-limit/connection failure; leave unset to disable fallback entirely |

The instance running live on this host points `LITELLM_BASE_URL` at
`https://llm-34a13a96.bunsenbrenner.org/v1` with `LITELLM_DEFAULT_MODEL=bunsenbrenner-default`
— the portfolio-wide shared model alias (confirmed above: the live `/query` response
reports `"model": "bunsenbrenner-default"`). `GROQ_API_KEY` is unset on this instance
(`/health` reports `"groq": false`). `.env.example` itself still ships an older
placeholder model name as an illustrative default — it is not the value actually in use.

Other env vars, all optional with sensible defaults (see `app.py`'s module docstring):
`RAG_HOST`, `RAG_PORT`, `RAG_DB_PATH`, `RAG_MEMORY_PATH`, `RAG_FASTEMBED_CACHE_DIR`,
`RAG_TOP_K` (default 5), `RAG_MEMORY_TOP_K` (default 3).

## Dependencies

- **Embedding**: `BAAI/bge-small-en-v1.5` (33M params, 384-dim, MIT-licensed — checked
  directly on the model card, not secondhand) via [`fastembed`](https://github.com/qdrant/fastembed)
  (Apache-2.0, ONNX/CPU, no PyTorch). Runs locally; only the one-time model download
  needs network access.
- **Document connectors**: [Gutendex](https://gutendex.com)/gutenberg.org, [Open
  Library](https://openlibrary.org/developers/api), [Internet
  Archive](https://archive.org/developers/) — all free, public, key-less. The Internet
  Archive connector explicitly refuses any `access-restricted-item` rather than
  assuming its lending terms are safe to reuse.
- **Document store**: local SQLite (`rag/store.py`) — BLOB vector column, brute-force
  cosine search, no vector-DB dependency (demo-scale, not web-scale).
- **Interaction memory**: [LanceDB](https://pypi.org/project/lancedb/) embedded mode
  (Apache-2.0, no server process) — `rag/memory.py`, distinct from the document corpus.
- **LLM**: the shared portfolio litellm-proxy (`bunsenbrenner-default` alias, see
  Configuration above), with an opt-in Groq fallback that is **not yet verified with a
  real completion** — see Known limitations.
- Python 3.12, FastAPI + uvicorn (`requirements.txt`).

## Reachability

Directly over HTTP from this host: `127.0.0.1:8081` (see Endpoints above). There is no
public `*.bunsenbrenner.org` subdomain for this service — no DNS/tunnel currently
resolves for it.

It is additionally exposed as a `text_generation` service over an Agent-Fabric ct-agent
channel to a specific peer session, via a channel identity, handler script, and
pidfile-watchdog kept at `/home/becke/rag-channel-identity/` on the host — **outside
this git repo**, not tracked here. That handler shells out to this service's own
`/query` HTTP endpoint (currently pointed at `127.0.0.1:8081`), so it adds no separate
capability, only a different transport. Confirmed alive on this host as of this check
(accept process running, listening for channel admissions).

## Tests

`tests/test_memory.py` (3 tests, real embedder + real embedded LanceDB, nothing mocked)
and `tests/test_provider_pool.py` (10 tests, mocked HTTP — routing/fallback logic only).
Ran live: 13 passed, 2026-09-01. The document-corpus path (`rag/store.py`,
`rag/gutendex.py`, `rag/embedder.py`) and `app.py` itself have no automated tests yet —
only manual verification via the scripts and the live `/query` runs above.

## What's not built yet

- **The Groq fallback path is unverified with a real key.** The routing logic is unit
  tested and the endpoint shape was live-confirmed (a placeholder key got a real `401`
  from Groq, proving the URL/request format), but no actual successful Groq completion
  has been observed — getting a real free key needs an interactive signup step.
- **No marketplace manifest packaging.** Not signed, not installable/uninstallable via
  ct-agent's installer-engine, no `dev_sign`/`activate()`/verify-output cycle run.
- **No Rust port.** This is Python end to end (`fastembed`, FastAPI). If a packaged
  manifest eventually needs a single Rust binary (matching the marketplace's
  `harness-memory` convention), that's a rewrite, not a wrapper.
- **No auth on `/query`.** Fine for a same-host/loopback demo; would need at least a
  bearer token before being reachable from outside a private network. The Agent-Fabric
  channel handler (see Reachability) also does not add its own auth layer beyond the
  channel's own admission handshake.
- **Partial test coverage** — see Tests above.
- **No production-quality error handling, retries, or rate-limit handling** for the
  Gutendex/gutenberg.org calls.
- **No LICENSE file** in this repo — to be decided by operator/project convention, not
  silently defaulted. The embedding model (MIT) and the ingested text (Project
  Gutenberg / Open Library CC0 / an Internet Archive Gutenberg mirror, all public
  domain) are independently license-clean regardless of this repo's own license.

## Repo layout

```
rag/
  embedder.py         — local BGE-small embedding wrapper (fastembed)
  gutendex.py         — Gutendex connector (fetch, boilerplate-strip, chunk_text — shared by all three connectors)
  openlibrary.py      — Open Library connector
  internetarchive.py  — Internet Archive connector
  store.py            — SQLite BLOB-vector store + brute-force cosine search (document corpus)
  memory.py           — LanceDB interaction memory: remember()/recall()
  provider_pool.py    — local litellm-proxy → Groq free-fallback LLM router
scripts/
  ingest.py                 — CLI: fetch from gutenberg/openlibrary/archive by id, embed, store
  query.py                  — CLI: embed a query, retrieve top-k passages
  verify_provider_pool.py   — CLI: live-check whichever LLM backends are configured
tests/
  test_memory.py, test_provider_pool.py
static/index.html   — single-page frontend (DE/EN, dark-first)
app.py               — FastAPI /query + /health service, wires everything above together
Dockerfile, docker-compose.yml, requirements.txt, .env.example
```
