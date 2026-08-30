# CADS-DEMO-distributed-rag

**Status: bootstrap / proof-of-concept. Not a finished demo. Not yet packaged as a
marketplace manifest. Read "What's NOT built yet" below before assuming this is done.**

Part of the bunsenbrenner.org demo portfolio
([marketplace#33](https://github.com/scimbe/CADS-agent-marketplace/issues/33)). The
target audience is devs/power-users; the point is to show real platform maturity — a
distributed-harness-style memory/context system, not just a scripted demo.

This bootstrap proves the hardest constraint from the issue first: a **fully local
embedding pipeline** (no API key, no external service, no per-request cost) feeding
**three real, free, key-less connectors** (Gutendex, Open Library, Internet Archive)
to public-domain book/library sources, with **genuine retrieval** verified by a human
reading the retrieved text.

## The "really free, no costs under any circumstances" requirement

Every component that ingestion/retrieval depends on for this bootstrap runs locally or
against a free, key-less public API:

- **Embedding model**: runs on this machine, once downloaded and cached — no API key,
  no per-call billing, works fully offline after the first run.
- **Connectors**: [Gutendex](https://gutendex.com) + gutenberg.org (full book text),
  [Open Library](https://openlibrary.org/developers/api) (bibliographic/descriptive
  text), and [Internet Archive](https://archive.org/developers/) (full book text from
  its own freely-downloadable, non-lending-restricted items) — all free, public,
  key-less APIs, no authentication, no rate-limit-behind-a-paywall.
- **Store**: a local SQLite file — no hosted database.

Nothing in this bootstrap calls any paid service, any team member's personal API key,
or any hosted "embed_text"-style channel service another maintainer runs. That would
have violated the issue's explicit "no hard external dependency" requirement — this
must be installable by *anyone* via ct-agent, not just people with access to our
internal services.

## Embedding model: verified license

**Model: [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5)**
(33M parameters, 384-dim output, English).

**License: MIT.** Verified directly on the model's Hugging Face model card on
2026-08-30 (not from any secondhand description): *"FlagEmbedding is licensed under the
[MIT License](https://github.com/FlagOpen/FlagEmbedding/blob/master/LICENSE). The
released models can be used for commercial purposes free of charge."* This project has
been burned before by trusting a model's marketing blurb over its actual license (a 3B
variant of an otherwise-permissive model family turned out to be non-commercial-only),
so this was checked at the source, not assumed from the issue's own research notes.

The issue's other candidate, `nomic-ai/nomic-embed-text-v1.5` (137M params, 8K context),
was also checked at the source and is genuinely **Apache-2.0** — also fine license-wise.
BGE-small was picked over it for this bootstrap because it's smaller, has no required
query/document prefix convention to get right, and is a very well-trodden path for
offline ONNX inference. Nomic Embed remains a legitimate option if a future iteration
needs its longer context window.

## Inference path: `fastembed`, not `sentence-transformers` or a Rust port

Chosen: **[`fastembed`](https://github.com/qdrant/fastembed)** (Apache-2.0, maintained
by Qdrant), which runs BGE-small as a quantized ONNX graph via `onnxruntime`. Reasoning:

- **vs. `sentence-transformers`**: `sentence-transformers` pulls in full PyTorch, a
  multi-GB dependency, to run a 33M-parameter model. `fastembed`'s ONNX path is a small,
  fast, CPU-friendly install with no GPU/CUDA story to get right for a demo that has to
  install cleanly for "anyone."
- **vs. a hand-rolled Rust path (`candle` or raw `ort`)**: this project's own
  `crates/harness-memory` (in `CADS-agent-marketplace`) already establishes the target
  storage pattern in Rust (SQLite BLOB + brute-force cosine — mirrored here, see below),
  and the eventual packaged manifest may want a Rust binary to match that convention.
  For *this bootstrap*, Python + `fastembed` was faster to stand up and verify actually
  working end to end within one sitting, and is explicitly not the final answer — see
  "Rust port" under What's NOT built yet.

Verified working (2026-08-30): loaded `BAAI/bge-small-en-v1.5` via `fastembed`, embedded
a sample sentence, got back a real `float32` vector of shape `(384,)`, L2-normalized
(`norm == 1.0`). No mocking — this actually ran.

## Architecture

```
Gutendex          Open Library          Internet Archive
(full book text)  (descriptive text)    (full book text, non-restricted items only)
        │                │                       │
        ▼                ▼                       ▼
rag/gutendex.py   rag/openlibrary.py      rag/internetarchive.py
        │                │                       │
        └────────────────┴───────────┬───────────┘
                                      ▼
                    chunk_text()  — ~800-char paragraph-aligned
                                    passages with overlap (defined in
                                    rag/gutendex.py, shared by all three)
                                      │
                                      ▼
rag/embedder.py  — BAAI/bge-small-en-v1.5 via fastembed (local, ONNX, no network
                    call at inference time)
                                      │  384-dim float32 vector per passage
                                      ▼
rag/store.py     — SQLite, one row per passage, embedding stored as a packed
                    float32 BLOB, brute-force cosine search over all rows
```

All three connectors feed the identical downstream pipeline via
`scripts/ingest.py <source> <id>` (`source` is `gutenberg`/`openlibrary`/`archive`) —
see "Reproduce it yourself" below.

This mirrors `CADS-agent-marketplace`'s own `crates/harness-memory/src/db.rs` pattern:
plain SQLite + a BLOB vector column + brute-force cosine, not a vector-DB dependency —
that crate's own reasoning applies here too (sub-millisecond to low-millisecond search
at the scale of one ingested book's worth of passages; this is demo-scale, not
web-scale).

The issue's own architecture additionally describes a separate **memory layer**
(ChromaDB/LanceDB, for cross-session interaction memory, distinct from this document
corpus — now built, see "Long-term interaction memory" below) and a **dynamic
free-provider LLM pool** (OpenRouter/Groq/Cerebras/Cloudflare Workers AI, as
resilience/fallback alongside the team's own `local-devstral-small2` — also built,
see below).

## What's actually proven working (ran, not just written)

1. **Local embedding pipeline** — loaded the model, embedded real text, got a real
   384-dim normalized vector back. See "Inference path" above.
2. **Gutendex connector, end to end** — fetched real metadata and the real plain-text
   body of *The Adventures of Sherlock Holmes* (Project Gutenberg id 1661) from
   `gutendex.com` + `gutenberg.org`, no key, no auth.
3. **Chunk → embed → store** — the fetched book's boilerplate-stripped text was split
   into passages and every passage was embedded locally and stored in a SQLite file
   with its vector as a BLOB.
4. **Retrieval, verified by reading the result, not trusting the score** — a natural-
   language query embedded with the same local model retrieved the top-matching stored
   passage(s) via brute-force cosine similarity; the retrieved text was read and
   confirmed to genuinely relate to the query (see `scripts/query.py` output / the
   ingest+query run recorded in the PR/commit this README ships with).
5. **Open Library connector, end to end** — a real live `GET
   https://openlibrary.org/search.json?q=pride+and+prejudice` returned work key
   `/works/OL66554W`; `GET https://openlibrary.org/works/OL66554W.json` returned
   *Pride and Prejudice*'s real description, first-sentence excerpt ("It is a truth
   universally acknowledged...") and subject tags, no key, no auth. Assembled into a
   1812-char document, split into 3 passages, embedded locally, and stored (source
   `openlibrary:OL66554W`) via `scripts/ingest.py openlibrary OL66554W`.
6. **Internet Archive connector, end to end** — a real live `GET
   https://archive.org/advancedsearch.php?q=(pride and prejudice) AND
   mediatype:texts` located item `prideandprejudic42671gut` (an archive.org-hosted
   Project Gutenberg mirror, `access-restricted-item` absent); `GET
   https://archive.org/metadata/prideandprejudic42671gut` found a real downloadable
   `42671.txt` (format `Text`, 725,090 bytes); fetching it returned the genuine full
   novel text (verified: contains "It is a truth universally acknowledged..." at
   character offset 1903, and the standard Project Gutenberg header/footer markers,
   correctly stripped by the shared `strip_gutenberg_boilerplate()` down to 705,474
   chars of body text). Split into 1202 passages, embedded locally, and stored
   (source `archive:prideandprejudic42671gut`) via
   `scripts/ingest.py archive prideandprejudic42671gut`.
7. **Cross-connector retrieval, verified by reading the result** — with all three
   connectors' data in one store (2167 chunks total: Gutendex + Open Library +
   Internet Archive), `scripts/query.py "Mr. Darcy's pride and his feelings for
   Elizabeth Bennet"` retrieved, in its top 5, genuinely on-topic passages from
   *both* new connectors alongside Gutendex-sourced ones — `archive:` passages
   quoting Darcy/Elizabeth's actual dialogue (similarity 0.79/0.78/0.78) and the
   `openlibrary:` descriptive passage (similarity 0.78) — read in full and confirmed
   on-topic, not assumed from the score.

Reproduce it yourself:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/ingest.py gutenberg 1661             # fetches + embeds + stores one real book
.venv/bin/python scripts/ingest.py openlibrary OL66554W       # real bibliographic/descriptive text
.venv/bin/python scripts/ingest.py archive prideandprejudic42671gut  # real full book text
.venv/bin/python scripts/query.py "Sherlock Holmes deduces something about a client from small details"
```

## Long-term interaction memory (distinct from the document corpus)

**`rag/memory.py`** — the issue's separate "context-optimization" memory layer:
top-K semantic recall over past query/answer interactions, instead of a
linear-growing context window. This is *not* the document corpus (`rag/store.py`
holds ingested book passages) — it's what was asked and answered before, so a later
turn can pull back a handful of relevant prior interactions rather than replaying
the whole conversation history.

**Library: [LanceDB](https://pypi.org/project/lancedb/)** (embedded/local mode, no
server process). License-checked directly at the source on 2026-08-30, not assumed:
**Apache License, Version 2.0**, confirmed both on the PyPI project page and by
fetching `LICENSE` from
[github.com/lancedb/lancedb](https://github.com/lancedb/lancedb/blob/main/LICENSE).
The issue's other named option, **ChromaDB**, was also checked the same way and is
also genuinely Apache-2.0 for its local/self-hosted modes (its optional "Chroma
Cloud" tier is paid, but never required). LanceDB was picked because its embedded
mode — `lancedb.connect(<local path>)`, a local directory of Lance/Arrow files, no
daemon — is the closer match to this repo's existing footprint (`rag/store.py` is
one local SQLite file, same shape: a local store, no server to run). No network
call, no API key, no billing relationship — same "really free" guarantee as the
rest of this repo; confirmed by running `remember()`/`recall()` end to end with
DNS resolution actively blocked at the socket layer and no failure.

Each interaction is embedded on its `query` text with the same local `Embedder`
(`rag/embedder.py`, BAAI/bge-small-en-v1.5 via fastembed) already used for the
document corpus — mirrors `rag/store.py`'s convention of embedding the text you'd
actually search against.

Verified: `tests/test_memory.py` (3 unit tests, real embedder + real embedded
LanceDB table, nothing mocked). Several genuinely distinct interactions (pasta
recipe, French Revolution date, Python asyncio, Mount Kilimanjaro's height) plus
two Sherlock-Holmes-deduction interactions were stored; a new Holmes-deduction
query recalled both Holmes interactions in the top two slots, strictly ahead of
every distractor (cosine similarity ≈0.72–0.74 for the Holmes interactions vs.
≈0.52 for the closest distractor). Same discipline as the corpus retrieval check
described above: the recalled `(query, answer)` pairs were read back and confirmed
by identity to be the genuinely on-topic interactions, not just "the score was
highest."

## Dynamic free-provider LLM fallback pool

**`rag/provider_pool.py`** — the issue's router that tries the team's local
litellm-proxy first (`LITELLM_BASE_URL`/`LITELLM_API_KEY`/`LITELLM_DEFAULT_MODEL`,
the same convention every other CADS-DEMO-* repo uses) and falls through to ONE
external free provider on a rate-limit/connection failure — never a hard dependency;
with `GROQ_API_KEY` unset, a local outage just surfaces as a normal error instead of
silently reaching a third party.

**Provider chosen: [Groq](https://groq.com).** Re-checked against the issue's own
`free-ai-tools` list (2026-08-30, not the stale 2026-08-28 research) alongside
OpenRouter/Cerebras/Cloudflare Workers AI. Groq wins for a single always-on fallback:
14,400 req/day on `llama-3.1-8b-instant` (vs. OpenRouter's 50 req/day shared across
all free models), no credit card for free-tier signup, and a drop-in OpenAI-compatible
endpoint (`https://api.groq.com/openai/v1/chat/completions`) — same request/response
shape as the local litellm-proxy, so the router needs no per-provider translation.

Verified: the fallback-routing logic itself (10 unit tests in
`tests/test_provider_pool.py`, mocked — local success, local rate-limit/connection/5xx
→ Groq fallback, a non-retryable local failure like 401 correctly *not* triggering a
fallback, no-backend-configured refusing before any network call, per-backend model
isolation). Also live-confirmed real network reachability of Groq's exact endpoint
(`scripts/verify_provider_pool.py` against `https://api.groq.com/openai/v1/chat/completions`
with a placeholder key returned a real `401 Unauthorized` — the right shape of failure,
proving the URL/path/request format are correct). **Not verified:** an actual successful
completion from Groq, because obtaining a real free API key requires an interactive
Google/GitHub-OAuth or email-OTP signup step this session couldn't complete headlessly
(confirmed by navigating to `console.groq.com/keys`) without acting on the operator's
real identity/inbox, which was out of scope here. Get a free key (\~30s, no card) at
<https://console.groq.com/keys> and re-run `scripts/verify_provider_pool.py` to close
this gap.

## The `/query` HTTP service

`app.py` wires every piece above into ONE real web service (FastAPI + uvicorn — the
repo had no framework implied by its existing dependencies, so FastAPI was the
reasonable default per marketplace#33's own instructions), matching the "small HTTP
service, not just a CLI script" pattern `crates/harness-memory` already established
elsewhere in this project (`main.rs`/`lib.rs`: env-configured, `axum::serve`) and the
single-`app.py`-plus-Dockerfile shape of
`manifests/litellm-proof/heartbeat-proxy` in `CADS-agent-marketplace`.

```
                 rag/store.py (SQLite)          rag/memory.py (LanceDB)
                 document-corpus chunks          past (query, answer) pairs
                        │  search()                    │  recall() / remember()
                        ▼                               ▼
question ──► rag/embedder.py ──► POST /query (app.py) ──► rag/provider_pool.py
             (embed once,        1. embed the question       (local litellm-proxy,
              reused for both    2. retrieve top-k chunks      Groq fallback)
              store + memory)    3. recall relevant memory          │
                                 4. build a strictly-grounded        ▼
                                    prompt from the chunks    synthesized, cited answer
                                 5. call the LLM
                                 6. remember (question, answer)
```

**`POST /query`** — `{"question": "...", "top_k": 5, "memory_top_k": 3}` (both
optional). Retrieves the `top_k` most similar document-corpus chunks, recalls up to
`memory_top_k` semantically similar past interactions from the memory layer (surfaced
in the response for observability; deliberately *not* injected into the LLM prompt, so
a prior answer can never masquerade as a "fact present in the retrieved passages"),
asks the LLM to synthesize an answer, then remembers the new (question, answer) pair.
Returns `{question, answer, backend, model, sources: [...], memory: {...}}` — every
`sources[]` entry carries the real `source` id (e.g. `gutenberg:1661`), title, passage
index, similarity, and the actual retrieved text, so an answer's grounding can always
be checked by a human, not just trusted.

**Grounding is enforced at the prompt level**: the system prompt explicitly restricts
the model to facts stated in the numbered SOURCE PASSAGES, requires a `[n]` citation
after every factual claim, and forbids outside/background knowledge or invented facts
— see `_SYSTEM_PROMPT` in `app.py`. If no relevant chunks are retrieved at all, the
service returns a fixed "I don't have any retrieved passages relevant to this
question" answer *without ever calling the LLM* (nothing to hallucinate from).

**`GET /health`** — corpus chunk count, memory-interaction count, and which LLM
backends are currently configured (`{"local-litellm": true, "groq": false}` etc.).

### Real acceptance-criterion runs (verified by reading, not by response-code)

Ran the service — once directly with `.venv/bin/python app.py`, once inside a built
`docker build -t distributed-rag .` container with `./data` volume-mounted — against the
real 2167-chunk corpus (Gutendex + Open Library + Internet Archive) and a real
`local-devstral-small2` backend. Three real questions, read in full against their real
retrieved chunks, not just checked for a 200:

1. *"How does Sherlock Holmes deduce facts about a client from small physical
   details, according to Watson?"* (Gutendex path) → answer: *"...Holmes quickly noted
   that Mr. Jabez Wilson had done manual labor, took snuff, was a Freemason, had been
   in China, and had done a considerable amount of writing lately... [3]"* — `[3]` is
   `gutenberg:1661` passage 87: *"Beyond the obvious facts that he has at some time
   done manual labour, that he takes snuff, that he is a Freemason, that he has been
   in China, and that he has done a considerable amount of writing lately, I can
   deduce nothing else."* Every clause in the answer is a direct paraphrase of that
   one cited passage — genuinely grounded.
2. *"What is the opening line of Pride and Prejudice, and what does Mr. Darcy think
   of Elizabeth Bennet when they first meet at the ball?"* (Open Library path) →
   answer quotes the opening line verbatim, cited `[1]` = `openlibrary:OL66554W`
   passage 1, which literally contains that exact sentence — **and** the model
   correctly refused the second half ("the passages do not explicitly state his
   thoughts when they first meet... I cannot provide an answer to that part of the
   question based on the given passages") because none of the retrieved chunks
   covered that scene, even though it's a famous, easily-guessable line from the real
   book. This is the grounding constraint actually doing its job, not just the happy
   path.
3. *"What does Open Library say Pride and Prejudice is about, and who are its main
   characters?"* (Open Library path, run inside the Docker container) → every clause
   of the answer (1813, novel of manners, Jane Austen, Elizabeth Bennet as dynamic
   protagonist, the entail forcing the Bennet daughters to marry well) is a direct
   match against the single cited passage `[1]` = `openlibrary:OL66554W` passage 0,
   word for word.

`GET /health` after these runs: `{"corpus_chunks": 2167, "memory_interactions": 3, ...}`
— confirming `remember()` fired on every call, and a later call's `memory.recalled[]`
genuinely returned the earlier (question, answer) pairs by real cosine similarity
(0.71 for a genuinely related follow-up, 0.45 for an unrelated one).

Reproduce it yourself:

```bash
# Directly:
.venv/bin/python app.py            # binds 0.0.0.0:8080 by default (RAG_PORT to override)
curl -s localhost:8080/query -H 'content-type: application/json' \
  -d '{"question": "your question about ingested content"}'

# As a container (persists the corpus + memory + model cache under ./data):
docker build -t distributed-rag .
docker run -p 8080:8080 -v "$PWD/data:/app/data" --env-file .env distributed-rag
# or:
docker compose up --build
```

## What's explicitly NOT built yet

This proves the hardest, most load-bearing pieces (local embeddings + three real free
connectors + genuine, cross-connector, LLM-synthesized *and cited* retrieval) as one
real containerized HTTP service. Everything below is real, scoped, remaining work —
do not mistake this repo for a finished, packaged demo:

- **The Groq fallback path is unverified with a real key** — see above.
- **No marketplace manifest packaging.** Not yet signed, not yet installable/
  uninstallable via ct-agent's installer-engine, no `dev_sign`/`activate()`/verify-output
  cycle run against it. The issue's acceptance criterion's *second* half — install
  *and* uninstall via the real installer-engine pipeline with real verify output — is
  unmet; the first half (a real question returns a real, demonstrably grounded answer)
  is now met, see above.
- **No Rust port.** This bootstrap is Python (`fastembed`, FastAPI). If the final
  packaged manifest needs a single Rust binary (matching `harness-memory`'s
  convention), that's a rewrite, not a wrapper, and hasn't been started.
- **No auth on `/query`.** Fine for a same-host demo container; would need at least a
  bearer token before being reachable from outside a private network.
- **Partial test coverage.** `rag/memory.py` and `rag/provider_pool.py` have real
  unit tests (`tests/test_memory.py`, `tests/test_provider_pool.py`); the document
  corpus path (`rag/store.py`, `rag/gutendex.py`, `rag/embedder.py`) and `app.py`
  itself still don't — only manually verified via `scripts/ingest.py`/`scripts/query.py`
  and the real `/query` runs above.
- **No production-quality error handling, retries, or rate-limit handling** for the
  Gutendex/gutenberg.org calls.

## Repo layout

```
rag/
  embedder.py         — local BGE-small embedding wrapper (fastembed)
  gutendex.py         — Gutendex connector: fetch, boilerplate-strip, chunk (chunk_text
                         is shared by all three document connectors below)
  openlibrary.py      — Open Library connector: search + fetch descriptive/bibliographic text
  internetarchive.py  — Internet Archive connector: search + fetch full text (non-restricted items only)
  store.py            — SQLite BLOB-vector store + brute-force cosine search (document corpus)
  memory.py           — LanceDB interaction/fact memory: remember()/recall() (distinct from the corpus)
  provider_pool.py    — local litellm-proxy -> Groq free-fallback LLM router
scripts/
  ingest.py                 — CLI: fetch from gutenberg/openlibrary/archive by id, embed, store
  query.py                  — CLI: embed a query, retrieve top-k passages
  verify_provider_pool.py   — CLI: live-check whichever LLM backends are configured
tests/
  test_memory.py           — memory-layer unit tests (real embedder + real LanceDB, no mocking)
  test_provider_pool.py    — routing-logic unit tests (mocked, no network)
app.py             — FastAPI /query + /health HTTP service, wires all of the above together
Dockerfile         — packages app.py as a single container (python:3.12-slim)
docker-compose.yml — runs it as one Compose service, ./data volume-mounted for persistence
requirements.txt
.env.example   — LITELLM_*/GROQ_* config template for provider_pool.py
```

## License

Code in this repo: to be decided by the operator/project convention (no LICENSE file
yet — flagging this as an open item, not silently defaulting to one).

The embedding model (`BAAI/bge-small-en-v1.5`) is MIT-licensed — see above. Gutendex-
sourced book text is public domain in the U.S. (Project Gutenberg's own terms).
Open Library's catalog data is released under a CC0 public-domain dedication (see
[openlibrary.org/developers/api](https://openlibrary.org/developers/api)). The
Internet Archive item used here (`prideandprejudic42671gut`) is itself a Project
Gutenberg mirror, same public-domain terms as above; the connector deliberately
refuses any `access-restricted-item` (lending-library) item rather than assuming its
terms.
