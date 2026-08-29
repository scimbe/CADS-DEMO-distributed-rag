# CADS-DEMO-distributed-rag

**Status: bootstrap / proof-of-concept. Not a finished demo. Not yet packaged as a
marketplace manifest. Read "What's NOT built yet" below before assuming this is done.**

Part of the bunsenbrenner.org demo portfolio
([marketplace#33](https://github.com/scimbe/CADS-agent-marketplace/issues/33)). The
target audience is devs/power-users; the point is to show real platform maturity — a
distributed-harness-style memory/context system, not just a scripted demo.

This bootstrap proves the hardest constraint from the issue first: a **fully local
embedding pipeline** (no API key, no external service, no per-request cost) feeding a
**real, free, key-less connector** to a public-domain book source, with **genuine
retrieval** verified by a human reading the retrieved text.

## The "really free, no costs under any circumstances" requirement

Every component that ingestion/retrieval depends on for this bootstrap runs locally or
against a free, key-less public API:

- **Embedding model**: runs on this machine, once downloaded and cached — no API key,
  no per-call billing, works fully offline after the first run.
- **Connector**: [Gutendex](https://gutendex.com) + gutenberg.org — free, public-domain
  book metadata and text, no authentication, no rate-limit-behind-a-paywall.
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
Gutendex (free, key-less API)
        │  fetch book text by Gutenberg id
        ▼
rag/gutendex.py  — strip Project Gutenberg boilerplate, chunk into ~800-char
                    paragraph-aligned passages with overlap
        │
        ▼
rag/embedder.py  — BAAI/bge-small-en-v1.5 via fastembed (local, ONNX, no network
                    call at inference time)
        │  384-dim float32 vector per passage
        ▼
rag/store.py     — SQLite, one row per passage, embedding stored as a packed
                    float32 BLOB, brute-force cosine search over all rows
```

This mirrors `CADS-agent-marketplace`'s own `crates/harness-memory/src/db.rs` pattern:
plain SQLite + a BLOB vector column + brute-force cosine, not a vector-DB dependency —
that crate's own reasoning applies here too (sub-millisecond to low-millisecond search
at the scale of one ingested book's worth of passages; this is demo-scale, not
web-scale).

The issue's own architecture additionally describes a separate **memory layer**
(ChromaDB/LanceDB, for cross-session interaction memory, distinct from this document
corpus) and a **dynamic free-provider LLM pool** (OpenRouter/Groq/Cerebras/Cloudflare
Workers AI, as resilience/fallback alongside the team's own `local-devstral-small2`).
Neither is built in this bootstrap — see below.

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

Reproduce it yourself:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/ingest.py 1661          # fetches + embeds + stores one real book
.venv/bin/python scripts/query.py "Sherlock Holmes deduces something about a client from small details"
```

## What's explicitly NOT built yet

This is a bootstrap that proves the two hardest, most load-bearing pieces (local
embeddings + a real free connector + genuine retrieval). Everything below is real,
scoped, remaining work — do not mistake this repo for a finished demo:

- **Only one connector is implemented.** Open Library and Internet Archive (the other
  two connectors named in the issue) are not built. Only Gutendex/Project Gutenberg.
- **No cross-session memory layer.** The issue's separate "context-optimization"
  memory system (ChromaDB or LanceDB, top-K retrieval over past interactions, distinct
  from this document corpus) is not built at all.
- **No dynamic free-provider LLM pool.** The issue's router that falls back from the
  team's local model to a free external provider (OpenRouter/Groq/Cerebras/Cloudflare
  Workers AI, from the `free-ai-tools` list) on rate-limit/failure is not built.
- **No conversational/answering layer.** This retrieves relevant passages; it does not
  yet feed them to any LLM to produce a synthesized natural-language answer.
- **No marketplace manifest packaging.** Not yet signed, not yet installable/
  uninstallable via ct-agent's installer-engine, no `dev_sign`/`activate()`/verify-output
  cycle run against it. The issue's acceptance criterion — install *and* uninstall via
  the real installer-engine pipeline with real verify output — is unmet.
- **No Rust port.** This bootstrap is Python (`fastembed`). If the final packaged
  manifest needs a single Rust binary (matching `harness-memory`'s convention), that's
  a rewrite, not a wrapper, and hasn't been started.
- **No containerization.** No Dockerfile, no Compose file yet, despite the issue
  suggesting `InstallerKind::Compose` or `Binary` as the eventual packaging shape.
- **Minimal test coverage.** No automated tests yet (the harness-memory crate this
  mirrors has real unit tests over its cosine search; this bootstrap doesn't yet).
- **No production-quality error handling, retries, or rate-limit handling** for the
  Gutendex/gutenberg.org calls.

## Repo layout

```
rag/
  embedder.py   — local BGE-small embedding wrapper (fastembed)
  gutendex.py   — Gutendex connector: fetch, boilerplate-strip, chunk
  store.py      — SQLite BLOB-vector store + brute-force cosine search
scripts/
  ingest.py     — CLI: fetch a Gutenberg book by id, embed, store
  query.py      — CLI: embed a query, retrieve top-k passages
requirements.txt
```

## License

Code in this repo: to be decided by the operator/project convention (no LICENSE file
yet — flagging this as an open item, not silently defaulting to one).

The embedding model (`BAAI/bge-small-en-v1.5`) is MIT-licensed — see above. Gutendex-
sourced book text is public domain in the U.S. (Project Gutenberg's own terms).
