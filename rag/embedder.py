"""Local embedding model wrapper.

Model: BAAI/bge-small-en-v1.5 (33M params, 384-dim output).
License: MIT -- verified directly on the model card at
https://huggingface.co/BAAI/bge-small-en-v1.5 on 2026-08-30 ("FlagEmbedding is licensed
under the MIT License ... The released models can be used for commercial purposes free
of charge."). No API key, no network call at inference time (only the one-time model
download, which is cached locally after the first run), no per-request cost -- satisfies
the demo's "really free, no costs under any circumstances" requirement.

Inference path: `fastembed` (Apache-2.0, https://github.com/qdrant/fastembed), which runs
the model as a quantized ONNX graph via onnxruntime. Chosen over `sentence-transformers`
(which pulls in full PyTorch, a much heavier dependency for a 33M-parameter model) and
over a hand-rolled Rust/candle path (more work than this bootstrap needed to prove the
approach; see README "Rust port" note for the intended production path mirroring
crates/harness-memory's language).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


class Embedder:
    """Thin wrapper around fastembed's TextEmbedding, loaded once and reused."""

    def __init__(self, model_name: str = MODEL_NAME, cache_dir: str | None = None) -> None:
        # cache_dir lets the HTTP service (app.py) point fastembed's one-time model
        # download at a persistent, volume-mounted path in the container instead of
        # fastembed's own default (a /tmp location, which is wiped on container
        # restart). Left as None (fastembed's own default) for every existing caller
        # (scripts/ingest.py, scripts/query.py, tests/test_memory.py) -- no behavior
        # change for them.
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed(self, texts: Iterable[str]) -> list[np.ndarray]:
        """Return one L2-normalized float32 vector per input text, same order as input."""
        texts = list(texts)
        if not texts:
            return []
        return list(self._model.embed(texts))

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
