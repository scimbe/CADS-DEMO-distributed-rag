"""Unit test for rag/memory.py -- the long-term interaction/fact memory layer
(top-K semantic recall over past query/answer pairs, distinct from the document
corpus in rag/store.py).

Uses the REAL local embedder (BAAI/bge-small-en-v1.5 via fastembed) and a REAL
embedded LanceDB table on a temp dir -- no mocking of the embedding or the vector
search. A mocked embedder would prove nothing about whether recall actually works;
the whole point of this module is real semantic retrieval.

Same discipline as the corpus retrieval check described in the README ("retrieved
text was read and confirmed to genuinely relate to the query"): this test doesn't
just assert a similarity score crossed some threshold -- it reads back the recalled
query/answer text and asserts it is *actually* the on-topic interaction among several
genuinely distinct distractors, by identity (id / substring), not by score alone.

Run: .venv/bin/python -m pytest tests/test_memory.py -v
  (or: .venv/bin/python -m unittest tests.test_memory -v)
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.embedder import Embedder
from rag.memory import Interaction, MemoryStore

# One shared, real embedder instance for the whole test module -- loading the model
# is the slow part (first run downloads + caches it); reusing it across test methods
# keeps the suite fast without changing what's being tested.
_EMBEDDER = Embedder()

# Several genuinely distinct past interactions, covering unrelated topics, so a
# correct top-1 match is a real signal rather than a coin flip between near-duplicates.
_DISTRACTORS = [
    Interaction(
        query="What's a good base recipe for a simple tomato pasta sauce?",
        answer="Saute garlic in olive oil, add crushed tomatoes, simmer 20 minutes, "
        "season with salt, basil, and a pinch of sugar to cut the acidity.",
    ),
    Interaction(
        query="When did the French Revolution begin?",
        answer="The French Revolution began in 1789, with the storming of the "
        "Bastille on 14 July.",
    ),
    Interaction(
        query="What's the difference between Python's asyncio.gather and asyncio.wait?",
        answer="asyncio.gather runs coroutines concurrently and returns results in "
        "input order (or raises on the first exception by default); asyncio.wait "
        "returns sets of done/pending tasks and gives more control over cancellation "
        "and timeouts, but doesn't preserve result order automatically.",
    ),
    Interaction(
        query="How tall is Mount Kilimanjaro?",
        answer="Mount Kilimanjaro's summit, Uhuru Peak, is about 5,895 metres "
        "(19,341 feet) above sea level, the highest peak in Africa.",
    ),
]

_TARGET = Interaction(
    query="How does Sherlock Holmes figure out details about a client just by observing them?",
    answer="Holmes reasons from small, easily-overlooked physical details -- callouses, "
    "mud splashes, wear patterns on clothing, posture -- to infer a person's "
    "occupation and recent history, then states the inference as if it were obvious.",
)


class MemoryStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="rag-memory-test-")
        self.store = MemoryStore(self._tmpdir, _EMBEDDER)

    def test_recall_returns_empty_list_on_empty_store(self) -> None:
        self.assertEqual(self.store.recall("anything", top_k=3), [])
        self.assertEqual(self.store.count(), 0)

    def test_remember_then_recall_finds_the_genuinely_relevant_interaction(self) -> None:
        for interaction in _DISTRACTORS:
            self.store.remember(interaction)
        target_id = self.store.remember(_TARGET)
        self.assertEqual(self.store.count(), len(_DISTRACTORS) + 1)

        results = self.store.recall(
            "What method does Sherlock Holmes use to deduce facts about a stranger "
            "from small details?",
            top_k=3,
        )

        self.assertGreaterEqual(len(results), 1)
        top = results[0]

        # Human-verified, not just "the score was highest": actually read back the
        # recalled text and confirm it is the target interaction, by identity --
        # not a distractor that happened to score close.
        self.assertEqual(top.id, target_id)
        self.assertEqual(top.query, _TARGET.query)
        self.assertIn("Holmes", top.answer)
        self.assertIn("infer", top.answer)

        # None of the unrelated distractor topics (pasta, French Revolution,
        # asyncio, Kilimanjaro) leaked into the top spot.
        for distractor in _DISTRACTORS:
            self.assertNotEqual(top.answer, distractor.answer)

        # Similarity is a real cosine similarity in a sane range, and the target
        # is scored strictly higher than every distractor also returned.
        self.assertGreater(top.similarity, 0.0)
        self.assertLessEqual(top.similarity, 1.0 + 1e-6)
        for other in results[1:]:
            self.assertLess(other.similarity, top.similarity)

    def test_recall_ranks_multiple_similar_interactions_above_unrelated_ones(self) -> None:
        for interaction in _DISTRACTORS:
            self.store.remember(interaction)
        self.store.remember(_TARGET)
        self.store.remember(
            Interaction(
                query="What technique did Sherlock Holmes use to read a visitor's "
                "recent past from their appearance?",
                answer="He examined details like calloused hands, sunburn patterns, "
                "or scuffed shoes and reasoned backward to the person's likely job "
                "or recent travels.",
            )
        )

        results = self.store.recall("Explain Sherlock Holmes's method of deduction", top_k=2)
        self.assertEqual(len(results), 2)
        top_answers = {r.answer for r in results}
        self.assertTrue(
            any("Holmes" in a and "infer" in a for a in top_answers)
            or any("Holmes" in a and "reasoned" in a for a in top_answers)
        )
        # Both top-2 slots went to the two Holmes-related interactions, not a
        # distractor -- confirms top-K ranking, not just top-1 luck. Checked on the
        # query text (both Holmes interactions' queries name him explicitly; one
        # answer paraphrases without repeating the name, which is expected -- the
        # match is semantic, not a keyword match).
        for r in results:
            self.assertIn("Holmes", r.query)


if __name__ == "__main__":
    unittest.main()
