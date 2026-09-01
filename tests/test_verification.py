"""Unit tests for rag/verification.py -- the programmatic grounding-verification
layer (marketplace#40).

The lexical-recall tests are fully real/deterministic -- no network, no LLM,
nothing mocked, matching tests/test_memory.py's discipline for the parts of the
system that don't need it. The LLM-entailment path is exercised the same way
tests/test_provider_pool.py exercises provider_pool.chat() -- injected via
verify_claim/verify_answer's `chat_fn` parameter, so no real network call is made
and no real provider needs to be configured for the suite to run.

Run: .venv/bin/python -m pytest tests/test_verification.py -v
  (or: .venv/bin/python -m unittest tests.test_verification -v)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.provider_pool import ChatResult, ProviderPoolError
from rag.verification import Claim, lexical_recall, split_claims, verify_answer, verify_claim


def _llm_stub(verdict_word: str):
    """A fake `chat_fn` that returns exactly the one-word verdict a real LLM
    backend is instructed to return, without touching the network."""

    def _fn(messages, **kwargs):
        return ChatResult(content=verdict_word, backend="stub", model="stub", raw={})

    return _fn


def _raising_stub(messages, **kwargs):
    raise ProviderPoolError("no backend configured")


class SplitClaimsTest(unittest.TestCase):
    def test_splits_sentences_and_extracts_citations_in_order(self) -> None:
        answer = (
            "Holmes deduced the visitor's occupation from calluses on her hand [1]. "
            "He also noted mud on her boots, which pointed to a specific district [2][3]. "
            "The passages do not say what she was carrying."
        )
        claims = split_claims(answer)
        self.assertEqual(len(claims), 3)
        self.assertEqual(claims[0].citations, [1])
        self.assertEqual(claims[1].citations, [2, 3])
        self.assertEqual(claims[2].citations, [])
        self.assertIn("calluses", claims[0].text)

    def test_empty_answer_yields_no_claims(self) -> None:
        self.assertEqual(split_claims(""), [])
        self.assertEqual(split_claims("   "), [])

    def test_duplicate_citations_in_one_sentence_are_deduplicated_and_ordered(self) -> None:
        claims = split_claims("A fact stated twice for emphasis [2][1][2].")
        self.assertEqual(claims[0].citations, [2, 1])


class LexicalRecallTest(unittest.TestCase):
    def test_high_overlap_scores_high(self) -> None:
        claim = "Holmes deduced her occupation from calluses on her hand."
        passage = "From the calluses on her hand, Holmes deduced her occupation at once."
        self.assertGreaterEqual(lexical_recall(claim, passage), 0.72)

    def test_unrelated_passage_scores_low(self) -> None:
        claim = "Holmes deduced her occupation from calluses on her hand."
        passage = "The French Revolution began in 1789 with the storming of the Bastille."
        self.assertLessEqual(lexical_recall(claim, passage), 0.35)

    def test_claim_with_no_content_words_scores_perfectly(self) -> None:
        # A citation marker alone, or a claim made entirely of stopwords, has
        # nothing in it that could be contradicted.
        self.assertEqual(lexical_recall("[1]", "anything at all"), 1.0)


class VerifyClaimTest(unittest.TestCase):
    def test_well_grounded_claim_is_supported_by_lexical_check_alone(self) -> None:
        claim = Claim(text="Holmes deduced her occupation from calluses on her hand [1].", citations=[1])
        passages = {1: "From the calluses on her hand, Holmes deduced her occupation at once."}
        result = verify_claim(claim, passages, chat_fn=_raising_stub)
        self.assertEqual(result.verdict, "supported")
        self.assertEqual(result.method, "lexical")
        self.assertGreater(result.confidence, 0.7)

    def test_deliberately_ungrounded_claim_is_caught_by_lexical_check_alone(self) -> None:
        claim = Claim(
            text="Holmes then traveled to Paris and met with the President of France [1].", citations=[1]
        )
        passages = {1: "From the calluses on her hand, Holmes deduced her occupation at once."}
        result = verify_claim(claim, passages, chat_fn=_raising_stub)
        self.assertEqual(result.verdict, "not_supported")
        self.assertEqual(result.method, "lexical")

    def test_paraphrase_in_the_ambiguous_band_falls_through_to_llm_yes(self) -> None:
        # Shares 2 of 5 content words with the passage ("holmes", "hand") -- recall
        # 0.4, inside the (0.35, 0.72) ambiguous band (confirmed explicitly in
        # test_paraphrase_in_the_ambiguous_band_falls_through_to_llm_yes below).
        claim = Claim(text="Holmes worked out her job by looking at her hand [1].", citations=[1])
        passages = {1: "From the calluses on her hand, Holmes deduced her occupation at once."}
        # Confirm this really lands in the ambiguous band before trusting the LLM path.
        recall = lexical_recall(claim.text, passages[1])
        self.assertTrue(0.35 < recall < 0.72, f"test fixture not actually ambiguous: recall={recall}")

        result = verify_claim(claim, passages, chat_fn=_llm_stub("yes"))
        self.assertEqual(result.verdict, "supported")
        self.assertEqual(result.method, "llm")

    def test_ambiguous_claim_llm_says_no(self) -> None:
        # Shares 2 of 5 content words with the passage ("holmes", "hand") -- recall
        # 0.4, inside the (0.35, 0.72) ambiguous band (confirmed explicitly in
        # test_paraphrase_in_the_ambiguous_band_falls_through_to_llm_yes below).
        claim = Claim(text="Holmes worked out her job by looking at her hand [1].", citations=[1])
        passages = {1: "From the calluses on her hand, Holmes deduced her occupation at once."}
        result = verify_claim(claim, passages, chat_fn=_llm_stub("no"))
        self.assertEqual(result.verdict, "not_supported")
        self.assertEqual(result.method, "llm")

    def test_ambiguous_claim_llm_says_partial(self) -> None:
        # Shares 2 of 5 content words with the passage ("holmes", "hand") -- recall
        # 0.4, inside the (0.35, 0.72) ambiguous band (confirmed explicitly in
        # test_paraphrase_in_the_ambiguous_band_falls_through_to_llm_yes below).
        claim = Claim(text="Holmes worked out her job by looking at her hand [1].", citations=[1])
        passages = {1: "From the calluses on her hand, Holmes deduced her occupation at once."}
        result = verify_claim(claim, passages, chat_fn=_llm_stub("partial"))
        self.assertEqual(result.verdict, "partially_supported")
        self.assertEqual(result.method, "llm")

    def test_unparsed_llm_output_falls_back_to_lexical_signal(self) -> None:
        # Shares 2 of 5 content words with the passage ("holmes", "hand") -- recall
        # 0.4, inside the (0.35, 0.72) ambiguous band (confirmed explicitly in
        # test_paraphrase_in_the_ambiguous_band_falls_through_to_llm_yes below).
        claim = Claim(text="Holmes worked out her job by looking at her hand [1].", citations=[1])
        passages = {1: "From the calluses on her hand, Holmes deduced her occupation at once."}
        result = verify_claim(claim, passages, chat_fn=_llm_stub("I am not sure, it depends"))
        self.assertEqual(result.method, "lexical_fallback")
        self.assertIn(result.verdict, {"partially_supported", "not_supported"})

    def test_failed_llm_call_falls_back_to_lexical_signal(self) -> None:
        # Shares 2 of 5 content words with the passage ("holmes", "hand") -- recall
        # 0.4, inside the (0.35, 0.72) ambiguous band (confirmed explicitly in
        # test_paraphrase_in_the_ambiguous_band_falls_through_to_llm_yes below).
        claim = Claim(text="Holmes worked out her job by looking at her hand [1].", citations=[1])
        passages = {1: "From the calluses on her hand, Holmes deduced her occupation at once."}
        result = verify_claim(claim, passages, chat_fn=_raising_stub)
        self.assertEqual(result.method, "lexical_fallback")

    def test_citation_pointing_nowhere_is_uncited(self) -> None:
        claim = Claim(text="A claim citing a source that wasn't retrieved [9].", citations=[9])
        result = verify_claim(claim, passages_by_citation={}, chat_fn=_raising_stub)
        self.assertEqual(result.verdict, "uncited")
        self.assertIsNone(result.confidence)

    def test_uncited_sentence_is_uncited(self) -> None:
        claim = Claim(text="This part restates the question without citing anything.", citations=[])
        result = verify_claim(claim, passages_by_citation={1: "irrelevant"}, chat_fn=_raising_stub)
        self.assertEqual(result.verdict, "uncited")

    def test_multi_citation_claim_is_checked_against_combined_evidence(self) -> None:
        # Neither passage alone contains both facts; together they do.
        claim = Claim(text="Her hand had calluses and her boots had mud [1][2].", citations=[1, 2])
        passages = {
            1: "There were distinct calluses visible on her right hand.",
            2: "Fresh mud was caked onto the heel and toe of her boots.",
        }
        result = verify_claim(claim, passages, chat_fn=_raising_stub)
        self.assertEqual(result.verdict, "supported")


class VerifyAnswerTest(unittest.TestCase):
    def test_verifies_every_claim_in_a_multi_sentence_answer(self) -> None:
        answer = (
            "Holmes deduced her occupation from calluses on her hand [1]. "
            "He then flew to the moon to confirm his theory [1]."
        )
        passages = {1: "From the calluses on her hand, Holmes deduced her occupation at once."}
        results = verify_answer(answer, passages, chat_fn=_raising_stub)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].verdict, "supported")
        self.assertEqual(results[1].verdict, "not_supported")


if __name__ == "__main__":
    unittest.main()
