"""Programmatic grounding verification for /query's answers (marketplace#40).

`app.py`'s system prompt *asks* the LLM to only state facts present in the
retrieved passages and to cite them -- but that is enforcement by instruction,
not proof. This module independently checks, after the answer already exists,
whether each cited claim is actually supported by the passage(s) it cites, using
a mechanism that is never the same call that produced the answer -- so it can't
inherit the same hallucination it exists to catch.

## Technique: cheap lexical recall first, a narrow LLM entailment call only when
## the lexical signal is ambiguous

Two techniques were considered up front:

1. **Deterministic lexical/n-gram overlap** (free, instant, fully explainable,
   no extra network call): score how many of a claim's content words actually
   appear in its cited passage(s). Cheap and has no failure mode of its own, but
   a real LLM answer routinely *paraphrases* its source ("he inferred it from
   the mud on her boots" vs. the passage's own wording of the same fact) --
   scoring paraphrases by raw word overlap alone produces false negatives.
2. **LLM-based entailment**: a second, narrowly-scoped chat call --
   "does this PASSAGE support this CLAIM, yes/no/partial" -- with the passage
   and the claim as the ONLY inputs (never the original question, never the
   original system prompt, never asked to re-answer anything). Handles
   paraphrase correctly, but costs a network round trip and is itself an LLM
   judging another LLM's output, so it should never be the *only* check, and
   should never be given enough context to reproduce the original mistake.

This module uses a **hybrid** of the two, in that order. Every cited claim first
gets a fast, deterministic lexical-recall score. A clearly-high score is accepted
as supported, and a clearly-low score is rejected as unsupported, without ever
calling an LLM -- both ends of that range are unambiguous enough that a second
opinion would only cost time. Only the genuinely ambiguous middle band
(paraphrase-shaped: some, but not most, of the claim's words show up verbatim)
falls through to the narrow LLM entailment call. This keeps the common cases
free and instant, spends the LLM call only where it earns its keep, and keeps
that call's inputs narrow enough that it cannot repeat the primary answer's own
mistake.

## Scope

Only *cited* sentences (containing at least one `[n]` marker) are verified --
this module checks whether a citation actually backs up what it's attached to,
which is the specific gap marketplace#40 calls out. An uncited sentence (e.g. a
plain refusal, or connective prose) is reported with verdict "uncited" and is
excluded from the `all_supported` rollup: whether the *system prompt's*
citation requirement itself was followed for every factual sentence is a
different, broader hallucination-detection problem than "is this citation
correct," and isn't what this module claims to check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.provider_pool import ChatResult, ProviderPoolError, chat

_CITATION_RE = re.compile(r"\[(\d+)\]")
_WORD_RE = re.compile(r"[a-z0-9]+")

# Sentence boundary: a `.`/`!`/`?` followed by whitespace and then something that
# looks like the start of a new sentence (a capital letter, a digit, or a citation
# bracket) -- avoids splitting mid-sentence on things like "Mr. Holmes" (lowercase
# continuation) or "3.5 million" (digit immediately after the period, no space).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])")

# A short function-word stoplist -- content words are what should genuinely
# transfer from a passage into a faithful paraphrase; function words are common
# enough in any two unrelated sentences that counting them inflates every score.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "because", "of", "in",
    "on", "at", "to", "for", "with", "from", "by", "is", "are", "was", "were", "be",
    "been", "being", "it", "its", "this", "that", "these", "those", "he", "she",
    "they", "him", "her", "them", "his", "hers", "their", "as", "not", "no", "do",
    "does", "did", "has", "have", "had", "can", "could", "would", "should", "will",
    "shall", "may", "might", "than", "which", "who", "whom", "whose", "what",
    "when", "where", "why", "how", "also", "into", "about", "over", "under",
    "after", "before", "between", "up", "down", "out", "off", "again", "further",
    "there", "here", "i", "you", "we", "our", "your", "my", "me", "us", "s", "t",
}

# Lexical recall >= this -> accepted as supported without an LLM call.
LEXICAL_SUPPORTED_THRESHOLD = 0.72
# Lexical recall <= this -> rejected as not supported without an LLM call.
LEXICAL_UNSUPPORTED_THRESHOLD = 0.35

# Confidence assigned to each possible LLM verdict word.
_LLM_CONFIDENCE = {"yes": 0.9, "partial": 0.55, "no": 0.1}

_VERIFY_SYSTEM_PROMPT = (
    "You are a strict fact-checking classifier. You will be given a PASSAGE and a "
    "CLAIM. Decide whether the PASSAGE supports the CLAIM. Respond with exactly one "
    "word and nothing else: "
    "'yes' if the passage fully supports the claim, "
    "'partial' if the passage supports only part of the claim or supports it with a "
    "meaningfully different nuance, or "
    "'no' if the passage does not support the claim, or the claim contradicts the "
    "passage, or the passage is unrelated to the claim. "
    "Do not answer any question in the CLAIM. Do not explain your reasoning. Do not "
    "add any other text."
)


def _content_words(text: str) -> list[str]:
    text = _CITATION_RE.sub(" ", text)
    return [w for w in _WORD_RE.findall(text.lower()) if len(w) > 1 and w not in _STOPWORDS]


def lexical_recall(claim_text: str, passage_text: str) -> float:
    """Fraction of `claim_text`'s content words that also appear in `passage_text`.

    A recall score, not a symmetric similarity -- what matters for grounding is
    whether the claim's own words are backed by the passage, not how much of the
    (usually much longer) passage the claim happens to use.
    """
    claim_words = _content_words(claim_text)
    if not claim_words:
        # Nothing but function words/citation markers -- no content to contradict.
        return 1.0
    passage_words = set(_content_words(passage_text))
    hits = sum(1 for w in claim_words if w in passage_words)
    return hits / len(claim_words)


@dataclass
class Claim:
    text: str
    citations: list[int] = field(default_factory=list)


def split_claims(answer: str) -> list[Claim]:
    """Split an answer into sentence-level claims, each carrying the citation
    numbers (`[n]`) that appear in it, in first-seen order with duplicates removed.
    """
    answer = answer.strip()
    if not answer:
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(answer) if s.strip()]
    claims = []
    for s in sentences:
        seen: list[int] = []
        for n in _CITATION_RE.findall(s):
            num = int(n)
            if num not in seen:
                seen.append(num)
        claims.append(Claim(text=s, citations=seen))
    return claims


@dataclass
class ClaimVerification:
    text: str
    citations: list[int]
    verdict: str  # "supported" | "partially_supported" | "not_supported" | "uncited"
    confidence: float | None
    method: str  # "lexical" | "llm" | "lexical_fallback" | "uncited"


def _llm_verdict(claim_text: str, passage_text: str, *, chat_fn) -> tuple[str, float, str] | None:
    """Ask a second, narrowly-scoped LLM call whether `passage_text` supports
    `claim_text`. Returns None if the call failed or its output didn't parse as
    one of the expected verdict words -- callers should fall back to the lexical
    signal rather than trust a call that didn't behave as instructed.
    """
    messages = [
        {"role": "system", "content": _VERIFY_SYSTEM_PROMPT},
        {"role": "user", "content": f"PASSAGE:\n{passage_text}\n\nCLAIM:\n{claim_text}"},
    ]
    try:
        result: ChatResult = chat_fn(messages, temperature=0, max_tokens=5)
    except ProviderPoolError:
        return None
    raw = result.content.strip().lower()
    first_word = re.sub(r"[^a-z]", "", raw.split()[0]) if raw.split() else ""
    if first_word.startswith("yes"):
        return "supported", _LLM_CONFIDENCE["yes"], "llm"
    if first_word.startswith("partial"):
        return "partially_supported", _LLM_CONFIDENCE["partial"], "llm"
    if first_word.startswith("no"):
        return "not_supported", _LLM_CONFIDENCE["no"], "llm"
    return None


def verify_claim(claim: Claim, passages_by_citation: dict[int, str], *, chat_fn=chat) -> ClaimVerification:
    cited_texts = [passages_by_citation[c] for c in claim.citations if c in passages_by_citation]
    if not cited_texts:
        return ClaimVerification(claim.text, claim.citations, "uncited", None, "uncited")

    # Citations are combined into one evidence pool before scoring: a claim citing
    # [1][3] may genuinely need words from both passages together, and checking it
    # against the union is the correct question ("do the cited passages, together,
    # support this claim?"), not against each passage in isolation.
    combined_passage = "\n\n".join(cited_texts)
    recall = lexical_recall(claim.text, combined_passage)

    if recall >= LEXICAL_SUPPORTED_THRESHOLD:
        return ClaimVerification(claim.text, claim.citations, "supported", round(recall, 3), "lexical")
    if recall <= LEXICAL_UNSUPPORTED_THRESHOLD:
        return ClaimVerification(claim.text, claim.citations, "not_supported", round(recall, 3), "lexical")

    llm_result = _llm_verdict(claim.text, combined_passage, chat_fn=chat_fn)
    if llm_result is None:
        verdict = "partially_supported" if recall >= 0.5 else "not_supported"
        return ClaimVerification(claim.text, claim.citations, verdict, round(recall, 3), "lexical_fallback")
    verdict, confidence, method = llm_result
    return ClaimVerification(claim.text, claim.citations, verdict, confidence, method)


def verify_answer(answer: str, passages_by_citation: dict[int, str], *, chat_fn=chat) -> list[ClaimVerification]:
    """Verify every cited claim in `answer` against `passages_by_citation`
    (citation number -> passage text, matching the `[n]` markers the system
    prompt asks the LLM to use)."""
    return [verify_claim(c, passages_by_citation, chat_fn=chat_fn) for c in split_claims(answer)]
