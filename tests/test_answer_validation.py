"""
test_answer_validation.py
=========================
Tests for the answer-validation / fallback logic in src/rag_pipeline.py.

Regression coverage for the bug where a good answer was shown TOGETHER with
"The answer was not found in the uploaded documents.", where empty/citation-only
model output reached the user, and where general-knowledge answers were not
rejected.  These tests exercise the real `validate_answer` / `answer_question`
code with synthetic chunks and a scripted stand-in for the chat model.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag_pipeline import (  # noqa: E402
    NOT_FOUND_MESSAGE,
    answer_question,
    find_supporting_chunks,
    validate_answer,
)

FALLBACK_LOWER = "not found in the uploaded documents"


@pytest.fixture
def results():
    return [
        (Document(
            page_content=("Artificial intelligence (AI) refers to the simulation of human intelligence "
                          "processes by computer systems. AI was founded as an academic discipline in "
                          "1956 at the Dartmouth Conference, where John McCarthy coined the term."),
            metadata={"source": "01_intro_to_ai.pdf", "page": 1}), 0.34),
        (Document(
            page_content=("Backpropagation: The algorithm used to calculate gradients and update weights. "
                          "Gradient Descent minimises the loss function."),
            metadata={"source": "03_deep_learning_rag.pdf", "page": 1}), 0.71),
    ]


def test_grounded_answer_with_appended_fallback_keeps_answer(results):
    """THE reported bug: real answer followed by the fallback sentence."""
    raw = ("John McCarthy coined the term Artificial Intelligence in 1956 at the Dartmouth Conference.\n\n"
           "The answer was not found in the uploaded documents.")
    v = validate_answer(raw, results)
    assert v["found"] is True
    assert v["status"] == "grounded_fallback_stripped"
    assert "McCarthy" in v["answer"] and "1956" in v["answer"]
    assert FALLBACK_LOWER not in v["answer"].lower()
    assert "01_intro_to_ai.pdf, Page 1" in v["answer"]


def test_sentinel_appended_to_answer_is_stripped(results):
    raw = "Backpropagation calculates gradients and updates weights. NOT_FOUND"
    v = validate_answer(raw, results)
    assert v["found"] is True
    assert "NOT_FOUND" not in v["answer"]
    assert "03_deep_learning_rag.pdf, Page 1" in v["answer"]


@pytest.mark.parametrize("raw", [
    "NOT_FOUND",
    "NOTFOUND",
    "Not Found",
    "The answer was not found in the uploaded documents.",
    "the answer was not found in the uploaded documents",
    "NOT_FOUND.",
])
def test_model_refusal_returns_exact_fallback(raw, results):
    v = validate_answer(raw, results)
    assert v["found"] is False
    assert v["status"] == "model_not_found"
    assert v["answer"] == NOT_FOUND_MESSAGE
    assert v["supporting_sources"] == []


@pytest.mark.parametrize("raw", ["", "   ", "[Source: 03_deep_learning_rag.pdf, Page 2]",
                                 "Sources: 01_intro_to_ai.pdf, Page 1"])
def test_empty_or_citation_only_output_is_flagged(raw, results):
    v = validate_answer(raw, results)
    assert v["found"] is False
    assert v["status"] == "empty_output"
    assert v["answer"] == NOT_FOUND_MESSAGE


def test_general_knowledge_answer_is_rejected(results):
    """An answer unsupported by the retrieved PDF chunks must not be shown."""
    raw = "Quantum computers use qubits that can exist in superposition, enabling parallel computation."
    v = validate_answer(raw, results)
    assert v["found"] is False
    assert v["status"] == "ungrounded_rejected"
    assert v["answer"] == NOT_FOUND_MESSAGE
    assert v["grounding_ratio"] < 0.5


def test_model_citation_is_replaced_by_verified_citation(results):
    """A wrong page cited by the model is replaced by the chunk that supports the answer."""
    raw = "John McCarthy coined the term in 1956. Sources: 03_deep_learning_rag.pdf, Page 9"
    v = validate_answer(raw, results)
    assert v["found"] is True
    assert "Page 9" not in v["answer"]
    assert v["supporting_sources"][0]["source"] == "01_intro_to_ai.pdf"
    assert v["supporting_sources"][0]["page"] == 1


def test_weakly_related_chunk_is_not_cited(results):
    """Only chunks that really back the answer are cited (not every retrieved chunk)."""
    v = validate_answer("Backpropagation calculates gradients and updates weights.", results)
    assert [s["source"] for s in v["supporting_sources"]] == ["03_deep_learning_rag.pdf"]


def test_find_supporting_chunks_orders_by_support(results):
    sup = find_supporting_chunks("backpropagation gradients weights", results)
    assert sup[0][0].metadata["source"] == "03_deep_learning_rag.pdf"


# --- answer_question integration with a scripted chat model ------------------

class ScriptedLLM:
    """Stand-in for ChatOllama that returns scripted replies and records calls."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        reply = MagicMock()
        reply.content = self.replies.pop(0)
        return reply


def _vs(results):
    vs = MagicMock()
    vs.similarity_search_with_score.return_value = results
    return vs


def test_chunks_are_shown_before_llm_is_called(results):
    order = []
    llm = ScriptedLLM(["John McCarthy coined the term in 1956 at the Dartmouth Conference."])
    original = llm.invoke
    llm.invoke = lambda m: (order.append("llm"), original(m))[1]
    out = answer_question("Who coined AI?", _vs(results), top_k=2, llm=llm,
                          on_retrieved=lambda r: order.append(f"chunks:{len(r)}"))
    assert order == ["chunks:2", "llm"]
    assert out["found"] is True


def test_empty_output_is_retried_once_then_succeeds(results):
    llm = ScriptedLLM(["[Source: 01_intro_to_ai.pdf, Page 1]",
                       "AI was founded as an academic discipline in 1956 at the Dartmouth Conference."])
    out = answer_question("When was AI founded?", _vs(results), top_k=2, llm=llm)
    assert llm.calls == 2
    assert out["found"] is True and "1956" in out["answer"]


def test_empty_output_after_retry_gives_fallback(results):
    llm = ScriptedLLM(["", ""])
    out = answer_question("When was AI founded?", _vs(results), top_k=2, llm=llm)
    assert llm.calls == 2
    assert out["found"] is False
    assert out["answer"] == NOT_FOUND_MESSAGE
    assert out["status"] == "empty_output_after_retry"


def test_page_metadata_survives_to_result(results):
    llm = ScriptedLLM(["Backpropagation calculates gradients and updates weights."])
    out = answer_question("What is backpropagation?", _vs(results), top_k=2, llm=llm)
    pages = {(s["source"], s["page"]) for s in out["sources"]}
    assert ("01_intro_to_ai.pdf", 1) in pages and ("03_deep_learning_rag.pdf", 1) in pages
    assert "[Source 1: 01_intro_to_ai.pdf, Page 1]" in out["context"]


def test_answer_never_contains_fallback_when_found(results):
    llm = ScriptedLLM(["John McCarthy coined the term in 1956. The answer was not found in the uploaded documents."])
    out = answer_question("Who coined AI?", _vs(results), top_k=2, llm=llm)
    assert out["found"] is True
    assert FALLBACK_LOWER not in out["answer"].lower()
