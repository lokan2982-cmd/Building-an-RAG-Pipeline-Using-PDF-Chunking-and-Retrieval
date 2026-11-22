"""
test_live_ollama.py
===================
LIVE end-to-end test against the real local Ollama server (nomic-embed-text +
llama3.2), the real FAISS index and the real PDFs in data/pdfs/.

Asks six questions whose answers ARE present in the uploaded PDFs and checks:
  * the top-3 retrieved chunks carry source + page metadata,
  * a grounded answer is returned (found=True),
  * the fallback sentence is NOT shown alongside the answer,
  * the answer contains the fact stated in the PDF,
  * verified source/page citations are attached.
It also asks one question that is NOT in the PDFs and checks the fallback is shown.

Skipped automatically when Ollama is not running. Writes the real results to
outputs/answer_validation_results.txt.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.embeddings import is_model_installed, list_installed_ollama_models  # noqa: E402
from src.rag_pipeline import (  # noqa: E402
    NOT_FOUND_MESSAGE,
    answer_question,
    build_pipeline,
    get_llm,
)

try:
    _installed = list_installed_ollama_models(timeout=2.0)
    OLLAMA_READY = is_model_installed("llama3.2", _installed) and is_model_installed("nomic-embed-text", _installed)
except ConnectionError:
    OLLAMA_READY = False

pytestmark = pytest.mark.skipif(not OLLAMA_READY, reason="Local Ollama / required models not available")

# (question, keyword(s) that the PDF text states and a correct answer must contain)
PRESENT_QUESTIONS = [
    ("Who coined the term artificial intelligence and in what year?", ["mccarthy", "1956"]),
    ("What is the difference between narrow AI and general AI?", ["specific task", "any intellectual task"]),
    ("Who coined the term machine learning?", ["samuel"]),
    ("What are activation functions in neural networks?", ["relu"]),
    ("What techniques help prevent overfitting?", ["dropout"]),
    ("What does FAISS do?", ["nearest"]),  # PDF: "enables fast nearest-neighbour search"
]


@pytest.fixture(scope="module")
def live():
    vs, chunks, stats = build_pipeline(str(ROOT / "data" / "pdfs"), str(ROOT / "vectorstore"))
    return vs, get_llm()


@pytest.fixture(scope="module")
def report():
    lines = ["LIVE ANSWER-VALIDATION RESULTS (real Ollama llama3.2 + nomic-embed-text + FAISS)", "=" * 72]
    yield lines
    out = ROOT / "outputs" / "answer_validation_results.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


@pytest.mark.parametrize("question,keywords", PRESENT_QUESTIONS)
def test_question_present_in_pdfs_gets_grounded_answer(live, report, question, keywords):
    vs, llm = live
    shown = []
    r = answer_question(question, vs, top_k=3, llm=llm, on_retrieved=lambda res: shown.append(res))

    report.append(f"\nQ: {question}")
    for i, (doc, score) in enumerate(r["retrieved_chunks"], 1):
        report.append(f"  chunk[{i}] {doc.metadata['source']} page={doc.metadata['page']} "
                      f"L2={score:.4f} :: {doc.page_content[:80]!r}".replace("\n", " "))
    report.append(f"  raw LLM output : {r['raw_answer']!r}")
    report.append(f"  status={r['status']} found={r['found']} grounding_ratio={r['grounding_ratio']}")
    report.append("  FINAL ANSWER   : " + r["answer"].replace("\n", " | "))

    # top-3 chunks were displayed (callback) before the answer, with metadata
    assert len(shown) == 1 and len(shown[0]) == 3
    for doc, _ in shown[0]:
        assert doc.metadata["source"].endswith(".pdf")
        assert isinstance(doc.metadata["page"], int) and doc.metadata["page"] >= 1

    assert r["found"] is True, f"status={r['status']} raw={r['raw_answer']!r}"
    assert NOT_FOUND_MESSAGE.lower() not in r["answer"].lower()
    answer_lower = r["answer"].lower()
    assert all(k in answer_lower for k in keywords), f"missing {keywords} in {r['answer']!r}"
    assert "Sources:" in r["answer"] and "Page" in r["answer"]
    assert r["supporting_sources"], "no supporting source chunks attached"


def test_question_absent_from_pdfs_gets_fallback(live, report):
    vs, llm = live
    r = answer_question("What is quantum computing?", vs, top_k=3, llm=llm)
    report.append("\nQ: What is quantum computing?   (NOT in the PDFs)")
    report.append(f"  raw LLM output : {r['raw_answer']!r}")
    report.append(f"  status={r['status']} found={r['found']}")
    report.append("  FINAL ANSWER   : " + r["answer"])
    assert r["found"] is False
    assert r["answer"] == NOT_FOUND_MESSAGE
