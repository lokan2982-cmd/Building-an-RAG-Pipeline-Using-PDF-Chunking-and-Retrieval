"""
test_faiss_real.py
==================
Tests exercising REAL FAISS indexing, disk persistence, reload, retrieval,
metadata preservation, and smart rebuild logic without requiring network calls.

Uses a deterministic local test embedder (no paid APIs) for fast unit-test
execution, plus graceful Ollama error-path testing.
"""

import hashlib
import re
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag_pipeline import NOT_FOUND_MESSAGE, answer_question  # noqa: E402
from src.retriever import retrieve_relevant_chunks  # noqa: E402
from src.vector_store import (  # noqa: E402
    create_vectorstore,
    get_or_create_vectorstore,
    load_vectorstore,
    source_changed,
)


class DeterministicTestEmbeddings(Embeddings):
    """Lightweight deterministic bag-of-words embeddings used only inside tests."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


@pytest.fixture
def docs():
    return [
        Document(
            page_content="FAISS performs fast similarity search over dense vectors.",
            metadata={"source": "a.pdf", "page": 3},
        ),
        Document(
            page_content="Photosynthesis converts sunlight into chemical energy in plants.",
            metadata={"source": "b.pdf", "page": 7},
        ),
        Document(
            page_content="Neural networks learn by backpropagation and gradient descent.",
            metadata={"source": "c.pdf", "page": 2},
        ),
    ]


@pytest.fixture
def pdf_dir(tmp_path):
    d = tmp_path / "pdfs"
    d.mkdir()
    (d / "x.pdf").write_bytes(b"one")
    return d


def test_real_index_roundtrip_and_metadata(tmp_path, docs, pdf_dir):
    emb = DeterministicTestEmbeddings()
    vs = create_vectorstore(docs, emb, str(tmp_path / "vs"), str(pdf_dir))
    assert vs.index.ntotal == 3

    loaded = load_vectorstore(str(tmp_path / "vs"), emb)
    assert loaded.index.ntotal == 3

    res = retrieve_relevant_chunks("similarity search vectors FAISS", loaded, top_k=3)
    assert len(res) == 3
    top_doc, _ = res[0]
    assert top_doc.metadata == {"source": "a.pdf", "page": 3}


def test_no_reembedding_when_unchanged_and_rebuild_when_changed(tmp_path, docs, pdf_dir, monkeypatch):
    emb = DeterministicTestEmbeddings()
    calls = {"n": 0}
    orig = emb.embed_documents

    def counting(texts):
        calls["n"] += 1
        return orig(texts)

    monkeypatch.setattr(emb, "embed_documents", counting)
    vs_path = str(tmp_path / "vs")

    # Initial build
    get_or_create_vectorstore(docs, emb, vs_path, str(pdf_dir))
    assert calls["n"] == 1

    # Unchanged PDFs -> loads existing index without re-embedding
    get_or_create_vectorstore(docs, emb, vs_path, str(pdf_dir))
    assert calls["n"] == 1
    assert source_changed(vs_path, str(pdf_dir)) is False

    # Add a PDF -> automatically rebuilds
    (pdf_dir / "y.pdf").write_bytes(b"two")
    assert source_changed(vs_path, str(pdf_dir)) is True
    get_or_create_vectorstore(docs, emb, vs_path, str(pdf_dir))
    assert calls["n"] == 2

    # Force rebuild
    get_or_create_vectorstore(docs, emb, vs_path, str(pdf_dir), force_rebuild=True)
    assert calls["n"] == 3


def test_answer_when_ollama_unreachable_returns_helpful_error(tmp_path, docs, pdf_dir):
    emb = DeterministicTestEmbeddings()
    vs = create_vectorstore(docs, emb, str(tmp_path / "vs"), str(pdf_dir))

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = ConnectionError("Connection refused")

    with patch("src.rag_pipeline.get_llm", return_value=mock_llm):
        out = answer_question("What is FAISS?", vs, top_k=2)

    assert "[Ollama Error]" in out["answer"]
    assert "llama3.2" in out["answer"]
    assert out["sources"][0]["source"] == "a.pdf"
    assert out["sources"][0]["page"] == 3


def test_no_results_returns_not_found():
    class EmptyVS:
        def similarity_search_with_score(self, q, k):
            return []

    out = answer_question("anything", EmptyVS(), top_k=3)
    assert out["answer"] == NOT_FOUND_MESSAGE
