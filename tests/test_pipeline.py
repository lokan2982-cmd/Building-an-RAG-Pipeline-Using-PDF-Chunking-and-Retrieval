"""
test_pipeline.py
================
Automated unit and integration tests for the offline Ollama RAG pipeline.

Uses mocks for Ollama HTTP calls so tests run quickly and deterministically
without depending on an external API or running Ollama daemon.

Run with:
    pytest tests -v
"""

import importlib
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory and remove it after the test."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_documents():
    """Return a small list of sample LangChain Documents with metadata."""
    from langchain_core.documents import Document

    return [
        Document(
            page_content="Artificial intelligence (AI) is intelligence demonstrated by machines.",
            metadata={"source": "ai_intro.pdf", "page": 1},
        ),
        Document(
            page_content="Machine learning is a subset of AI that enables systems to learn from data.",
            metadata={"source": "ai_intro.pdf", "page": 2},
        ),
        Document(
            page_content="Deep learning uses neural networks with many layers to solve complex problems.",
            metadata={"source": "deep_learning.pdf", "page": 1},
        ),
    ]


@pytest.fixture
def mock_embeddings():
    """Return a mock embeddings object producing fixed 768-dim float vectors."""
    import numpy as np

    rng = np.random.default_rng(42)
    emb = MagicMock()
    emb.model = "nomic-embed-text"
    emb.embed_documents.side_effect = lambda texts: [
        rng.random(768).tolist() for _ in texts
    ]
    emb.embed_query.side_effect = lambda text: rng.random(768).tolist()
    return emb


# ===========================================================================
# 1. PDF Loading tests
# ===========================================================================

class TestPDFLoader:

    def test_missing_directory_raises(self):
        """FileNotFoundError when PDF directory does not exist."""
        from src.pdf_loader import load_pdfs_from_directory

        with pytest.raises(FileNotFoundError):
            load_pdfs_from_directory("/nonexistent/path/to/pdfs")

    def test_empty_directory_raises(self, temp_dir):
        """ValueError when no PDF files are found."""
        from src.pdf_loader import load_pdfs_from_directory

        with pytest.raises(ValueError, match="No PDF files found"):
            load_pdfs_from_directory(temp_dir)

    def test_loads_valid_pdf_and_preserves_metadata(self, temp_dir):
        """Loads a PDF file and converts 0-indexed page metadata to 1-indexed."""
        from langchain_core.documents import Document
        from src.pdf_loader import load_pdfs_from_directory

        pdf_path = Path(temp_dir) / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")

        with patch("src.pdf_loader.PyPDFLoader") as MockLoader:
            mock_instance = MockLoader.return_value
            mock_instance.load.return_value = [
                Document(
                    page_content="Hello World from test PDF.",
                    metadata={"source": str(pdf_path), "page": 0},
                )
            ]
            docs = load_pdfs_from_directory(temp_dir)

        assert len(docs) == 1
        assert docs[0].metadata["source"] == "test.pdf"
        assert docs[0].metadata["page"] == 1

    def test_corrupted_pdf_handled_gracefully(self, temp_dir):
        """A corrupted PDF is skipped without crashing the loader."""
        from src.pdf_loader import load_pdfs_from_directory

        corrupted = Path(temp_dir) / "corrupt.pdf"
        corrupted.write_bytes(b"THIS IS NOT A VALID PDF FILE")

        docs = load_pdfs_from_directory(temp_dir)
        assert docs == []

    def test_loads_project_sample_pdfs(self):
        """All 3 sample PDFs in data/pdfs/ load cleanly with metadata."""
        from src.pdf_loader import load_pdfs_from_directory

        sample_dir = PROJECT_ROOT / "data" / "pdfs"
        if not sample_dir.exists():
            pytest.skip("Sample PDF directory not present")

        docs = load_pdfs_from_directory(str(sample_dir))
        assert len(docs) >= 3
        for doc in docs:
            assert doc.page_content.strip()
            assert doc.metadata["source"].endswith(".pdf")
            assert isinstance(doc.metadata["page"], int) and doc.metadata["page"] >= 1


# ===========================================================================
# 2. Text Chunking tests
# ===========================================================================

class TestTextChunker:

    def test_basic_chunking(self, sample_documents):
        """Chunking produces at least as many chunks as source pages."""
        from src.text_chunker import chunk_documents

        chunks = chunk_documents(sample_documents, chunk_size=100, chunk_overlap=20)
        assert len(chunks) >= len(sample_documents)

    def test_default_chunk_settings_1000_200(self, sample_documents, monkeypatch):
        """Default chunking uses chunk_size=1000 and chunk_overlap=200."""
        from src.text_chunker import calculate_chunk_statistics, chunk_documents

        monkeypatch.delenv("CHUNK_SIZE", raising=False)
        monkeypatch.delenv("CHUNK_OVERLAP", raising=False)
        chunks = chunk_documents(sample_documents)
        stats = calculate_chunk_statistics(sample_documents, chunks)
        assert stats["chunk_size_setting"] == 1000
        assert stats["chunk_overlap_setting"] == 200

    def test_chunks_preserve_metadata(self, sample_documents):
        """Every chunk preserves source and page metadata and adds chunk_index."""
        from src.text_chunker import chunk_documents

        chunks = chunk_documents(sample_documents, chunk_size=50, chunk_overlap=10)
        for i, chunk in enumerate(chunks):
            assert "source" in chunk.metadata
            assert "page" in chunk.metadata
            assert chunk.metadata["chunk_index"] == i

    def test_chunk_size_respected(self, sample_documents):
        """No chunk exceeds the configured chunk_size."""
        from src.text_chunker import chunk_documents

        chunk_size = 150
        chunks = chunk_documents(sample_documents, chunk_size=chunk_size, chunk_overlap=20)
        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size

    def test_invalid_overlap_raises(self, sample_documents):
        """Overlap >= chunk_size raises ValueError."""
        from src.text_chunker import chunk_documents

        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_documents(sample_documents, chunk_size=100, chunk_overlap=100)

    def test_statistics_calculation(self, sample_documents):
        """Statistics dict accurately computes total chunks and average chunk size."""
        from src.text_chunker import calculate_chunk_statistics, chunk_documents

        chunks = chunk_documents(sample_documents, chunk_size=100, chunk_overlap=20)
        stats = calculate_chunk_statistics(sample_documents, chunks)

        expected_avg = round(sum(len(c.page_content) for c in chunks) / len(chunks), 1)
        assert stats["num_documents"] == 2
        assert stats["num_pages"] == len(sample_documents)
        assert stats["num_chunks"] == len(chunks)
        assert stats["avg_chunk_size"] == expected_avg

    def test_empty_documents_returns_empty(self):
        """Chunking an empty list returns an empty list."""
        from src.text_chunker import chunk_documents

        assert chunk_documents([]) == []


# ===========================================================================
# 3. Ollama Embeddings & Chat Configuration tests (No Azure/OpenAI)
# ===========================================================================

class TestOllamaConfiguration:

    def test_no_azure_or_openai_in_active_src(self):
        """Active src/ modules must not import langchain_openai or Azure OpenAI."""
        src_dir = PROJECT_ROOT / "src"
        for py_file in src_dir.glob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            assert "langchain_openai" not in content, f"Found langchain_openai in {py_file.name}"
            assert "AzureOpenAI" not in content, f"Found AzureOpenAI in {py_file.name}"
            assert "AzureChatOpenAI" not in content, f"Found AzureChatOpenAI in {py_file.name}"

    def test_default_ollama_embeddings(self, monkeypatch):
        """get_embeddings() returns OllamaEmbeddings with nomic-embed-text by default."""
        from langchain_ollama import OllamaEmbeddings
        from src import embeddings as emb_module

        monkeypatch.delenv("OLLAMA_EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        importlib.reload(emb_module)

        emb = emb_module.get_embeddings()
        assert isinstance(emb, OllamaEmbeddings)
        assert emb.model == "nomic-embed-text"
        assert emb.base_url == "http://localhost:11434"

    def test_custom_ollama_embeddings_env(self, monkeypatch):
        """OLLAMA_EMBEDDING_MODEL and OLLAMA_BASE_URL override defaults."""
        from src import embeddings as emb_module

        monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/")
        emb = emb_module.get_embeddings()
        assert emb.model == "nomic-embed-text:latest"
        assert emb.base_url == "http://127.0.0.1:11434"

    def test_invalid_ollama_url_raises(self, monkeypatch):
        """Non-HTTP OLLAMA_BASE_URL raises ValueError."""
        from src import embeddings as emb_module

        monkeypatch.setenv("OLLAMA_BASE_URL", "localhost:11434")
        with pytest.raises(ValueError, match="Invalid OLLAMA_BASE_URL"):
            emb_module.get_embeddings()

    def test_validate_embedding_config_no_api_key_required(self, monkeypatch):
        """validate_embedding_config confirms Ollama requires no API key."""
        from src import embeddings as emb_module

        monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        cfg = emb_module.validate_embedding_config(check_server=False)
        assert cfg["provider"] == "ollama"
        assert cfg["model"] == "nomic-embed-text"
        assert cfg["api_key_required"] is False

    def test_is_model_installed_matches_with_or_without_tag(self):
        """is_model_installed matches 'nomic-embed-text' against 'nomic-embed-text:latest'."""
        from src.embeddings import is_model_installed

        installed = ["nomic-embed-text:latest", "llama3.2:latest"]
        assert is_model_installed("nomic-embed-text", installed) is True
        assert is_model_installed("llama3.2", installed) is True
        assert is_model_installed("nonexistent-model", installed) is False

    def test_default_chat_ollama(self, monkeypatch):
        """get_llm() returns ChatOllama configured with llama3.2 and temperature=0."""
        from langchain_ollama import ChatOllama
        from src.rag_pipeline import get_llm

        monkeypatch.delenv("OLLAMA_CHAT_MODEL", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_TEMPERATURE", raising=False)

        llm = get_llm()
        assert isinstance(llm, ChatOllama)
        assert llm.model == "llama3.2"
        assert llm.temperature == 0.0

    def test_list_models_connection_error(self):
        """Unreachable Ollama URL raises a clear ConnectionError."""
        from src.embeddings import list_installed_ollama_models

        with patch("urllib.request.urlopen", side_effect=OSError("Connection refused")):
            with pytest.raises(ConnectionError, match="Could not connect to local Ollama"):
                list_installed_ollama_models(base_url="http://localhost:11434")


# ===========================================================================
# 4. FAISS Vector Store tests
# ===========================================================================

class TestVectorStore:

    def test_create_vectorstore_calls_faiss(self, temp_dir, sample_documents, mock_embeddings):
        """create_vectorstore embeds chunks via FAISS.from_documents and saves locally."""
        from src.text_chunker import chunk_documents
        from src.vector_store import create_vectorstore

        chunks = chunk_documents(sample_documents, chunk_size=100, chunk_overlap=10)

        with patch("src.vector_store.FAISS") as MockFAISS:
            mock_vs = MagicMock()
            MockFAISS.from_documents.return_value = mock_vs

            vs = create_vectorstore(chunks, mock_embeddings, temp_dir, temp_dir)
            MockFAISS.from_documents.assert_called_once_with(chunks, mock_embeddings)
            mock_vs.save_local.assert_called_once()
            assert vs is mock_vs

    def test_empty_chunks_raises(self, temp_dir, mock_embeddings):
        """Creating a vectorstore with zero chunks raises ValueError."""
        from src.vector_store import create_vectorstore

        with pytest.raises(ValueError):
            create_vectorstore([], mock_embeddings, temp_dir, temp_dir)

    def test_load_missing_index_raises(self, temp_dir, mock_embeddings):
        """Loading a non-existent index raises FileNotFoundError."""
        from src.vector_store import load_vectorstore

        with pytest.raises(FileNotFoundError):
            load_vectorstore(temp_dir, mock_embeddings)

    def test_source_hash_changes_on_pdf_or_model_change(self, temp_dir, monkeypatch):
        """Fingerprint changes when PDFs or embedding model change."""
        from src.vector_store import _compute_pdf_hash

        monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        hash1 = _compute_pdf_hash(temp_dir)

        fake_pdf = Path(temp_dir) / "doc.pdf"
        fake_pdf.write_bytes(b"fake content")
        hash2 = _compute_pdf_hash(temp_dir)
        assert hash1 != hash2

        monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "other-embed-model")
        hash3 = _compute_pdf_hash(temp_dir)
        assert hash2 != hash3


# ===========================================================================
# 5. Retrieval tests
# ===========================================================================

class TestRetriever:

    def test_retrieval_returns_top_k_results(self, sample_documents):
        """similarity_search_with_score is called with k=top_k."""
        from src.retriever import retrieve_relevant_chunks

        mock_vs = MagicMock()
        fake_results = [
            (sample_documents[0], 0.12),
            (sample_documents[1], 0.25),
            (sample_documents[2], 0.31),
        ]
        mock_vs.similarity_search_with_score.return_value = fake_results

        results = retrieve_relevant_chunks("What is AI?", mock_vs, top_k=3)
        mock_vs.similarity_search_with_score.assert_called_once_with("What is AI?", k=3)
        assert len(results) == 3

    def test_empty_query_returns_empty(self):
        """An empty query returns [] without hitting FAISS."""
        from src.retriever import retrieve_relevant_chunks

        mock_vs = MagicMock()
        results = retrieve_relevant_chunks("   ", mock_vs, top_k=3)
        assert results == []
        mock_vs.similarity_search_with_score.assert_not_called()

    def test_context_formatting_includes_source_and_page(self, sample_documents):
        """format_context_for_llm includes source filename and page number."""
        from src.retriever import format_context_for_llm

        results = [(sample_documents[0], 0.1), (sample_documents[1], 0.2)]
        context = format_context_for_llm(results)
        assert "ai_intro.pdf" in context
        assert "Page 1" in context
        assert "Page 2" in context

    def test_no_results_context(self):
        """Empty results return a clear fallback context message."""
        from src.retriever import format_context_for_llm

        assert "No relevant" in format_context_for_llm([])


# ===========================================================================
# 6. End-to-end pipeline integration tests (mocked Ollama)
# ===========================================================================

class TestPipelineIntegration:

    @patch("src.rag_pipeline.get_embeddings")
    @patch("src.rag_pipeline.get_or_create_vectorstore")
    @patch("src.rag_pipeline.load_pdfs_from_directory")
    def test_build_pipeline_mocked(
        self,
        mock_load,
        mock_vs_factory,
        mock_embeddings_factory,
        sample_documents,
        mock_embeddings,
        temp_dir,
    ):
        """build_pipeline orchestrates load -> chunk -> embed -> FAISS and returns stats."""
        from src.rag_pipeline import build_pipeline

        mock_load.return_value = sample_documents
        mock_embeddings_factory.return_value = mock_embeddings
        mock_vs = MagicMock()
        mock_vs.index.ntotal = len(sample_documents)
        mock_vs_factory.return_value = mock_vs

        vs, chunks, stats = build_pipeline(
            pdf_dir=temp_dir,
            vectorstore_path=temp_dir,
            force_rebuild=True,
        )

        assert vs is mock_vs
        assert stats["num_pages"] == len(sample_documents)
        assert stats["embedding_model"] == "nomic-embed-text"
        assert stats["chat_model"] == "llama3.2"

    @patch("src.rag_pipeline.get_llm")
    def test_answer_question_mocked(self, mock_llm_factory, sample_documents):
        """answer_question retrieves chunks, calls ChatOllama, and returns answer + sources."""
        from src.rag_pipeline import answer_question

        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.return_value = [
            (sample_documents[0], 0.15),
        ]

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "AI is intelligence demonstrated by machines. (Source: ai_intro.pdf, Page 1)"
        mock_llm.invoke.return_value = mock_response
        mock_llm_factory.return_value = mock_llm

        result = answer_question("What is AI?", mock_vs, top_k=1)

        assert result["question"] == "What is AI?"
        assert "intelligence demonstrated by machines" in result["answer"]
        assert result["sources"][0]["source"] == "ai_intro.pdf"
        assert result["sources"][0]["page"] == 1
