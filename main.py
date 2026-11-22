"""
main.py
=======
Command-line interface for the offline PDF RAG Assistant (Ollama + FAISS + LangChain).

Usage:
    python main.py
    python main.py --rebuild
    python main.py --debug      (log retrieved chunks, raw LLM output, answer validation)

Runs completely offline using local Ollama models:
  - Embeddings : nomic-embed-text (OllamaEmbeddings)
  - Chat LLM   : llama3.2 (ChatOllama)

Type 'exit' or 'quit' to stop.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure UTF-8 console output on Windows without crashing on non-UTF-8 terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is on sys.path so src.* imports work from any directory
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

from src.embeddings import get_embedding_model_name, get_ollama_base_url  # noqa: E402
from src.rag_pipeline import (  # noqa: E402
    answer_question,
    build_pipeline,
    enable_debug_logging,
    get_chat_model_name,
    get_llm,
)
from src.retriever import display_retrieved_chunks  # noqa: E402

BANNER = """
============================================================
         Welcome to PDF RAG Assistant (Local Ollama)
   Ask questions about your uploaded PDF documents offline
   Type 'exit' or 'quit' to stop.
============================================================
"""


def print_separator(char="-", width=60):
    print(char * width)


def _resolve_path(path_value: str) -> str:
    p = Path(path_value)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


def main():
    print(BANNER)

    pdf_dir = _resolve_path(os.getenv("PDF_DIR", "data/pdfs"))
    vectorstore_path = _resolve_path(os.getenv("VECTORSTORE_PATH", "vectorstore"))
    force_rebuild = "--rebuild" in sys.argv
    debug = "--debug" in sys.argv or os.getenv("RAG_DEBUG", "").lower() in {"1", "true", "yes", "on"}
    if debug:
        enable_debug_logging()
    top_k = int(os.getenv("TOP_K_RESULTS", 3))

    embed_model = get_embedding_model_name()
    chat_model = get_chat_model_name()
    ollama_url = get_ollama_base_url()

    print("Initialising offline RAG pipeline ...")
    print(f"  Ollama URL       : {ollama_url}")
    print(f"  Embedding Model  : {embed_model}")
    print(f"  Chat Model       : {chat_model}")
    print_separator()

    try:
        vectorstore, chunks, stats = build_pipeline(
            pdf_dir=pdf_dir,
            vectorstore_path=vectorstore_path,
            force_rebuild=force_rebuild,
        )
        llm = get_llm()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Please add PDF files to the data/pdfs/ directory and try again.")
        sys.exit(1)
    except ValueError as e:
        print(f"\n[CONFIG ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        print("\nMake sure Ollama is running locally and both models are downloaded:")
        print(f"  ollama pull {embed_model}")
        print(f"  ollama pull {chat_model}")
        sys.exit(1)

    # Display pipeline statistics
    print("\nPipeline Statistics")
    print_separator()
    print(f"  PDF Documents    : {stats['num_documents']}")
    print(f"  Total Pages      : {stats['num_pages']}")
    print(f"  Total Chunks     : {stats['num_chunks']}")
    print(f"  Avg Chunk Size   : {stats['avg_chunk_size']} characters")
    print(f"  Chunk Settings   : size={stats['chunk_size_setting']}, overlap={stats['chunk_overlap_setting']}")
    print(f"  Embeddings       : {stats.get('num_embeddings', stats['num_chunks'])} ({embed_model})")
    print(f"  FAISS Vectors    : {stats.get('faiss_vectors', stats['num_chunks'])}")
    print(f"  Top-K Retrieval  : {top_k}")
    print_separator()
    print("\nRAG system ready (running 100% offline via Ollama). Ask your questions!\n")

    # Interactive Q&A loop
    while True:
        try:
            question = input("Enter your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not question:
            print("  (Please enter a question.)\n")
            continue

        if question.lower() in {"exit", "quit", "q", "bye"}:
            print("\nGoodbye!")
            break

        print("\nRetrieving relevant documents ...")

        def show_chunks(results):
            # Called by answer_question right after FAISS retrieval and BEFORE
            # the LLM is invoked, so the top chunks are visible first.
            display_retrieved_chunks(results)
            print("\nGenerating answer from the retrieved context ...")

        try:
            result = answer_question(
                question, vectorstore, top_k=top_k, llm=llm, on_retrieved=show_chunks
            )
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}\n")
            continue

        # Display the validated answer
        print(f"\nAI Answer ({chat_model}):")
        print_separator()
        print(result["answer"])
        print_separator()

        # Source references: only chunks that actually support a found answer
        if result["found"] and result["supporting_sources"]:
            print("\nSource Document References (chunks supporting the answer):")
            for i, src in enumerate(result["supporting_sources"], 1):
                print(
                    f"  [{i}] {src['source']} - Page {src['page']}  "
                    f"(L2 distance: {src['score']}, answer support: {src['support']})"
                )
        if debug:
            print(f"\n[debug] status={result['status']} grounding_ratio={result['grounding_ratio']}")

        print()


if __name__ == "__main__":
    main()
