"""
retriever.py
============
Performs semantic search over the FAISS vectorstore to retrieve
the most relevant document chunks for a given user query.

Features:
  - Converts query to embedding and searches FAISS
  - Returns top-K results (configurable via env or argument)
  - Displays source document name, page number, and content preview
  - Optionally returns similarity scores
"""

import os
import logging
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)


def retrieve_relevant_chunks(
    query: str,
    vectorstore: FAISS,
    top_k: int = None,
) -> List[Tuple[Document, float]]:
    """
    Search the FAISS vectorstore for chunks semantically similar to `query`.

    Args:
        query       (str)  : The user's question.
        vectorstore (FAISS): Loaded FAISS vectorstore.
        top_k       (int)  : Number of results to return.  Defaults to env
                             TOP_K_RESULTS or 3.

    Returns:
        List[Tuple[Document, float]]: Each element is (document_chunk, score).
                                      Lower score = more similar (L2 distance).
    """
    top_k = top_k or int(os.getenv("TOP_K_RESULTS", 3))

    if not query or not query.strip():
        logger.warning("Empty query received — skipping retrieval.")
        return []

    logger.info(f"Retrieving top {top_k} chunks for query: '{query[:80]}…'")

    try:
        results = vectorstore.similarity_search_with_score(query, k=top_k)
        logger.info(f"Retrieved {len(results)} chunk(s).")
        return results
    except Exception as e:
        logger.error(f"Retrieval failed: {type(e).__name__}: {e}")
        raise


def display_retrieved_chunks(
    results: List[Tuple[Document, float]],
    show_full_content: bool = False,
) -> None:
    """
    Print retrieved chunks in a human-readable format.

    Args:
        results           : Output from retrieve_relevant_chunks().
        show_full_content : If True, prints the full chunk text; otherwise
                            shows a 400-character preview.
    """
    if not results:
        print("No relevant chunks found.")
        return

    print("\n" + "=" * 60)
    print(f"TOP {len(results)} RETRIEVED CHUNKS")
    print("=" * 60)

    for rank, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        chunk_idx = doc.metadata.get("chunk_index", "N/A")
        content = doc.page_content

        print(f"\n[Rank {rank}]")
        print(f"  Source     : {source}")
        print(f"  Page       : {page}")
        print(f"  Chunk #    : {chunk_idx}")
        print(f"  Similarity : {score:.4f} (lower = more similar)")
        print(f"  Length     : {len(content)} characters")

        if show_full_content:
            print(f"  Content    :\n{content}")
        else:
            preview = content[:400].replace("\n", " ")
            print(f"  Preview    : {preview}{'...' if len(content) > 400 else ''}")

    print("=" * 60)


def format_context_for_llm(results: List[Tuple[Document, float]]) -> str:
    """
    Format retrieved chunks into a single context string for the LLM prompt.

    Args:
        results: Output from retrieve_relevant_chunks().

    Returns:
        str: Formatted context block with source metadata.
    """
    if not results:
        return "No relevant document chunks were found."

    parts = []
    for rank, (doc, _score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        parts.append(
            f"[Source {rank}: {source}, Page {page}]\n"
            f"{doc.page_content.strip()}"
        )

    return "\n\n---\n\n".join(parts)
