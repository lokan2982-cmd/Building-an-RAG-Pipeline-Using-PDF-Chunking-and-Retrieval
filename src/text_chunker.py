"""
text_chunker.py
===============
Splits loaded PDF documents into smaller, overlapping chunks
using LangChain's RecursiveCharacterTextSplitter.

Features:
  - Configurable chunk_size and chunk_overlap (via args or env vars)
  - Preserves all original document metadata in every chunk
  - Displays statistics: document count, page count, chunk count, avg chunk size
  - Shows sample chunks for inspection
"""

import os
import logging
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """
    Split a list of Document pages into overlapping chunks.

    Args:
        documents    (List[Document]): Full-page documents from pdf_loader.
        chunk_size   (int, optional): Characters per chunk. Defaults to env
                                      CHUNK_SIZE or 1000.
        chunk_overlap(int, optional): Overlap characters. Defaults to env
                                      CHUNK_OVERLAP or 200.

    Returns:
        List[Document]: Chunked documents, each carrying the original metadata
                        plus a 'chunk_index' field.
    """
    # Resolve configuration from args → env → defaults
    chunk_size = chunk_size if chunk_size is not None else int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap = chunk_overlap if chunk_overlap is not None else int(os.getenv("CHUNK_OVERLAP", 200))
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    if not documents:
        logger.warning("No documents provided for chunking.")
        return []

    logger.info(
        f"Chunking {len(documents)} page(s) with "
        f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Split on paragraphs → sentences → words → chars
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    # Tag each chunk with its index and the settings used (for traceability)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_size_setting"] = chunk_size
        chunk.metadata["chunk_overlap_setting"] = chunk_overlap

    logger.info(f"Created {len(chunks)} chunk(s).")
    return chunks


def calculate_chunk_statistics(
    original_docs: List[Document],
    chunks: List[Document],
) -> dict:
    """
    Compute and return statistics about the chunking operation.

    Args:
        original_docs (List[Document]): The full-page documents before chunking.
        chunks        (List[Document]): The chunks produced after splitting.

    Returns:
        dict: Statistics dictionary with keys:
              - num_documents, num_pages, num_chunks,
                avg_chunk_size, min_chunk_size, max_chunk_size,
                chunk_size_setting, chunk_overlap_setting
    """
    unique_sources = {doc.metadata.get("source", "unknown") for doc in original_docs}
    num_pages = len(original_docs)
    num_chunks = len(chunks)

    if chunks:
        sizes = [len(c.page_content) for c in chunks]
        avg_size = sum(sizes) / len(sizes)
        min_size = min(sizes)
        max_size = max(sizes)
    else:
        avg_size = min_size = max_size = 0

    stats = {
        "num_documents": len(unique_sources),
        "num_pages": num_pages,
        "num_chunks": num_chunks,
        "avg_chunk_size": round(avg_size, 1),
        "min_chunk_size": min_size,
        "max_chunk_size": max_size,
        # Report the settings actually used to create the chunks
        "chunk_size_setting": chunks[0].metadata.get("chunk_size_setting") if chunks else int(os.getenv("CHUNK_SIZE", 1000)),
        "chunk_overlap_setting": chunks[0].metadata.get("chunk_overlap_setting") if chunks else int(os.getenv("CHUNK_OVERLAP", 200)),
    }
    return stats


def display_chunk_statistics(stats: dict) -> None:
    """Print chunking statistics to stdout in a readable format."""
    print("\n" + "=" * 60)
    print("CHUNKING STATISTICS")
    print("=" * 60)
    print(f"  PDF Documents      : {stats['num_documents']}")
    print(f"  Total Pages        : {stats['num_pages']}")
    print(f"  Total Chunks       : {stats['num_chunks']}")
    print(f"  Avg Chunk Size     : {stats['avg_chunk_size']} characters")
    print(f"  Min Chunk Size     : {stats['min_chunk_size']} characters")
    print(f"  Max Chunk Size     : {stats['max_chunk_size']} characters")
    print(f"  Chunk Size Setting : {stats['chunk_size_setting']}")
    print(f"  Overlap Setting    : {stats['chunk_overlap_setting']}")
    print("=" * 60)


def display_sample_chunks(chunks: List[Document], n: int = 3) -> None:
    """
    Print the first `n` chunks to the console for inspection.

    Args:
        chunks (List[Document]): All chunks.
        n      (int): Number of samples to display (default 3).
    """
    sample_count = min(n, len(chunks))
    print(f"\n--- Sample Chunks (showing {sample_count} of {len(chunks)}) ---")
    for i in range(sample_count):
        chunk = chunks[i]
        print(f"\nChunk #{i + 1}")
        print(f"  Source   : {chunk.metadata.get('source', 'N/A')}")
        print(f"  Page     : {chunk.metadata.get('page', 'N/A')}")
        print(f"  Size     : {len(chunk.page_content)} chars")
        preview = chunk.page_content[:300].replace("\n", " ")
        print(f"  Content  : {preview}{'...' if len(chunk.page_content) > 300 else ''}")
    print("-" * 60)


if __name__ == "__main__":
    # Quick standalone test
    from src.pdf_loader import load_pdfs_from_directory

    docs = load_pdfs_from_directory("data/pdfs")
    chunks = chunk_documents(docs)
    stats = calculate_chunk_statistics(docs, chunks)
    display_chunk_statistics(stats)
    display_sample_chunks(chunks)
