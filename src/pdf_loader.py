"""
pdf_loader.py
=============
Handles loading and validating PDF documents using LangChain's PyPDFLoader.

Features:
  - Loads multiple PDFs from a directory
  - Preserves source filename and page number as metadata
  - Gracefully handles empty, corrupted, and unreadable PDFs
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_pdfs_from_directory(pdf_dir: str) -> List[Document]:
    """
    Load all PDF files from the specified directory.

    Args:
        pdf_dir (str): Path to the directory containing PDF files.

    Returns:
        List[Document]: A list of LangChain Document objects, one per page,
                        with metadata containing 'source' and 'page'.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        ValueError: If no PDF files are found in the directory.
    """
    pdf_path = Path(pdf_dir)

    # Validate directory
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    if not pdf_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {pdf_dir}")

    # Collect all PDF files
    pdf_files = sorted(pdf_path.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(
            f"No PDF files found in '{pdf_dir}'.\n"
            "Please add at least one PDF document to the data/pdfs/ folder."
        )

    logger.info(f"Found {len(pdf_files)} PDF file(s) in '{pdf_dir}'")

    all_documents: List[Document] = []
    load_summary: Dict[str, Any] = {}

    for pdf_file in pdf_files:
        logger.info(f"Loading: {pdf_file.name}")
        docs = _load_single_pdf(pdf_file)

        if docs:
            all_documents.extend(docs)
            load_summary[pdf_file.name] = {
                "pages": len(docs),
                "status": "success",
            }
            logger.info(f"  → Loaded {len(docs)} page(s) from '{pdf_file.name}'")
        else:
            load_summary[pdf_file.name] = {
                "pages": 0,
                "status": "empty_or_failed",
            }
            logger.warning(f"  → No content extracted from '{pdf_file.name}' (empty or corrupted)")

    logger.info(
        f"\nLoad summary: {len(all_documents)} total page(s) from "
        f"{sum(1 for v in load_summary.values() if v['status'] == 'success')} "
        f"of {len(pdf_files)} file(s)."
    )

    return all_documents


def _load_single_pdf(pdf_path: Path) -> List[Document]:
    """
    Load a single PDF file safely.

    Args:
        pdf_path (Path): Path object pointing to a PDF file.

    Returns:
        List[Document]: Pages extracted from the PDF. Empty list on failure.
    """
    try:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()

        # Filter out pages with no meaningful text content
        valid_docs = []
        for doc in docs:
            text = doc.page_content.strip()
            if text:  # Skip blank pages
                # Ensure metadata has the filename (not full path)
                doc.metadata["source"] = pdf_path.name
                # Page is 0-indexed from PyPDFLoader; store as 1-indexed for readability
                if "page" in doc.metadata:
                    doc.metadata["page"] = doc.metadata["page"] + 1
                valid_docs.append(doc)

        return valid_docs

    except Exception as e:
        logger.error(f"Failed to load '{pdf_path.name}': {type(e).__name__}: {e}")
        return []


def display_document_summary(documents: List[Document]) -> None:
    """
    Print a human-readable summary of loaded documents.

    Args:
        documents (List[Document]): List of loaded document pages.
    """
    if not documents:
        print("No documents loaded.")
        return

    # Group by source file
    sources: Dict[str, List[Document]] = {}
    for doc in documents:
        src = doc.metadata.get("source", "unknown")
        sources.setdefault(src, []).append(doc)

    print("\n" + "=" * 60)
    print("LOADED DOCUMENTS SUMMARY")
    print("=" * 60)
    print(f"Total documents (files): {len(sources)}")
    print(f"Total pages loaded     : {len(documents)}")
    print("-" * 60)

    for filename, pages in sources.items():
        print(f"  File : {filename}")
        print(f"  Pages: {len(pages)}")
        # Show a preview of the first page
        preview = pages[0].page_content[:200].replace("\n", " ")
        print(f"  Preview: {preview}...")
        print()

    print("=" * 60)


if __name__ == "__main__":
    # Quick test when run directly
    import sys
    pdf_directory = sys.argv[1] if len(sys.argv) > 1 else "data/pdfs"
    docs = load_pdfs_from_directory(pdf_directory)
    display_document_summary(docs)
