"""
vector_store.py
===============
Creates, persists, and loads the FAISS vector database.

Features:
  - Create a FAISS vectorstore from document chunks + embeddings
  - Save the index locally to disk (avoids re-embedding unchanged documents)
  - Load an existing saved index
  - Rebuild the index on demand when PDFs change
  - Change detection via a content hash of PDF files, embedding model, and chunk settings
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

# Sub-folder names inside the vectorstore directory
INDEX_SUBDIR = "faiss_index"
HASH_FILE = "source_hash.json"


def _compute_pdf_hash(pdf_dir: str) -> str:
    """
    Compute a fingerprint of the PDF directory contents (filename + MD5 of
    each file's bytes) combined with the embedding model identifier and
    chunk settings.  If any of these change, the index must be rebuilt.
    """
    from src.embeddings import get_embedding_identifier  # local import avoids cycles

    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        return ""
    entries = sorted(
        (f.name, hashlib.md5(f.read_bytes()).hexdigest())
        for f in pdf_path.glob("*.pdf")
        if f.is_file()
    )
    fingerprint = {
        "files": entries,
        "embedding": get_embedding_identifier(),
        "chunk_size": os.getenv("CHUNK_SIZE", "1000"),
        "chunk_overlap": os.getenv("CHUNK_OVERLAP", "200"),
    }
    raw = json.dumps(fingerprint, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()


def _save_hash(vectorstore_path: str, pdf_dir: str) -> None:
    """Persist the current PDF directory hash next to the FAISS index."""
    hash_path = Path(vectorstore_path) / HASH_FILE
    current_hash = _compute_pdf_hash(pdf_dir)
    hash_path.write_text(json.dumps({"hash": current_hash}))


def _load_saved_hash(vectorstore_path: str) -> Optional[str]:
    """Load the previously saved PDF directory hash, or None if not found."""
    hash_path = Path(vectorstore_path) / HASH_FILE
    if hash_path.exists():
        try:
            data = json.loads(hash_path.read_text())
            return data.get("hash")
        except Exception:
            return None
    return None


def source_changed(vectorstore_path: str, pdf_dir: str) -> bool:
    """
    Return True if the PDF directory has changed since the last index build,
    or if no saved index exists.

    Args:
        vectorstore_path (str): Directory where the FAISS index is stored.
        pdf_dir          (str): Directory containing PDF source files.

    Returns:
        bool: True → rebuild needed.  False → reuse existing index.
    """
    saved_hash = _load_saved_hash(vectorstore_path)
    if saved_hash is None:
        return True  # No previous index → must build
    current_hash = _compute_pdf_hash(pdf_dir)
    changed = saved_hash != current_hash
    if changed:
        logger.info("PDF sources have changed — vectorstore rebuild required.")
    else:
        logger.info("PDF sources are unchanged — existing vectorstore is current.")
    return changed


def create_vectorstore(
    chunks: List[Document],
    embeddings,
    vectorstore_path: str,
    pdf_dir: str = "data/pdfs",
) -> FAISS:
    """
    Build a new FAISS vectorstore from document chunks and save it to disk.

    Args:
        chunks           (List[Document]): Chunked documents to embed.
        embeddings       : LangChain embeddings object (OllamaEmbeddings).
        vectorstore_path (str): Directory path where the index will be saved.
        pdf_dir          (str): Source PDF directory (for change detection).

    Returns:
        FAISS: The in-memory FAISS vectorstore.
    """
    if not chunks:
        raise ValueError("Cannot create a vectorstore with zero chunks.")

    logger.info(f"Embedding {len(chunks)} chunk(s) and building FAISS index …")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Persist to disk
    index_path = Path(vectorstore_path) / INDEX_SUBDIR
    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))
    _save_hash(vectorstore_path, pdf_dir)

    logger.info(f"FAISS index saved to: {index_path}")
    return vectorstore


def load_vectorstore(vectorstore_path: str, embeddings) -> FAISS:
    """
    Load a previously saved FAISS vectorstore from disk.

    Args:
        vectorstore_path (str): Directory containing the saved FAISS index.
        embeddings       : LangChain embeddings object (must match the one used
                           when the index was created).

    Returns:
        FAISS: Loaded vectorstore.

    Raises:
        FileNotFoundError: If the index directory does not exist.
    """
    index_path = Path(vectorstore_path) / INDEX_SUBDIR
    if not index_path.exists():
        raise FileNotFoundError(
            f"No saved FAISS index found at '{index_path}'.\n"
            "Run the pipeline first to build the vectorstore."
        )

    logger.info(f"Loading FAISS index from: {index_path}")
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,  # Required in newer LangChain
    )
    logger.info("FAISS index loaded successfully.")
    return vectorstore


def get_or_create_vectorstore(
    chunks: List[Document],
    embeddings,
    vectorstore_path: str,
    pdf_dir: str = "data/pdfs",
    force_rebuild: bool = False,
) -> FAISS:
    """
    Load an existing FAISS index if the PDFs haven't changed; otherwise create
    a new one.

    Args:
        chunks           (List[Document]): Chunks to embed (used only when building).
        embeddings       : LangChain embeddings object.
        vectorstore_path (str): Directory path for the FAISS index.
        pdf_dir          (str): Source PDF directory.
        force_rebuild    (bool): If True, always rebuild regardless of hash.

    Returns:
        FAISS: Vectorstore (loaded or freshly created).
    """
    index_path = Path(vectorstore_path) / INDEX_SUBDIR
    needs_build = force_rebuild or not index_path.exists() or source_changed(vectorstore_path, pdf_dir)

    if needs_build:
        logger.info("Building new vectorstore …")
        return create_vectorstore(chunks, embeddings, vectorstore_path, pdf_dir)
    else:
        logger.info("Reusing existing vectorstore (PDFs unchanged).")
        return load_vectorstore(vectorstore_path, embeddings)
