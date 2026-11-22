"""
rag_pipeline.py
===============
Orchestrates the full offline RAG pipeline using local Ollama models:
  PDF Loading -> Text Chunking -> OllamaEmbeddings (nomic-embed-text)
  -> FAISS Vector Store -> Semantic Retrieval -> ChatOllama (llama3.2) Answer
  -> Answer VALIDATION (fallback handling + grounding check)

No OpenAI or Azure OpenAI API keys are required.

Answer validation (see `validate_answer`)
-----------------------------------------
Small local models sometimes (a) append the "not found" sentence after a good
answer, (b) return only a citation / an empty answer, or (c) answer from general
knowledge.  Every model output is therefore validated against the retrieved PDF
chunks before being shown:
  * the fallback sentence is stripped when a real, grounded answer is present;
  * an output that contains only the fallback is treated as "not found";
  * an empty output is retried once, then treated as "not found";
  * an answer whose content words are not supported by the retrieved chunks
    (i.e. likely general knowledge) is rejected and replaced by the fallback;
  * source names / page numbers are attached from the chunks that actually
    support the answer, not from the model's own (possibly wrong) citation.
"""

import logging
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.embeddings import (
    DEFAULT_OLLAMA_BASE_URL,
    get_embeddings,
    get_ollama_base_url,
)
from src.pdf_loader import load_pdfs_from_directory
from src.retriever import format_context_for_llm, retrieve_relevant_chunks
from src.text_chunker import calculate_chunk_statistics, chunk_documents
from src.vector_store import get_or_create_vectorstore

load_dotenv()
logger = logging.getLogger(__name__)

# Dedicated logger for retrieval / validation diagnostics (enable with RAG_DEBUG=1
# or `python main.py --debug`).
debug_logger = logging.getLogger("rag.debug")

DEFAULT_CHAT_MODEL = "llama3.2"

# Minimum share of the answer's content words that must appear in the retrieved
# chunks for the answer to count as "grounded in the PDFs".
MIN_GROUNDING_RATIO = float(os.getenv("MIN_GROUNDING_RATIO", "0.5"))

# --- Prompt template ----------------------------------------------------------
NOT_FOUND_MESSAGE = "The answer was not found in the uploaded documents."

# The model is asked to emit a short sentinel (not the long user-facing sentence)
# when the context lacks the answer.  A single sentinel is far less likely to be
# appended to a real answer than a long sentence quoted twice in the prompt.
NOT_FOUND_SENTINEL = "NOT_FOUND"

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered context passages provided by the user.

Rules:
1. Use only facts that are explicitly written in the context. Never use outside knowledge.
2. If the context contains the answer, reply with a concise, direct answer in your own
   words or quoting the context. Do NOT write {NOT_FOUND_SENTINEL} in that case.
3. If the context does not contain the answer, reply with exactly one word: {NOT_FOUND_SENTINEL}
4. Never combine an answer with {NOT_FOUND_SENTINEL}. Choose one.
5. Do not add a sources list; sources are added automatically."""

USER_PROMPT_TEMPLATE = """Context passages from the uploaded PDF documents:
{context}

---
Question: {question}

Answer using only the context passages above, or reply {sentinel} if they do not contain the answer."""

# Used for one retry when the model returns an empty / citation-only answer.
RETRY_PROMPT_TEMPLATE = """Context passages from the uploaded PDF documents:
{context}

---
Question: {question}

Copy the sentence or sentences from the context passages that answer the question.
If no sentence in the context answers it, reply {sentinel}."""


# --- Debug logging ------------------------------------------------------------

def enable_debug_logging() -> None:
    """Send retrieval / validation diagnostics (logger 'rag.debug') to stderr."""
    if not any(getattr(h, "_rag_debug", False) for h in debug_logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler._rag_debug = True  # type: ignore[attr-defined]
        handler.setFormatter(logging.Formatter("[RAG-DEBUG] %(message)s"))
        debug_logger.addHandler(handler)
    debug_logger.setLevel(logging.DEBUG)
    debug_logger.propagate = False


if os.getenv("RAG_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
    enable_debug_logging()


# --- Answer validation --------------------------------------------------------

# Matches the user-facing fallback sentence (tolerant to small wording changes)
# and the sentinel token.
_FALLBACK_RE = re.compile(
    r"(?:the\s+)?answer\s+(?:was|is|could)\s*(?:not|n't)\s+(?:be\s+)?(?:found|available|present)"
    r"(?:\s+in\s+(?:the\s+)?(?:uploaded\s+|provided\s+)?(?:documents?|context|pdfs?))?\s*[.!]?"
    r"|\bNOT[_\s-]?FOUND\b\.?",
    re.IGNORECASE,
)

_CITATION_PATTERNS = [
    re.compile(r"\[\s*sources?\b[^\]]*\]", re.IGNORECASE),          # [Source: a.pdf, Page 1]
    re.compile(r"\(\s*sources?\b[^)]*\)", re.IGNORECASE),           # (Source: a.pdf, Page 1)
    re.compile(r"^\s*sources?\s*[:\-].*$", re.IGNORECASE | re.MULTILINE),  # Sources: a.pdf, Page 1
    re.compile(r"\bsources?\s*:\s*[^\n]*?\.pdf[^\n]*", re.IGNORECASE),     # ...1956. Sources: a.pdf, Page 1
]

_STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "that", "this", "with", "from", "have", "has",
    "had", "its", "their", "they", "them", "can", "will", "would", "should", "could", "which",
    "what", "when", "where", "who", "whom", "how", "why", "not", "but", "also", "such", "into",
    "than", "then", "there", "these", "those", "been", "being", "does", "did", "use", "used",
    "using", "about", "over", "more", "most", "some", "any", "each", "other", "between",
    # words models add when quoting a context ("according to the provided context ...")
    "according", "provided", "context", "passage", "passages", "document", "documents",
    "uploaded", "based", "states", "stated", "mentioned", "source", "sources", "page", "pages",
    "answer", "question", "refers", "refer",
}


def _content_words(text: str) -> set:
    """Lower-cased, lightly stemmed content words (stop-words removed)."""
    words = set()
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if len(w) < 3 or w in _STOPWORDS:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        words.add(w)
    return words


def _strip_citations(text: str) -> str:
    """Remove model-written citation markers (we attach verified citations ourselves)."""
    for pattern in _CITATION_PATTERNS:
        text = pattern.sub("", text)
    return text


def _clean(text: str) -> str:
    """Collapse whitespace and trim stray punctuation left by removals."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(" \n\t-:;,")


def find_supporting_chunks(
    answer_text: str,
    results: List[Tuple[Document, float]],
    min_support: float = 0.5,
    relative_to_best: float = 0.75,
) -> List[Tuple[Document, float, float]]:
    """
    Return retrieved chunks that support `answer_text`, as (doc, l2_score, support).

    `support` is the share of the answer's content words found in the chunk.
    A chunk is kept when its support is at least `min_support` AND at least
    `relative_to_best` x the best chunk's support (so weakly-related chunks are
    not cited); the single best chunk is always kept.
    """
    answer_words = _content_words(answer_text)
    if not answer_words:
        return []
    scored = []
    for doc, score in results:
        chunk_words = _content_words(doc.page_content)
        support = len(answer_words & chunk_words) / len(answer_words)
        scored.append((doc, float(score), support))
    best = max((s for _, _, s in scored), default=0.0)
    if best <= 0:
        return []
    cutoff = max(min_support, relative_to_best * best)
    chosen = [t for t in scored if t[2] >= cutoff or t[2] == best]
    return sorted(chosen, key=lambda t: -t[2])


def validate_answer(
    raw_answer: str,
    results: List[Tuple[Document, float]],
    min_grounding: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Validate a raw model output against the retrieved chunks.

    Returns a dict with:
      answer          : final text to show the user (fallback sentence when not found)
      found           : True if a grounded answer from the PDFs is being returned
      status          : one of
                          'grounded'                    clean grounded answer
                          'grounded_fallback_stripped'  grounded answer; stray fallback text removed
                          'model_not_found'             model said the answer is not in the context
                          'ungrounded_rejected'         answer not supported by the chunks
                          'empty_output'                nothing usable (caller may retry)
      grounding_ratio : share of answer content words found in retrieved chunks (or None)
      supporting_sources : list of {'source','page','score','support'} for supporting chunks
    """
    threshold = MIN_GROUNDING_RATIO if min_grounding is None else min_grounding
    raw = (raw_answer or "").strip()

    has_fallback = bool(_FALLBACK_RE.search(raw))
    body = _FALLBACK_RE.sub("", raw)
    body = _clean(_strip_citations(body))
    answer_words = _content_words(body)

    base: Dict[str, Any] = {
        "raw_answer": raw_answer,
        "grounding_ratio": None,
        "supporting_sources": [],
    }

    if not answer_words:
        status = "model_not_found" if has_fallback else "empty_output"
        return {**base, "answer": NOT_FOUND_MESSAGE, "found": False, "status": status}

    context_words = set()
    for doc, _ in results:
        context_words |= _content_words(doc.page_content)
    ratio = len(answer_words & context_words) / len(answer_words)
    base["grounding_ratio"] = round(ratio, 3)

    if ratio < threshold:
        return {**base, "answer": NOT_FOUND_MESSAGE, "found": False, "status": "ungrounded_rejected"}

    supporting = find_supporting_chunks(body, results)
    sources_out, seen = [], set()
    for doc, score, support in supporting:
        key = (doc.metadata.get("source", "Unknown"), doc.metadata.get("page", "N/A"))
        if key in seen:
            continue
        seen.add(key)
        sources_out.append(
            {"source": key[0], "page": key[1], "score": round(score, 4), "support": round(support, 3)}
        )
    citation = "; ".join(f"{s['source']}, Page {s['page']}" for s in sources_out)
    final = body + (f"\n\nSources: {citation}" if citation else "")

    return {
        **base,
        "answer": final,
        "found": True,
        "status": "grounded_fallback_stripped" if has_fallback else "grounded",
        "supporting_sources": sources_out,
    }


# --- LLM factory --------------------------------------------------------------

def get_chat_model_name(model: Optional[str] = None) -> str:
    """Return the configured local Ollama chat model name."""
    name = model or os.getenv("OLLAMA_CHAT_MODEL", DEFAULT_CHAT_MODEL)
    return name.strip()


def get_llm(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
) -> ChatOllama:
    """
    Create and return a local ChatOllama instance.

    Args:
        model       (str, optional): Ollama chat model name (default: llama3.2).
        base_url    (str, optional): Ollama server URL (default: http://localhost:11434).
        temperature (float, optional): Sampling temperature (default 0.0 = deterministic).
    """
    resolved_model = get_chat_model_name(model)
    resolved_url = get_ollama_base_url(base_url)
    resolved_temp = (
        float(temperature)
        if temperature is not None
        else float(os.getenv("OLLAMA_TEMPERATURE", "0"))
    )

    if not resolved_model:
        raise ValueError(
            "OLLAMA_CHAT_MODEL cannot be empty. Set it to 'llama3.2' in your .env file."
        )
    if not resolved_url.startswith(("http://", "https://")):
        raise ValueError(
            f"Invalid OLLAMA_BASE_URL='{resolved_url}'. "
            f"Expected a URL such as '{DEFAULT_OLLAMA_BASE_URL}'."
        )

    logger.info(
        f"Initialising ChatOllama: model='{resolved_model}', "
        f"base_url='{resolved_url}', temperature={resolved_temp}"
    )
    return ChatOllama(model=resolved_model, base_url=resolved_url, temperature=resolved_temp)


# Backward-compatible alias
_get_llm = get_llm


# --- Core pipeline functions --------------------------------------------------

def _invoke_llm(llm: ChatOllama, template: str, question: str, context: str) -> str:
    """Call the chat model once and return its raw text."""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=template.format(
                context=context, question=question, sentinel=NOT_FOUND_SENTINEL
            )
        ),
    ]
    response = llm.invoke(messages)
    return (response.content if hasattr(response, "content") else str(response)).strip()


def build_pipeline(
    pdf_dir: str = "data/pdfs",
    vectorstore_path: str = "vectorstore",
    force_rebuild: bool = False,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> Tuple[FAISS, List[Document], Dict[str, Any]]:
    """
    Load PDFs, split into chunks, embed via OllamaEmbeddings, and create/load FAISS.

    Returns:
        Tuple[FAISS, List[Document], dict]: (vectorstore, chunks, pipeline statistics)
    """
    logger.info("Step 1: Loading PDF documents ...")
    documents = load_pdfs_from_directory(pdf_dir)

    logger.info("Step 2: Chunking documents ...")
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    stats = calculate_chunk_statistics(documents, chunks)

    logger.info("Step 3: Initialising OllamaEmbeddings ...")
    embeddings = get_embeddings()

    logger.info("Step 4: Building / loading FAISS vectorstore ...")
    vectorstore = get_or_create_vectorstore(
        chunks=chunks,
        embeddings=embeddings,
        vectorstore_path=vectorstore_path,
        pdf_dir=pdf_dir,
        force_rebuild=force_rebuild,
    )

    stats["num_embeddings"] = stats["num_chunks"]
    stats["faiss_vectors"] = (
        int(vectorstore.index.ntotal) if hasattr(vectorstore, "index") else stats["num_chunks"]
    )
    stats["embedding_model"] = getattr(embeddings, "model", "nomic-embed-text")
    stats["chat_model"] = get_chat_model_name()
    return vectorstore, chunks, stats


def _log_retrieved(question: str, results: List[Tuple[Document, float]]) -> None:
    """Debug-log the retrieved chunks (rank, source, page, L2 distance, preview)."""
    if not debug_logger.isEnabledFor(logging.DEBUG):
        return
    debug_logger.debug(f"QUESTION: {question!r}  -> {len(results)} chunk(s) retrieved")
    for rank, (doc, score) in enumerate(results, 1):
        preview = doc.page_content[:110].replace("\n", " ")
        debug_logger.debug(
            f"  chunk[{rank}] {doc.metadata.get('source')} page={doc.metadata.get('page')} "
            f"chunk_index={doc.metadata.get('chunk_index')} L2={float(score):.4f} "
            f"len={len(doc.page_content)} :: {preview!r}"
        )


def answer_question(
    question: str,
    vectorstore: FAISS,
    top_k: Optional[int] = None,
    llm: Optional[ChatOllama] = None,
    on_retrieved: Optional[Callable[[List[Tuple[Document, float]]], None]] = None,
) -> Dict[str, Any]:
    """
    Retrieve the top-K chunks, generate an answer with local ChatOllama, and
    validate it against the retrieved PDF context.

    Args:
        question     : The user's question.
        vectorstore  : Loaded FAISS vectorstore.
        top_k        : Number of chunks to retrieve (default 3).
        llm          : Optional pre-initialised ChatOllama instance.
        on_retrieved : Optional callback called with the retrieved
                       [(Document, score), ...] BEFORE the LLM is invoked
                       (used by the CLI to show the chunks first).

    Returns:
        dict with keys:
          question, answer, found, status, grounding_ratio, raw_answer,
          sources (all retrieved chunks), supporting_sources (chunks that back
          the answer), retrieved_chunks, context
    """
    top_k = top_k if top_k is not None else int(os.getenv("TOP_K_RESULTS", 3))

    # 1) Semantic retrieval
    results = retrieve_relevant_chunks(question, vectorstore, top_k=top_k)
    _log_retrieved(question, results)
    if on_retrieved is not None:
        on_retrieved(results)

    context = format_context_for_llm(results)

    sources: List[Dict[str, Any]] = []
    for doc, score in results:
        entry = {
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "N/A"),
            "score": round(float(score), 4),
        }
        if entry not in sources:
            sources.append(entry)

    # 2) Generation + validation
    if not results:
        verdict: Dict[str, Any] = {
            "answer": NOT_FOUND_MESSAGE, "found": False, "status": "no_chunks_retrieved",
            "grounding_ratio": None, "raw_answer": "", "supporting_sources": [],
        }
    else:
        try:
            chat_model = llm if llm is not None else get_llm()
            raw = _invoke_llm(chat_model, USER_PROMPT_TEMPLATE, question, context)
            debug_logger.debug(f"RAW LLM OUTPUT: {raw!r}")
            verdict = validate_answer(raw, results)

            if verdict["status"] == "empty_output":
                debug_logger.debug("Empty/citation-only model output -> retrying once (extractive prompt)")
                raw = _invoke_llm(chat_model, RETRY_PROMPT_TEMPLATE, question, context)
                debug_logger.debug(f"RAW LLM OUTPUT (retry): {raw!r}")
                verdict = validate_answer(raw, results)
                if verdict["status"] == "empty_output":
                    verdict["status"] = "empty_output_after_retry"
        except Exception as exc:
            logger.error(f"Ollama LLM call failed: {type(exc).__name__}: {exc}")
            verdict = {
                "answer": (
                    f"[Ollama Error] Could not generate answer ({type(exc).__name__}: {exc}). "
                    f"Ensure Ollama is running and model '{get_chat_model_name()}' is pulled "
                    f"(`ollama pull {get_chat_model_name()}`)."
                ),
                "found": False, "status": "llm_error", "grounding_ratio": None,
                "raw_answer": "", "supporting_sources": [],
            }

    debug_logger.debug(
        f"VALIDATION: status={verdict['status']} found={verdict['found']} "
        f"grounding_ratio={verdict['grounding_ratio']} "
        f"supporting={[(s['source'], s['page'], s['support']) for s in verdict['supporting_sources']]}"
    )

    return {
        "question": question,
        "answer": verdict["answer"],
        "found": verdict["found"],
        "status": verdict["status"],
        "grounding_ratio": verdict["grounding_ratio"],
        "raw_answer": verdict["raw_answer"],
        "sources": sources,
        "supporting_sources": verdict["supporting_sources"],
        "retrieved_chunks": results,
        "context": context,
    }
