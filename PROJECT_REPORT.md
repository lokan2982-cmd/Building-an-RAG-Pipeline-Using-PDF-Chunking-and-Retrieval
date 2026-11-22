# PROJECT REPORT

## Building an Offline RAG Pipeline Using PDF Chunking, FAISS, and Local Ollama

---

**Course:** Applied AI / Machine Learning Engineering  
**Project Title:** Building an RAG Pipeline Using PDF Chunking and Retrieval  
**Technology Stack:** Python, LangChain, PyPDFLoader, RecursiveCharacterTextSplitter, FAISS, Local Ollama (`nomic-embed-text` & `llama3.2`), Jupyter Notebook  
**Execution Mode:** 100% Local / Offline (No OpenAI or Azure OpenAI API keys required)  
**Date:** November 2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Project Objective](#2-project-objective)
3. [Importance of RAG](#3-importance-of-rag)
4. [Technologies Used](#4-technologies-used)
5. [System Architecture](#5-system-architecture)
6. [Implementation Methodology](#6-implementation-methodology)
7. [PDF Loading Process](#7-pdf-loading-process)
8. [Text Chunking Methodology](#8-text-chunking-methodology)
9. [Embedding Generation (Local Ollama)](#9-embedding-generation-local-ollama)
10. [FAISS Vector Database](#10-faiss-vector-database)
11. [Semantic Retrieval](#11-semantic-retrieval)
12. [Local LLM Answer Generation (`ChatOllama`)](#12-local-llm-answer-generation-chatollama)
13. [Experimental Results and Metrics](#13-experimental-results-and-metrics)
14. [Sample Queries and Outputs](#14-sample-queries-and-outputs)
15. [Challenges Faced](#15-challenges-faced)
16. [Limitations](#16-limitations)
17. [Future Improvements](#17-future-improvements)
18. [Conclusion](#18-conclusion)

---

## 1. Introduction

Large Language Models (LLMs) excel at language understanding and synthesis, yet standalone LLMs suffer from fixed training cutoffs, lack of access to private or domain-specific PDFs, and a tendency to hallucinate plausible-sounding but incorrect facts.

Retrieval-Augmented Generation (RAG) resolves these issues by pairing an external semantic search store with an LLM: at query time, only the most relevant passages from the uploaded documents are retrieved and supplied as grounding context for the model's answer.

This project implements a complete, working, **100% offline RAG pipeline** built with Python, LangChain, FAISS, and **local Ollama models** (`nomic-embed-text` for vector embeddings and `llama3.2` for answer generation). Migrating from cloud APIs (Azure OpenAI / OpenAI) to local Ollama eliminates API costs, rate-limit errors, and external data exposure while maintaining strong semantic retrieval and grounded, cited answers.

---

## 2. Project Objective

1. Build a modular, end-to-end RAG pipeline with zero cloud API dependencies (no OpenAI or Azure OpenAI keys).
2. Load multiple PDF documents using `PyPDFLoader`, preserving filename (`source`) and page number (`page`) metadata.
3. Split extracted text into overlapping chunks (`chunk_size=1000`, `chunk_overlap=200`) using `RecursiveCharacterTextSplitter`.
4. Generate dense vector embeddings locally using `OllamaEmbeddings` (`nomic-embed-text`).
5. Create, persist, reload, and intelligently rebuild a local **FAISS** vector database.
6. Retrieve the top 3 matching chunks using semantic similarity search with distance scores and source metadata.
7. Generate grounded answers locally using `ChatOllama` (`llama3.2`) with explicit source citations and strict refusal when answers are absent from the documents.
8. Validate the entire system through automated tests (`pytest`), a CLI (`main.py`), a batch runner (`run_demo.py`), and an executed 18-step Jupyter Notebook (`RAG_Pipeline.ipynb`).

---

## 3. Importance of RAG

| Standalone LLM Limitation | How Local RAG Solves It |
|---------------------------|-------------------------|
| **Static training cutoff** | Answers reflect whatever PDFs are placed in `data/pdfs/`. |
| **Hallucination** | Context-only prompting grounds responses in retrieved passages and enforces refusal when evidence is absent. |
| **Context window constraints** | Only the top-K relevant chunks (here top 3) are sent to the LLM instead of whole documents. |
| **Lack of verifiability** | Every retrieved chunk retains its source PDF name and page number for transparent citation. |
| **Cloud cost & privacy** | Local Ollama models run completely offline on the user's machine with zero API fees and zero data leaving the device. |

---

## 4. Technologies Used

| Technology | Verified Installed Version | Role |
|------------|----------------------------|------|
| **Python** | 3.14.6 (`.venv`, Windows) | Core programming language |
| **Ollama** | 0.40.17 (local daemon) | Local model runtime (`nomic-embed-text`, `llama3.2`) |
| **langchain** | 1.5.2 | RAG orchestration framework |
| **langchain-core** | 1.8.4 | Document and message abstractions |
| **langchain-community** | 0.4.2 | `PyPDFLoader` and `FAISS` vectorstore wrapper |
| **langchain-ollama** | 1.1.0 | `OllamaEmbeddings` and `ChatOllama` integrations |
| **langchain-text-splitters** | 1.1.3 | `RecursiveCharacterTextSplitter` |
| **ollama** (Python SDK) | 0.6.3 | HTTP client used by `langchain-ollama` |
| **pypdf** | 7.1.1 | PDF text extraction backend |
| **faiss-cpu** | 1.16.0 | Vector similarity indexing and disk persistence |
| **python-dotenv** | 1.2.4 | Optional `.env` configuration loader |
| **pytest / pytest-mock** | 9.2.0 / 3.17.0 | Automated unit and integration testing |
| **ipykernel / nbconvert / nbclient** | 7.4.0 / 7.17.2 / 0.11.0 | Jupyter Notebook execution |

No OpenAI (`openai`, `langchain-openai`) or Azure OpenAI dependencies are imported or required anywhere in the active RAG pipeline.

---

## 5. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                OFFLINE LOCAL OLLAMA RAG PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INGESTION PHASE (runs on first start or whenever PDFs/config change)   │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌─────────────────────┐ │
│  │PDF Files │──▶│pdf_loader│──▶│text_chunk │──▶│   embeddings.py     │ │
│  │data/pdfs/│   │PyPDF     │   │1000 / 200 │   │OllamaEmbeddings     │ │
│  └──────────┘   └──────────┘   └───────────┘   │(nomic-embed-text)   │ │
│                                                └──────────┬──────────┘ │
│                                                           ▼            │
│                                                ┌─────────────────────┐ │
│                                                │   vector_store.py   │ │
│                                                │FAISS IndexFlatL2    │ │
│                                                │vectorstore/faiss_.. │ │
│                                                └──────────┬──────────┘ │
│                                                           │            │
│  QUERY PHASE (100% offline per user question)             │            │
│  ┌──────────┐   ┌──────────────────┐   ┌──────────────┐   │            │
│  │  User    │──▶│ Query Embedding  │──▶│ retriever.py │◄──┘            │
│  │ Question │   │ nomic-embed-text │   │ Top-3 Chunks │                │
│  └──────────┘   └──────────────────┘   └──────┬───────┘                │
│                                               ▼                        │
│                                        ┌─────────────────────┐         │
│                                        │   rag_pipeline.py   │         │
│                                        │ChatOllama (llama3.2)│         │
│                                        │temperature = 0      │         │
│                                        └──────────┬──────────┘         │
│                                                   ▼                    │
│                                        Grounded Answer + PDF Citations │
└─────────────────────────────────────────────────────────────────────────┘
```

### Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `src/pdf_loader.py` | Loads all `.pdf` files from `data/pdfs/`, skips blank/corrupted files safely, normalises `source` filename and 1-indexed `page`. |
| `src/text_chunker.py` | Splits pages with `RecursiveCharacterTextSplitter(1000, 200)`, preserves metadata, computes chunk statistics. |
| `src/embeddings.py` | Initialises local `OllamaEmbeddings(model="nomic-embed-text")` and checks local Ollama server/model availability. |
| `src/vector_store.py` | Creates, saves, and loads the FAISS index; hashes PDF contents + embedding model + chunk settings to prevent redundant re-embedding. |
| `src/retriever.py` | Runs `similarity_search_with_score(query, k=3)`, formats retrieved chunks with source/page headers for the LLM. |
| `src/rag_pipeline.py` | Orchestrates the pipeline and calls local `ChatOllama(model="llama3.2", temperature=0)` with strict grounding prompts. |
| `main.py` | Interactive offline CLI assistant (`--rebuild` supported). |
| `run_demo.py` | Non-interactive verification script that records actual metrics and query outputs to `outputs/`. |

---

## 6. Implementation Methodology

The migration from Azure OpenAI / OpenAI to local Ollama was completed methodically:

1. **Dependency cleanup**: Removed `langchain-openai`, `openai`, and `tiktoken` from `requirements.txt` and all active `src/` modules; added `langchain-ollama` and `ollama`.
2. **Local embeddings**: Replaced cloud embedding factories in `src/embeddings.py` with `OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")`, plus server/model validation helpers.
3. **Local chat generation**: Replaced `ChatOpenAI` / `AzureChatOpenAI` in `src/rag_pipeline.py` with `ChatOllama(model="llama3.2", temperature=0)` and tightened the context-only grounding prompt.
4. **Metadata & change tracking**: Preserved `source`, `page`, and `chunk_index` metadata through FAISS, and included the Ollama embedding model identifier (`ollama:nomic-embed-text`) in the vectorstore fingerprint so switching embedding models automatically triggers a rebuild.
5. **CLI, demo, tests & notebook**: Updated `main.py`, `run_demo.py`, `tests/test_pipeline.py`, `tests/test_faiss_real.py`, and `notebooks/RAG_Pipeline.ipynb`.
6. **Live validation**: Ran all 34 automated tests, executed `run_demo.py --rebuild`, smoke-tested `main.py` interactively, and executed every cell of `RAG_Pipeline.ipynb` against local Ollama.

---

## 7. PDF Loading Process

Implemented in `src/pdf_loader.py` using `PyPDFLoader`:
- Iterates through all sorted `*.pdf` files in `data/pdfs/`.
- Extracts text per page, skipping empty/blank pages.
- Normalises `metadata["source"]` to the portable filename (e.g., `01_intro_to_ai.pdf`) and converts `metadata["page"]` from 0-indexed to 1-indexed (`Page 1`, `Page 2`, ...).
- Catches exceptions on corrupted/unreadable PDFs, logging a warning rather than crashing the batch load.

---

## 8. Text Chunking Methodology

Implemented in `src/text_chunker.py` using `RecursiveCharacterTextSplitter`:
- **Separators (ordered):** `["\n\n", "\n", ". ", " ", ""]` — preserves paragraphs and sentences before falling back to word/character splits.
- **Defaults:** `chunk_size=1000` characters, `chunk_overlap=200` characters (20% overlap to retain context across chunk boundaries). Both are configurable via `.env` or arguments, with validation ensuring `chunk_overlap < chunk_size`.
- **Metadata:** Every chunk retains `source` and `page`, plus `chunk_index`, `chunk_size_setting`, and `chunk_overlap_setting`.

---

## 9. Embedding Generation (Local Ollama)

Implemented in `src/embeddings.py` using `langchain_ollama.OllamaEmbeddings`:
- **Model:** `nomic-embed-text` (local, 274 MB, **768-dimensional** dense semantic vectors).
- **Server:** Local Ollama REST endpoint (`http://localhost:11434/api/embed`).
- **Offline benefits:** Zero API keys, zero per-token costs, zero rate-limit rejections, and complete data privacy.
- **Diagnostics:** `list_installed_ollama_models()` and `validate_embedding_config(check_server=True)` inspect `/api/tags` to verify the local daemon and required models are available.

---

## 10. FAISS Vector Database

Implemented in `src/vector_store.py` using `langchain_community.vectorstores.FAISS`:
- Builds an exact L2 flat index (`IndexFlatL2`) over all 768-d chunk vectors via `FAISS.from_documents()`.
- Saves the index locally to `vectorstore/faiss_index/` (`index.faiss` + `index.pkl`).
- Stores a SHA/MD5 fingerprint (`vectorstore/source_hash.json`) combining every PDF filename and byte-level hash, the embedding model identifier (`ollama:nomic-embed-text`), and chunking settings.
- Automatically reuses the saved index when nothing changed and automatically rebuilds when PDFs or settings change (or when `--rebuild` is passed).

---

## 11. Semantic Retrieval

Implemented in `src/retriever.py`:
- Embeds the incoming question with `nomic-embed-text`.
- Searches FAISS using `vectorstore.similarity_search_with_score(query, k=3)`.
- Returns `(Document, score)` tuples ordered by ascending L2 distance (lower score = closer semantic match).
- Formats the retrieved passages into a structured context block tagged with `[Source N: <filename>, Page <page>]`.

---

## 12. Local LLM Answer Generation (`ChatOllama`)

Implemented in `src/rag_pipeline.py`:
- Uses `ChatOllama(model="llama3.2", base_url="http://localhost:11434", temperature=0)`.
- `temperature=0` produces deterministic, factual answers grounded strictly in the retrieved context.
- System prompt enforces two critical rules:
  1. Answer **only** using facts explicitly stated in the retrieved chunks, citing source filename and page number.
  2. If the context does not contain the answer, respond with **exact text**: `"The answer was not found in the uploaded documents."`

---

## 13. Experimental Results and Metrics

All values below were measured directly from live offline execution on this machine using `run_demo.py --rebuild` and the executed `notebooks/RAG_Pipeline.ipynb` (raw files: `outputs/metrics.json` and `outputs/sample_results.txt`).

| Metric | Actual Measured Value |
|--------|-----------------------|
| PDF documents processed | 3 (`01_intro_to_ai.pdf`, `02_machine_learning.pdf`, `03_deep_learning_rag.pdf`) |
| Total pages processed | 6 (2 pages per PDF) |
| Total text chunks | 11 (`01_intro_to_ai.pdf`: 3, `02_machine_learning.pdf`: 4, `03_deep_learning_rag.pdf`: 4) |
| Average chunk size | 790.1 characters |
| Minimum / maximum chunk size | 345 / 995 characters |
| Chunk size / overlap settings | 1000 / 200 characters |
| Embedding model | `nomic-embed-text` (local Ollama, **768 dimensions**) |
| Chat LLM model | `llama3.2` (local Ollama, `temperature=0.0`) |
| Embeddings generated | 11 (one per chunk) |
| Vectors stored in FAISS (`IndexFlatL2`) | 11 |
| Pipeline build time (load + chunk + embed + FAISS save) | 4.46 seconds |
| Top-K chunks retrieved per query | 3 |
| Average query + generation latency (CPU) | 23.46 seconds (range: 20.69s – 25.65s) |
| Automated test suite (`pytest tests -v`) | **34 passed, 0 failed** (1.43s) |
| Jupyter Notebook (`RAG_Pipeline.ipynb`) | **17 / 17 code cells executed, 0 errors** |

---

## 14. Sample Queries and Outputs

All query results below come verbatim from `outputs/sample_results.txt` (real `nomic-embed-text` FAISS retrieval + real `llama3.2` generation):

| # | Query | Top Retrieved Chunk | L2 Distance | Latency | Grounded? |
|---|-------|---------------------|-------------|---------|-----------|
| 1 | What is Artificial Intelligence? | `01_intro_to_ai.pdf`, Page 1 (Chunk #0) | **0.3375** | 25.65s | Yes, cited |
| 2 | What are the types of machine learning? | `02_machine_learning.pdf`, Page 1 (Chunk #3) | **0.4786** | 20.69s | Yes, cited |
| 3 | What is Retrieval-Augmented Generation (RAG)? | `03_deep_learning_rag.pdf`, Page 2 (Chunk #9) | **0.5284** | 24.18s | Yes, cited |
| 4 | What is backpropagation? | `03_deep_learning_rag.pdf`, Page 1 (Chunk #8) | **0.6416** | 24.58s | Yes, cited |
| 5 | What is quantum computing? *(not in PDFs)* | `01_intro_to_ai.pdf`, Page 1 (Chunk #1) | **0.9403** | 22.21s | Correctly refused |

### Exact Recorded `llama3.2` Responses

**Query 1 — What is Artificial Intelligence?**
> Artificial Intelligence (AI) refers to the simulation of human intelligence processes by computer systems. These processes include learning, reasoning, and self-correction.
>
> Sources: `01_intro_to_ai.pdf`, Page 1

**Query 2 — What are the types of machine learning?**
> The types of machine learning are:
> 1. Supervised Learning
> 2. Unsupervised Learning
> 3. Reinforcement Learning
>
> Sources: `02_machine_learning.pdf`, Page 1

*(Analytical note: `02_machine_learning.pdf` Page 1 also lists Semi-supervised Learning at the bottom of the page, which fell into Chunk #4 — outside the top-3 chunks retrieved for this query. This illustrates a real trade-off between chunk size and recall.)*

**Query 3 — What is Retrieval-Augmented Generation (RAG)?**
> Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models (LLMs) by retrieving relevant documents from a knowledge base and incorporating them as context before generating a response.
>
> Sources: `03_deep_learning_rag.pdf`, Page 2

**Query 4 — What is backpropagation?**
> Backpropagation is the algorithm used to calculate gradients and update weights in a neural network, propagating the error signal from the output layer back through the network.
>
> Sources: `03_deep_learning_rag.pdf`, Page 1

**Query 5 — What is quantum computing? (Out-of-domain test)**
> The answer was not found in the uploaded documents.

### Key Retrieval Analysis
- Every in-domain question retrieved the exact target document and page at **Rank 1**, with L2 distances between **0.3375** and **0.6416**.
- The out-of-domain question (`What is quantum computing?`) yielded much higher L2 distances (**0.9403 – 0.9437**), and `llama3.2` faithfully obeyed the system prompt by returning the exact refusal phrase with zero hallucination.

---

## 15. Challenges Faced

1. **Migrating away from cloud API limitations**: Initial cloud testing hit OpenAI credit limits (`429`) and incomplete Azure deployment credentials. Migrating to local Ollama (`nomic-embed-text` + `llama3.2`) solved both cost and offline reliability completely.
2. **Python 3.14 & LangChain 1.x package compatibility**: Older LangChain 0.x import paths (`langchain.vectorstores`, `langchain.text_splitter`, `langchain.chains`) are deprecated or removed. The pipeline uses current standalone packages (`langchain-ollama`, `langchain-text-splitters`, `langchain-core`, `langchain-community`).
3. **FAISS embedding-model mismatch protection**: Switching from one embedding dimension (e.g., 1024-d or 1536-d) to `nomic-embed-text` (768-d) breaks an existing FAISS index unless rebuilt. Including `ollama:<model>` in `vectorstore/source_hash.json` made model switches rebuild cleanly and automatically.
4. **Windows console encoding**: Windows terminals can default to legacy encodings (`cp1252`). Reconfiguring stdout to UTF-8 with safe fallback in `main.py` and `run_demo.py` prevented encoding crashes.
5. **Chunk boundary recall**: As seen in Query 2, a bullet point at the end of a page can spill into the next chunk when `chunk_size=1000` splits a long section; tuning `TOP_K_RESULTS` or `CHUNK_SIZE` directly controls this trade-off.

---

## 16. Limitations

| Limitation | Impact | Possible Mitigation |
|------------|--------|---------------------|
| **CPU inference latency** | ~20–25s per answer for `llama3.2` on CPU-only laptops | Use a GPU-accelerated machine or stream tokens to the user |
| **No OCR for scanned PDFs** | Image-only PDFs produce empty text in `PyPDFLoader` | Pre-process scanned documents with Tesseract OCR or `unstructured` |
| **Small test corpus** | 3 sample PDFs (6 pages, 11 chunks) | Add larger real-world manuals/reports to `data/pdfs/` |
| **Flat FAISS index** | Brute-force L2 scan scales linearly with corpus size | Switch to FAISS `IVFFlat` or `HNSW` for 100k+ chunks |
| **No conversational memory** | Each query is answered independently | Add multi-turn history / query rewriting |
| **No hard similarity threshold** | Out-of-domain queries still pass top-3 chunks to the LLM | Add an optional L2 distance cutoff to skip LLM calls when distances exceed ~0.85 |

---

## 17. Future Improvements

1. **Streaming CLI responses**: Stream tokens from `ChatOllama.stream()` for immediate visual feedback on CPU.
2. **Distance threshold guard**: Short-circuit to `"The answer was not found..."` when the top FAISS L2 distance exceeds a calibrated threshold, saving ~22s per out-of-domain query.
3. **Hybrid search (BM25 + FAISS)**: Combine keyword matching with `nomic-embed-text` dense retrieval for exact acronyms and IDs.
4. **OCR and table parsing**: Integrate `pdfplumber` or `pytesseract` for tables and scanned documents.
5. **Conversational memory & web UI**: Add multi-turn follow-ups and a local Streamlit/Gradio interface.
6. **Automated RAG evaluation**: Add faithfulness and context-recall evaluation using RAGAS or custom test sets.

---

## 18. Conclusion

This project delivers a fully functional, validated, **100% offline PDF RAG Assistant** powered by **LangChain**, **FAISS**, and **local Ollama** (`nomic-embed-text` + `llama3.2`). Every stage — PDF ingestion with metadata preservation, recursive character chunking (`1000/200`), local embedding generation (`768-d`), FAISS disk persistence with automatic change detection, top-3 semantic search, and strictly grounded local LLM answering — has been executed and verified against live local models with **34/34 passing tests**, **17/17 executed notebook cells**, and real, reproducible outputs.
