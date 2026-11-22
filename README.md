# PDF RAG Assistant — Offline Local Ollama + FAISS + LangChain

A complete, **100% offline** Retrieval-Augmented Generation (RAG) pipeline that answers questions from PDF documents using **local Ollama models**, **LangChain**, and **FAISS** — with **zero OpenAI or Azure OpenAI API keys required**.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Project Structure](#project-structure)
4. [Setup & Installation](#setup--installation)
5. [Downloading Local Ollama Models](#downloading-local-ollama-models)
6. [Configuration (Optional `.env`)](#configuration-optional-env)
7. [Adding PDF Documents](#adding-pdf-documents)
8. [Running the Pipeline](#running-the-pipeline)
9. [Running the Jupyter Notebook](#running-the-jupyter-notebook)
10. [Running Automated Tests](#running-automated-tests)
11. [Rebuilding the Vector Database](#rebuilding-the-vector-database)
12. [Architecture Overview](#architecture-overview)
13. [Troubleshooting](#troubleshooting)

---

## Project Overview

This project implements an end-to-end offline RAG pipeline that:

- **Loads** multiple PDF documents from `data/pdfs/` using LangChain's `PyPDFLoader`
- **Preserves** source document filename and 1-indexed page number metadata on every page and chunk
- **Splits** text into overlapping segments (`chunk_size=1000`, `chunk_overlap=200`) using `RecursiveCharacterTextSplitter`
- **Generates** 768-dimensional semantic embeddings locally using `OllamaEmbeddings` (`nomic-embed-text`)
- **Stores** embeddings and metadata locally in a persistent **FAISS** vector database
- **Retrieves** the top 3 semantically matching chunks for any question
- **Generates** grounded, source-cited answers locally using `ChatOllama` (`llama3.2`), refusing to hallucinate when information is absent

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.10 or newer (tested on Python 3.14.6) |
| **Ollama** | Installed locally from [https://ollama.com](https://ollama.com) |
| **Ollama Models** | `nomic-embed-text` (embeddings) and `llama3.2` (chat LLM) |
| **API Keys** | **None required** — runs completely offline after models are pulled |

---

## Project Structure

```
rag-pdf-assistant/
│
├── data/
│   └── pdfs/                    <- Place your PDF files here (3 sample PDFs included)
│
├── vectorstore/                 <- Persisted FAISS index + source hash (auto-created)
│
├── notebooks/
│   └── RAG_Pipeline.ipynb       <- Complete executed 18-step Jupyter Notebook
│
├── src/
│   ├── __init__.py
│   ├── pdf_loader.py            <- PyPDFLoader + metadata + error handling
│   ├── text_chunker.py          <- RecursiveCharacterTextSplitter (1000/200) + stats
│   ├── embeddings.py            <- Local OllamaEmbeddings (nomic-embed-text)
│   ├── vector_store.py          <- FAISS create / save / load / smart rebuild
│   ├── retriever.py             <- Top-K semantic search + context formatting
│   └── rag_pipeline.py          <- Full pipeline orchestration + ChatOllama (llama3.2)
│
├── outputs/
│   ├── sample_results.txt       <- Real measured execution outputs
│   └── metrics.json             <- Machine-readable metrics and query results
│
├── tests/
│   ├── test_pipeline.py         <- Unit & mocked integration tests
│   └── test_faiss_real.py       <- Real FAISS persistence, reload & rebuild tests
│
├── create_sample_pdfs.py        <- Script to regenerate the 3 sample PDFs
├── run_demo.py                  <- Non-interactive demo runner (writes outputs/)
├── main.py                      <- Interactive CLI entry point
├── .env.example                 <- Optional Ollama environment settings template
├── .gitignore
├── requirements.txt
├── README.md
└── PROJECT_REPORT.md
```

---

## Setup & Installation

### Step 1 — Open the project folder

```powershell
cd D:\Ass-1\rag-pdf-assistant
```

### Step 2 — Create and activate a Python virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## Downloading Local Ollama Models

Install [Ollama](https://ollama.com), ensure the Ollama service is running (`ollama serve` if not already running as a background app), and pull both required models once:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

Verify both models are available locally:

```bash
ollama list
```

You should see `nomic-embed-text` and `llama3.2` listed. Once downloaded, **no internet connection or API keys are needed**.

---

## Configuration (Optional `.env`)

The pipeline works out of the box with built-in defaults (`http://localhost:11434`, `nomic-embed-text`, `llama3.2`). To customise settings, copy `.env.example` to `.env`:

```powershell
copy .env.example .env
```

Available settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama server address |
| `OLLAMA_CHAT_MODEL` | `llama3.2` | Local Ollama model for answer generation |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Local Ollama model for vector embeddings |
| `OLLAMA_TEMPERATURE` | `0` | Sampling temperature (`0` = deterministic, factual) |
| `CHUNK_SIZE` | `1000` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlapping characters between consecutive chunks |
| `TOP_K_RESULTS` | `3` | Number of matching chunks retrieved per query |
| `PDF_DIR` | `data/pdfs` | Directory containing input PDF documents |
| `VECTORSTORE_PATH` | `vectorstore` | Directory where the FAISS index is saved |

---

## Adding PDF Documents

Place any 2–3 (or more) PDF files inside `data/pdfs/`. Three sample PDFs are already included:
- `01_intro_to_ai.pdf`
- `02_machine_learning.pdf`
- `03_deep_learning_rag.pdf`

If you ever want to regenerate those sample PDFs:

```bash
python create_sample_pdfs.py
```

---

## Running the Pipeline

### Interactive CLI

```bash
python main.py
```

The assistant displays document/chunk statistics, retrieves the top 3 matching chunks with source filename, page number, and L2 distance, and prints the grounded answer generated by local `llama3.2`. Ask as many questions as you like; type `exit` or `quit` to stop.

### Non-Interactive Batch Demo

```bash
python run_demo.py
```

Runs 5 sample questions (4 in-domain + 1 out-of-domain) and writes actual metrics and outputs to `outputs/sample_results.txt` and `outputs/metrics.json`.

---

## Running the Jupyter Notebook

Launch Jupyter and open `notebooks/RAG_Pipeline.ipynb`:

```bash
jupyter notebook notebooks/RAG_Pipeline.ipynb
```

Or execute it headlessly from the command line:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 RAG_Pipeline.ipynb
```

The notebook walks through all 18 steps with explanatory Markdown cells, comments, and live executed outputs.

---

## Running Automated Tests

```bash
pytest tests -v
```

All 34 automated tests run locally with zero external API dependencies, covering PDF loading, error handling, chunking & statistics, metadata preservation, Ollama configuration validation, real FAISS persistence/reloading/change detection, semantic retrieval, and end-to-end RAG orchestration.

---

## Answer Validation & Debugging

Small local models can occasionally append the "not found" sentence to a correct answer, return an
empty/citation-only reply, or answer from general knowledge. Every `llama3.2` output is therefore
validated against the retrieved PDF chunks (`validate_answer` in `src/rag_pipeline.py`):

| Model output | Result shown to the user |
|--------------|--------------------------|
| Grounded answer (+ stray "not found" text) | Answer only; fallback text stripped; `Sources:` added from the chunks that support it |
| Only "not found" / `NOT_FOUND` | `The answer was not found in the uploaded documents.` |
| Empty or citation-only | Retried once with an extractive prompt, then the fallback |
| Answer not supported by the retrieved chunks (general knowledge) | Rejected, fallback shown (`MIN_GROUNDING_RATIO`, default `0.5`) |

The top 3 retrieved chunks (source, page, L2 distance) are printed **before** the answer is generated.
To see retrieved chunks, raw LLM output and validation decisions, run:

```bash
python main.py --debug        # or set RAG_DEBUG=1
```

Live verification results (real Ollama) are written to `outputs/answer_validation_results.txt`
by `pytest tests/test_live_ollama.py` (skipped automatically if Ollama is not running).

---

## Rebuilding the Vector Database

The pipeline hashes PDF file contents, the active Ollama embedding model, and chunk settings into `vectorstore/source_hash.json`.

| Scenario | Behaviour |
|----------|-----------|
| First run (no index yet) | Embeds all chunks via `nomic-embed-text` and saves FAISS index |
| PDFs & settings unchanged | Loads existing FAISS index directly (skips re-embedding) |
| PDFs added, edited, or removed | Automatically rebuilds and re-saves FAISS index |
| Embedding model / chunk size changed | Automatically rebuilds and re-saves FAISS index |
| `python main.py --rebuild` or `python run_demo.py --rebuild` | Forces a full rebuild on demand |

---

## Architecture Overview

```
PDFs (data/pdfs/)
   │
   ▼
PyPDFLoader  (extracts text + preserves source filename & 1-indexed page number)
   │
   ▼
RecursiveCharacterTextSplitter  (chunk_size=1000, chunk_overlap=200)
   │
   ▼
OllamaEmbeddings  (local nomic-embed-text, 768-dim vectors)
   │
   ▼
FAISS Vector Store  (IndexFlatL2, persisted to vectorstore/faiss_index/)
   │
   ▼  ◄── User Question (embedded via nomic-embed-text)
Semantic Retrieval  (top 3 matching chunks + L2 distance + source metadata)
   │
   ▼
ChatOllama  (local llama3.2, temperature=0, strict context-only prompt)
   │
   ▼
Grounded Answer + Source Document & Page References
```

---

## Troubleshooting

### `Could not connect to local Ollama server at http://localhost:11434`
Start Ollama (launch the Ollama desktop app or run `ollama serve` in a terminal), then retry.

### `model "nomic-embed-text" not found` or `model "llama3.2" not found`
Download the missing local model once:
```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

### Slow response times on CPU
`llama3.2` runs locally on your CPU if no compatible GPU is available (~20–25 seconds per answer on typical laptops). Keeping `TOP_K_RESULTS=3` and `CHUNK_SIZE=1000` keeps prompts compact and responsive.

### `No PDF files found`
Ensure at least one readable `.pdf` file is placed in `data/pdfs/`. Scanned image-only PDFs contain no embedded text layer and will be skipped with a warning.

### PowerShell blocks `.venv\Scripts\Activate.ps1`
Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating `.venv`.
