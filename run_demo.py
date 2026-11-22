"""
run_demo.py
===========
Runs the complete local Ollama RAG pipeline non-interactively and saves REAL
measured metrics and query outputs to:
  - outputs/sample_results.txt
  - outputs/metrics.json

Usage:
    python run_demo.py
    python run_demo.py --rebuild

Nothing is hardcoded or fabricated: every metric, similarity score, and answer
is produced by actual execution against local Ollama (`nomic-embed-text` + `llama3.2`).
"""

import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from src.embeddings import get_embeddings, validate_embedding_config  # noqa: E402
from src.rag_pipeline import (  # noqa: E402
    answer_question,
    build_pipeline,
    get_chat_model_name,
    get_llm,
)

QUESTIONS = [
    "What is Artificial Intelligence?",
    "What are the types of machine learning?",
    "What is Retrieval-Augmented Generation (RAG)?",
    "What is backpropagation?",
    "What is quantum computing?",  # Not in the uploaded documents
]


def main():
    force_rebuild = "--rebuild" in sys.argv
    out = []
    p = out.append

    cfg = validate_embedding_config(check_server=True)
    chat_model = get_chat_model_name()

    p("RAG PIPELINE - ACTUAL EXECUTION RESULTS (LOCAL OLLAMA)")
    p("=" * 70)
    p(f"Embedding config : {cfg}")
    p(f"Chat model       : {chat_model} (local Ollama)")

    # Measure actual embedding dimension from Ollama
    emb = get_embeddings()
    sample_vec = emb.embed_query("dimension check")
    embedding_dim = len(sample_vec)

    t0 = time.time()
    vs, chunks, stats = build_pipeline(
        str(ROOT / os.getenv("PDF_DIR", "data/pdfs")),
        str(ROOT / os.getenv("VECTORSTORE_PATH", "vectorstore")),
        force_rebuild=force_rebuild,
    )
    build_seconds = round(time.time() - t0, 2)

    top_k = int(os.getenv("TOP_K_RESULTS", 3))
    stats["embedding_dimensions"] = embedding_dim
    stats["faiss_vectors"] = int(vs.index.ntotal)
    stats["top_k"] = top_k
    stats["build_seconds"] = build_seconds

    p(f"Pipeline build   : {build_seconds}s")
    p("")
    p("METRICS (computed from actual execution)")
    for k, v in stats.items():
        p(f"  {k:24s}: {v}")

    llm = get_llm()
    answers = []

    for idx, q in enumerate(QUESTIONS, start=1):
        qt0 = time.time()
        r = answer_question(q, vs, top_k=top_k, llm=llm)
        elapsed = round(time.time() - qt0, 2)

        p("")
        p("-" * 70)
        p(f"QUERY {idx}: {q}  ({elapsed}s)")
        p("Top matching chunks:")
        chunk_records = []
        for rank, (doc, score) in enumerate(r["retrieved_chunks"], 1):
            snippet = doc.page_content[:180].replace("\n", " ")
            p(
                f"  [{rank}] {doc.metadata['source']} (Page {doc.metadata['page']}, "
                f"Chunk #{doc.metadata.get('chunk_index', 'N/A')}) | "
                f"L2 distance={score:.4f} | {snippet}..."
            )
            chunk_records.append(
                {
                    "rank": rank,
                    "source": doc.metadata["source"],
                    "page": doc.metadata["page"],
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "score": round(float(score), 4),
                    "preview": snippet,
                }
            )
        p(f"AI ANSWER ({chat_model}):")
        p("  " + r["answer"].replace("\n", "\n  "))
        p(
            "RETRIEVED SOURCES: "
            + "; ".join(
                f"{s['source']} (Page {s['page']}, L2={s['score']})"
                for s in r["sources"]
            )
        )

        answers.append(
            {
                "question": q,
                "elapsed_seconds": elapsed,
                "top_chunks": chunk_records,
                "answer": r["answer"],
                "sources": r["sources"],
            }
        )

    outputs_dir = ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    (outputs_dir / "sample_results.txt").write_text("\n".join(out), encoding="utf-8")
    (outputs_dir / "metrics.json").write_text(
        json.dumps(
            {
                "metrics": stats,
                "embedding_config": cfg,
                "chat_model": chat_model,
                "answers": answers,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("\n".join(out))


if __name__ == "__main__":
    main()
