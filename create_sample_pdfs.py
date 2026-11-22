"""
create_sample_pdfs.py
=====================
Helper script to generate 3 sample PDF files for testing the RAG pipeline.
Run this script once to populate the data/pdfs/ directory.

Usage:
    python create_sample_pdfs.py
"""

import sys
from pathlib import Path

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False


def create_pdfs_with_fpdf(output_dir: Path):
    """Create sample PDFs using the fpdf2 library."""

    # ── PDF 1: Introduction to Artificial Intelligence ───────────────────
    pdf1 = FPDF()
    pdf1.set_auto_page_break(auto=True, margin=15)
    pdf1.add_page()
    pdf1.set_font("Helvetica", "B", 16)
    pdf1.cell(0, 10, "Introduction to Artificial Intelligence", new_x="LMARGIN", new_y="NEXT")
    pdf1.set_font("Helvetica", size=11)
    pdf1.multi_cell(0, 8,
        "Chapter 1: What is Artificial Intelligence?\n\n"
        "Artificial intelligence (AI) refers to the simulation of human intelligence processes "
        "by computer systems. These processes include learning (the acquisition of information "
        "and rules for using the information), reasoning (using rules to reach approximate or "
        "definite conclusions), and self-correction.\n\n"
        "AI was founded as an academic discipline in 1956 at the Dartmouth Conference, where "
        "John McCarthy coined the term. The field grew out of the belief that every aspect of "
        "learning or feature of intelligence can in principle be so precisely described that a "
        "machine can be made to simulate it.\n\n"
        "Chapter 2: Types of Artificial Intelligence\n\n"
        "AI systems are broadly classified into two categories: Narrow AI (also called Weak AI) "
        "and General AI (also called Strong AI). Narrow AI is designed to perform a specific task, "
        "such as facial recognition or internet searches. General AI, by contrast, is a system "
        "that can perform any intellectual task that a human being can.\n\n"
        "Current AI systems are all narrow AI, including even very powerful systems like GPT-4 "
        "and other large language models.\n\n"
        "Chapter 3: Applications of AI\n\n"
        "AI has found applications across many domains:\n"
        "- Healthcare: AI assists in medical imaging, drug discovery, and personalised medicine.\n"
        "- Finance: Fraud detection, algorithmic trading, and credit scoring.\n"
        "- Transportation: Self-driving cars and traffic optimisation systems.\n"
        "- Education: Personalised learning platforms and intelligent tutoring systems.\n"
        "- Natural Language Processing: Chatbots, translation, and document summarisation.\n"
    )
    pdf1.add_page()
    pdf1.set_font("Helvetica", "B", 14)
    pdf1.cell(0, 10, "Chapter 4: Challenges and Ethics in AI", new_x="LMARGIN", new_y="NEXT")
    pdf1.set_font("Helvetica", size=11)
    pdf1.multi_cell(0, 8,
        "The development of AI raises several ethical concerns that researchers and policymakers "
        "must address:\n\n"
        "Bias and Fairness: AI systems can perpetuate or even amplify biases present in training "
        "data. For example, facial recognition systems have been shown to have higher error rates "
        "for darker-skinned individuals.\n\n"
        "Privacy: AI systems often require large amounts of data, raising concerns about how "
        "personal data is collected, used, and stored.\n\n"
        "Transparency: Many AI systems, particularly deep learning models, operate as 'black boxes', "
        "making it difficult to understand how they arrive at decisions.\n\n"
        "Job Displacement: Automation through AI threatens to displace workers in many sectors.\n\n"
        "Safety: Ensuring AI systems behave safely and as intended is a critical research area."
    )
    out1 = output_dir / "01_intro_to_ai.pdf"
    pdf1.output(str(out1))
    print(f"Created: {out1}")

    # ── PDF 2: Machine Learning Fundamentals ─────────────────────────────
    pdf2 = FPDF()
    pdf2.set_auto_page_break(auto=True, margin=15)
    pdf2.add_page()
    pdf2.set_font("Helvetica", "B", 16)
    pdf2.cell(0, 10, "Machine Learning Fundamentals", new_x="LMARGIN", new_y="NEXT")
    pdf2.set_font("Helvetica", size=11)
    pdf2.multi_cell(0, 8,
        "Chapter 1: Introduction to Machine Learning\n\n"
        "Machine Learning (ML) is a subset of artificial intelligence that provides systems the "
        "ability to automatically learn and improve from experience without being explicitly "
        "programmed. ML focuses on the development of computer programs that can access data and "
        "use it to learn for themselves.\n\n"
        "Arthur Samuel coined the term 'machine learning' in 1959, defining it as the 'field of "
        "study that gives computers the ability to learn without being explicitly programmed.'\n\n"
        "Chapter 2: Types of Machine Learning\n\n"
        "Supervised Learning: The algorithm is trained on labelled data. Examples include "
        "classification (spam detection) and regression (house price prediction).\n\n"
        "Unsupervised Learning: The algorithm finds patterns in unlabelled data. Examples include "
        "clustering (customer segmentation) and dimensionality reduction (PCA).\n\n"
        "Reinforcement Learning: An agent learns by interacting with an environment, receiving "
        "rewards or penalties for actions. Used in game playing (AlphaGo) and robotics.\n\n"
        "Semi-supervised Learning: Combines small amounts of labelled data with large amounts of "
        "unlabelled data during training.\n"
    )
    pdf2.add_page()
    pdf2.set_font("Helvetica", "B", 14)
    pdf2.cell(0, 10, "Chapter 3: Key ML Algorithms", new_x="LMARGIN", new_y="NEXT")
    pdf2.set_font("Helvetica", size=11)
    pdf2.multi_cell(0, 8,
        "Linear Regression: Models the relationship between a dependent variable and one or more "
        "independent variables using a linear equation.\n\n"
        "Decision Trees: A flowchart-like model that makes decisions by splitting data based on "
        "feature values. Easy to interpret and visualise.\n\n"
        "Random Forests: An ensemble of decision trees, combining their predictions to improve "
        "accuracy and reduce overfitting.\n\n"
        "Support Vector Machines (SVM): Finds the optimal hyperplane that separates classes in "
        "high-dimensional space.\n\n"
        "Neural Networks: Inspired by the human brain, they consist of layers of interconnected "
        "neurons that learn complex patterns from data.\n\n"
        "k-Nearest Neighbors (kNN): Classifies a data point based on the majority class of its "
        "k nearest neighbours in the feature space.\n\n"
        "Chapter 4: Model Evaluation Metrics\n\n"
        "Accuracy: The proportion of correct predictions to total predictions.\n"
        "Precision: The proportion of true positives among all positive predictions.\n"
        "Recall: The proportion of true positives among all actual positives.\n"
        "F1-Score: The harmonic mean of precision and recall.\n"
        "ROC-AUC: Measures the ability of a classifier to distinguish between classes.\n"
        "Mean Squared Error (MSE): Used for regression, measures average squared difference "
        "between predicted and actual values.\n"
    )
    out2 = output_dir / "02_machine_learning.pdf"
    pdf2.output(str(out2))
    print(f"Created: {out2}")

    # ── PDF 3: Deep Learning and Neural Networks ──────────────────────────
    pdf3 = FPDF()
    pdf3.set_auto_page_break(auto=True, margin=15)
    pdf3.add_page()
    pdf3.set_font("Helvetica", "B", 16)
    pdf3.cell(0, 10, "Deep Learning and Neural Networks", new_x="LMARGIN", new_y="NEXT")
    pdf3.set_font("Helvetica", size=11)
    pdf3.multi_cell(0, 8,
        "Chapter 1: What is Deep Learning?\n\n"
        "Deep learning is a subset of machine learning that uses artificial neural networks with "
        "multiple layers (hence 'deep') to progressively extract higher-level features from raw "
        "input. For example, in image processing, lower layers may identify edges, while higher "
        "layers identify human faces.\n\n"
        "Deep learning has revolutionised fields such as computer vision, natural language "
        "processing, speech recognition, and game playing.\n\n"
        "Chapter 2: Neural Network Architecture\n\n"
        "A neural network consists of:\n"
        "Input Layer: Receives raw input data (pixels, words, sensor readings).\n"
        "Hidden Layers: Intermediate layers that learn representations of the data. The number "
        "of hidden layers determines the 'depth' of the network.\n"
        "Output Layer: Produces the final prediction or classification.\n\n"
        "Activation Functions: Functions like ReLU, Sigmoid, and Tanh introduce non-linearity "
        "into the network, allowing it to learn complex patterns.\n\n"
        "Chapter 3: Training Neural Networks\n\n"
        "Backpropagation: The algorithm used to calculate gradients and update weights. It "
        "propagates the error signal from the output layer back through the network.\n\n"
        "Gradient Descent: An optimisation algorithm that minimises the loss function by "
        "iteratively updating model parameters in the direction of steepest descent.\n\n"
        "Overfitting and Regularisation: When a model memorises training data, it fails to "
        "generalise. Techniques like dropout, L1/L2 regularisation, and data augmentation "
        "help prevent overfitting.\n"
    )
    pdf3.add_page()
    pdf3.set_font("Helvetica", "B", 14)
    pdf3.cell(0, 10, "Chapter 4: Large Language Models and RAG", new_x="LMARGIN", new_y="NEXT")
    pdf3.set_font("Helvetica", size=11)
    pdf3.multi_cell(0, 8,
        "Large Language Models (LLMs) such as GPT-4, Claude, and Gemini are deep learning models "
        "trained on vast amounts of text data. They are capable of generating human-like text, "
        "answering questions, translating languages, and much more.\n\n"
        "Retrieval-Augmented Generation (RAG) is a technique that enhances LLMs by retrieving "
        "relevant documents from a knowledge base and incorporating them as context before "
        "generating a response. This addresses the limitation of LLMs having a fixed training "
        "knowledge cutoff and prevents hallucination by grounding responses in retrieved facts.\n\n"
        "RAG Pipeline Components:\n"
        "1. Document Loading: PDFs, Word documents, and web pages are loaded and parsed.\n"
        "2. Text Chunking: Documents are split into smaller, overlapping segments.\n"
        "3. Embedding Generation: Each chunk is converted into a vector representation.\n"
        "4. Vector Storage: Embeddings are stored in a vector database like FAISS or Pinecone.\n"
        "5. Semantic Retrieval: User queries are embedded and matched against stored vectors.\n"
        "6. Answer Generation: Retrieved chunks are passed as context to an LLM for answering.\n\n"
        "FAISS (Facebook AI Similarity Search) is an efficient library for similarity search and "
        "clustering of dense vectors. It enables fast nearest-neighbour search, which is essential "
        "for the retrieval step in a RAG pipeline.\n\n"
        "LangChain is a popular framework for building LLM applications, providing abstractions "
        "for document loaders, text splitters, embeddings, vector stores, and chains."
    )
    out3 = output_dir / "03_deep_learning_rag.pdf"
    pdf3.output(str(out3))
    print(f"Created: {out3}")


def create_pdfs_with_pypdf(output_dir: Path):
    """
    Fallback: create minimal PDFs with pypdf (content is limited without
    a layout library, so we write raw PDF syntax).
    """
    pdfs = {
        "01_intro_to_ai.pdf": (
            "Introduction to Artificial Intelligence\n\n"
            "AI refers to the simulation of human intelligence by machines. "
            "Founded in 1956 at the Dartmouth Conference, AI includes learning, "
            "reasoning, and self-correction. Applications span healthcare, finance, "
            "transportation, and natural language processing."
        ),
        "02_machine_learning.pdf": (
            "Machine Learning Fundamentals\n\n"
            "Machine Learning is a subset of AI that enables systems to learn from "
            "experience. Types include supervised, unsupervised, and reinforcement learning. "
            "Key algorithms: linear regression, decision trees, random forests, SVM, neural networks."
        ),
        "03_deep_learning_rag.pdf": (
            "Deep Learning and RAG\n\n"
            "Deep learning uses multi-layer neural networks. RAG (Retrieval-Augmented Generation) "
            "enhances LLMs by retrieving relevant documents from a vector database before "
            "generating answers. It prevents hallucination and grounds responses in facts. "
            "FAISS enables fast similarity search over dense embeddings."
        ),
    }

    for filename, content in pdfs.items():
        # Write a minimal valid PDF manually
        lines = content.split("\n")
        pdf_content = _make_simple_pdf(lines)
        out = output_dir / filename
        out.write_bytes(pdf_content)
        print(f"Created (minimal): {out}")


def _make_simple_pdf(lines):
    """Generate minimal valid PDF bytes for text content."""
    # Build PDF page content
    text_lines = []
    y = 750
    for line in lines:
        line = line.strip()
        if not line:
            y -= 10
            continue
        # Escape PDF special chars
        line = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        text_lines.append(f"BT /F1 11 Tf 50 {y} Td ({line}) Tj ET")
        y -= 15

    page_stream = "\n".join(text_lines)
    stream_bytes = page_stream.encode("latin-1", errors="replace")
    stream_len = len(stream_bytes)

    # Minimal PDF structure
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        + f"4 0 obj\n<< /Length {stream_len} >>\nstream\n".encode()
        + stream_bytes
        + b"\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    )

    # Cross-reference table
    xref_offset = len(pdf)
    objects = [
        0,
        pdf.index(b"1 0 obj"),
        pdf.index(b"2 0 obj"),
        pdf.index(b"3 0 obj"),
        pdf.index(b"4 0 obj"),
        pdf.index(b"5 0 obj"),
    ]
    xref = b"xref\n"
    xref += f"0 {len(objects)}\n".encode()
    xref += b"0000000000 65535 f \n"
    for offset in objects[1:]:
        xref += f"{offset:010d} 00000 n \n".encode()

    trailer = (
        f"trailer\n<< /Size {len(objects)} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()

    return pdf + xref + trailer


if __name__ == "__main__":
    output_dir = Path("data/pdfs")
    output_dir.mkdir(parents=True, exist_ok=True)

    if FPDF_AVAILABLE:
        print("Using fpdf2 to create sample PDFs …")
        create_pdfs_with_fpdf(output_dir)
    else:
        print("fpdf2 not installed. Creating minimal PDFs …")
        create_pdfs_with_pypdf(output_dir)

    print(f"\nDone! PDF files are in: {output_dir.resolve()}")
    print("Run the RAG pipeline with: python main.py")
