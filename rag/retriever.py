import faiss
import json
import numpy as np
from pathlib import Path
import ollama

# Cache per subject
_INDEXES = {}
_METADATA = {}

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def _load_subject(subject: str):
    """
    Lazy-load FAISS index and metadata for a given subject
    """
    if subject in _INDEXES:
        return

    subject_dir = DATA_DIR / subject
    if not subject_dir.exists():
        raise ValueError(f"Subject '{subject}' not found")

    index_path = subject_dir / "faiss.index"
    meta_path = subject_dir / "metadata.json"

    if not index_path.exists() or not meta_path.exists():
        raise ValueError(f"Index or metadata missing for subject '{subject}'")

    _INDEXES[subject] = faiss.read_index(str(index_path))
    with open(meta_path) as f:
        _METADATA[subject] = json.load(f)


def retrieve(query: str, subject: str, k: int = 4):
    """
    Retrieve top-k relevant chunks for a query from a given subject
    """
    _load_subject(subject)

    emb = ollama.embeddings(
        model="nomic-embed-text",
        prompt=query
    )["embedding"]

    index = _INDEXES[subject]
    metadata = _METADATA[subject]

    D, I = index.search(
        np.array([emb], dtype="float32"),
        k
    )

    chunks = []
    sources = []

    for idx in I[0]:
        meta = metadata[idx]
        chunks.append(meta["text"])
        sources.append({
            "subject": subject,
            "source": meta.get("source", "unknown"),
            "page": meta.get("page"),
            "chunk_id": meta.get("id", idx)
        })

    context = "\n\n".join(chunks)
    context = context[:40000]  # safety cap

    return context, sources

