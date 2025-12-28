import faiss
import json
import numpy as np
from pathlib import Path
import ollama

_index = None
_metadata = None

DATA_DIR = Path("data")

def _load_index():
    global _index, _metadata
    if _index is None:
        _index = faiss.read_index(str(DATA_DIR / "faiss.index"))
        with open(DATA_DIR / "metadata.json") as f:
            _metadata = json.load(f)

def retrieve(query: str, k: int = 4) -> str:
    _load_index()

    emb = ollama.embeddings(
        model="nomic-embed-text",
        prompt=query
    )["embedding"]

    D, I = _index.search(
        np.array([emb], dtype="float32"),
        k
    )

    chunks = []
    sources = []
    for idx in I[0]:
        meta = _metadata[idx]
        chunks.append(_metadata[idx]["text"])
        sources.append({
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", None),
            "chunk_id": idx
        })

    context = "\n\n".join(chunks)
    context = context[:40000]

    return context, sources
