import faiss
import json
import ollama
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

index = faiss.read_index(str(BASE_DIR / "data/faiss.index"))
metadata = json.load(open(BASE_DIR / "data/metadata.json"))

def retrieve(query, k=5):
    q_emb = ollama.embeddings(
        model="nomic-embed-text", 
        prompt=query
    )["embedding"]

    q_vec = np.array([q_emb], dtype='float32')
    distances, indices = index.search(q_vec, k)

    return [metadata[i] for i in indices[0]]

if __name__ == "__main__":
    query = "Explain deadlock prevention and avoidance"
    results = retrieve(query)

    for res in results:
        print("\nSource:", res["source"])
        print("Text:", res["text"][:400])