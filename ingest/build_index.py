import faiss
import json
from pathlib import Path
import ollama
from chunker import chunk_text
from tqdm import tqdm
import numpy as np
RAW_DIR = Path("../data/raw_text")
INDEX_PATH = Path("../data/faiss.index")
META_PATH = Path("../data/metadata.json")

DIM = 768
index = faiss.IndexFlatL2(DIM)

metadata = []
chunk_id = 0

for txt_file in RAW_DIR.glob("*.txt"):
    text = txt_file.read_text(encoding="utf-8")
    chunks = chunk_text(text)

    for chunk in tqdm(chunks, desc=f"Processing {txt_file.name}"):
        emb = ollama.embeddings(
            model = "nomic-embed-text", 
            prompt = chunk
        )["embedding"]

        index.add(np.array([emb], dtype='float32'))

        metadata.append({
            "id": chunk_id,
            "source": txt_file.name,
            "text": chunk
        })
        chunk_id += 1
# Save the FAISS index
INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(INDEX_PATH))
json.dump(metadata, open(META_PATH, "w"), indent=2)

print("Index built successfully")

