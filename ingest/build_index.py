import faiss
import json
from pathlib import Path
import ollama
import pdfplumber
from chunker import chunk_text
from tqdm import tqdm
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR   = BASE_DIR / "notes"
INDEX_PATH = BASE_DIR / "data" / "faiss.index"
META_PATH  = BASE_DIR / "data" / "metadata.json"


DIM = 768
index = faiss.IndexFlatL2(DIM)

metadata = []
chunk_id = 0

for pdf_file in RAW_DIR.glob("*.pdf"):
    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                continue

            chunks = chunk_text(text)

            for chunk in tqdm(
                chunks,
                desc=f"{pdf_file.name} (page {page_num + 1})"
            ):
                emb = ollama.embeddings(
                    model="nomic-embed-text",
                    prompt=chunk
                )["embedding"]

                index.add(np.array([emb], dtype="float32"))

                metadata.append({
                    "id": chunk_id,
                    "source": pdf_file.name,
                    "page": page_num + 1,     
                    "text": chunk
                })

                chunk_id += 1

INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(INDEX_PATH))

with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print("Index built successfully")
