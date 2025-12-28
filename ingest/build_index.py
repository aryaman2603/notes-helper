import faiss
import json
import numpy as np
import ollama
import pdfplumber
from pathlib import Path
from tqdm import tqdm
from chunker import chunk_text


BASE_DIR = Path(__file__).resolve().parent.parent

NOTES_DIR = BASE_DIR / "notes"     
DATA_DIR  = BASE_DIR / "data"

DIM = 768


def build_subject_index(subject: str, pdf_dir: Path):
    print(f"\n📚 Building index for subject: {subject}")

    subject_data_dir = DATA_DIR / subject
    subject_data_dir.mkdir(parents=True, exist_ok=True)

    index_path = subject_data_dir / "faiss.index"
    meta_path  = subject_data_dir / "metadata.json"

    index = faiss.IndexFlatL2(DIM)
    metadata = []
    chunk_id = 0

    for pdf_file in pdf_dir.glob("*.pdf"):
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text:
                    continue

                chunks = chunk_text(text)

                for chunk in tqdm(
                    chunks,
                    desc=f"{pdf_file.name} (page {page_num + 1})",
                    leave=False
                ):
                    emb = ollama.embeddings(
                        model="nomic-embed-text",
                        prompt=chunk
                    )["embedding"]

                    index.add(np.array([emb], dtype="float32"))

                    metadata.append({
                        "id": chunk_id,
                        "subject": subject,
                        "source": pdf_file.name,
                        "page": page_num + 1,
                        "text": chunk
                    })

                    chunk_id += 1

    faiss.write_index(index, str(index_path))
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Finished {subject}: {len(metadata)} chunks indexed")


# ---------- Main ----------
def main():
    if not NOTES_DIR.exists():
        raise RuntimeError("notes/ directory not found")

    for subject_dir in NOTES_DIR.iterdir():
        if subject_dir.is_dir():
            build_subject_index(
                subject=subject_dir.name,
                pdf_dir=subject_dir
            )

    print("\nAll subjects indexed successfully")


if __name__ == "__main__":
    main()

