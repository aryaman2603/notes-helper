import faiss
import json
import numpy as np
import ollama
import pdfplumber
from pathlib import Path
from tqdm import tqdm
from chunker import chunk_text
import pytesseract
from pdf2image import convert_from_path


BASE_DIR = Path(__file__).resolve().parent.parent
NOTES_DIR = BASE_DIR / "notes"
DATA_DIR = BASE_DIR / "data"

DIM = 768

def smart_extract(pdf_path, page_index, plum_page):
    text = plum_page.extract_text() or ""
    clean_text = text.strip()

    if len(clean_text)<20:
        try:
            images = convert_from_path(str(pdf_path),first_page=page_index+1, last_page=page_index+1, dpi=300)
            if images:
                ocr_text = pytesseract.image_to_string(images[0])
                return f"[OCR_EXTRACT]{ocr_text}"
        except Exception as e:
            pass
    return text


def build_subject_index(subject: str, pdf_dir: Path):
    print(f"\nProcessing subject: {subject}")

    subject_data_dir = DATA_DIR / subject
    subject_data_dir.mkdir(parents=True, exist_ok=True)

    index_path = subject_data_dir / "faiss.index"
    meta_path = subject_data_dir / "metadata.json"

    
    if index_path.exists() and meta_path.exists():
        print(" Loading existing index")
        index = faiss.read_index(str(index_path))
        metadata = json.load(open(meta_path))

        existing_keys = {
            (m["source"], m["page"], m["text"])
            for m in metadata
        }

        chunk_id = max(m["id"] for m in metadata) + 1 if metadata else 0
    else:
        print("🆕 Creating new index")
        index = faiss.IndexFlatL2(DIM)
        metadata = []
        existing_keys = set()
        chunk_id = 0

    new_chunks = 0

    
    for pdf_file in pdf_dir.glob("*.pdf"):
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue

                chunks = chunk_text(text)

                for chunk in tqdm(
                    chunks,
                    desc=f"{pdf_file.name} (page {page_num})",
                    leave=False
                ):
                    key = (pdf_file.name, page_num, chunk)
                    if key in existing_keys:
                        continue

                    emb = ollama.embeddings(
                        model="nomic-embed-text",
                        prompt=chunk
                    )["embedding"]

                    index.add(np.array([emb], dtype="float32"))

                    metadata.append({
                        "id": chunk_id,
                        "subject": subject,
                        "source": pdf_file.name,
                        "page": page_num,
                        "text": chunk
                    })

                    existing_keys.add(key)
                    chunk_id += 1
                    new_chunks += 1

    \
    faiss.write_index(index, str(index_path))
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ {subject}: added {new_chunks} new chunks")


def main():
    if not NOTES_DIR.exists():
        raise RuntimeError("notes/ directory not found")

    for subject_dir in NOTES_DIR.iterdir():
        if subject_dir.is_dir():
            build_subject_index(subject_dir.name, subject_dir)

    print("\n Incremental ingestion complete")


if __name__ == "__main__":
    main()


