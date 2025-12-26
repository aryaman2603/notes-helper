import pdfplumber
from pathlib import Path

NOTES_DIR = Path("../notes")
OUT_DIR = Path("../data/raw_text")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_pdf(pdf_path):
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

if __name__ == "__main__":
    for pdf in NOTES_DIR.glob("*.pdf"):
        print(f"Extracting text from {pdf.name}...")
        txt = extract_pdf(pdf)
        (OUT_DIR / f"{pdf.stem}.txt").write_text(txt, encoding="utf-8")
        