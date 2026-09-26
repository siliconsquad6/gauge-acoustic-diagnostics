"""Split the manuals into page-tagged chunks and embed them locally for the repair chatbot.
Run from the gauge folder (after bash get_manuals.sh):
  python app/ingest_manuals.py
Output: manuals/index.npz (embeddings) + manuals/chunks.json (text, title, page, machine)
"""
import json
from pathlib import Path
import numpy as np, fitz  # pymupdf
from sentence_transformers import SentenceTransformer

G = Path(__file__).resolve().parent.parent
M = G / "manuals"
EMB = "BAAI/bge-small-en-v1.5"
SIZE, OVERLAP = 900, 150

chunks = []
for src in json.loads((M / "sources.json").read_text()):
    pdf = M / src["file"]
    if not pdf.exists() or pdf.stat().st_size < 20_000:
        print("skip (missing or blocked):", pdf); continue
    doc = fitz.open(pdf)
    n0 = len(chunks)
    for i, page in enumerate(doc):
        text = " ".join(page.get_text().split())
        for s in range(0, max(len(text) - OVERLAP, 1), SIZE - OVERLAP):
            piece = text[s:s + SIZE]
            if len(piece) > 120:
                chunks.append({"text": piece, "page": i + 1, "title": src["title"],
                               "file": src["file"], "machine": src["machine"]})
    print(f"{src['title']}: {len(doc)} pages -> {len(chunks) - n0} chunks")

model = SentenceTransformer(EMB, device="cuda")
emb = model.encode([c["text"] for c in chunks], batch_size=64, normalize_embeddings=True, show_progress_bar=True)
np.savez(M / "index.npz", emb=emb.astype(np.float32))
(M / "chunks.json").write_text(json.dumps(chunks))
print(f"indexed {len(chunks)} chunks -> manuals/index.npz")
