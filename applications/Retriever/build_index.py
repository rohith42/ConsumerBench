import os, orjson
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CORPUS_JSONL = "ragqa_arena_tech_corpus.jsonl"
OUT_DIR = "rag_index"
os.makedirs(OUT_DIR, exist_ok=True)

INDEX_PATH = os.path.join(OUT_DIR, "corpus.faiss")
DOCS_PATH  = os.path.join(OUT_DIR, "docs.jsonl")
META_PATH  = os.path.join(OUT_DIR, "meta.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # CPU-friendly
MAX_CHARS = 6000
BATCH = 64

texts = []
with open(CORPUS_JSONL, "rb") as f_in, open(DOCS_PATH, "wb") as f_docs:
    for i, line in enumerate(f_in):
        t = orjson.loads(line).get("text", "")[:MAX_CHARS]
        texts.append(t)
        f_docs.write(orjson.dumps({"doc_id": i, "text": t}) + b"\n")

model = SentenceTransformer(MODEL_NAME)
emb = model.encode(
    texts, batch_size=BATCH, show_progress_bar=True,
    normalize_embeddings=True
).astype("float32")

dim = emb.shape[1]
index = faiss.IndexFlatIP(dim)  # cosine via normalized vecs
index.add(emb)
faiss.write_index(index, INDEX_PATH)

with open(META_PATH, "wb") as f:
    f.write(orjson.dumps(
        {"model": MODEL_NAME, "dim": dim, "num_docs": len(texts), "max_chars": MAX_CHARS},
        option=orjson.OPT_INDENT_2
    ))

print(f"Saved: {INDEX_PATH}, {DOCS_PATH}, {META_PATH}")