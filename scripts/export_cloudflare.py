import json
from pathlib import Path

import chromadb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
col = chromadb.PersistentClient(path=str(ROOT / "data" / "chroma")).get_collection("developers")
data = col.get(include=["embeddings", "documents", "metadatas"])

vecs = np.array(data["embeddings"], dtype="float32")
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)      # unit length: dot product = cosine similarity

out = ROOT / "public" / "data"
(out / "docs").mkdir(parents=True, exist_ok=True)
vecs.astype("<f4").tofile(out / "vectors.bin")            # raw little-endian float32

for i, (m, text) in enumerate(zip(data["metadatas"], data["documents"])):
    doc = {"login": m["login"], "github_url": m["github_url"], "label": m["label"], "text": text}
    (out / "docs" / f"{i}.json").write_text(json.dumps(doc))

print("exported", len(vecs), "developers; vectors.bin bytes:", (out / "vectors.bin").stat().st_size)