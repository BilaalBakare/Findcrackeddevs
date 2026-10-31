"""
Embed developer documents and store them in a local Chroma vector database.

Usage (from project root):
    python src/processing/build_vector_database.py build
    python src/processing/build_vector_database.py search "React frontend with Tailwind"
    python src/processing/build_vector_database.py search "builds Discord bots" --label automation_bots -k 5
"""
import argparse
from pathlib import Path

import chromadb
import pandas as pd
from fastembed import TextEmbedding

ROOT = Path(__file__).resolve().parents[2]
DOCS_FILE = ROOT / "data" / "processed" / "documents.parquet"
CHROMA_DIR = ROOT / "data" / "chroma"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION = "developers"
BATCH = 128


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION,
        # cosine distance = 1 - cosine similarity. Record the model name too, so
        # we can detect if someone queries with a different (incompatible) model.
        metadata={"hnsw:space": "cosine", "embed_model": MODEL_NAME},
    )


def build():
    docs = pd.read_parquet(DOCS_FILE)
    print(f"Loaded {len(docs):,} documents. Loading model {MODEL_NAME} ...")
    model = TextEmbedding(MODEL_NAME)
    col = get_collection()

    for start in range(0, len(docs), BATCH):
        chunk = docs.iloc[start:start + BATCH]
        texts = chunk["text"].tolist()
        vectors = [v.tolist() for v in model.embed(texts)]   # generator -> list

        col.upsert(                                           # insert or update by id
            ids=[str(a) for a in chunk["actor_id"]],
            embeddings=vectors,
            documents=texts,
            metadatas=[
                {
                    "login": r.login,
                    "label": r.label,
                    "actor_id": int(r.actor_id),
                    "github_url": r.github_url,
                    "n_repos_in_doc": int(r.n_repos_in_doc),
                    "n_words": int(r.n_words),
                }
                for r in chunk.itertuples(index=False)
            ],
        )
        print(f"  embedded and stored {min(start + BATCH, len(docs)):,}/{len(docs):,}")

    print(f"\nCollection '{COLLECTION}' now holds {col.count():,} vectors")


def search(query, k=10, label=None):
    col = get_collection()
    stored = col.metadata.get("embed_model")
    if stored != MODEL_NAME:
        raise RuntimeError(f"Index built with {stored}, but querying with {MODEL_NAME}")

    model = TextEmbedding(MODEL_NAME)
    qvec = next(iter(model.query_embed(query))).tolist()      # query_embed, not embed

    res = col.query(
        query_embeddings=[qvec],
        n_results=k,
        where={"label": label} if label else None,
    )

    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({**meta, "similarity": 1 - dist, "text": doc})
    return hits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["build", "search"])
    p.add_argument("query", nargs="?")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--label", default=None)
    args = p.parse_args()

    if args.mode == "build":
        build()
        return

    if not args.query:
        p.error("search needs a query string")
    for i, h in enumerate(search(args.query, args.k, args.label), 1):
        print(f"\n{i}. {h['login']}  [{h['label']}]  similarity={h['similarity']:.3f}")
        print(f"   {h['github_url']}")
        for line in h["text"].splitlines()[2:4]:              # first two repos
            print(f"   {line[:140]}")


if __name__ == "__main__":
    main()