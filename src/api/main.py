import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import chromadb
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastembed import TextEmbedding
from pydantic import BaseModel, Field
import json

from src.generation.report import SYSTEM, build_prompt, call_llm

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "BAAI/bge-small-en-v1.5"

RATE_LIMIT = 5            # requests per IP
RATE_WINDOW = 60          # per this many seconds
CACHE_TTL = 3600          # seconds
CACHE_MAX = 200

state = {}
embed_lock = threading.Lock()   # one embedding at a time keeps memory low
hits_by_ip = defaultdict(deque)
cache = {}


@asynccontextmanager
async def lifespan(app):
    # heavy things load ONCE, when the server starts
    state["model"] = TextEmbedding(MODEL_NAME, threads=2)
    client = chromadb.PersistentClient(path=str(ROOT / "data" / "chroma"))
    state["col"] = client.get_collection("developers")
    print("ready, vectors:", state["col"].count())
    yield


app = FastAPI(title="FindMeTheCrackedDevs", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(","),
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=300)
    k: int = Field(default=5, ge=1, le=10)


def check_rate_limit(ip):
    now = time.time()
    q = hits_by_ip[ip]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(429, "Too many requests. Try again in a minute.")
    q.append(now)


def retrieve(query, k):
    with embed_lock:
        qvec = next(iter(state["model"].query_embed(query))).tolist()
    res = state["col"].query(query_embeddings=[qvec], n_results=k)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({**meta, "similarity": round(1 - dist, 3), "text": doc})
    return hits


@app.get("/api/health")
def health():
    return {"status": "ok", "vectors": state["col"].count()}


@app.post("/api/search")
def search(req: SearchRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    check_rate_limit(ip)

    key = (req.query.strip().lower(), req.k)
    cached = cache.get(key)
    if cached and time.time() - cached[0] < CACHE_TTL:
        return cached[1]

    hits = retrieve(req.query, req.k)

    try:
        raw = call_llm(build_prompt(req.query, hits))
        parsed = {item["id"]: item for item in json.loads(raw)["results"]}
    except requests.Timeout:
        raise HTTPException(504, "The language model timed out. Please retry.")
    except (requests.RequestException, RuntimeError):
        raise HTTPException(502, "The language model service failed.")
    except (json.JSONDecodeError, KeyError, TypeError):
        parsed = {}   # model broke the format: still return the retrieved people

    results = []
    for i, h in enumerate(hits, 1):
        item = parsed.get(i, {})
        results.append({
            "login": h["login"],
            "github_url": h["github_url"],
            "label": h["label"],
            "similarity": h["similarity"],
            "fit": item.get("fit", "unknown"),
            "summary": item.get("summary", "No summary generated."),
        })

    response = {"query": req.query, "results": results}
    if len(cache) >= CACHE_MAX:
        cache.pop(next(iter(cache)))
    cache[key] = (time.time(), response)
    return response


# serve the frontend from the same server (no CORS headaches); keep this LAST
if (ROOT / "frontend").exists():
    app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")