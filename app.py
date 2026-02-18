import time
import hashlib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from cachetools import LRUCache
from sentence_transformers import SentenceTransformer

app = FastAPI()

TTL_SECONDS = 24 * 3600
CACHE_SIZE = 1500
MODEL_COST_PER_1M = 0.60
AVG_TOKENS = 800

cache = LRUCache(maxsize=CACHE_SIZE)
timestamps = {}
embeddings_store = {}

model = SentenceTransformer("all-MiniLM-L6-v2")

stats = {
    "total": 0,
    "hits": 0,
    "misses": 0,
    "cached_tokens": 0
}

class Query(BaseModel):
    query: str
    application: str

def md5_key(q: str) -> str:
    return hashlib.md5(q.encode("utf-8")).hexdigest()

def cosine_sim(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def fake_llm_call(query: str) -> str:
    time.sleep(1.2)
    return f"AI response for: {query}"

@app.post("/")
def handle_query(req: Query):
    start = time.time()
    stats["total"] += 1

    key = md5_key(req.query)
    now = time.time()

    if key in timestamps and now - timestamps[key] > TTL_SECONDS:
        cache.pop(key, None)
        embeddings_store.pop(key, None)
        timestamps.pop(key, None)

    if key in cache:
        stats["hits"] += 1
        stats["cached_tokens"] += AVG_TOKENS
        return {
            "answer": cache[key],
            "cached": True,
            "latency": int((time.time() - start) * 1000),
            "cacheKey": key
        }

    query_emb = model.encode(req.query)
    for k, emb in embeddings_store.items():
        if cosine_sim(query_emb, emb) > 0.95:
            stats["hits"] += 1
            stats["cached_tokens"] += AVG_TOKENS
            return {
                "answer": cache[k],
                "cached": True,
                "latency": int((time.time() - start) * 1000),
                "cacheKey": k
            }

    stats["misses"] += 1
    answer = fake_llm_call(req.query)

    cache[key] = answer
    timestamps[key] = now
    embeddings_store[key] = query_emb

    return {
        "answer": answer,
        "cached": False,
        "latency": int((time.time() - start) * 1000),
        "cacheKey": key
    }

@app.get("/analytics")
def analytics():
    hit_rate = stats["hits"] / max(stats["total"], 1)
    baseline_cost = (stats["total"] * AVG_TOKENS * MODEL_COST_PER_1M) / 1_000_000
    actual_cost = ((stats["total"] - stats["hits"]) * AVG_TOKENS * MODEL_COST_PER_1M) / 1_000_000
    savings = baseline_cost - actual_cost

    return {
        "hitRate": round(hit_rate, 2),
        "totalRequests": stats["total"],
        "cacheHits": stats["hits"],
        "cacheMisses": stats["misses"],
        "cacheSize": len(cache),
        "costSavings": round(savings, 2),
        "savingsPercent": int(hit_rate * 100),
        "strategies": [
            "exact match",
            "semantic similarity",
            "LRU eviction",
            "TTL expiration"
        ]
    }
