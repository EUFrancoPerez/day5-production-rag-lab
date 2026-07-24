"""
main.py - FastAPI application wiring together the RAG pipeline, rate
limiter, semantic cache, and evaluation suite into a production-ready API.
"""
from __future__ import annotations

import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import rag
from rate_limiter import rate_limiter, RateLimitExceeded
from semantic_cache import semantic_cache
from evaluation import run_evaluation, EvalResult

app = FastAPI(title="Production RAG API")

_embedder = None

_INJECTION_PATTERNS = [
    re.compile(r"ignore (all|any|previous) instructions", re.IGNORECASE),
    re.compile(r"you are now", re.IGNORECASE),
    re.compile(r"system prompt", re.IGNORECASE),
    re.compile(r"reveal (your|the) (prompt|instructions)", re.IGNORECASE),
]


def sanitize_query(raw_query):
    """Basic prompt-injection defense: flag and neutralize common override
    attempts before the query reaches the model. This is a lightweight
    input-side filter; see PROMPT_INJECTION_DEFENSES.md for the full
    defense-in-depth strategy (sandwich defense, output validation, etc.)."""
    cleaned = raw_query.strip()
    for pattern in _INJECTION_PATTERNS:
        cleaned = pattern.sub("[filtered]", cleaned)
    return cleaned


def _get_embedder():
    """Lazily load the local embedding model once at module level."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def embed_query(query):
    model = _get_embedder()
    return model.encode(query).tolist()


async def retrieve(query, limit=5):
    """Placeholder hybrid retrieval. Replace with the Lab 04 hybrid search
    (BM25 + vector store) implementation."""
    results = []
    for i in range(1, limit + 1):
        results.append({
            "id": "doc_" + str(i),
            "text": "Placeholder chunk " + str(i) + " relevant to: " + query,
            "metadata": {"title": "Document " + str(i)},
        })
    return results


class QueryRequest(BaseModel):
    query: str
    limit: int = 5
    use_cache: bool = True
    client_id: str = "default"


class IndexRequest(BaseModel):
    doc_id: str
    title: str
    text: str


class EvaluateRequest(BaseModel):
    dataset: list
    k: int = 5


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        rate_limiter.check_rate_limit(request.client_id)
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc))

    safe_query = sanitize_query(request.query)
    query_embedding = embed_query(safe_query)

    if request.use_cache:
        cached = semantic_cache.get(query_embedding)
        if cached is not None:
            result = dict(cached)
            result["cache_hit"] = True
            return result

    chunks = await retrieve(safe_query, request.limit)
    response = await rag.generate_answer(safe_query, chunks, cache_hit=False)

    result = {
        "answer": response.answer,
        "sources": [s.__dict__ for s in response.sources],
        "latency_ms": response.latency_ms,
        "cache_hit": False,
    }

    if request.use_cache:
        semantic_cache.put(safe_query, query_embedding, result)

    approx_tokens = len(safe_query.split()) + len(response.answer.split())
    rate_limiter.record_tokens(request.client_id, approx_tokens)

    return result


@app.post("/index")
async def index_endpoint(request: IndexRequest):
    # Plug in the Lab 04 indexing pipeline (chunk -> embed -> upsert) here.
    return {"status": "acknowledged", "doc_id": request.doc_id}


@app.post("/evaluate")
async def evaluate_endpoint(request: EvaluateRequest):
    async def generation_fn(query, chunks):
        return await rag.generate_answer(query, chunks, cache_hit=False)

    async def retrieval_fn(query):
        return await retrieve(query, request.k)

    result = await run_evaluation(request.dataset, retrieval_fn, generation_fn, request.k)
    return {
        "precision_at_k": result.precision_at_k,
        "mrr": result.mrr,
        "faithfulness": result.faithfulness,
        "relevance": result.relevance,
        "details": result.details,
    }


@app.get("/metrics")
async def metrics_endpoint():
    return {
        "rag": rag.metrics.as_dict(),
        "cache": {"size": semantic_cache.size},
    }


@app.get("/health")
async def health_endpoint():
    return {"status": "ok"}
