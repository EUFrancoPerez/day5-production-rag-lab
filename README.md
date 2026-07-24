# day5-production-rag-lab

Production RAG system with evaluation, rate limiting, semantic caching,
and observability - Taller Academy Day 5 lab (Frontier Engineer
Onboarding).

## What this is

An implementation of Lab 05 (Production RAG System with Evaluation) plus
the Day 5 in-class exercises: the Production Readiness Checklist
assessment and the four Extra Exercises.

## Structure

- rag.py - RAG pipeline core: system prompt, RAGResponse, RAGMetrics,
  build_context, generate_answer (Step 1).
- rate_limiter.py - sliding-window InMemoryRateLimiter with per-minute,
  per-hour, and token-usage tracking (Step 2).
- semantic_cache.py - embedding-similarity SemanticCache with TTL
  expiry and eviction (Step 3).
- evaluation.py - Precision@K, MRR, LLM-as-judge faithfulness and
  relevance, and the run_evaluation orchestrator (Step 4).
- main.py - FastAPI app wiring everything together: /query, /index,
  /evaluate, /metrics, /health (Step 5), plus a basic sanitize_query
  prompt-injection filter.
- circuit_breaker.py - Extra Exercise 4: per-model circuit breaker with
  closed/open/half-open states and fallback routing.
- tests/test_rag.py - unit tests for the rate limiter, semantic cache,
  and evaluation metrics (the "How to verify" checks from the lab).
- tests/eval_dataset.json - a 10-question evaluation dataset for
  POST /evaluate.
- PRODUCTION_READINESS.md - the graded checklist assessment of the
  sample customer support chatbot.
- PROMPT_INJECTION_DEFENSES.md - the defense-in-depth write-up required
  by the lab deliverables.
- EXTRA_EXERCISES.md - status and notes for the four bonus exercises.

## Setup

    uv add fastapi uvicorn google-generativeai sentence-transformers pinecone
    uv run uvicorn main:app --reload --port 8000

Requires a GOOGLE_API_KEY environment variable (free tier at
aistudio.google.com) for the Gemini calls in rag.py and evaluation.py.

## Running the tests

    pytest tests/

## Notes on scope

The retrieve() function in main.py is a placeholder that returns
synthetic chunks; it is meant to be replaced with the Lab 04 hybrid
(BM25 + vector store) retriever. Live deployment to Railway/Render is
described in the lab instructions but is an infrastructure step outside
this repository; the application is deployment-ready as-is (a single
FastAPI app with a health check, so any standard container/PaaS deploy
flow applies).
