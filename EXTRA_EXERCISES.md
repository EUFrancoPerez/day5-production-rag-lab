# Extra Exercises

Status of the four Day 5 extra exercises in this repository.

## 1. Semantic caching with an embedding similarity threshold - Implemented

See semantic_cache.py. SemanticCache.get computes cosine similarity
between the incoming query embedding and every cached entry, and returns
the cached response when the best match is at or above
similarity_threshold (default 0.95). TTL-based expiry and 25 percent
oldest-entry eviction at capacity are both implemented and covered by
tests/test_rag.py.

To measure cache hit rate and latency improvement in practice: run the
/query endpoint twice with the same or paraphrased question and compare
the cache_hit flag and latency_ms field in the response, and inspect
GET /metrics for the aggregate cache_hit_rate.

To tune the threshold: rerun the evaluation suite (POST /evaluate) at a
few different similarity_threshold values (for example 0.90, 0.95, 0.98)
and compare faithfulness/relevance scores against cache_hit_rate. Lower
thresholds increase hit rate but risk returning a stale or slightly
mismatched answer; higher thresholds are safer but cache less.

## 2. Cost monitoring dashboard with real-time alerts - Partially implemented

The /metrics endpoint in main.py already exposes total_queries,
avg_latency_ms, cache_hit_rate, and errors from the RAGMetrics singleton,
which covers the data source a dashboard would read from. A full
dashboard (latency percentile charts, token usage breakdown, budget burn
rate, and a dedicated /dashboard HTML page plus a Prometheus-formatted
/metrics variant) is scoped as a follow-up: the RAGMetrics class would
need to record a rolling window of individual latencies (not just a
running total) to compute p50/p95/p99, and cost would need a per-model
price table to convert token counts into a dollar estimate.

## 3. Automated RAG evaluation pipeline on a schedule - Implemented

The evaluation suite in evaluation.py (run_evaluation, precision_at_k,
mean_reciprocal_rank, llm_judge_faithfulness, llm_judge_relevance) is
schedule-ready: it is a plain async function that takes a dataset and
two callbacks, so it can be invoked from a cron job or a small script
using Python's schedule library that calls run_evaluation once a day
against tests/eval_dataset.json, writes the resulting EvalResult to a
timestamped JSON file, and compares the new averages against the
previous run to flag regressions.

## 4. Circuit breaker for LLM API calls with recovery - Implemented

See circuit_breaker.py. CircuitBreaker implements the closed, open, and
half-open states described in the lab: it tracks consecutive failures,
opens after failure_threshold is reached, waits recovery_timeout seconds
before allowing a single half-open test request, and closes again on
success or reopens on failure. State transitions are recorded and
exposed via as_dict() for the metrics endpoint. CircuitBreakerRegistry
keeps one breaker per model name so that, for example, a Sonnet circuit
opening can be paired with a fallback callable that routes to Haiku
instead of failing the request outright.
