# Production Readiness Checklist - Sample Customer Support Chatbot

System under review: a Gemini-based customer support chatbot answering
questions about returns, shipping status, and account issues. It calls an
order database via function calling and serves roughly 5,000 queries/day.

Each item below is rated Meets, Partial, or Fails, with a short
justification and a remediation plan for any gap.

## 1. Rate limiting configured - Partial

At 5,000 queries/day (roughly 3-4 per minute on average, with bursts),
the system likely has no explicit per-user or global limiter today, since
none is mentioned in the system description. Without it, a single user or
a scripted client could exhaust the Gemini API quota and degrade service
for everyone else.

Remediation: add the InMemoryRateLimiter from this repo (rate_limiter.py)
in front of the chat endpoint, with a conservative per-user limit (for
example 20 requests/minute) and a global limit aligned to the Gemini API
tier. Return a clear 429 with a retry-after hint instead of a raw error.

## 2. Caching strategy implemented - Fails

Support questions repeat heavily (order status, return policy, etc.), so
without caching every repeated question re-invokes the model and the
order-database function call, wasting cost and latency.

Remediation: add the SemanticCache module (semantic_cache.py) in front of
generation for read-only questions (policy/FAQ style), keyed on query
embedding similarity, plus exact-match response caching for identical
FAQ text. Cache invalidation should be tied to a policy-version flag so
that whenever the returns/shipping policy changes, the cache is flushed
or the version key changes so stale entries stop matching.

## 3. Prompt injection defenses in place - Fails

A support bot with function-calling access to an order database is a
high-value injection target: a malicious message could try to get the
model to reveal other customers' order data or ignore its instructions.
The description gives no indication of input sanitization or output
checks.

Remediation: apply the layered defenses documented in
PROMPT_INJECTION_DEFENSES.md - input-side pattern filtering (see
sanitize_query in main.py), a sandwich defense that repeats the system
instructions after the user content, and output validation that checks
the model is not echoing back data outside the current user's order
scope before the function-call result is returned to the user.

## 4. Cost monitoring active - Partial

At 5,000 queries/day cost is meaningful but not necessarily tracked yet.
Nothing in the description mentions per-request cost logging or budget
alerts.

Remediation: log token usage and estimated cost per request in the
RAGMetrics tracker (rag.py), aggregate daily/monthly totals, and add
threshold alerts (for example, alert at 80 percent of monthly budget).
Route simple FAQ-style questions to a cheaper/faster model and reserve
the larger model for ambiguous or multi-step account issues.

## 5. Evaluation pipeline running - Fails

There is no mention of a golden Q&A set or scheduled evaluation, so
regressions after a prompt or model change would only be caught by user
complaints.

Remediation: build the evaluation suite in evaluation.py (Precision@K,
MRR, LLM-as-judge faithfulness and relevance) against a golden dataset of
at least 10 representative support questions, and run it automatically
before any prompt/model deployment plus on a nightly schedule (see
scheduled evaluation pattern in EXTRA_EXERCISES.md), alerting if average
scores drop below a threshold.

## 6. Error handling and fallbacks - Partial

Function calling against a live order database can fail (timeouts,
schema mismatches); the description does not say how those failures are
surfaced to the user today.

Remediation: wrap model and database calls with retry and exponential
backoff for transient errors, add the CircuitBreaker (circuit_breaker.py)
around the Gemini call so a sustained outage trips to a fallback message
or a simpler model instead of hanging, and ensure end users only ever see
a friendly generic error, never a raw stack trace or database error.

## 7. API key rotation plan - Fails

No secrets-management strategy is described.

Remediation: move the Gemini and database credentials into a secrets
manager (not .env files on disk), define a rotation schedule (for
example every 90 days), and provision at least two active keys so a
rotation never causes downtime.

## 8. Observability and tracing - Partial

Metrics such as latency and cache-hit rate can be exposed via the
/metrics endpoint in main.py, but full request/response tracing and
anomaly alerting are not described as existing today.

Remediation: log every LLM call (prompt, response, latency, token
count) to a structured trace store, build a dashboard with latency
percentiles (p50/p95/p99), error rate, and cost, and add alerting for
cost spikes, quality drops (from the evaluation pipeline), and error
bursts.

## Summary

The most urgent gaps to close first are prompt injection defenses (item
3) and the evaluation pipeline (item 5), since they directly affect data
safety and the ability to detect regressions. Rate limiting, caching, and
observability are close behind, since they control cost and reliability
at the current 5,000 queries/day volume.
