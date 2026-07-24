"""
tests/test_rag.py - Unit tests for rate_limiter, semantic_cache, and
evaluation modules (Steps 2-4 of Lab 05). Run with: pytest tests/
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rate_limiter import InMemoryRateLimiter, RateLimitConfig, RateLimitExceeded
from semantic_cache import SemanticCache
from evaluation import precision_at_k, mean_reciprocal_rank


def test_rate_limiter_blocks_after_limit():
    limiter = InMemoryRateLimiter(RateLimitConfig(requests_per_minute=3))
    for _ in range(3):
        assert limiter.check_rate_limit("test") is True
    try:
        limiter.check_rate_limit("test")
        assert False, "expected RateLimitExceeded"
    except RateLimitExceeded:
        pass


def test_rate_limiter_resets_after_window():
    limiter = InMemoryRateLimiter(RateLimitConfig(requests_per_minute=2))
    now = time.time()
    limiter._minute_windows["test"] = [now - 61, now - 62]
    assert limiter.check_rate_limit("test") is True


def test_rate_limiter_token_tracking():
    limiter = InMemoryRateLimiter()
    limiter.record_tokens("test", 500)
    limiter.record_tokens("test", 500)
    assert limiter.get_token_usage("test") == 1000


def test_semantic_cache_hit_on_identical_embedding():
    cache = SemanticCache(similarity_threshold=0.95)
    embedding = [1.0, 0.0, 0.0]
    cache.put("hello", embedding, {"answer": "hi"})
    result = cache.get(embedding)
    assert result == {"answer": "hi"}


def test_semantic_cache_miss_on_dissimilar_embedding():
    cache = SemanticCache(similarity_threshold=0.95)
    cache.put("hello", [1.0, 0.0, 0.0], {"answer": "hi"})
    result = cache.get([0.0, 0.0, 1.0])
    assert result is None


def test_semantic_cache_ttl_expiration():
    cache = SemanticCache()
    cache.put("hello", [1.0, 0.0], {"answer": "hi"}, ttl_seconds=1)
    time.sleep(1.5)
    assert cache.get([1.0, 0.0]) is None
    assert cache.size == 0


def test_semantic_cache_eviction():
    cache = SemanticCache(max_entries=4)
    for i in range(4):
        cache.put("q" + str(i), [float(i), 0.0], {"answer": str(i)})
    assert cache.size == 4
    cache.put("q_new", [9.0, 0.0], {"answer": "new"})
    assert cache.size == 4


def test_precision_at_k():
    assert round(precision_at_k(["a", "b", "c"], {"a", "c"}, 3), 4) == 0.6667
    assert precision_at_k(["a", "b", "c"], {"d"}, 3) == 0.0
    assert precision_at_k([], {"a"}, 3) == 0.0


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert mean_reciprocal_rank(["a", "b", "c"], {"d"}) == 0.0
    assert mean_reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
