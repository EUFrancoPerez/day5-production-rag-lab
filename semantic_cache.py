"""
semantic_cache.py - Embedding-similarity cache for RAG responses.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


@dataclass
class CacheEntry:
    query: str
    embedding: list
    response: dict
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 3600


class SemanticCache:
    def __init__(self, similarity_threshold=0.95, max_entries=1000):
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self._entries = []

    @staticmethod
    def cosine_similarity(a, b):
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for x, y in zip(a, b):
            dot += x * y
            norm_a += x * x
            norm_b += y * y
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

    def _evict_expired(self):
        now = time.time()
        self._entries = [
            e for e in self._entries if now - e.created_at <= e.ttl_seconds
        ]

    def get(self, query_embedding):
        self._evict_expired()

        best_entry = None
        best_similarity = -1.0
        for entry in self._entries:
            similarity = self.cosine_similarity(query_embedding, entry.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry

        if best_entry is not None and best_similarity >= self.similarity_threshold:
            return best_entry.response
        return None

    def put(self, query, query_embedding, response, ttl_seconds=3600):
        if len(self._entries) >= self.max_entries:
            evict_count = max(1, self.max_entries // 4)
            self._entries = self._entries[evict_count:]

        self._entries.append(
            CacheEntry(
                query=query,
                embedding=query_embedding,
                response=response,
                ttl_seconds=ttl_seconds,
            )
        )

    @property
    def size(self):
        return len(self._entries)


semantic_cache = SemanticCache()
