"""
rate_limiter.py - In-memory sliding-window rate limiter with per-client
request and token tracking.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass


class RateLimitExceeded(Exception):
    """Raised when a client exceeds the configured rate limit."""

    def __init__(self, message, status_code=429):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class RateLimitConfig:
    requests_per_minute: int = 20
    requests_per_hour: int = 200
    tokens_per_minute: int = 100000


class InMemoryRateLimiter:
    """Sliding-window rate limiter tracked per client_id, in memory."""

    def __init__(self, config=None):
        self.config = config or RateLimitConfig()
        self._minute_windows = defaultdict(list)
        self._hour_windows = defaultdict(list)
        # token usage stored as (timestamp, token_count) tuples
        self._token_windows = defaultdict(list)

    @staticmethod
    def _prune(timestamps, now, window_seconds):
        return [t for t in timestamps if now - t <= window_seconds]

    def check_rate_limit(self, client_id):
        """Check (and record) a request for client_id. Raises
        RateLimitExceeded if either the per-minute or per-hour limit is
        exceeded."""
        now = time.time()

        self._minute_windows[client_id] = self._prune(self._minute_windows[client_id], now, 60)
        self._hour_windows[client_id] = self._prune(self._hour_windows[client_id], now, 3600)

        if len(self._minute_windows[client_id]) >= self.config.requests_per_minute:
            raise RateLimitExceeded(
                "Rate limit exceeded: more than " + str(self.config.requests_per_minute) + " requests per minute."
            )
        if len(self._hour_windows[client_id]) >= self.config.requests_per_hour:
            raise RateLimitExceeded(
                "Rate limit exceeded: more than " + str(self.config.requests_per_hour) + " requests per hour."
            )

        self._minute_windows[client_id].append(now)
        self._hour_windows[client_id].append(now)
        return True

    def record_tokens(self, client_id, token_count):
        now = time.time()
        self._token_windows[client_id].append((now, token_count))

    def get_token_usage(self, client_id):
        now = time.time()
        window = [
            (t, count) for (t, count) in self._token_windows[client_id]
            if now - t <= 60
        ]
        self._token_windows[client_id] = window
        return sum(count for _, count in window)


rate_limiter = InMemoryRateLimiter()
