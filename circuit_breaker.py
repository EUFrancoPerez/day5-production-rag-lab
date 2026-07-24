"""
circuit_breaker.py - Extra Exercise 4: Circuit breaker for LLM API calls
with automatic recovery, per-model instances, and fallback routing.
"""
from __future__ import annotations

import time
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Per-model circuit breaker.

    Closed: requests pass through normally, failures are tracked.
    Open: requests are rejected immediately with a fallback; a recovery
        timer determines when to try again.
    Half-open: a single test request is allowed through. Success closes
        the circuit, failure reopens it.
    """

    def __init__(self, name, failure_threshold=5, recovery_timeout=30, fallback=None):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.fallback = fallback

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.opened_at = None
        self.state_transitions = []

    def _transition(self, new_state):
        self.state_transitions.append({
            "from": self.state.value,
            "to": new_state.value,
            "at": time.time(),
        })
        self.state = new_state

    def _ready_to_try_half_open(self):
        if self.opened_at is None:
            return True
        return (time.time() - self.opened_at) >= self.recovery_timeout

    async def call(self, fn, *args, **kwargs):
        """Execute fn(*args, **kwargs) through the circuit breaker."""
        if self.state == CircuitState.OPEN:
            if self._ready_to_try_half_open():
                self._transition(CircuitState.HALF_OPEN)
            else:
                return await self._fallback_response()

        try:
            result = await fn(*args, **kwargs)
        except Exception:
            self._on_failure()
            if self.state == CircuitState.OPEN:
                return await self._fallback_response()
            raise
        else:
            self._on_success()
            return result

    def _on_success(self):
        self.failure_count = 0
        if self.state != CircuitState.CLOSED:
            self._transition(CircuitState.CLOSED)
        self.opened_at = None

    def _on_failure(self):
        self.failure_count += 1
        if self.state == CircuitState.HALF_OPEN:
            self._transition(CircuitState.OPEN)
            self.opened_at = time.time()
        elif self.failure_count >= self.failure_threshold:
            self._transition(CircuitState.OPEN)
            self.opened_at = time.time()

    async def _fallback_response(self):
        if self.fallback is not None:
            return await self.fallback()
        return {"error": "circuit_open", "circuit": self.name}

    def as_dict(self):
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "transitions": self.state_transitions[-20:],
        }


class CircuitBreakerRegistry:
    """Holds one circuit breaker per model, so a failure on one model
    (e.g. Sonnet) can route to a fallback model (e.g. Haiku)."""

    def __init__(self):
        self._breakers = {}

    def get_or_create(self, model_name, **kwargs):
        if model_name not in self._breakers:
            self._breakers[model_name] = CircuitBreaker(model_name, **kwargs)
        return self._breakers[model_name]

    def metrics(self):
        return {name: cb.as_dict() for name, cb in self._breakers.items()}


registry = CircuitBreakerRegistry()
