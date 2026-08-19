"""Circuit Breaker pattern for upstream LLM provider resilience."""

import time
import threading
from typing import Optional, Dict, Any


class CircuitBreakerOpenException(Exception):
    pass


class CircuitBreaker:
    STATE_CLOSED = "CLOSED"
    STATE_OPEN = "OPEN"
    STATE_HALF_OPEN = "HALF_OPEN"

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 30.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_seconds
        
        self.state = self.STATE_CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.total_calls = 0
        self.last_failure_time: Optional[float] = None
        self.last_state_change: float = time.time()
        self._lock = threading.Lock()

    def can_execute(self) -> bool:
        with self._lock:
            now = time.time()

            if self.state == self.STATE_CLOSED:
                return True

            if self.state == self.STATE_OPEN:
                if now - self.last_state_change >= self.recovery_timeout:
                    self.state = self.STATE_HALF_OPEN
                    self.last_state_change = now
                    return True
                return False

            if self.state == self.STATE_HALF_OPEN:
                return True

            return False

    def record_success(self):
        with self._lock:
            self.total_calls += 1
            self.success_count += 1
            if self.state == self.STATE_HALF_OPEN:
                self.state = self.STATE_CLOSED
                self.failure_count = 0
                self.last_state_change = time.time()
            elif self.state == self.STATE_CLOSED:
                self.failure_count = 0

    def record_failure(self, error: Optional[Exception] = None):
        with self._lock:
            self.total_calls += 1
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == self.STATE_CLOSED and self.failure_count >= self.failure_threshold:
                self.state = self.STATE_OPEN
                self.last_state_change = time.time()
            elif self.state == self.STATE_HALF_OPEN:
                self.state = self.STATE_OPEN
                self.last_state_change = time.time()

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "state": self.state,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "total_calls": self.total_calls,
                "last_failure_time": self.last_failure_time,
                "last_state_change": self.last_state_change,
            }
