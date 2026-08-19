"""In-Memory Sliding-Window Rate Limiter per API key."""

import time
import threading
from collections import deque
from typing import Dict, Tuple, Optional


class SlidingWindowRateLimiter:
    """
    In-memory sliding-window rate limiter.
    Tracks exact request timestamps in a sliding time window (e.g. 60 requests per 60s) per API key.
    """

    def __init__(
        self,
        window_seconds: int = 60,
        max_requests: int = 60,
        enabled: bool = True,
    ):
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self.enabled = enabled
        self._windows: Dict[str, deque] = {}
        self._lock = threading.Lock()

    def check_limit(self, client_id: str) -> Tuple[bool, float]:
        """
        Evaluates if request from client_id is within the sliding window quota.
        Returns:
            - allowed (bool)
            - retry_after_seconds (float)
        """
        if not self.enabled:
            return True, 0.0

        now = time.time()
        window_start = now - self.window_seconds

        with self._lock:
            if client_id not in self._windows:
                self._windows[client_id] = deque()

            req_deque = self._windows[client_id]

            # Evict timestamps outside current sliding window
            while req_deque and req_deque[0] <= window_start:
                req_deque.popleft()

            # Check if threshold reached
            if len(req_deque) >= self.max_requests:
                oldest_in_window = req_deque[0]
                retry_after = max(0.1, (oldest_in_window + self.window_seconds) - now)
                return False, round(retry_after, 2)

            # Record current request timestamp
            req_deque.append(now)
            return True, 0.0

    def get_client_usage(self, client_id: str) -> int:
        """Returns active request count in current sliding window for client."""
        now = time.time()
        window_start = now - self.window_seconds
        with self._lock:
            req_deque = self._windows.get(client_id)
            if not req_deque:
                return 0
            # Clean expired
            while req_deque and req_deque[0] <= window_start:
                req_deque.popleft()
            return len(req_deque)

    def clear(self):
        with self._lock:
            self._windows.clear()
