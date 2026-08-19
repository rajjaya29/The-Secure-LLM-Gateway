"""In-Memory Sliding-Window Rate Limiter per API Key.

NOTE: This rate limiter is process-local / in-memory. It manages rate-limiting
quotas on a single gateway instance using an efficient collections.deque
timestamp window per API key identifier.
"""

import time
import threading
from collections import deque
from typing import Dict, Tuple


class SlidingWindowRateLimiter:
    """
    In-memory sliding-window rate limiter.
    Maintains a rolling timestamp window for each API key identity.
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

    def check_limit(self, key_id: str) -> Tuple[bool, float, int]:
        """
        Checks if the request from key_id is permitted under the sliding window.
        Returns:
            - allowed (bool): True if allowed, False if exceeded.
            - retry_after (float): Seconds until the oldest request leaves the window.
            - remaining (int): Remaining requests allowed in the current window.
        """
        if not self.enabled:
            return True, 0.0, self.max_requests

        now = time.time()
        window_start = now - self.window_seconds

        with self._lock:
            if key_id not in self._windows:
                self._windows[key_id] = deque()

            req_deque = self._windows[key_id]

            # 1. Evict timestamps older than the sliding window boundary
            while req_deque and req_deque[0] <= window_start:
                req_deque.popleft()

            current_count = len(req_deque)

            # 2. Check if the threshold is exceeded
            if current_count >= self.max_requests:
                oldest_timestamp = req_deque[0]
                retry_after = max(0.1, (oldest_timestamp + self.window_seconds) - now)
                return False, round(retry_after, 2), 0

            # 3. Record new timestamp
            req_deque.append(now)
            remaining = self.max_requests - len(req_deque)
            return True, 0.0, remaining

    def get_remaining(self, key_id: str) -> int:
        """Returns the remaining request quota in the active sliding window."""
        now = time.time()
        window_start = now - self.window_seconds
        with self._lock:
            req_deque = self._windows.get(key_id)
            if not req_deque:
                return self.max_requests
            while req_deque and req_deque[0] <= window_start:
                req_deque.popleft()
            return max(0, self.max_requests - len(req_deque))

    def clear(self):
        """Clears all rate-limiting state."""
        with self._lock:
            self._windows.clear()
