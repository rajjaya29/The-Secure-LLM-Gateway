"""Unit tests for In-Memory Sliding-Window Rate Limiting."""

import time
import pytest
from app.resilience.rate_limiter import SlidingWindowRateLimiter


def test_sliding_window_rate_limiter_burst_and_window():
    limiter = SlidingWindowRateLimiter(window_seconds=1, max_requests=3, enabled=True)
    key_a = "key_client_a"
    key_b = "key_client_b"

    # 1. Key A: Requests 1, 2, 3 allowed
    for i in range(3):
        allowed, wait, rem = limiter.check_limit(key_a)
        assert allowed is True
        assert rem == (2 - i)

    # 2. Key A: Request 4 in same 1-second window rejected
    allowed, wait, rem = limiter.check_limit(key_a)
    assert allowed is False
    assert wait > 0
    assert rem == 0

    # 3. Independent limit for Key B: Key B can still make requests
    allowed_b, wait_b, rem_b = limiter.check_limit(key_b)
    assert allowed_b is True
    assert rem_b == 2

    # 4. Wait for window expiration
    time.sleep(1.1)

    # Key A is allowed again
    allowed, wait, rem = limiter.check_limit(key_a)
    assert allowed is True
    assert rem == 2
