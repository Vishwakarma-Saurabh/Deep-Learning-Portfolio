"""
Token Bucket Rate Limiter - Prevents API abuse.
Each user gets tokens that refill over time.
"""

import time
from collections import defaultdict
from typing import Dict


class TokenBucket:
    """Token bucket algorithm for rate limiting."""
    
    def __init__(self, rate: int = 30, per_seconds: int = 60):
        self.rate = rate
        self.per_seconds = per_seconds
        self.tokens = rate
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if allowed."""
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        refill_amount = elapsed * (self.rate / self.per_seconds)
        self.tokens = min(self.rate, self.tokens + refill_amount)
        self.last_refill = now
    
    def tokens_remaining(self) -> int:
        self._refill()
        return int(self.tokens)


class RateLimiter:
    """Per-user rate limiting."""
    
    def __init__(self, default_rate: int = 30):
        self.default_rate = default_rate
        self.buckets: Dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(rate=default_rate)
        )
    
    def is_allowed(self, user_id: str) -> bool:
        """Check if user is allowed to make a request."""
        return self.buckets[user_id].consume(1)
    
    def get_remaining(self, user_id: str) -> int:
        """Get remaining tokens for user."""
        return self.buckets[user_id].tokens_remaining()
    
    def get_retry_after(self, user_id: str) -> float:
        """Seconds until next token available."""
        bucket = self.buckets[user_id]
        if bucket.tokens >= 1:
            return 0
        return bucket.per_seconds / bucket.rate


# Global rate limiter
rate_limiter = RateLimiter(default_rate=30)


if __name__ == "__main__":
    limiter = RateLimiter(default_rate=5)
    
    for i in range(10):
        allowed = limiter.is_allowed("user123")
        remaining = limiter.get_remaining("user123")
        print(f"Request {i+1}: {'✅' if allowed else '❌'} (Remaining: {remaining})")