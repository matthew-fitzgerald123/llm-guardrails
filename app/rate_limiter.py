from __future__ import annotations
import redis as redis_lib
import time
from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class RateLimitResult:
    allowed: bool
    remaining: int
    reset_in_seconds: int
    limit: int

class RateLimiter:
    def __init__(self):
        self.redis = redis_lib.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379")
        )
        self.rpm = int(os.getenv("RATE_LIMIT_RPM", 60))

    def check(self, client_id: str) -> RateLimitResult:
        """
        Sliding window rate limiter using Redis.
        Window: 60 seconds, limit: RATE_LIMIT_RPM requests.
        """
        now = time.time()
        window_start = now - 60
        key = f"ratelimit:{client_id}"

        pipe = self.redis.pipeline()
        # Remove requests outside the window
        pipe.zremrangebyscore(key, 0, window_start)
        # Count requests in window
        pipe.zcard(key)
        # Add current request
        pipe.zadd(key, {str(now): now})
        # Set expiry
        pipe.expire(key, 60)
        results = pipe.execute()

        count = results[1]
        allowed = count < self.rpm
        remaining = max(0, self.rpm - count - 1)

        # Estimate reset time — oldest request in window + 60s
        oldest = self.redis.zrange(key, 0, 0, withscores=True)
        reset_in = 60
        if oldest:
            reset_in = max(0, int(60 - (now - oldest[0][1])))

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_in_seconds=reset_in,
            limit=self.rpm,
        )

# Global instance
rate_limiter = RateLimiter()
