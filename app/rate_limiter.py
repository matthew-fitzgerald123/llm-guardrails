from __future__ import annotations
import redis as redis_lib
import time, os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# Tier format: "free:20,standard:100,premium:500"
# Falls back to RATE_LIMIT_RPM for any unrecognised tier.
_DEFAULT_RPM = int(os.getenv("RATE_LIMIT_RPM", 60))

def _parse_tiers() -> dict[str, int]:
    raw = os.getenv("RATE_LIMIT_TIERS", "")
    tiers: dict[str, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if ":" in part:
            name, limit = part.split(":", 1)
            try:
                tiers[name.strip()] = int(limit.strip())
            except ValueError:
                pass
    return tiers

_TIERS = _parse_tiers()
_DEFAULT_TIER = os.getenv("RATE_LIMIT_DEFAULT_TIER", "free")


def _rpm_for_tier(tier: str) -> int:
    return _TIERS.get(tier, _DEFAULT_RPM)


@dataclass
class RateLimitResult:
    allowed: bool
    remaining: int
    reset_in_seconds: int
    limit: int
    tier: str


class RateLimiter:
    def __init__(self):
        self.redis = redis_lib.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379")
        )

    def check(self, client_id: str, tier: str | None = None) -> RateLimitResult:
        resolved_tier = tier or _DEFAULT_TIER
        rpm = _rpm_for_tier(resolved_tier)

        now = time.time()
        window_start = now - 60
        key = f"ratelimit:{client_id}"

        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, 60)
        results = pipe.execute()

        count = results[1]
        allowed = count < rpm
        remaining = max(0, rpm - count - 1)

        oldest = self.redis.zrange(key, 0, 0, withscores=True)
        reset_in = 60
        if oldest:
            reset_in = max(0, int(60 - (now - oldest[0][1])))

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_in_seconds=reset_in,
            limit=rpm,
            tier=resolved_tier,
        )


rate_limiter = RateLimiter()
