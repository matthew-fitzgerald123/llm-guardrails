from __future__ import annotations
import redis as redis_lib
import time, os, uuid
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


# Atomic sliding-window rate-limit check via Lua script.
# The script runs entirely inside Redis, so concurrent requests cannot race past
# the limit. Denied requests are never added to the window, so they do not
# consume future capacity.
_RATE_LIMIT_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window_start = tonumber(ARGV[2])
local rpm = tonumber(ARGV[3])
local member = ARGV[4]

redis.call('ZREMRANGEBYSCORE', key, 0, window_start)
local count = redis.call('ZCARD', key)

if count < rpm then
    redis.call('ZADD', key, now, member)
    redis.call('EXPIRE', key, 65)
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    return {1, count + 1, oldest[2] or tostring(now)}
else
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    return {0, count, oldest[2] or tostring(now)}
end
"""


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
        self._script = self.redis.register_script(_RATE_LIMIT_SCRIPT)

    def check(self, client_id: str, tier: str | None = None) -> RateLimitResult:
        resolved_tier = tier or _DEFAULT_TIER
        rpm = _rpm_for_tier(resolved_tier)

        now = time.time()
        window_start = now - 60
        key = f"ratelimit:{client_id}"
        member = f"{now:.6f}:{uuid.uuid4().hex}"

        result = self._script(keys=[key], args=[now, window_start, rpm, member])

        allowed = bool(result[0])
        count = int(result[1])
        oldest_score = float(result[2])

        remaining = max(0, rpm - count)
        reset_in = max(0, int(60 - (now - oldest_score)))

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_in_seconds=reset_in,
            limit=rpm,
            tier=resolved_tier,
        )


rate_limiter = RateLimiter()
