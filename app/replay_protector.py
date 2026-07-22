from __future__ import annotations
import os
import redis as redis_lib
from dotenv import load_dotenv

load_dotenv()

NONCE_TTL = int(os.getenv("NONCE_TTL_SECONDS", 300))  # 5-minute replay window


class ReplayProtector:
    def __init__(self):
        self.redis = redis_lib.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379")
        )

    def check_and_store(self, nonce: str) -> bool:
        """Return True if nonce is fresh. False means it was seen before (replay)."""
        key = f"nonce:{nonce}"
        stored = self.redis.set(key, 1, ex=NONCE_TTL, nx=True)
        return stored is not None

    def is_replay(self, nonce: str) -> bool:
        return not self.check_and_store(nonce)

    def peek(self, nonce: str) -> bool:
        """Return True if nonce has been stored, without modifying Redis state."""
        return bool(self.redis.exists(f"nonce:{nonce}"))


replay_protector = ReplayProtector()
