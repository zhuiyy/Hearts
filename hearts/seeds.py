"""Named RNG streams: no global random state and no randomized Python hash()."""

import hashlib
import json
import random
from dataclasses import dataclass

SEED_VERSION = "hearts-seeds-v1"


@dataclass(frozen=True, slots=True)
class Seeds:
    root: int

    def __post_init__(self) -> None:
        if type(self.root) is not int or self.root < 0:
            raise ValueError("root seed must be a non-negative integer")

    def derive(self, *path: str | int) -> int:
        """The same root and typed path always produce the same 128-bit seed."""
        if any(type(part) not in (str, int) for part in path):
            raise TypeError("seed path components must be strings or integers")
        payload = json.dumps([SEED_VERSION, self.root, *path], ensure_ascii=True)
        return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:16], "big")

    def rng(self, *path: str | int) -> random.Random:
        return random.Random(self.derive(*path))
