"""Deterministic hashing helpers used for caching and shingling."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


def content_hash(text: str, *, length: int = 32) -> str:
    """Return a hex SHA-256 digest of ``text`` truncated to ``length`` characters."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:length]


def stable_hash(text: str, *, seed: int = 0) -> int:
    """Return a 64-bit integer hash of ``text`` that is stable across processes.

    Python's built-in ``hash`` is randomised per process for strings, which makes it
    unsuitable for MinHash signatures that must be reproducible.
    """
    payload = f"{seed}:{text}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def combined_hash(parts: Iterable[str]) -> str:
    """Hash several strings into one digest (order-sensitive)."""
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()
