"""Near-duplicate detection with word shingles, MinHash signatures and LSH banding."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from newsroom.preprocessing.text import clean_text, tokenize
from newsroom.utils.hashing import stable_hash

_PRIME_ABOVE_32 = 4294967311  # smallest prime larger than 2**32
_MAX_HASH = (1 << 32) - 1


def shingles(text: str, k: int = 3) -> set[str]:
    """Return the set of ``k``-word shingles of the normalised text."""
    if k < 1:
        raise ValueError("k must be >= 1")
    tokens = tokenize(clean_text(text, numbers=False))
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard similarity of two sets (0 when both are empty)."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class MinHasher:
    """MinHash signatures using ``num_perm`` universal hash functions ``(a*x + b) mod p``."""

    def __init__(self, num_perm: int = 128, *, seed: int = 7, shingle_size: int = 3) -> None:
        if num_perm < 4:
            raise ValueError("num_perm must be >= 4")
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        rng = np.random.RandomState(seed)
        # a, b < 2^32 and x < 2^32 keep a*x + b below 2^64, so uint64 arithmetic never overflows.
        self._a = rng.randint(1, _MAX_HASH, size=num_perm, dtype=np.int64).astype(np.uint64)
        self._b = rng.randint(0, _MAX_HASH, size=num_perm, dtype=np.int64).astype(np.uint64)

    def signature(self, text: str) -> np.ndarray:
        """Return a ``uint64`` signature of length ``num_perm``."""
        grams = shingles(text, self.shingle_size)
        sig = np.full(self.num_perm, _MAX_HASH, dtype=np.uint64)
        if not grams:
            return sig
        x = np.asarray([stable_hash(g) & _MAX_HASH for g in grams], dtype=np.uint64)
        hashes = (x[:, None] * self._a[None, :] + self._b[None, :]) % np.uint64(_PRIME_ABOVE_32)
        return np.minimum(sig, hashes.min(axis=0)).astype(np.uint64)

    @staticmethod
    def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """Estimate Jaccard similarity as the fraction of matching signature slots."""
        if sig_a.shape != sig_b.shape:
            raise ValueError("Signatures must have the same length")
        return float(np.mean(sig_a == sig_b))


@dataclass(frozen=True)
class DuplicatePair:
    """Two documents judged near-duplicates."""

    left: str
    right: str
    similarity: float


class NearDuplicateDetector:
    """Find near-duplicate documents with MinHash + LSH candidate generation.

    ``bands * rows == num_perm``. Candidate pairs that share at least one band bucket are
    verified with the estimated Jaccard similarity against ``threshold``.
    """

    def __init__(
        self,
        threshold: float = 0.8,
        *,
        num_perm: int = 128,
        bands: int = 32,
        shingle_size: int = 3,
        seed: int = 7,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if num_perm % bands != 0:
            raise ValueError("num_perm must be divisible by bands")
        self.threshold = threshold
        self.bands = bands
        self.rows = num_perm // bands
        self.hasher = MinHasher(num_perm, seed=seed, shingle_size=shingle_size)
        self._signatures: dict[str, np.ndarray] = {}
        self._buckets: dict[tuple[int, bytes], list[str]] = {}

    def __len__(self) -> int:
        return len(self._signatures)

    def add(self, doc_id: str, text: str) -> None:
        """Index a document."""
        if doc_id in self._signatures:
            raise ValueError(f"Duplicate doc_id: {doc_id!r}")
        sig = self.hasher.signature(text)
        self._signatures[doc_id] = sig
        for band in range(self.bands):
            key = (band, sig[band * self.rows : (band + 1) * self.rows].tobytes())
            self._buckets.setdefault(key, []).append(doc_id)

    def add_many(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        if len(ids) != len(texts):
            raise ValueError("ids and texts must be the same length")
        for doc_id, text in zip(ids, texts, strict=True):
            self.add(doc_id, text)

    def candidates(self, doc_id: str) -> set[str]:
        """Ids sharing at least one LSH bucket with ``doc_id``."""
        sig = self._signatures[doc_id]
        out: set[str] = set()
        for band in range(self.bands):
            key = (band, sig[band * self.rows : (band + 1) * self.rows].tobytes())
            out.update(self._buckets.get(key, ()))
        out.discard(doc_id)
        return out

    def query(self, text: str) -> list[tuple[str, float]]:
        """Return indexed documents similar to ``text`` (above threshold), best first."""
        sig = self.hasher.signature(text)
        cands: set[str] = set()
        for band in range(self.bands):
            key = (band, sig[band * self.rows : (band + 1) * self.rows].tobytes())
            cands.update(self._buckets.get(key, ()))
        scored = [(c, self.hasher.estimate_jaccard(sig, self._signatures[c])) for c in cands]
        return sorted([(c, s) for c, s in scored if s >= self.threshold], key=lambda kv: -kv[1])

    def find_duplicates(self) -> list[DuplicatePair]:
        """All near-duplicate pairs among indexed documents, sorted by similarity."""
        seen: set[tuple[str, str]] = set()
        pairs: list[DuplicatePair] = []
        for doc_id in self._signatures:
            for other in self.candidates(doc_id):
                key = (doc_id, other) if doc_id < other else (other, doc_id)
                if key in seen:
                    continue
                seen.add(key)
                sim = self.hasher.estimate_jaccard(
                    self._signatures[key[0]], self._signatures[key[1]]
                )
                if sim >= self.threshold:
                    pairs.append(DuplicatePair(key[0], key[1], sim))
        return sorted(pairs, key=lambda p: (-p.similarity, p.left, p.right))

    def clusters(self) -> list[list[str]]:
        """Connected components of the duplicate graph (union-find), singletons excluded."""
        parent: dict[str, str] = {d: d for d in self._signatures}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for pair in self.find_duplicates():
            ra, rb = find(pair.left), find(pair.right)
            if ra != rb:
                parent[rb] = ra
        groups: dict[str, list[str]] = {}
        for d in self._signatures:
            groups.setdefault(find(d), []).append(d)
        return sorted((sorted(g) for g in groups.values() if len(g) > 1), key=lambda g: g[0])


def cosine_duplicates(
    embeddings: np.ndarray, ids: Sequence[str], threshold: float = 0.95
) -> list[DuplicatePair]:
    """Dense alternative: pairs whose cosine similarity is >= ``threshold``."""
    x = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.where(norms == 0, 1.0, norms)
    sims = x @ x.T
    iu = np.triu_indices(len(ids), k=1)
    mask = sims[iu] >= threshold
    pairs = [
        DuplicatePair(ids[i], ids[j], float(sims[i, j]))
        for i, j in zip(iu[0][mask], iu[1][mask], strict=True)
    ]
    return sorted(pairs, key=lambda p: -p.similarity)


__all__ = [
    "DuplicatePair",
    "MinHasher",
    "NearDuplicateDetector",
    "cosine_duplicates",
    "jaccard",
    "shingles",
]
