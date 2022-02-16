from __future__ import annotations

import numpy as np
import pytest

from newsroom.similarity.dedup import (
    DuplicatePair,
    MinHasher,
    NearDuplicateDetector,
    cosine_duplicates,
    jaccard,
    shingles,
)

BASE = (
    "The central bank raised interest rates by a quarter point on Wednesday citing persistent "
    "inflation and a tight labour market while officials signalled further increases remain possible"
)
NEAR = BASE.replace("Wednesday", "Thursday").replace("possible", "likely")
OTHER = "The striker scored twice as the home side won the derby in front of a sold-out stadium on Sunday"


def test_shingles_and_jaccard() -> None:
    s = shingles("the quick brown fox", k=2)
    assert s == {"the quick", "quick brown", "brown fox"}
    assert shingles("hi", k=3) == {"hi"}
    assert shingles("", k=3) == set()
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert jaccard(set(), set()) == 0.0
    with pytest.raises(ValueError):
        shingles("x", k=0)


def test_minhash_estimates_jaccard() -> None:
    hasher = MinHasher(num_perm=256, shingle_size=2)
    sig_a, sig_b, sig_c = hasher.signature(BASE), hasher.signature(NEAR), hasher.signature(OTHER)
    assert sig_a.shape == (256,) and sig_a.dtype == np.uint64
    true_ab = jaccard(shingles(BASE, 2), shingles(NEAR, 2))
    est_ab = hasher.estimate_jaccard(sig_a, sig_b)
    assert abs(est_ab - true_ab) < 0.15
    assert hasher.estimate_jaccard(sig_a, sig_c) < 0.1
    assert np.array_equal(hasher.signature(BASE), sig_a)  # deterministic
    with pytest.raises(ValueError):
        MinHasher(num_perm=2)
    with pytest.raises(ValueError):
        hasher.estimate_jaccard(sig_a, sig_a[:10])


def test_detector_finds_near_duplicates() -> None:
    detector = NearDuplicateDetector(threshold=0.5, num_perm=128, bands=32, shingle_size=2)
    detector.add_many(["base", "near", "other"], [BASE, NEAR, OTHER])
    assert len(detector) == 3
    pairs = detector.find_duplicates()
    assert pairs and pairs[0] == DuplicatePair("base", "near", pairs[0].similarity)
    assert all("other" not in (p.left, p.right) for p in pairs)
    assert "near" in detector.candidates("base")
    assert detector.clusters() == [["base", "near"]]
    hits = detector.query(NEAR)
    assert hits and hits[0][0] in {"near", "base"}
    assert detector.query(OTHER)[0][0] == "other"


def test_detector_validation() -> None:
    with pytest.raises(ValueError):
        NearDuplicateDetector(threshold=0)
    with pytest.raises(ValueError):
        NearDuplicateDetector(num_perm=100, bands=32)
    detector = NearDuplicateDetector()
    detector.add("a", BASE)
    with pytest.raises(ValueError):
        detector.add("a", BASE)
    with pytest.raises(ValueError):
        detector.add_many(["b"], [BASE, NEAR])


def test_cosine_duplicates() -> None:
    emb = np.array([[1.0, 0.0], [0.99, 0.05], [0.0, 1.0]])
    pairs = cosine_duplicates(emb, ["a", "b", "c"], threshold=0.95)
    assert len(pairs) == 1 and (pairs[0].left, pairs[0].right) == ("a", "b")
