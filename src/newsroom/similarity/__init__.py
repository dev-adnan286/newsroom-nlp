"""Document similarity and near-duplicate detection."""

from newsroom.similarity.dedup import (
    DuplicatePair,
    MinHasher,
    NearDuplicateDetector,
    jaccard,
    shingles,
)

__all__ = ["DuplicatePair", "MinHasher", "NearDuplicateDetector", "jaccard", "shingles"]
