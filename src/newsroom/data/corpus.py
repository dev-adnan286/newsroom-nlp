"""Synthetic-but-realistic labelled news corpus.

Documents are assembled from per-category sentence templates whose slots are filled from
category vocabularies. A shared pool of generic "newsroom" sentences and light cross-category
vocabulary leakage keep the task non-trivial. Generation is fully deterministic for a given
seed, so tests and demos are reproducible without any network access.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from newsroom.utils.logging import get_logger

logger = get_logger(__name__)

CATEGORIES: tuple[str, ...] = ("technology", "sports", "politics", "business", "health", "science")
REQUIRED_COLUMNS: tuple[str, ...] = ("text", "label", "title", "published_at")

# --------------------------------------------------------------------------------------
# Vocabularies. Each category maps slot names to candidate fillers.
# --------------------------------------------------------------------------------------
_VOCAB: dict[str, dict[str, list[str]]] = {
    "technology": {
        "org": [
            "Nimbus Labs",
            "Quantara",
            "Vektor Systems",
            "OpenGrid",
            "Hollow Pixel",
            "Arcline AI",
        ],
        "product": [
            "smartphone",
            "cloud platform",
            "open-source framework",
            "chip architecture",
            "developer toolkit",
            "wearable headset",
            "language model",
            "browser engine",
        ],
        "tech": [
            "machine learning",
            "edge computing",
            "quantum processors",
            "5G connectivity",
            "generative AI",
            "semiconductor fabrication",
            "encryption",
            "robotics",
        ],
        "person": [
            "the chief technology officer",
            "lead engineer Priya Raman",
            "founder Tomas Weil",
            "analyst Dana Ortiz",
        ],
        "verb": ["unveiled", "open-sourced", "shipped", "benchmarked", "patched", "rolled out"],
        "issue": [
            "a zero-day vulnerability",
            "latency regressions",
            "battery drain",
            "data privacy concerns",
            "supply shortages of GPUs",
        ],
    },
    "sports": {
        "org": [
            "Riverside United",
            "the Harbour City Hawks",
            "Northgate FC",
            "the Summit Rockets",
            "Valley Athletic",
            "Coastal Rovers",
        ],
        "product": [
            "championship",
            "playoff series",
            "derby",
            "season opener",
            "cup final",
            "qualifying round",
            "grand prix",
            "marathon",
        ],
        "tech": [
            "a late penalty",
            "a stoppage-time header",
            "a triple-double",
            "a record lap",
            "a comeback in the fourth quarter",
            "back-to-back aces",
        ],
        "person": [
            "striker Malik Osei",
            "captain Elena Voss",
            "head coach Ray Delgado",
            "goalkeeper Sam Ito",
            "sprinter Nia Bekele",
        ],
        "verb": ["defeated", "edged", "outscored", "held off", "swept", "rallied past"],
        "issue": [
            "a hamstring injury",
            "a controversial offside call",
            "a transfer dispute",
            "a suspension for two matches",
            "fatigue after a congested schedule",
        ],
    },
    "politics": {
        "org": [
            "the Senate",
            "the Parliament",
            "the ruling coalition",
            "the opposition party",
            "the Election Commission",
            "the city council",
        ],
        "product": [
            "budget bill",
            "immigration reform",
            "campaign platform",
            "coalition agreement",
            "constitutional amendment",
            "trade sanctions package",
            "housing policy",
        ],
        "tech": [
            "a bipartisan compromise",
            "a no-confidence vote",
            "a filibuster",
            "a snap election",
            "a leaked memo",
            "an executive order",
        ],
        "person": [
            "Prime Minister Ada Kovac",
            "Senator Luis Marquez",
            "Governor Ruth Anand",
            "Minister Femi Adeyemi",
            "spokesperson Ivan Petrov",
        ],
        "verb": ["passed", "vetoed", "debated", "postponed", "ratified", "rejected"],
        "issue": [
            "voter turnout",
            "lobbying disclosures",
            "gerrymandered districts",
            "the cabinet reshuffle",
            "diplomatic tensions with neighbours",
        ],
    },
    "business": {
        "org": [
            "Meridian Holdings",
            "Cobalt Retail Group",
            "Atlas Logistics",
            "Brightwater Capital",
            "Sable Foods",
            "Orion Energy",
        ],
        "product": [
            "quarterly earnings",
            "merger",
            "initial public offering",
            "dividend",
            "share buyback",
            "supply contract",
            "loan facility",
            "profit forecast",
        ],
        "tech": [
            "rising interest rates",
            "inflation data",
            "a stronger dollar",
            "weak consumer demand",
            "tighter credit conditions",
            "record revenue",
        ],
        "person": [
            "chief executive Hana Sato",
            "CFO Marcus Bell",
            "investor Greta Lindqvist",
            "central bank governor Omar Haddad",
            "analyst Wei Zhang",
        ],
        "verb": ["reported", "acquired", "forecast", "slashed", "raised", "restructured"],
        "issue": [
            "shareholder pressure",
            "layoffs across regional offices",
            "a debt downgrade",
            "antitrust scrutiny",
            "currency volatility",
        ],
    },
    "health": {
        "org": [
            "St. Aldric Hospital",
            "the National Health Agency",
            "Cedar Clinic Network",
            "the World Health Council",
            "Lumen Pharmaceuticals",
            "the vaccine task force",
        ],
        "product": [
            "vaccine",
            "clinical trial",
            "screening programme",
            "antibiotic",
            "mental health service",
            "insulin therapy",
            "wellness guideline",
        ],
        "tech": [
            "a measles outbreak",
            "seasonal influenza",
            "rising obesity rates",
            "a shortage of nurses",
            "long waiting times",
            "an antimicrobial resistance study",
        ],
        "person": [
            "Dr. Amina Yusuf",
            "surgeon Leo Brandt",
            "epidemiologist Sofia Neri",
            "nurse practitioner Grace Odum",
            "health minister Karl Jensen",
        ],
        "verb": ["approved", "recommended", "launched", "expanded", "suspended", "monitored"],
        "issue": [
            "side effects reported by patients",
            "hospital bed capacity",
            "insurance coverage gaps",
            "misinformation about treatments",
            "rural access to care",
        ],
    },
    "science": {
        "org": [
            "the Kepler Institute",
            "a team at Tidewater University",
            "the European Space Consortium",
            "the Deep Ocean Observatory",
            "the Polar Research Station",
            "the Genome Atlas project",
        ],
        "product": [
            "telescope",
            "particle detector",
            "climate model",
            "fossil discovery",
            "CRISPR experiment",
            "rover mission",
            "coral reef survey",
        ],
        "tech": [
            "dark matter",
            "gene expression",
            "exoplanet atmospheres",
            "melting permafrost",
            "protein folding",
            "gravitational waves",
            "volcanic activity",
        ],
        "person": [
            "astrophysicist Dr. Ines Farah",
            "geologist Ben Okafor",
            "biologist Mei Tanaka",
            "climatologist Erik Sund",
            "lead researcher Dr. Nadia Volkov",
        ],
        "verb": ["observed", "measured", "published", "simulated", "sequenced", "confirmed"],
        "issue": [
            "funding cuts to basic research",
            "peer review delays",
            "sample contamination",
            "sensor calibration errors",
            "competing hypotheses",
        ],
    },
}

_TEMPLATES: dict[str, list[str]] = {
    "technology": [
        "{org} {verb} a new {product} built around {tech}, promising faster performance for developers.",
        "According to {person}, the {product} will rely on {tech} to cut costs for enterprise customers.",
        "Security researchers flagged {issue} in the latest {product}, prompting {org} to release an update.",
        "The {product} race intensified as {org} demonstrated {tech} at its annual developer conference.",
        "{person} said adoption of {tech} across the {product} ecosystem is accelerating despite {issue}.",
        "Investors questioned whether {tech} can scale after {org} delayed its {product} roadmap.",
    ],
    "sports": [
        "{org} {verb} their rivals in the {product} thanks to {tech} from {person}.",
        "{person} shrugged off {issue} to deliver a decisive performance in the {product}.",
        "A tense {product} ended with {tech}, sending {org} fans into celebration.",
        "The coach praised {person} after {org} {verb} a stubborn opponent to reach the {product}.",
        "{org} face uncertainty ahead of the {product} because of {issue} in the squad.",
        "Fans debated {tech} for days as {org} {verb} a title contender in front of a sold-out stadium.",
    ],
    "politics": [
        "{org} {verb} the {product} after a heated session dominated by {tech}.",
        "{person} defended the {product}, arguing that {issue} demanded urgent action.",
        "Critics said {tech} exposed deep divisions inside {org} over the {product}.",
        "Negotiations over the {product} stalled as {org} weighed {tech} and {issue}.",
        "{person} announced that {org} would revisit the {product} before the next election cycle.",
        "Polls suggest {issue} could shape how voters judge the {product} championed by {person}.",
    ],
    "business": [
        "{org} {verb} its {product} as {tech} weighed on margins, sending shares lower in early trading.",
        "{person} told analysts the {product} reflects {tech} rather than {issue}.",
        "Markets reacted to the {product} from {org}, with traders citing {tech} and {issue}.",
        "{org} {verb} guidance for the year, blaming {tech} and warning about {issue}.",
        "The {product} announced by {org} values the deal well above estimates, according to {person}.",
        "Economists said {tech} will keep pressure on the {product} outlook for firms like {org}.",
    ],
    "health": [
        "{org} {verb} a {product} aimed at tackling {tech} among older patients.",
        "{person} warned that {issue} could undermine the new {product} rolled out this spring.",
        "Data from the {product} showed promising results against {tech}, {org} said.",
        "Doctors at {org} reported {tech} straining wards, renewing calls to expand the {product}.",
        "{person} urged families to join the {product}, citing {tech} and {issue}.",
        "Regulators reviewed the {product} after {org} disclosed {issue} during monitoring.",
    ],
    "science": [
        "{org} {verb} evidence of {tech} using a {product} designed for high-precision measurements.",
        "{person} explained that the {product} data could reshape theories about {tech}.",
        "A peer-reviewed paper from {org} links {tech} to patterns first seen by the {product}.",
        "Despite {issue}, {org} {verb} results from the {product} that surprised the field.",
        "{person} cautioned that {tech} findings from the {product} still need independent replication.",
        "The {product} returned its first readings on {tech}, {org} announced on Tuesday.",
    ],
}

_TITLE_TEMPLATES: dict[str, list[str]] = {
    "technology": [
        "{org} bets on {tech} with new {product}",
        "{product} update targets {issue}",
        "Why {tech} matters for the next {product}",
    ],
    "sports": [
        "{org} {verb} rivals in {product} thriller",
        "{person} stars as {org} reach {product}",
        "{product} drama: {tech} decides it",
    ],
    "politics": [
        "{org} {verb} {product} amid {tech}",
        "{person} stakes future on {product}",
        "{product} faces test over {issue}",
    ],
    "business": [
        "{org} {verb} {product} as {tech} bites",
        "{person}: {product} shows resilience",
        "{product} rattles markets amid {issue}",
    ],
    "health": [
        "{org} {verb} {product} against {tech}",
        "{person} calls for wider {product}",
        "{product} under review after {issue}",
    ],
    "science": [
        "{org} {verb} signs of {tech}",
        "{product} sheds light on {tech}",
        "{person} on {tech}: what we know",
    ],
}

# Templates shared by every category: only the vocabulary filling the slots carries signal.
_SHARED_TEMPLATES: list[str] = [
    "{org} {verb} the {product} on Tuesday, a move {person} described as long overdue.",
    "{person} said the {product} would be judged on {tech} rather than on {issue}.",
    "Questions about {issue} overshadowed the {product} unveiled by {org} this week.",
    "Observers linked the {product} to {tech}, while {org} pointed instead to {issue}.",
    "In a statement, {org} {verb} its position on the {product} and thanked {person}.",
    "The {product} drew attention from {person}, who cited {tech} as the deciding factor.",
    "{org} faces a difficult week after {issue} and renewed scrutiny of the {product}.",
]

_GENERIC_SENTENCES: list[str] = [
    "Officials declined to comment further when contacted on Monday.",
    "The announcement was made at a press briefing attended by local reporters.",
    "Analysts expect more details to emerge over the coming weeks.",
    "A spokesperson said the situation remained under review.",
    "Reaction on social media was swift and divided.",
    "The story was first reported by regional outlets before being confirmed.",
    "Further updates are expected once a formal statement is released.",
    "Observers described the development as significant for the sector.",
    "Sources familiar with the matter spoke on condition of anonymity.",
    "The full report is scheduled for publication early next month.",
]


@dataclass(frozen=True)
class CorpusSpec:
    """Parameters controlling synthetic corpus generation."""

    n_docs: int = 2000
    seed: int = 42
    min_sentences: int = 3
    max_sentences: int = 6
    generic_probability: float = 0.4
    leak_probability: float = 0.3
    shared_template_probability: float = 0.6
    start_date: datetime = datetime(2025, 1, 1)
    days_span: int = 540
    categories: Sequence[str] = CATEGORIES

    def __post_init__(self) -> None:
        if self.n_docs <= 0:
            raise ValueError("n_docs must be positive")
        if self.min_sentences < 1 or self.max_sentences < self.min_sentences:
            raise ValueError("invalid sentence range")
        unknown = set(self.categories) - set(CATEGORIES)
        if unknown:
            raise ValueError(f"unknown categories: {sorted(unknown)}")


class CorpusGenerator:
    """Deterministic generator of labelled pseudo-news documents.

    Example:
        >>> df = CorpusGenerator(CorpusSpec(n_docs=50)).generate()
        >>> sorted(df.columns) == ["label", "published_at", "text", "title"]
        True
    """

    def __init__(self, spec: CorpusSpec | None = None) -> None:
        self.spec = spec or CorpusSpec()
        self._rng = np.random.RandomState(self.spec.seed)

    def _fill(self, template: str, category: str) -> str:
        vocab = _VOCAB[category]
        rng = self._rng

        def _slot(match: re.Match[str]) -> str:
            key = match.group(1)
            source = category
            if rng.random_sample() < self.spec.leak_probability:
                source = str(rng.choice([c for c in CATEGORIES if c != category]))
            choices = _VOCAB[source].get(key) or vocab[key]
            return str(choices[rng.randint(len(choices))])

        return re.sub(r"\{(\w+)\}", _slot, template)

    def _topical_sentence(self, category: str) -> str:
        """A sentence built from either a category-specific or a shared template."""
        if self._rng.random_sample() < self.spec.shared_template_probability:
            templates = _SHARED_TEMPLATES
        else:
            templates = _TEMPLATES[category]
        return self._fill(templates[self._rng.randint(len(templates))], category)

    def _sentence(self, category: str) -> str:
        if self._rng.random_sample() < self.spec.generic_probability:
            return _GENERIC_SENTENCES[self._rng.randint(len(_GENERIC_SENTENCES))]
        return self._topical_sentence(category)

    def _title(self, category: str) -> str:
        templates = _TITLE_TEMPLATES[category]
        raw = self._fill(templates[self._rng.randint(len(templates))], category)
        return raw[0].upper() + raw[1:]

    def _document(self, category: str) -> dict[str, Any]:
        n_sent = self._rng.randint(self.spec.min_sentences, self.spec.max_sentences + 1)
        sentences = [self._topical_sentence(category)]
        sentences.extend(self._sentence(category) for _ in range(n_sent - 1))
        offset_days = int(self._rng.randint(self.spec.days_span))
        offset_minutes = int(self._rng.randint(24 * 60))
        published = self.spec.start_date + timedelta(days=offset_days, minutes=offset_minutes)
        return {
            "title": self._title(category),
            "text": " ".join(s[0].upper() + s[1:] for s in sentences),
            "label": category,
            "published_at": published,
        }

    def generate(self) -> pd.DataFrame:
        """Generate the corpus as a DataFrame with ``text, label, title, published_at``."""
        cats = list(self.spec.categories)
        labels = [cats[i % len(cats)] for i in range(self.spec.n_docs)]
        self._rng.shuffle(labels)  # type: ignore[arg-type]
        rows = [self._document(label) for label in labels]
        df = pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS))
        df["published_at"] = pd.to_datetime(df["published_at"])
        df.insert(0, "doc_id", [f"doc-{i:05d}" for i in range(len(df))])
        logger.info("Generated %d synthetic documents across %d categories", len(df), len(cats))
        return df


def generate_corpus(n_docs: int = 2000, seed: int = 42, **kwargs: Any) -> pd.DataFrame:
    """Convenience wrapper around :class:`CorpusGenerator`."""
    return CorpusGenerator(CorpusSpec(n_docs=n_docs, seed=seed, **kwargs)).generate()


def save_corpus_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Persist a corpus DataFrame to CSV (UTF-8, ISO timestamps)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if "published_at" in out.columns:
        out["published_at"] = pd.to_datetime(out["published_at"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    out.to_csv(p, index=False, encoding="utf-8")
    logger.info("Saved %d rows to %s", len(out), p)
    return p


def load_corpus_csv(path: str | Path, *, require_label: bool = True) -> pd.DataFrame:
    """Load a corpus CSV and validate the expected columns."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Corpus file not found: {p}")
    df = pd.read_csv(p, encoding="utf-8")
    missing = [c for c in ("text",) if c not in df.columns]
    if require_label and "label" not in df.columns:
        missing.append("label")
    if missing:
        raise ValueError(f"Corpus at {p} is missing required columns: {missing}")
    if "title" not in df.columns:
        df["title"] = df["text"].str.slice(0, 60)
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    else:
        df["published_at"] = pd.NaT
    if "doc_id" not in df.columns:
        df.insert(0, "doc_id", [f"doc-{i:05d}" for i in range(len(df))])
    df["text"] = df["text"].fillna("").astype(str)
    return df


def load_twenty_newsgroups(
    subset: str = "train",
    categories: Sequence[str] | None = None,
    *,
    data_home: str | Path | None = None,
    remove_headers: bool = True,
) -> pd.DataFrame:
    """Load the classic 20 Newsgroups dataset via scikit-learn (requires network on first use).

    This loader is intentionally opt-in; tests and demos never call it.
    """
    from sklearn.datasets import fetch_20newsgroups

    remove = ("headers", "footers", "quotes") if remove_headers else ()
    bunch = fetch_20newsgroups(
        subset=subset,
        categories=list(categories) if categories else None,
        remove=remove,
        data_home=str(data_home) if data_home else None,
    )
    names = list(bunch.target_names)
    df = pd.DataFrame(
        {
            "text": [str(t) for t in bunch.data],
            "label": [names[i] for i in bunch.target],
        }
    )
    df["title"] = df["text"].str.strip().str.split("\n").str[0].str.slice(0, 80)
    df["published_at"] = pd.NaT
    df.insert(0, "doc_id", [f"ng-{i:05d}" for i in range(len(df))])
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    logger.info("Loaded %d 20-newsgroups documents (%s)", len(df), subset)
    return df


__all__ = [
    "CATEGORIES",
    "REQUIRED_COLUMNS",
    "CorpusGenerator",
    "CorpusSpec",
    "generate_corpus",
    "load_corpus_csv",
    "load_twenty_newsgroups",
    "save_corpus_csv",
]
