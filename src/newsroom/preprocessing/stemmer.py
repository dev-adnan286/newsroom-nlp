"""A compact Porter-style suffix stripping stemmer implemented from the algorithm's rules.

The implementation follows the five classic steps (1a, 1b, 1c, 2, 3, 4, 5a, 5b) using the
``m`` (measure) notion from Porter's 1980 paper. It is intentionally self-contained so the
package works fully offline; ``nltk`` can be used instead via :func:`get_stemmer` when the
optional dependency is installed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

_VOWELS = set("aeiou")


class PorterStemmer:
    """Rule-based English stemmer.

    Example:
        >>> PorterStemmer().stem("relational")
        'relat'
    """

    def __init__(self, cache_size: int = 65536) -> None:
        self._cache: dict[str, str] = {}
        self._cache_limit = cache_size

    _STEP2 = (
        ("ational", "ate"),
        ("tional", "tion"),
        ("enci", "ence"),
        ("anci", "ance"),
        ("izer", "ize"),
        ("abli", "able"),
        ("alli", "al"),
        ("entli", "ent"),
        ("eli", "e"),
        ("ousli", "ous"),
        ("ization", "ize"),
        ("ation", "ate"),
        ("ator", "ate"),
        ("alism", "al"),
        ("iveness", "ive"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("aliti", "al"),
        ("iviti", "ive"),
        ("biliti", "ble"),
    )
    _STEP3 = (
        ("icate", "ic"),
        ("ative", ""),
        ("alize", "al"),
        ("iciti", "ic"),
        ("ical", "ic"),
        ("ful", ""),
        ("ness", ""),
    )
    _STEP4 = (
        "al",
        "ance",
        "ence",
        "er",
        "ic",
        "able",
        "ible",
        "ant",
        "ement",
        "ment",
        "ent",
        "ou",
        "ism",
        "ate",
        "iti",
        "ous",
        "ive",
        "ize",
    )

    # -- helpers ---------------------------------------------------------------------
    @staticmethod
    def _is_consonant(word: str, i: int) -> bool:
        ch = word[i]
        if ch in _VOWELS:
            return False
        if ch == "y":
            return i == 0 or not PorterStemmer._is_consonant(word, i - 1)
        return True

    @classmethod
    def _measure(cls, stem: str) -> int:
        """Count VC sequences in ``stem`` (Porter's ``m``)."""
        m = 0
        prev_vowel = False
        for i in range(len(stem)):
            is_v = not cls._is_consonant(stem, i)
            if not is_v and prev_vowel:
                m += 1
            prev_vowel = is_v
        return m

    @classmethod
    def _contains_vowel(cls, stem: str) -> bool:
        return any(not cls._is_consonant(stem, i) for i in range(len(stem)))

    @classmethod
    def _ends_double_consonant(cls, word: str) -> bool:
        return len(word) >= 2 and word[-1] == word[-2] and cls._is_consonant(word, len(word) - 1)

    @classmethod
    def _cvc(cls, word: str) -> bool:
        if len(word) < 3:
            return False
        n = len(word)
        return (
            cls._is_consonant(word, n - 1)
            and not cls._is_consonant(word, n - 2)
            and cls._is_consonant(word, n - 3)
            and word[-1] not in "wxy"
        )

    # -- steps -----------------------------------------------------------------------
    def _step1a(self, w: str) -> str:
        if w.endswith("sses"):
            return w[:-2]
        if w.endswith("ies"):
            return w[:-2]
        if w.endswith("ss"):
            return w
        if w.endswith("s"):
            return w[:-1]
        return w

    def _step1b(self, w: str) -> str:
        if w.endswith("eed"):
            stem = w[:-3]
            return w[:-1] if self._measure(stem) > 0 else w
        changed = False
        if w.endswith("ed") and self._contains_vowel(w[:-2]):
            w, changed = w[:-2], True
        elif w.endswith("ing") and self._contains_vowel(w[:-3]):
            w, changed = w[:-3], True
        if changed:
            if w.endswith(("at", "bl", "iz")):
                return w + "e"
            if self._ends_double_consonant(w) and w[-1] not in "lsz":
                return w[:-1]
            if self._measure(w) == 1 and self._cvc(w):
                return w + "e"
        return w

    def _step1c(self, w: str) -> str:
        if w.endswith("y") and self._contains_vowel(w[:-1]):
            return w[:-1] + "i"
        return w

    def _apply_rules(self, w: str, rules: Iterable[tuple[str, str]], min_measure: int) -> str:
        for suffix, replacement in rules:
            if w.endswith(suffix):
                stem = w[: -len(suffix)]
                if self._measure(stem) > min_measure - 1:
                    return stem + replacement
                return w
        return w

    def _step2(self, w: str) -> str:
        return self._apply_rules(w, self._STEP2, 1)

    def _step3(self, w: str) -> str:
        return self._apply_rules(w, self._STEP3, 1)

    def _step4(self, w: str) -> str:
        for suffix in self._STEP4:
            if w.endswith(suffix):
                stem = w[: -len(suffix)]
                if self._measure(stem) > 1:
                    return stem
                return w
        if w.endswith("ion"):
            stem = w[:-3]
            if stem and stem[-1] in "st" and self._measure(stem) > 1:
                return stem
        return w

    def _step5a(self, w: str) -> str:
        if w.endswith("e"):
            stem = w[:-1]
            m = self._measure(stem)
            if m > 1 or (m == 1 and not self._cvc(stem)):
                return stem
        return w

    def _step5b(self, w: str) -> str:
        if self._measure(w) > 1 and self._ends_double_consonant(w) and w.endswith("l"):
            return w[:-1]
        return w

    def stem(self, word: str) -> str:
        """Return the stem of a single lowercase ``word`` (memoised per instance)."""
        if len(word) <= 2:
            return word
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        w = word
        for step in (
            self._step1a,
            self._step1b,
            self._step1c,
            self._step2,
            self._step3,
            self._step4,
            self._step5a,
            self._step5b,
        ):
            w = step(w)
        if len(self._cache) < self._cache_limit:
            self._cache[word] = w
        return w

    def __call__(self, word: str) -> str:
        return self.stem(word)


def get_stemmer(prefer_nltk: bool = False) -> Callable[[str], str]:
    """Return a stemming callable; uses NLTK's Porter stemmer if requested and installed."""
    if prefer_nltk:
        try:
            from nltk.stem import PorterStemmer as _NltkPorter  # type: ignore

            nltk_stemmer = _NltkPorter()
            return nltk_stemmer.stem
        except ImportError:  # pragma: no cover - depends on optional dependency
            pass
    return PorterStemmer().stem


def stem_tokens(tokens: Iterable[str], stemmer: Callable[[str], str] | None = None) -> list[str]:
    """Stem every token with ``stemmer`` (default: built-in :class:`PorterStemmer`)."""
    fn = stemmer or _DEFAULT_STEMMER.stem
    return [fn(t) for t in tokens]


_DEFAULT_STEMMER = PorterStemmer()

__all__ = ["PorterStemmer", "get_stemmer", "stem_tokens"]
