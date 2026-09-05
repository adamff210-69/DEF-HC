"""Reproducible label-preserving perturbation suite (spec Exp-F).

Pure, deterministic transforms — never alter labels, never recursive, all
bounded.  Used by the robustness experiment to measure clean → perturbed
degradation and normalization recovery.  Generated perturbations are for
EVALUATION ONLY (never trained on).
"""

from __future__ import annotations

import base64

from defend_hc2.normalize import fold_leetspeak

_ZW = "​"


def zero_width_sprinkle(text: str, every: int = 3) -> str:
    """Insert a zero-width space between every `every`-th character."""
    return _ZW.join(text[i: i + every] for i in range(0, len(text), every))


def leetspeak(text: str) -> str:
    """Letters -> common leet digits.

    Must remain a strict *sub-morphism* of ``fold_leetspeak``
    (fold∘leet = identity on substituted characters) so the robustness
    experiment measures the detector behaviour the docstring promises.
    Only uniquely-invertible glyphs are used — notably ``l`` is NOT
    folded to ``1`` here, because ``fold_leetspeak`` maps ``1 -> i`` and
    an ``l -> 1`` source would fold back as ``i`` (e.g. "all" -> "aii"),
    silently defeating every pattern match and producing a non-invertible
    transform that conflates detector blindness with a broken probe.
    """
    table = str.maketrans({"o": "0", "O": "0", "i": "1", "I": "1",
                           "e": "3", "E": "3", "a": "4", "A": "4", "s": "5",
                           "S": "5", "t": "7", "T": "7"})
    return text.translate(table)


def whitespace_fragment(text: str) -> str:
    """Legacy combined transform — superseded by the word/letter split.

    Kept so any artifact referencing the old name errors loudly instead of
    silently aliasing.
    """
    raise RuntimeError(
        "whitespace_fragment was split into word_whitespace and "
        "letter_spacing_extreme (pooled scoring hid a 0.99 success and an "
        "OOD failure in one number; update callers)")


def word_whitespace(text: str) -> str:
    """Insert 2-4 spaces between every word.

    Deterministic by design — the gap width cycles 2,3,4 by word index
    (a seeded RNG would give identical behavior only if re-seeded every
    call; an explicit cycle keeps transforms pure and reproducible).
    Word-level multi-spacing leaves subword tokenization mostly intact
    (embedder handles it; this meansure in-distribution spacing noise).
    Fallback: no whitespace in the input → return the original string.
    """
    words = text.split()
    if len(words) <= 1:
        return text
    out = words[0]
    for i, w in enumerate(words[1:]):
        out += " " * (2 + (i % 3)) + w
    return out


def letter_spacing_extreme(text: str) -> str:
    """Join EVERY character with a single space (letter-fragmentation).

    This destroys subword tokenization — out-of-distribution for the
    embedding model by construction.  Despaced-collapse recovery is
    lexical-literal only (glue strings are never embedded, by policy), so
    degraded scores on this transform are an expected, documented
    limitation without train-time augmentation — not a pipeline bug.
    """
    return " ".join(text)


def base64_wrap(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def casing_shuffle(text: str) -> str:
    return "".join(ch.upper() if i % 2 else ch.lower()
                   for i, ch in enumerate(text))


def delimiter_stuff(text: str, delim: str = "|") -> str:
    """Insert delimiters inside keyword territory (imperatives)."""
    out = text
    for word in ("ignore", "forget", "reveal", "override"):
        for variant in (word, word.capitalize(), word.upper()):
            out = out.replace(variant, variant[0] + delim + variant[1:])
    return out


TRANSFORMS = {
    "zero_width": zero_width_sprinkle,
    "leetspeak": leetspeak,
    "word_whitespace": word_whitespace,
    "letter_spacing_extreme": letter_spacing_extreme,
    "base64": base64_wrap,
    "casing": casing_shuffle,
    "delimiter": delimiter_stuff,
}


def recover_fold(text: str) -> str:
    """Escapes this module; normalization recovery measurement uses
    defend_hc2.normalize.fold_leetspeak + basic_normalize instead — this
    alias kept for transform registry completeness."""
    return fold_leetspeak(text)
