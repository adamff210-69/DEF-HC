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
    """Letters -> common leet digits (inverse of fold_leetspeak)."""
    table = str.maketrans({"o": "0", "O": "0", "i": "1", "I": "1", "l": "1",
                           "e": "3", "E": "3", "a": "4", "A": "4", "s": "5",
                           "S": "5", "t": "7", "T": "7"})
    return text.translate(table)


def whitespace_fragment(text: str) -> str:
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
    "whitespace": whitespace_fragment,
    "base64": base64_wrap,
    "casing": casing_shuffle,
    "delimiter": delimiter_stuff,
}


def recover_fold(text: str) -> str:
    """Escapes this module; normalization recovery measurement uses
    defend_hc2.normalize.fold_leetspeak + basic_normalize instead — this
    alias kept for transform registry completeness."""
    return fold_leetspeak(text)
