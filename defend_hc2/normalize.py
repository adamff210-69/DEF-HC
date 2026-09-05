"""Conservative attack-variant normalization (spec Phase 2).

Produces a small, bounded set of text *variants* so lexical / ML detectors
can score the surface an attacker actually controls:

* ``raw``        — the text as supplied;
* ``normalized`` — Unicode NFKC + zero-width/invisible char removal +
                   collapsed excessive whitespace;
* ``folded``     — leetspeak folding on top of ``normalized``;
* ``b64_i``      — safely decoded Base64-looking tokens (strict limits).

Everything is bounded: input is truncated, Base64 tokens and decoded output
are length-capped, decoding is never recursive and non-text decodes are
rejected.  The layer is deliberately conservative — it *adds* detector
views; it never rewrites what downstream layers see.
"""

from __future__ import annotations

import base64
import re
import unicodedata
from typing import Final

# ------------------------------------------------------------------ limits
MAX_INPUT_CHARS: Final = 8192
# Whole-message Base64 wraps of realistic prompts (system-prompt-style rows
# commonly run 1-3 KB) must still decode — Exp-F diagnosis showed 1024/512
# silently produced zero decoded variants for exactly those attacks.  Bounds
# stay hard (worst case 4 variants × ~3 KB decode + printability scan).
MAX_B64_TOKEN_CHARS: Final = 4096
MAX_B64_DECODED_CHARS: Final = 3072
MAX_B64_VARIANTS: Final = 4
MIN_B64_TOKEN_CHARS: Final = 24

# 200B-200F (zero-width space / non-joiner / joiner / marks), 2060 word
# joiner, FEFF BOM / zero-width no-break, 00AD soft hyphen.
_ZERO_WIDTH: Final = frozenset(
    "\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\u00ad"
)

_LEET_TABLE: Final = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
     "@": "a", "$": "s", "|": "l"}
)

_B64_TOKEN_RE: Final = re.compile(r"[A-Za-z0-9+/]{%d,}={0,2}" % MIN_B64_TOKEN_CHARS)
_WS_RUN_RE: Final = re.compile(r"[ \t]{2,}")
_NL_RUN_RE: Final = re.compile(r"\n{3,}")


def basic_normalize(text: str, max_chars: int = MAX_INPUT_CHARS) -> str:
    """NFKC + invisible-char removal + collapse excessive whitespace.

    Newlines are preserved (collapsed only beyond two) so line-anchored
    structural patterns (fake role headers) still work.
    """
    if not isinstance(text, str):
        raise TypeError("text must be str")
    text = text[:max_chars]
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch not in _ZERO_WIDTH)
    text = _NL_RUN_RE.sub("\n\n", text)
    text = _WS_RUN_RE.sub(" ", text)
    return text.strip()


def fold_leetspeak(text: str) -> str:
    """Map common leetspeak glyphs back to letters (0→o, 3→e, …)."""
    return text.translate(_LEET_TABLE)


def b64_variants(text: str) -> list[str]:
    """Decode plausible Base64 tokens under strict resource limits."""
    decoded: list[str] = []
    for match in _B64_TOKEN_RE.finditer(text[:MAX_INPUT_CHARS]):
        token = match.group(0)
        if len(token) > MAX_B64_TOKEN_CHARS or len(decoded) >= MAX_B64_VARIANTS:
            break
        # true Base64 length is a multiple of 4
        padded = token + "=" * (-len(token) % 4)
        try:
            raw = base64.b64decode(padded, validate=False)
            text_out = raw.decode("utf-8", errors="strict")
        except Exception:
            continue
        # accept only mostly-printable text decodes
        if not text_out:
            continue
        printable = sum(1 for c in text_out if c.isprintable() or c in "\n\t")
        if printable / len(text_out) < 0.90:
            continue
        value = text_out[:MAX_B64_DECODED_CHARS].strip()
        if value and value not in decoded:
            decoded.append(value)
    return decoded


def variants(text: str) -> dict[str, str]:
    """All detector views of ``text`` in deterministic order.

    Keys: ``raw`` (always), ``normalized``, ``folded``, ``b64_0``…
    Variants identical to an earlier view are omitted.
    """
    out: dict[str, str] = {"raw": text[:MAX_INPUT_CHARS]}
    normalized = basic_normalize(text)
    if normalized != out["raw"]:
        out["normalized"] = normalized
    folded = fold_leetspeak(normalized)
    if folded != normalized:
        out["folded"] = folded
    for i, dec in enumerate(b64_variants(normalized)):
        out[f"b64_{i}"] = dec
    return out
