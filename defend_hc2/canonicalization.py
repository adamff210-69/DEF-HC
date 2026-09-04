"""Layer 0 — canonicalization and schema validation.

Everything hashed or MAC-ed by DEFEND-HC2 goes through this module:

* text is normalized with Unicode NFKC and stripped of invalid control
  characters (including zero-width and bidirectional-override code points
  commonly abused to hide injected instructions);
* structured payloads are serialized with deterministic JSON
  (``sort_keys=True, separators=(",", ":"), ensure_ascii=False``);
* hashes are computed over canonical UTF-8 bytes with domain separation
  (every field length-prefixed, so framing is unambiguous).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import unicodedata
from typing import Any, Iterable

from defend_hc2.constants import TAG_PAYLOAD
from defend_hc2.exceptions import SchemaValidationError

# C0/C1 control characters other than \t \n \r, plus DEL.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Zero-width / direction-override code points frequently used to smuggle
# invisible instructions past human review.
_SNEAKY_CODEPOINTS = {
    "​",  # U+200B ZERO WIDTH SPACE
    "‌",  # U+200C ZERO WIDTH NON-JOINER
    "‍",  # U+200D ZERO WIDTH JOINER
    "⁠",  # U+2060 WORD JOINER
    "﻿",  # U+FEFF ZERO WIDTH NO-BREAK SPACE / BOM
    "‪",  # U+202A LEFT-TO-RIGHT EMBEDDING
    "‫",  # U+202B RIGHT-TO-LEFT EMBEDDING
    "‬",  # U+202C POP DIRECTIONAL FORMATTING
    "‭",  # U+202D LEFT-TO-RIGHT OVERRIDE
    "‮",  # U+202E RIGHT-TO-LEFT OVERRIDE
    "⁦",  # U+2066 LEFT-TO-RIGHT ISOLATE
    "⁧",  # U+2067 RIGHT-TO-LEFT ISOLATE
    "⁨",  # U+2068 FIRST STRONG ISOLATE
    "⁩",  # U+2069 POP DIRECTIONAL ISOLATE
}


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return Canonicalizer.normalize_text(value)
    if isinstance(value, dict):
        return {str(_normalize_scalar(k)): _normalize_scalar(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_scalar(v) for v in value]
    return value


class Canonicalizer:
    """Deterministic text/binary canonicalization (spec: Layer 0)."""

    # ------------------------------------------------------------------ text
    @staticmethod
    def normalize_text(text: str) -> str:
        """NFKC-normalize and remove invalid control characters."""
        if not isinstance(text, str):
            raise SchemaValidationError(f"expected str, got {type(text).__name__}")
        text = unicodedata.normalize("NFKC", text)
        text = _CONTROL_RE.sub("", text)
        for cp in _SNEAKY_CODEPOINTS:
            text = text.replace(cp, "")
        return text

    @staticmethod
    def normalize_obj(obj: Any) -> Any:
        """Recursively normalize every string in a JSON-like structure."""
        return _normalize_scalar(obj)

    # ------------------------------------------------------------------ json
    @staticmethod
    def canonical_json(obj: Any) -> str:
        """Deterministic JSON serialization (spec)."""
        return json.dumps(
            obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    @staticmethod
    def canonical_bytes(obj: Any) -> bytes:
        """Canonical UTF-8 bytes of a (normalized) JSON-like object."""
        return Canonicalizer.canonical_json(Canonicalizer.normalize_obj(obj)).encode("utf-8")

    # ---------------------------------------------------------------- framing
    @staticmethod
    def frame(parts: Iterable[bytes]) -> bytes:
        """Unambiguous concatenation: ``len(part) || part`` for each part.

        Instantiates the spec's ``a || b || c`` notation so that field
        boundaries can never shift (e.g. ``H("ab"||"c") != H("a"||"bc")``).
        """
        out = bytearray()
        for part in parts:
            out += len(part).to_bytes(4, "big")
            out += part
        return bytes(out)

    # ------------------------------------------------------------------ hash
    @staticmethod
    def sha3_256(*parts: bytes, tag: bytes | None = None) -> bytes:
        """SHA3-256 over framed parts with optional domain separation."""
        h = hashlib.sha3_256()
        h.update(Canonicalizer.frame(([tag] if tag else []) + list(parts)))
        return h.digest()

    @staticmethod
    def sha3_256_hex(*parts: bytes, tag: bytes | None = None) -> str:
        return Canonicalizer.sha3_256(*parts, tag=tag).hex()

    @staticmethod
    def hmac_sha3_256(key: bytes, *parts: bytes, tag: bytes | None = None) -> bytes:
        """Keyed HMAC-SHA3-256 over framed parts."""
        return hmac.new(
            key, Canonicalizer.frame(([tag] if tag else []) + list(parts)),
            hashlib.sha3_256,
        ).digest()

    @staticmethod
    def hmac_sha3_256_hex(key: bytes, *parts: bytes, tag: bytes | None = None) -> str:
        return Canonicalizer.hmac_sha3_256(key, *parts, tag=tag).hex()

    # ---------------------------------------------------------------- payload
    @staticmethod
    def payload_hash(payload: Any) -> str:
        """Domain-separated hash of a canonicalized payload (spec: L2)."""
        return Canonicalizer.sha3_256_hex(
            Canonicalizer.canonical_bytes(payload), tag=TAG_PAYLOAD
        )

    # --------------------------------------------------------------- schema
    @staticmethod
    def validate_schema(
        payload: dict[str, Any],
        required: dict[str, type | tuple[type, ...]],
        optional: dict[str, type | tuple[type, ...]] | None = None,
        name: str = "payload",
    ) -> None:
        """Minimal deterministic schema validation for request intake (L0).

        ``required``/``optional`` map field names to accepted Python types.
        Raises :class:`SchemaValidationError` on any violation.
        """
        if not isinstance(payload, dict):
            raise SchemaValidationError(f"{name}: expected object, got {type(payload).__name__}")
        optional = optional or {}
        allowed = set(required) | set(optional)
        unknown = set(payload) - allowed
        if unknown:
            raise SchemaValidationError(f"{name}: unknown fields {sorted(unknown)}")
        for field_name, types in required.items():
            if field_name not in payload:
                raise SchemaValidationError(f"{name}: missing required field {field_name!r}")
            if not isinstance(payload[field_name], types):
                raise SchemaValidationError(
                    f"{name}: field {field_name!r} has type "
                    f"{type(payload[field_name]).__name__}, expected {types}"
                )
        for field_name, types in optional.items():
            if field_name in payload and not isinstance(payload[field_name], types):
                raise SchemaValidationError(
                    f"{name}: optional field {field_name!r} has type "
                    f"{type(payload[field_name]).__name__}, expected {types}"
                )


def ct_equal(a: str, b: str) -> bool:
    """Constant-time equality for hex digests / MACs."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
