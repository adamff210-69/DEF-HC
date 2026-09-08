"""Layer 0 tests: canonicalization, framing, schema validation."""

from __future__ import annotations

import unicodedata

import pytest

from defend_hc2.canonicalization import Canonicalizer
from defend_hc2.exceptions import SchemaValidationError


class TestNormalizeText:
    def test_nfkc_fullwidth(self):
        # fullwidth Latin letters fold to ASCII under NFKC
        assert Canonicalizer.normalize_text("ＨＥＬＬＯ") == "HELLO"

    def test_nfkc_compatibility_ligature(self):
        assert unicodedata.normalize("NFKC", "ﬁle") == "file"
        assert Canonicalizer.normalize_text("ﬁle") == "file"

    def test_control_characters_removed(self):
        assert Canonicalizer.normalize_text("a\x07b\x1bc") == "abc"

    def test_whitespace_controls_preserved(self):
        assert Canonicalizer.normalize_text("a\tb\nc\r") == "a\tb\nc\r"

    def test_zero_width_removed(self):
        assert Canonicalizer.normalize_text("he​llo") == "hello"

    def test_bidi_override_removed(self):
        assert Canonicalizer.normalize_text("abc‮def‬") == "abcdef"

    def test_nbsp_becomes_space(self):
        assert Canonicalizer.normalize_text("a b") == "a b"

    def test_non_string_rejected(self):
        with pytest.raises(SchemaValidationError):
            Canonicalizer.normalize_text(123)  # type: ignore[arg-type]


class TestDeterministicJson:
    def test_key_ordering_and_separators(self):
        obj1 = {"b": 1, "a": [3, 2], "é": "x"}
        obj2 = {"é": "x", "a": [3, 2], "b": 1}
        assert Canonicalizer.canonical_json(obj1) == Canonicalizer.canonical_json(obj2)
        assert Canonicalizer.canonical_json(obj1) == '{"a":[3,2],"b":1,"é":"x"}'

    def test_no_ascii_escaping(self):
        assert "é" in Canonicalizer.canonical_json({"k": "é"})

    def test_canonical_bytes_utf8(self):
        assert Canonicalizer.canonical_bytes({"k": "é"}) == '{"k":"é"}'.encode("utf-8")

    def test_nested_strings_normalized_inside_bytes(self):
        b1 = Canonicalizer.canonical_bytes({"t": "Ａ"})   # fullwidth A
        b2 = Canonicalizer.canonical_bytes({"t": "A"})
        assert b1 == b2  # normalization happens before hashing


class TestHashingAndFraming:
    def test_sha3_256_length(self):
        assert len(Canonicalizer.sha3_256(b"x")) == 32

    def test_domain_separation_changes_digest(self):
        a = Canonicalizer.sha3_256(b"data", tag=b"DEFEND-HC2-PAYLOAD")
        b = Canonicalizer.sha3_256(b"data", tag=b"DEFEND-HC2-EVENT")
        assert a != b

    def test_framing_is_unambiguous(self):
        # ("ab","c") must differ from ("a","bc") — naive concat would collide
        a = Canonicalizer.sha3_256(b"ab", b"c")
        b = Canonicalizer.sha3_256(b"a", b"bc")
        assert a != b

    def test_hmac_matches_hashlib_reference(self):
        import hashlib
        import hmac as py_hmac

        key, msg = b"k" * 32, b"payload"
        ours = Canonicalizer.hmac_sha3_256(key, msg, tag=b"T")
        ref = py_hmac.new(
            key, Canonicalizer.frame([b"T", msg]), hashlib.sha3_256
        ).digest()
        assert ours == ref

    def test_payload_hash_stable_across_key_order(self):
        h1 = Canonicalizer.payload_hash({"a": 1, "b": {"y": 2, "x": 3}})
        h2 = Canonicalizer.payload_hash({"b": {"x": 3, "y": 2}, "a": 1})
        assert h1 == h2


class TestSchemaValidation:
    def test_required_and_types(self):
        Canonicalizer.validate_schema(
            {"s": "x", "i": 3}, {"s": str, "i": int}
        )

    def test_missing_field(self):
        with pytest.raises(SchemaValidationError):
            Canonicalizer.validate_schema({"s": "x"}, {"s": str, "i": int})

    def test_wrong_type(self):
        with pytest.raises(SchemaValidationError):
            Canonicalizer.validate_schema({"s": 1}, {"s": str})

    def test_unknown_field_rejected(self):
        with pytest.raises(SchemaValidationError):
            Canonicalizer.validate_schema({"s": "x", "evil": 1}, {"s": str})

    def test_optional_fields(self):
        Canonicalizer.validate_schema(
            {"s": "x", "n": "abc"}, {"s": str}, {"n": str}
        )
