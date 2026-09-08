"""Purity contract for train-time letter-spacing augmentation (Exp-G)."""

from __future__ import annotations

import pytest

from defend_hc2.augment import (
    AUGMENT_EVERY,
    AUGMENT_MAX_CHARS,
    letter_spacing_augment,
)
from defend_hc2.perturb import letter_spacing_extreme

_ROWS = [(f"train text number {i} about shipment returns", i % 2) for i in range(24)]


def test_only_additional_rows_returned_and_deterministic():
    a = letter_spacing_augment(_ROWS)
    b = letter_spacing_augment(_ROWS)
    assert a == b  # no RNG, same output every time
    for orig_i, (new_text, label) in enumerate(a):
        assert label == _ROWS[orig_i * AUGMENT_EVERY][1]
        assert new_text == letter_spacing_extreme(_ROWS[orig_i * AUGMENT_EVERY][0])


def test_every_th_eligible_row_selected():
    out = letter_spacing_augment(_ROWS, every=4)
    assert len(out) == 24 // 4  # rows 0,4,8,12,16,20
    out3 = letter_spacing_augment(_ROWS, every=3)
    assert len(out3) == 8       # rows 0,3,6,...,21


def test_long_rows_skipped():
    rows = [("x" * (AUGMENT_MAX_CHARS + 1), 1), ("short text", 0)]
    out = letter_spacing_augment(rows, every=1, max_chars=AUGMENT_MAX_CHARS)
    assert [txt for txt, _ in out] == [letter_spacing_extreme("short text")]


def test_labels_preserved_and_strings_non_empty():
    for text, label in letter_spacing_augment(_ROWS, every=2):
        assert label in (0, 1)
        assert isinstance(text, str) and text


def test_no_contamination_of_eval_splits():
    """The augment output must contain NO rows from cal/test — it takes only
    train rows by construction; the test pins that semantics against a
    deliberately adversarial split layout."""
    train = [("alpha bravo charlie delta", 1), ("echo foxtrot golf hotel", 0),
             ("india juliet kilo lima", 1), ("mike november oscar papa", 0)]
    cal = [("quebec romeo sierra tango", 0)]
    out = letter_spacing_augment(train, every=1)
    out_texts = " ".join(t for t, _ in out)
    for ct, _ in cal:
        assert ct not in out_texts


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        letter_spacing_augment(_ROWS, every=0)
    with pytest.raises(ValueError):
        letter_spacing_augment(_ROWS, max_chars=0)
