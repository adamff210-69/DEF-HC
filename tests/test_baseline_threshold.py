"""Regression tests for baseline threshold selection.

Same defect class as ``tests/test_policy_feasibility.py``, in the other
place it mattered.  ``run_baselines.py`` picked its operating threshold at
``recall@<target>``.  When the target exceeds what a detector can deliver on
the corpus, the only way to reach it is to drop the threshold to the bottom
of the score range -- "achieving" the recall by flagging every input.  The
comparison table then reported recall 1.0 / FPR 1.0 for every system, which
makes all detectors look identical and hides real differences that ROC-AUC
still shows.
"""
from __future__ import annotations

import random

import pytest

from scripts.run_baselines import metrics_at, select_threshold


def _scores(seed: int = 0):
    """A detector with genuine signal on one class and none on another."""
    rng = random.Random(seed)
    y: list[int] = []
    s: list[float] = []
    for _ in range(200):                       # benign, low scores
        y.append(0)
        s.append(rng.betavariate(2, 8) * 0.5)
    for _ in range(100):                       # in-domain, high scores
        y.append(1)
        s.append(0.5 + rng.betavariate(5, 2) * 0.5)
    for _ in range(100):                       # out-of-domain, indistinct
        y.append(1)
        s.append(rng.betavariate(2, 8) * 0.5)
    return y, s


def test_unreachable_recall_target_is_marked_degenerate():
    y, s = _scores()
    sel = select_threshold(y, s, target_recall=0.95)
    assert sel["feasible"] is False
    assert "DEGENERATE" in sel["note"]


def test_degenerate_note_points_at_the_threshold_free_metric():
    """A reader must be told what is still comparable."""
    y, s = _scores()
    sel = select_threshold(y, s, target_recall=0.95)
    assert "ROC-AUC" in sel["note"]
    assert "--fpr-budget" in sel["note"]


def test_reachable_recall_target_is_feasible():
    y, s = _scores()
    sel = select_threshold(y, s, target_recall=0.40)
    assert sel["feasible"] is True
    assert "DEGENERATE" not in sel["note"]


@pytest.mark.parametrize("budget", [0.0, 0.01, 0.05, 0.10, 0.50])
def test_fpr_budget_is_respected_when_feasible(budget):
    y, s = _scores()
    sel = select_threshold(y, s, fpr_budget=budget)
    if sel["feasible"]:
        m = metrics_at(y, s, sel["threshold"])
        assert (m["benign_fpr"] or 0) <= budget + 1e-9


def test_fpr_budget_never_flags_everything():
    y, s = _scores()
    sel = select_threshold(y, s, fpr_budget=0.05)
    m = metrics_at(y, s, sel["threshold"])
    assert (m["benign_fpr"] or 0) < 1.0
    assert (m["recall"] or 0) < 1.0          # honest, not a shrug


def test_recall_target_and_budget_give_different_operating_points():
    y, s = _scores()
    a = select_threshold(y, s, target_recall=0.95)["threshold"]
    b = select_threshold(y, s, fpr_budget=0.01)["threshold"]
    assert a != b


def test_every_result_carries_the_contract():
    y, s = _scores()
    for sel in (select_threshold(y, s, target_recall=0.95),
                select_threshold(y, s, target_recall=0.10),
                select_threshold(y, s, fpr_budget=0.05),
                select_threshold(y, s, fpr_budget=0.0)):
        assert set(sel) >= {"threshold", "objective", "feasible", "note",
                            "metrics"}
        assert isinstance(sel["feasible"], bool)
