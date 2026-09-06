"""Regression tests for policy-band selection under an unmet objective.

The defect these lock down: when the recall target was unreachable,
``select_policy`` fell back to *maximizing recall*, which walks straight to
the most aggressive bands on the grid.  That produced a flag-everything
operating point with a huge benign FPR, reported it with the same wording as
a genuine solution, and — because every infeasible target lands on the same
fallback — made two different targets emit byte-identical policies.
Everything measured downstream inherited it.
"""
from __future__ import annotations

import random

import pytest

from scripts.calibrate_policy import select_policy


def _corpus(seed: int = 0):
    """Separable in-domain attacks + an out-of-domain class the scorer is
    blind to (its scores sit inside the benign distribution)."""
    rng = random.Random(seed)
    risks: list[float] = []
    gold: list[int] = []
    cats: list[str] = []
    for _ in range(300):
        risks.append(rng.betavariate(2, 8) * 0.6)
        gold.append(0)
        cats.append("benign")
    for _ in range(150):
        risks.append(0.45 + rng.betavariate(5, 2) * 0.55)
        gold.append(1)
        cats.append("injection")
    for _ in range(150):
        risks.append(rng.betavariate(2, 8) * 0.6)
        gold.append(1)
        cats.append("harmful-content")
    return risks, gold, cats


def test_unreachable_recall_target_is_reported_infeasible():
    risks, gold, _ = _corpus()
    sel = select_policy(risks, gold, target_recall=0.95)
    assert sel["feasible"] is False
    assert "INFEASIBLE" in sel["note"]


def test_infeasible_result_states_what_was_attainable():
    """The note must be actionable, not just negative."""
    risks, gold, _ = _corpus()
    sel = select_policy(risks, gold, target_recall=0.95)
    assert str(sel["metrics"]["recall"]) in sel["note"]
    assert "fpr-budget" in sel["note"]


def test_feasible_target_is_marked_feasible():
    risks, gold, cats = _corpus()
    keep = [i for i, c in enumerate(cats) if c != "harmful-content"]
    sel = select_policy([risks[i] for i in keep], [gold[i] for i in keep],
                        target_recall=0.95)
    assert sel["feasible"] is True
    assert "INFEASIBLE" not in sel["note"]


def test_dropping_out_of_domain_class_restores_a_usable_operating_point():
    """The scorer is fine on its own domain; mixing in a class it cannot see
    is what destroys the operating point for every other class."""
    risks, gold, cats = _corpus()
    mixed = select_policy(risks, gold, target_recall=0.95)
    keep = [i for i, c in enumerate(cats) if c != "harmful-content"]
    scoped = select_policy([risks[i] for i in keep], [gold[i] for i in keep],
                           target_recall=0.95)
    assert scoped["metrics"]["benign_fpr"] < mixed["metrics"]["benign_fpr"]
    assert scoped["feasible"] and not mixed["feasible"]


@pytest.mark.parametrize("budget", [0.01, 0.05, 0.10, 0.25])
def test_fpr_budget_is_respected_whenever_feasible(budget):
    risks, gold, _ = _corpus()
    sel = select_policy(risks, gold, fpr_budget=budget)
    if sel["feasible"]:
        assert sel["metrics"]["benign_fpr"] <= budget + 1e-9


def test_fpr_budget_never_selects_a_flag_everything_policy():
    risks, gold, _ = _corpus()
    sel = select_policy(risks, gold, fpr_budget=0.05)
    assert sel["metrics"]["benign_fpr"] <= 0.05 + 1e-9
    # the old recall-first fallback landed around .31 benign FPR here
    assert sel["metrics"]["benign_fpr"] < 0.30


def test_impossible_fpr_budget_falls_back_to_lowest_fpr_not_highest_recall():
    risks, gold, _ = _corpus()
    sel = select_policy(risks, gold, fpr_budget=0.0)
    assert sel["feasible"] is False
    every = select_policy(risks, gold, target_recall=0.95)   # recall-first
    assert sel["metrics"]["benign_fpr"] < every["metrics"]["benign_fpr"]


def test_distinct_budgets_do_not_all_collapse_to_one_policy():
    """Two 'different' policies that are the same operating point compare
    nothing.  Over a spread of budgets we must see more than one point."""
    risks, gold, _ = _corpus()
    bands = {select_policy(risks, gold, fpr_budget=b)["bands"]
             for b in (0.01, 0.05, 0.10, 0.25, 0.50)}
    assert len(bands) > 1


def test_every_result_carries_the_feasibility_contract():
    risks, gold, _ = _corpus()
    for sel in (select_policy(risks, gold, target_recall=0.95),
                select_policy(risks, gold, target_recall=0.10),
                select_policy(risks, gold, fpr_budget=0.05),
                select_policy(risks, gold, fpr_budget=0.0)):
        assert set(sel) >= {"bands", "metrics", "feasible", "objective", "note"}
        assert isinstance(sel["feasible"], bool)
        assert len(sel["bands"]) == 3
        assert sel["bands"][0] < sel["bands"][1] < sel["bands"][2]
