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


#: Markers that mean "the objective was not met" — either the constraint is
#: unreachable, or it is reachable only at an FPR that makes it meaningless.
UNMET = ("INFEASIBLE", "DEGENERATE", "UNMET")


def test_unreachable_recall_target_is_reported_infeasible():
    risks, gold, _ = _corpus()
    sel = select_policy(risks, gold, target_recall=0.95)
    assert sel["feasible"] is False
    assert any(k in sel["note"] for k in UNMET)


def test_infeasible_result_states_what_was_attainable():
    """The note must be actionable, not just negative: it has to say what
    the cost of the request was, and how to ask for something reachable."""
    risks, gold, _ = _corpus()
    sel = select_policy(risks, gold, target_recall=0.95)
    assert any(k in sel["note"] for k in UNMET)
    assert "fpr-budget" in sel["note"]
    # the actual price of the unreachable request is quoted
    assert "benign" in sel["note"]


def test_feasible_target_is_marked_feasible():
    risks, gold, cats = _corpus()
    keep = [i for i, c in enumerate(cats) if c != "harmful-content"]
    sel = select_policy([risks[i] for i in keep], [gold[i] for i in keep],
                        target_recall=0.95)
    assert sel["feasible"] is True
    assert not any(k in sel["note"] for k in UNMET)


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
    # The grid can genuinely reach 0.0 benign FPR, so a negative budget is
    # used here to exercise the unreachable branch.
    sel = select_policy(risks, gold, fpr_budget=-1.0)
    assert sel["feasible"] is False
    every = select_policy(risks, gold, target_recall=0.95)   # recall-first
    assert sel["metrics"]["benign_fpr"] < every["metrics"]["benign_fpr"]


def test_infeasible_result_reports_what_is_achievable():
    """'No' is not an actionable answer: the caller needs the frontier."""
    risks, gold, _ = _corpus()
    for sel in (select_policy(risks, gold, fpr_budget=-1.0),
                select_policy(risks, gold, target_recall=0.95)):
        front = sel.get("achievable_frontier")
        assert front, "infeasible result must carry an achievable frontier"
        assert all({"bands", "benign_fpr", "recall", "precision"} <= set(f)
                   for f in front)
        # sorted by ascending benign FPR so the first row is the safest point
        fprs = [f["benign_fpr"] for f in front]
        assert fprs == sorted(fprs)


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


def test_candidate_bands_are_not_capped_by_a_hardcoded_ladder():
    """The bug this locks down bit twice.

    With a fixed sanitize ladder the whole achievable frontier landed on the
    ladder's own top value, which means the real optimum was above it and
    was never searched.  Candidates must come from the score distribution,
    so the reachable FPR floor is a property of the model, not of a
    constant in this file.
    """
    from scripts.calibrate_policy import sweep_grid, sweep_grid_from_scores

    risks, gold, _ = _corpus()
    fixed_max = max(b[0] for b in sweep_grid())
    derived = sweep_grid_from_scores(risks)
    assert max(b[0] for b in derived) > fixed_max
    assert all(s < q < r for s, q, r in derived)


def test_tight_budget_reachable_when_the_scores_support_it():
    """A budget must fail because the model cannot meet it, never because
    the search space stopped short."""
    risks, gold, cats = _corpus()
    keep = [i for i, c in enumerate(cats) if c != "harmful-content"]
    sel = select_policy([risks[i] for i in keep], [gold[i] for i in keep],
                        fpr_budget=0.01)
    assert sel["feasible"] is True
    assert sel["metrics"]["benign_fpr"] <= 0.01 + 1e-9


def test_frontier_spans_the_tradeoff_not_just_the_safe_corner():
    """Ten variations of 'detect almost nothing' is not a useful answer."""
    risks, gold, _ = _corpus()
    front = select_policy(risks, gold, fpr_budget=-1.0)["achievable_frontier"]
    recalls = [f["recall"] for f in front]
    assert max(recalls) - min(recalls) > 0.3, "frontier must show a trade-off"
    fprs = [f["benign_fpr"] for f in front]
    assert fprs == sorted(fprs)
