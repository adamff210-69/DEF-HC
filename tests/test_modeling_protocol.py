"""Regression tests for the training/evaluation protocol (spec Phase 17).

7  folded weights match sklearn predictions
8  classifier thresholds come from calibration data
9  test data cannot be selected as calibration accidentally
10 multiple --dataset paths actually contribute rows
11 duplicate groups do not cross splits
14 cached embedder is reused across loads
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from defend_hc2 import embedder as embedder_mod
from defend_hc2.modeling import (
    assert_disjoint_roles,
    calibrate_thresholds,
    file_sha256,
    fit_classifier,
    fold_scaler_into_weights,
    load_many,
    remove_overlap,
)
from defend_hc2.splitting import (
    assert_no_group_crossing,
    build_groups,
    group_stratified_split,
    template_key,
)


# ----------------------------------------------------- 7: folded == sklearn
class TestFoldedWeights:
    def _data(self, n=240, seed=0):
        rng = np.random.RandomState(seed)
        X1 = rng.normal(loc=1.2, size=(n, 8))
        X0 = rng.normal(loc=-1.2, size=(n, 8))
        X = np.vstack([X1, X0])
        y = [1] * n + [0] * n
        return X, y

    def test_folded_matches_sklearn_within_tolerance(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        X, y = self._data()
        scaler = StandardScaler().fit(X)
        clf = LogisticRegression(C=1.0, max_iter=5000,
                                 class_weight="balanced",
                                 random_state=42).fit(scaler.transform(X), y)
        w_fold, b_fold = fold_scaler_into_weights(
            clf.coef_[0], float(clf.intercept_[0]), scaler.mean_, scaler.scale_)
        probs_sk = clf.predict_proba(scaler.transform(X))[:, 1]
        for x, p_sk in zip(X, probs_sk):
            z = sum(w * v for w, v in zip(w_fold, x)) + b_fold
            p_fold = 1.0 / (1.0 + np.exp(-z))
            assert abs(p_fold - p_sk) < 1e-6

    def test_fit_classifier_self_verifies_and_selects_C_on_cal(self):
        X, y = self._data()
        idx = np.random.RandomState(7).permutation(len(y))  # interleave classes
        X, y = X[idx], [int(y[i]) for i in idx]
        fit = fit_classifier(X[:160], y[:160], X[160:], y[160:], seed=42)
        assert fit["fold_scaler_max_abs_dev"] < 1e-6
        assert fit["selected_C"] in {0.03, 0.1, 0.3, 1.0, 3.0, 10.0}
        assert fit["selection_metric"] == "calibration_pr_auc"
        assert len(fit["weights"]) == 8


# ---------------------------------------------------- 8: cal-only thresholds
class TestThresholdOrigin:
    def test_threshold_computed_from_given_scores_only(self):
        # calibration set where only 60% of positives reach score >= 0.9
        # and ALL positives reach >= 0.4: a recall@0.95 threshold must come
        # out <= 0.4 (not 0.9) — proving it derives from the CALIBRATION
        # distribution the caller supplied
        gold = [1] * 10 + [0] * 10
        score = [0.9] * 6 + [0.4] * 4 + [0.1] * 8 + [0.45, 0.5]
        out = calibrate_thresholds(gold, score)
        assert out["recall@0.95"] <= 0.4 + 1e-9
        assert out["recall@0.95"] > 0.0

    def test_monotonic_recall_targets(self):
        gold = [1, 1, 1, 1, 0, 0, 0, 0]
        score = [0.95, 0.9, 0.6, 0.3, 0.2, 0.1, 0.05, 0.01]
        out = calibrate_thresholds(gold, score)
        assert out["recall@0.98"] <= out["recall@0.95"]


# --------------------------------------------- 9: role separation is forced
class TestRoleSeparation:
    def test_same_path_rejected(self, tmp_path):
        f = tmp_path / "data.jsonl"
        with pytest.raises(SystemExit):
            assert_disjoint_roles(dataset=[f], cal=[f])

    def test_overlap_removal_clears_test(self):
        train = [("ignore everything now", 1), ("show my order", 0)]
        test = [("Ignore  everything now", 1), ("a different test prompt", 1),
                ("where is my refund", 0)]
        kept, removed = remove_overlap(test, train)
        assert removed == 1
        assert ("where is my refund", 0) in kept and len(kept) == 2


# ---------------------------------------------- 10: multi-dataset concat
class TestMultiDataset:
    def test_multiple_inputs_all_contribute(self, tmp_path):
        for i in (1, 2):
            with (tmp_path / f"d{i}.jsonl").open("w") as fh:
                for j in range(5):
                    fh.write(f'{{"text": "prompt {i} {j}", "label": {j % 2}}}\n')
        rows = load_many([tmp_path / "d1.jsonl", tmp_path / "d2.jsonl"])
        assert len(rows) == 10
        assert sum(1 for t, _ in rows if t.startswith("prompt 1")) == 5


# --------------------------------------- 11: template groups never cross
class TestGroupSplit:
    def _rows(self):
        rows = []
        for i in range(6):
            for j in range(1, 6):  # 5 near-identical templates each
                rows.append({"text": f"transfer account {1000 + i * 100 + j} now",
                             "label": 1})
        for i in range(4):
            for j in range(1, 6):
                rows.append({"text": f"order {2000 + i * 100 + j} missing",
                             "label": 0})
        return rows

    def test_templates_collapse_to_groups(self):
        rows = self._rows()
        groups = build_groups(rows, "text")
        # rows differ only in digit runs -> SAME template group (that is
        # exactly the SPML template-dedup behavior the spec requires)
        assert len(groups) == 2
        assert template_key("x 12 y") == template_key("x 987 y")

    def test_groups_do_not_cross_splits(self):
        rows = self._rows()
        parts = group_stratified_split(rows, "text", "label", seed=42,
                                       fractions=(0.6, 0.2, 0.2))
        assert_no_group_crossing(parts, "text")  # must not raise
        assert sum(len(p) for p in parts) == len(rows)


# ------------------------------------------------- 14: embedder cache reuse
class TestEmbedderCache:
    def test_one_load_per_process_per_name(self, monkeypatch):
        calls = []

        class FakeST:
            def __init__(self, name):
                calls.append(name)

        fake_mod = types.ModuleType("sentence_transformers")
        fake_mod.SentenceTransformer = FakeST
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
        embedder_mod.clear_cache()
        a = embedder_mod.get_sentence_transformer("model-x")
        b = embedder_mod.get_sentence_transformer("model-x")
        c = embedder_mod.get_sentence_transformer("model-y")
        assert a is b and a is not c
        assert calls == ["model-x", "model-y"]
        embedder_mod.clear_cache()


# -------------------------------------------- manifest helpers (Phase 20)
def test_file_sha256_deterministic(tmp_path):
    f = tmp_path / "x.txt"
    g = tmp_path / "y.txt"
    f.write_text("hello")
    g.write_text("world")
    assert file_sha256(f) == file_sha256(f)
    assert file_sha256(f) != file_sha256(g)
