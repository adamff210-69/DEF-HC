"""Shared ML-training and evaluation core (spec Phases 3, 5, 11–13, 20).

Everything here is importable by the scripts and unit-testable without a
network connection:

* :func:`fit_classifier` — sklearn ``StandardScaler`` + ``LogisticRegression``
  with ``class_weight="balanced"``, ``max_iter >= 5000``; ``C`` selected on
  **calibration PR-AUC** (never test); scaler folded back into raw-embedding
  weights at the end so the runtime needs only ``(weights, bias)``;
* :func:`calibrate_thresholds` — target-recall 0.95/0.98 and max-F1
  thresholds computed on calibration data only;
* full metric set (Phase 13) + bootstrap 95% CIs (>=1000 resamples, seed 42);
* IO + reproducibility helpers (Phase 20).

sklearn/numpy are imported lazily so the core library stays dependency-free.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

DEFAULT_CS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


# ==================================================================== data
def load_jsonl(path: Path) -> list[tuple[str, int]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows.append((str(row["text"]), int(row["label"])))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return rows


def load_many(paths: Iterable[Path], role: str = "") -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for path in paths:
        rows = load_jsonl(Path(path))
        pos = sum(y for _, y in rows)
        print(f"  {role + ' ' if role else ''}{path}: {len(rows)} rows "
              f"({pos} injection / {len(rows) - pos} benign)")
        out.extend(rows)
    return out


def norm_for_dedup(text: str) -> str:
    """Cheap duplicate key: lowercased, whitespace-collapsed."""
    return " ".join(text.split()).casefold()


def exact_duplicate_count(rows: Sequence[tuple[str, int]]) -> int:
    seen: set[str] = set()
    dups = 0
    for text, _ in rows:
        key = norm_for_dedup(text)
        if key in seen:
            dups += 1
        seen.add(key)
    return dups


def remove_overlap(
    rows: list[tuple[str, int]],
    *reference: list[tuple[str, int]],
) -> tuple[list[tuple[str, int]], int]:
    """Drop eval rows whose normalized text already appears in reference
    (train/cal) corpora.  Returns (cleaned_rows, removed_count)."""
    ref_keys = {norm_for_dedup(t) for ref in reference for t, _ in ref}
    kept = [r for r in rows if norm_for_dedup(r[0]) not in ref_keys]
    return kept, len(rows) - len(kept)


def assert_disjoint_roles(**roles: list[Path]) -> None:
    """A path used for one role must not appear in an incompatible role
    (Phase 11 guard)."""
    seen: dict[str, str] = {}
    for role, paths in roles.items():
        for p in paths or []:
            key = str(Path(p).resolve())
            if key in seen and seen[key] != role:
                raise SystemExit(
                    f"path-role violation: {p} used as both "
                    f"{seen[key]!r} and {role!r}"
                )
            seen[key] = role


# ================================================================== training
def fold_scaler_into_weights(
    coef: Sequence[float], intercept: float,
    mean: Sequence[float], scale: Sequence[float],
) -> tuple[list[float], float]:
    """Fold ``StandardScaler`` (z = (x - mean)/scale) into raw-space weights.

    sklearn:          p = sigma(beta . z + intercept)
    raw equivalent:   p = sigma(w_fold . x + b_fold), where
        w_fold = beta / scale
        b_fold = intercept - sum(beta * mean / scale)
    (spec Phase 3 exact formulas.)
    """
    w_fold = [float(b) / (float(s) if float(s) != 0.0 else 1.0) for b, s in zip(coef, scale)]
    b_fold = float(intercept) - sum(
        float(b) * float(m) / (float(s) if float(s) != 0.0 else 1.0)
        for b, m, s in zip(coef, mean, scale)
    )
    return w_fold, b_fold


def fit_classifier(
    X_train, y_train,
    X_cal, y_cal,
    seed: int = 42,
    class_balance: bool = True,
    Cs: Sequence[float] = DEFAULT_CS,
    verbose: bool = False,
) -> dict[str, Any]:
    """Train + select C on **calibration PR-AUC**; refit nothing on test.

    Returns a weights-compatible dict (folded raw-space weights) with full
    selection metadata.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import StandardScaler

    X_train = np.asarray(X_train, dtype=float)
    X_cal = np.asarray(X_cal, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    y_cal = np.asarray(y_cal, dtype=float)

    selection = []
    best: dict[str, Any] | None = None
    for C in Cs:
        scaler = StandardScaler().fit(X_train)
        Ztr = scaler.transform(X_train)
        clf = LogisticRegression(
            C=float(C), solver="liblinear", max_iter=5000,
            class_weight="balanced" if class_balance else None,
            random_state=seed,
        ).fit(Ztr, y_train)
        Zcal = scaler.transform(X_cal)
        cal_prob = clf.predict_proba(Zcal)[:, 1]
        pr_auc = float(average_precision_score(y_cal, cal_prob))
        selection.append({"C": float(C), "cal_pr_auc": round(pr_auc, 6)})
        if verbose:
            print(f"    C={C:<6} calibration PR-AUC={pr_auc:.4f}")
        if best is None or pr_auc > best["pr_auc"]:
            best = {"C": float(C), "pr_auc": pr_auc,
                    "scaler": scaler, "clf": clf}

    assert best is not None
    coef = best["clf"].coef_[0]
    intercept = float(best["clf"].intercept_[0])
    w_fold, b_fold = fold_scaler_into_weights(
        coef, intercept, best["scaler"].mean_, best["scaler"].scale_
    )

    # --- numerically verify sklearn-vs-folded equivalence on the cal set
    import math

    def folded_prob(x) -> float:
        z = sum(w * v for w, v in zip(w_fold, x)) + b_fold
        return 1.0 / (1.0 + math.exp(-z))

    Zcal = best["scaler"].transform(X_cal)
    sk_probs = best["clf"].predict_proba(Zcal)[:, 1]
    max_dev = max(
        abs(float(sp) - folded_prob(list(x))) for sp, x in zip(sk_probs, X_cal)
    )
    if max_dev > 1e-6:
        raise SystemExit(f"folded-weights verification FAILED (max dev {max_dev:.2e})")

    return {
        "weights": w_fold,
        "bias": b_fold,
        "dims": len(w_fold),
        "selected_C": best["C"],
        "selection_metric": "calibration_pr_auc",
        "selected_C_cal_pr_auc": round(best["pr_auc"], 6),
        "C_sweep": selection,
        "fold_scaler_max_abs_dev": float(max_dev),
        "class_balance": bool(class_balance),
        "seed": seed,
        "estimator": "sklearn LogisticRegression(liblinear, max_iter=5000) "
                     "+ StandardScaler (folded into weights)",
    }


# ================================================================ calibration
def calibrate_thresholds(
    cal_gold: Sequence[int],
    cal_score: Sequence[float],
    recall_targets: Sequence[float] = (0.95, 0.98),
) -> dict[str, float]:
    """All threshold candidates from **calibration data only** (Phase 5).

    Returns ``recall@<target>`` thresholds plus the max-F1 calibration
    threshold; the caller picks the deployment criterion in ADVANCE and
    records all three.
    """
    total_pos = max(1, sum(int(g) for g in cal_gold))
    out: dict[str, float] = {}
    for target in recall_targets:
        best = 0.0
        for t in sorted(set(cal_score)):
            tp = sum(g and s >= t for g, s in zip(cal_gold, cal_score))
            if tp / total_pos >= target - 1e-9:
                best = t
            else:
                break  # recall is non-increasing in t
        out[f"recall@{target}"] = float(best)
    # max calibration F1
    best_t, best_f1 = 0.0, -1.0
    for t in sorted(set(cal_score)):
        m = binary_metrics(cal_gold, cal_score, t)
        if m["f1"] > best_f1:
            best_t, best_f1 = float(t), m["f1"]
    out["max_f1_on_calibration"] = best_t
    return out


# =================================================================== metrics
def binary_metrics(gold: Sequence[int], score: Sequence[float], threshold: float) -> dict[str, float]:
    pred = [s >= threshold for s in score]
    tp = sum(1 for g, p in zip(gold, pred) if g and p)
    fp = sum(1 for g, p in zip(gold, pred) if not g and p)
    fn = sum(1 for g, p in zip(gold, pred) if g and not p)
    tn = sum(1 for g, p in zip(gold, pred) if not g and not p)
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)
    prec = tp / max(1, tp + fp)
    rec = tpr
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    f2 = 5 * prec * rec / max(1e-9, 4 * prec + rec)
    bal = (tpr + tnr) / 2
    acc = (tp + tn) / max(1, len(gold))
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0
    return {
        "threshold": round(float(threshold), 6),
        "accuracy": round(acc, 4), "balanced_accuracy": round(bal, 4),
        "precision": round(prec, 4), "recall": round(rec, 4),
        "specificity": round(tnr, 4), "fpr": round(fpr, 4), "fnr": round(fnr, 4),
        "f1": round(f1, 4), "f2": round(f2, 4), "mcc": round(mcc, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def roc_auc(gold: Sequence[int], score: Sequence[float]) -> float | None:
    if sum(gold) in (0, len(gold)):
        return None
    from sklearn.metrics import roc_auc_score

    return round(float(roc_auc_score(gold, score)), 4)


def pr_auc(gold: Sequence[int], score: Sequence[float]) -> float | None:
    if sum(gold) in (0, len(gold)):
        return None
    from sklearn.metrics import average_precision_score

    return round(float(average_precision_score(gold, score)), 4)


def fpr_at_tpr(gold: Sequence[int], score: Sequence[float], tpr_target: float) -> float | None:
    """FPR at the first operating point reaching ``tpr_target`` TPR."""
    if sum(gold) in (0, len(gold)):
        return None
    from sklearn.metrics import roc_curve

    fpr, tpr, _thr = roc_curve(gold, score)
    eligible = [f for f, t in zip(fpr, tpr) if t >= tpr_target]
    return round(min(eligible), 4) if eligible else None


def full_metric_report(
    gold: Sequence[int], score: Sequence[float], threshold: float
) -> dict[str, Any]:
    """Phase 13 complete row."""
    out: dict[str, Any] = binary_metrics(gold, score, threshold)
    out["roc_auc"] = roc_auc(gold, score)
    out["pr_auc"] = pr_auc(gold, score)
    out["fpr@95tpr"] = fpr_at_tpr(gold, score, 0.95)
    out["fpr@98tpr"] = fpr_at_tpr(gold, score, 0.98)
    out["base_rate"] = round(sum(gold) / max(1, len(gold)), 4)
    return out


_CI_METRICS = ("roc_auc", "pr_auc", "recall", "precision", "f1", "balanced_accuracy")


def bootstrap_cis(
    gold: Sequence[int],
    score: Sequence[float],
    threshold: float,
    resamples: int = 1000,
    seed: int = 42,
) -> dict[str, list[float | None]]:
    """95% bootstrap percentile CIs (>=1000 resamples, seeded, safe on
    single-class resamples)."""
    import random

    rng = random.Random(seed)
    n = len(gold)
    collected: dict[str, list[float]] = {k: [] for k in _CI_METRICS}
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        g = [gold[i] for i in idx]
        s = [score[i] for i in idx]
        if sum(g) in (0, len(g)):  # skip single-class resample safely
            continue
        m = binary_metrics(g, s, threshold)
        collected["roc_auc"].append(roc_auc(g, s))
        collected["pr_auc"].append(pr_auc(g, s))
        for k in ("recall", "precision", "f1", "balanced_accuracy"):
            collected[k].append(m[k])

    def pct(values: list[float], q: float) -> float | None:
        if not values:
            return None
        values = sorted(values)
        k = min(len(values) - 1, max(0, int(round(q / 100 * (len(values) - 1)))))
        return round(values[k], 4)

    return {
        k: [pct(v, 2.5), pct(v, 97.5)] if v else None
        for k, v in collected.items()
    }


# =========================================================== reproducibility
def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo: Path | str = ".") -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=str(repo), timeout=10,
        ).stdout.strip() or None
    except Exception:
        return None


def environment_block() -> dict[str, Any]:

    def _ver(name: str) -> str | None:
        try:
            mod = __import__(name)
            return getattr(mod, "__version__", None)
        except Exception:
            return None

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {n: _ver(n) for n in
                     ("numpy", "sklearn", "sentence_transformers", "torch")},
        "machine": os.uname().machine if hasattr(os, "uname") else None,
    }


def sha256_of_json(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        .encode("utf-8")
    ).hexdigest()
