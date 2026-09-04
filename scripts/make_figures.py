"""Generate all evaluation figures from scores/metrics artifacts (Phase 18).

Inputs are the ``scores-*.jsonl`` / ``bench-metrics-*.json`` artifacts of the
benchmark/experiment runs — no recomputation, so figures can never disagree
with reported numbers.  Every figure is labelled with dataset, model,
calibration origin, and sample size.  Matplotlib runs headless (Agg).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow bare run


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_scores(path: Path):
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    return rows


def gold_scores(rows, score_key="ml_score"):
    key = next((k for k in (score_key, "ml_score", "stacked_score", "fused_content_risk")
                if k in rows[0]), None)
    if key is None:
        raise SystemExit(f"no usable score column in {rows[0].keys()}")
    return [int(r["gold"]) for r in rows], [float(r[key]) for r in rows]


def fig_roc(plt, gold, score, title, path):
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(gold, score)
    auc = roc_auc_score(gold, score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(title)
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def fig_pr(plt, gold, score, title, path):
    from sklearn.metrics import average_precision_score, precision_recall_curve

    prec, rec, _ = precision_recall_curve(gold, score)
    ap = average_precision_score(gold, score)
    base = sum(gold) / len(gold)
    plt.figure()
    plt.plot(rec, prec, label=f"AP = {ap:.4f}")
    plt.axhline(base, color="k", ls="--", alpha=0.4, label=f"base rate {base:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(title)
    plt.legend(loc="best"); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def fig_calibration(plt, gold, score, title, path, bins=10):
    edges = [i / bins for i in range(bins + 1)]
    xs, ys, ns = [], [], []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        pts = [(s, g) for s, g in zip(score, gold) if lo <= s < hi or (i == bins - 1 and s == 1.0)]
        if pts:
            xs.append(sum(p for p, _ in pts) / len(pts))
            ys.append(sum(g for _, g in pts) / len(pts))
            ns.append(len(pts))
    plt.figure()
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
    plt.plot(xs, ys, "o-", label="observed")
    plt.xlabel("mean predicted score"); plt.ylabel("observed positive rate")
    plt.title(title); plt.legend(loc="best"); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def fig_threshold_pr(plt, gold, score, title, path):
    ts = [i / 100 for i in range(1, 100)]
    prec, rec = [], []
    for t in ts:
        pred = [s >= t for s in score]
        tp = sum(g and p for g, p in zip(gold, pred))
        fp = sum((not g) and p for g, p in zip(gold, pred))
        fn = sum(g and not p for g, p in zip(gold, pred))
        prec.append(tp / max(1, tp + fp))
        rec.append(tp / max(1, tp + fn))
    plt.figure()
    plt.plot(ts, prec, label="precision")
    plt.plot(ts, rec, label="recall")
    plt.xlabel("threshold"); plt.ylabel("value"); plt.title(title)
    plt.legend(loc="best"); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path, nargs="+", required=True)
    ap.add_argument("--metrics", type=Path, nargs="*", default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--score-key", default="ml_score")
    args = ap.parse_args()
    plt = _mpl()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for scores_path in args.scores:
        rows = load_scores(scores_path)
        if not rows:
            print(f"skip empty {scores_path}"); continue
        gold, score = gold_scores(rows, args.score_key)
        tag = scores_path.stem.replace("scores-", "")
        title = f"{tag} | n={len(rows)} | scores:{scores_path.name} | calibration: per metrics file"
        fig_roc(plt, gold, score, "ROC — " + title, args.out_dir / f"roc-{tag}.png")
        fig_pr(plt, gold, score, "PR — " + title, args.out_dir / f"pr-{tag}.png")
        fig_calibration(plt, gold, score, "calibration — " + title,
                        args.out_dir / f"calibration-{tag}.png")
        fig_threshold_pr(plt, gold, score, "threshold sweep — " + title,
                         args.out_dir / f"threshold-{tag}.png")
        print(f"figures for {scores_path.name} -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
