"""Benchmark the Layer-1 injection classifier on real labeled corpora.

Upgrades over the plain logistic benchmark:

* **multi-dataset training** — pass several ``--dataset`` JSONL files; they
  are concatenated (single-corpus quirks are the enemy of generalization);
* **class balancing** — inverse-frequency weighting in the logistic loss so
  the base rate of each corpus does not bias the decision boundary;
* **stacked meta-model** — a small logistic over ``[base_p, lexical,
  structural, obfuscation]`` features, trained on the *validation* split
  (the base embedder sees only the train split — honest stacking);
* **validation-calibrated threshold** — the operating point is chosen on the
  *validation* split to hit a target recall (e.g. catch 95% of attacks) with
  the best precision available — never tuned on the test set.

Every reported metric is computed on an untouched test split
(``--eval-file`` for an official/foreign corpus, else a random hold-out).

Input JSONL: ``{"text": "...", "label": 0|1}`` per line (1 = injection).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow bare run

_B64ISH = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{32,}={0,2}(?![A-Za-z0-9+/=])")
_LEET = re.compile(r"[0-9]{2,}|[a-zA-Z][0-9][a-zA-Z]|[a-zA-Z][0-9]{2,}")


def load_jsonl(path: Path) -> list[tuple[str, int]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows.append((str(row["text"]), int(row["label"])))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return rows


def load_many(paths: list[Path]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for path in paths:
        rows = load_jsonl(path)
        print(f"  loaded {len(rows):>6} rows from {path}")
        out.extend(rows)
    return out


# ------------------------------------------------------------------ metrics
def binary_metrics(gold, score, threshold):
    pred = [s >= threshold for s in score]
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum((not g) and p for g, p in zip(gold, pred))
    fn = sum(g and not p for g, p in zip(gold, pred))
    tn = sum((not g) and not p for g, p in zip(gold, pred))
    acc = (tp + tn) / max(1, len(gold))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"threshold": round(float(threshold), 4),
            "accuracy": round(acc, 4), "balanced_accuracy": round((rec + spec) / 2, 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def auc_rank(gold, score):
    pairs = sorted(zip(score, gold))
    rank_sum, n_pos = 0.0, sum(gold)
    n_neg = len(gold) - n_pos
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        rank_sum += sum(pairs[k][1] for k in range(i, j + 1)) * ((i + j) / 2 + 1)
        i = j + 1
    if not n_pos or not n_neg:
        return None
    return round((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg), 4)


def best_f1_metrics(gold, score):
    return max((binary_metrics(gold, score, t / 100) for t in range(0, 101)),
               key=lambda m: m["f1"])


def calibrate_for_recall(val_gold, val_score, target_recall: float) -> float:
    """Highest threshold whose validation recall still meets the target —
    i.e. best precision compatible with ``target_recall``.  Chosen on
    validation data only, never on test.

    Recall is non-increasing as the threshold rises, so we scan thresholds
    ascending and keep the last one that still achieves the target.
    """
    total_pos = max(1, sum(val_gold))
    best = 0.0
    for t in sorted(set(val_score)):
        tp = sum(g and s >= t for g, s in zip(val_gold, val_score))
        if tp / total_pos >= target_recall - 1e-9:
            best = t
        else:
            break  # from here recall can only drop further
    return best


# ------------------------------------------------------------ model helpers
def train_logistic(X, y, epochs, lr, l2, class_balance=False):
    """Plain numpy logistic regression with optional inverse-frequency
    class weighting. Returns (w, b)."""
    import numpy as np

    y = np.asarray(y, dtype=float)
    n = len(y)
    w_sample = np.ones(n)
    if class_balance:
        n_pos = max(1.0, float(y.sum()))
        n_neg = max(1.0, n - n_pos)
        w_sample[y == 1] = n / (2 * n_pos)
        w_sample[y == 0] = n / (2 * n_neg)
    w = np.zeros(X.shape[1])
    b = 0.0
    for epoch in range(epochs):
        z = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        err = (z - y) * w_sample
        grad_w = X.T @ err / n + l2 * w
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b
        if epoch % 100 == 0:
            eps = 1e-9
            loss = -float(np.mean(w_sample * (y * np.log(z + eps) + (1 - y) * np.log(1 - z + eps))))
            print(f"    epoch {epoch:4d}  train_loss={loss:.4f}")
    return w, b


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


# --------------------------------------------------------------------- main
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, nargs="+", required=True,
                   help="one or more JSONL training corpora (concatenated)")
    p.add_argument("--eval-file", type=Path, nargs="+", default=None,
                   help="optional JSONL used ONLY as the test set (official or "
                        "foreign-corpus split)")
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="fraction of the training pool used for stacking + "
                        "threshold calibration (never for the base model)")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0xDEF2)
    p.add_argument("--class-balance", action="store_true",
                   help="inverse-frequency weighting in the loss")
    p.add_argument("--target-recall", type=float, default=0.95,
                   help="calibrate the operating threshold on VALIDATION to "
                        "achieve this recall with best precision")
    p.add_argument("--out-weights", type=Path, required=True,
                   help="drop-in weights for ContentRiskAnalyzer (base model; "
                        "stacker + calibrated threshold stored alongside)")
    p.add_argument("--out-metrics", type=Path, default=None)
    p.add_argument("--out-scores", type=Path, default=None)
    args = p.parse_args()

    import random

    print("loading corpora:")
    data = load_many(args.dataset)
    rng = random.Random(args.seed)
    rng.shuffle(data)
    if args.eval_file:
        test = load_many(args.eval_file)
        pool = data
        split_desc = "official/foreign eval file"
    else:
        n_test = max(1, int(len(data) * args.test_frac))
        test, pool = data[:n_test], data[n_test:]
        split_desc = f"random {args.test_frac:.0%} hold-out"
    n_val = max(1, int(len(pool) * args.val_frac))
    val, train = pool[:n_val], pool[n_val:]
    y_train = [y for _, y in train]
    y_val = [y for _, y in val]
    y_test = [y for _, y in test]
    print(f"train {len(train)} ({sum(y_train)} inj) | "
          f"val {len(val)} ({sum(y_val)} inj) | "
          f"test {len(test)} ({sum(y_test)} inj) [{split_desc}]")
    print(f"test base rate: {sum(y_test) / max(1, len(test)):.3f}")

    # ------------------------------------------------------- embed + base
    import numpy as np
    from sentence_transformers import SentenceTransformer

    print(f"embedding with {args.model} ...")
    model = SentenceTransformer(args.model)
    X = np.asarray(model.encode(
        [t for t, _ in train + val + test],
        normalize_embeddings=True, convert_to_numpy=True, batch_size=64),
        dtype=np.float64)
    Xtr, Xva, Xte = (X[: len(train)],
                     X[len(train): len(train) + len(val)],
                     X[len(train) + len(val):])

    print("training base logistic (embeddings)...")
    w, b = train_logistic(Xtr, y_train, args.epochs, args.lr, args.l2,
                          class_balance=args.class_balance)
    base_va = 1.0 / (1.0 + np.exp(-(Xva @ w + b)))
    base_te = 1.0 / (1.0 + np.exp(-(Xte @ w + b)))

    # ------------------------------------------------- meta-features/stack
    from defend_hc2.content_risk import ContentRiskAnalyzer

    demo = ContentRiskAnalyzer(demo_mode=True)

    def meta_row(text: str, base_p: float) -> list[float]:
        lex, _ = demo.lexical_scan(text)
        struct, _ = demo._structural_features(text)
        obf = (1.0 if _B64ISH.search(text) else 0.0) + (1.0 if _LEET.search(text) else 0.0)
        return [1.0, float(base_p), lex, struct, obf]

    print("training stacked meta-model (val split)...")
    Zva = np.array([meta_row(t, p) for (t, _), p in zip(val, base_va)])
    Zte = np.array([meta_row(t, p) for (t, _), p in zip(test, base_te)])
    ws, bs = train_logistic(Zva, y_val, epochs=250, lr=0.5, l2=1e-3,
                            class_balance=args.class_balance)
    stack_va = np.array([sigmoid(float(z @ ws + bs)) for z in Zva])
    stack_te = np.array([sigmoid(float(z @ ws + bs)) for z in Zte])

    # --------------------------------------- calibration + final metrics
    thr = calibrate_for_recall(y_val, stack_va.tolist(), args.target_recall)
    print(f"calibrated threshold on validation "
          f"(target recall {args.target_recall}): t={thr:.4f}")

    lex_scores = [demo.lexical_scan(t)[0] for t, _ in test]
    demo_scores = [demo.injection_score_for(t)[0] for t, _ in test]

    base_va_list = base_va.tolist()
    results = {
        "datasets": [str(d) for d in args.dataset],
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "split": split_desc,
        "eval_files": [str(d) for d in args.eval_file] if args.eval_file else None,
        "class_balance": bool(args.class_balance),
        "positive_rate_test": round(sum(y_test) / max(1, len(test)), 4),
        "model": args.model, "seed": args.seed,
        "embedding_logistic_t0.5": {
            **binary_metrics(y_test, base_te.tolist(), 0.5),
            "auc": auc_rank(y_test, base_te.tolist()),
        },
        "embedding_logistic_best_f1": {
            **best_f1_metrics(y_test, base_te.tolist()),
            "auc": auc_rank(y_test, base_te.tolist()),
        },
        "stacked_meta_t0.5": {
            **binary_metrics(y_test, stack_te.tolist(), 0.5),
            "auc": auc_rank(y_test, stack_te.tolist()),
        },
        "stacked_meta_calibrated": {
            **binary_metrics(y_test, stack_te.tolist(), thr),
            "auc": auc_rank(y_test, stack_te.tolist()),
            "calibration": f"t chosen on val for recall>={args.target_recall}",
        },
        "demo_heuristic_fusion": {
            **best_f1_metrics(y_test, demo_scores), "auc": auc_rank(y_test, demo_scores),
        },
        "lexical_only": {
            **best_f1_metrics(y_test, lex_scores), "auc": auc_rank(y_test, lex_scores),
        },
    }

    if args.out_scores:
        args.out_scores.parent.mkdir(parents=True, exist_ok=True)
        with args.out_scores.open("w", encoding="utf-8") as fh:
            for (text, g), p_base, p_stack in zip(test, base_te.tolist(), stack_te.tolist()):
                fh.write(json.dumps({"text": text, "label": g,
                                     "ml_score": round(float(p_base), 6),
                                     "stacked_score": round(float(p_stack), 6)}) + "\n")

    print("\nheld-out test metrics:")
    for name, m in results.items():
        if isinstance(m, dict) and "f1" in m:
            print(f"  {name:<34} acc={m['accuracy']:.4f} bal={m['balanced_accuracy']:.4f} "
                  f"prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} "
                  f"auc={m['auc']} (t={m['threshold']})")

    # -------- weights: v1 drop-in (base logistic) + stacker + calibration
    args.out_weights.parent.mkdir(parents=True, exist_ok=True)
    args.out_weights.write_text(json.dumps({
        "format": "defend-hc2-weights/1", "model": args.model, "type": "logistic",
        "weights": [float(x) for x in w], "bias": float(b), "threshold": 0.5,
        "trained_at_ns": time.time_ns(),
        "trained_on": {"datasets": [str(d) for d in args.dataset],
                       "n_train": len(train), "seed": args.seed,
                       "class_balance": bool(args.class_balance)},
        "metrics": results["embedding_logistic_t0.5"],
        "stacked_meta": {
            "features": ["bias", "base_p", "lexical", "structural", "obfuscation"],
            "weights": [float(x) for x in ws], "bias": float(bs),
            "calibrated_threshold": float(thr),
            "target_recall": args.target_recall,
        },
    }))
    print(f"\nweights: {args.out_weights}")
    if args.out_metrics:
        args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
        args.out_metrics.write_text(json.dumps(results, indent=2))
        print(f"metrics: {args.out_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
