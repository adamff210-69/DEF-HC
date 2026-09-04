"""Benchmark the Layer-1 injection classifier on a real labeled corpus.

Trains the logistic layer over ``BAAI/bge-small-en-v1.5`` embeddings on a
train split, evaluates on a HELD-OUT test split (unlike train_classifier.py,
which is for quick weight generation), and compares three detectors on the
same split:

  1. lexical pattern bank (no ML)
  2. demo-mode heuristic fusion (no ML)
  3. embedding logistic classifier (ML, trained here)

Input JSONL: ``{"text": "...", "label": 0|1}`` per line (1 = injection).
Convert e.g. qualifire's benchmark with two Kaggle lines (see --help epilog).

Outputs trained weights (same format as train_classifier.py, loadable by
``ContentRiskAnalyzer(demo_mode=False)``) plus a metrics JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow bare run


def load_jsonl(path: Path) -> list[tuple[str, int]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append((str(row["text"]), int(row["label"])))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return rows


def binary_metrics(gold, score, threshold):
    pred = [s >= threshold for s in score]
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum((not g) and p for g, p in zip(gold, pred))
    fn = sum(g and not p for g, p in zip(gold, pred))
    tn = sum((not g) and not p for g, p in zip(gold, pred))
    acc = (tp + tn) / max(1, len(gold))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"threshold": threshold, "accuracy": round(acc, 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def auc_rank(gold, score):
    """Mann-Whitney AUC, ties averaged."""
    pairs = sorted(zip(score, gold))
    rank_sum, n_pos = 0.0, sum(gold)
    n_neg = len(gold) - n_pos
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        rank_sum += avg_rank * sum(pairs[k][1] for k in range(i, j + 1))
        i = j + 1
    if not n_pos or not n_neg:
        return None
    return round((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg), 4)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog="Example (Kaggle, qualifire benchmark):\n"
        "  from datasets import load_dataset; import json\n"
        "  ds = load_dataset('qualifire/prompt-injections-benchmark')['train']\n"
        "  with open('pi.jsonl','w') as f:\n"
        "      for r in ds:\n"
        "          f.write(json.dumps({'text': r['text'],\n"
        "             'label': 1 if r['label']=='jailbreak' else 0}) + '\\n')\n"
        "  python scripts/benchmark_classifier.py --dataset pi.jsonl "
        "--out-weights weights/bge-bench.json --out-metrics metrics.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--eval-file", type=Path, default=None,
                   help="optional JSONL used ONLY as the test set (e.g. an "
                        "official test split). Prevents near-duplicate leakage "
                        "that a random re-split can introduce.")
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0xDEF2)
    p.add_argument("--out-weights", type=Path, required=True)
    p.add_argument("--out-metrics", type=Path, default=None)
    args = p.parse_args()

    import random

    data = load_jsonl(args.dataset)
    if args.eval_file:
        train = data
        test = load_jsonl(args.eval_file)
        split_desc = "official eval file"
    else:
        rng = random.Random(args.seed)
        rng.shuffle(data)
        n_test = max(1, int(len(data) * args.test_frac))
        test, train = data[:n_test], data[n_test:]
        split_desc = f"random {args.test_frac:.0%} hold-out"
    y_train = [y for _, y in train]
    y_test = [y for _, y in test]
    print(f"dataset: train {len(train)} ({sum(y_train)} inj) | "
          f"held-out test {len(test)} ({sum(y_test)} inj) [{split_desc}]")

    # ------------------------------------------------- embeddings + training
    import numpy as np
    from sentence_transformers import SentenceTransformer

    print(f"embedding with {args.model} ...")
    model = SentenceTransformer(args.model)
    X = model.encode([t for t, _ in train + test],
                     normalize_embeddings=True, convert_to_numpy=True,
                     batch_size=64)
    X = np.asarray(X, dtype=np.float64)
    Xtr, Xte = X[: len(train)], X[len(train):]
    ytr = np.array(y_train, dtype=float)

    w = np.zeros(X.shape[1])
    b = 0.0
    for epoch in range(args.epochs):
        z = 1.0 / (1.0 + np.exp(-(Xtr @ w + b)))
        grad_w = Xtr.T @ (z - ytr) / len(train) + args.l2 * w
        grad_b = float((z - ytr).mean())
        w -= args.lr * grad_w
        b -= args.lr * grad_b
        if epoch % 50 == 0:
            eps = 1e-9
            loss = -float(np.mean(ytr * np.log(z + eps) + (1 - ytr) * np.log(1 - z + eps)))
            print(f"epoch {epoch:4d}  train_loss={loss:.4f}")

    ml_scores = (1.0 / (1.0 + np.exp(-(Xte @ w + b)))).tolist()

    # ------------------------------------------------------- baseline scores
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from defend_hc2.content_risk import ContentRiskAnalyzer

    demo = ContentRiskAnalyzer(demo_mode=True)
    lex_scores, demo_scores = [], []
    for text, _ in test:
        lex_scores.append(demo.lexical_scan(text)[0])
        demo_scores.append(demo.injection_score_for(text)[0])

    # threshold = 0.5 for ML; best-F1 threshold for the scores that need one
    def best_threshold(scores):
        best = max((binary_metrics(y_test, scores, t / 20) for t in range(0, 21)),
                   key=lambda m: m["f1"])
        return best

    results = {
        "dataset": str(args.dataset), "n_train": len(train), "n_test": len(test),
        "split": split_desc,
        "eval_file": str(args.eval_file) if args.eval_file else None,
        "model": args.model, "seed": args.seed,
        "embedding_logistic": {
            **binary_metrics(y_test, ml_scores, 0.5),
            "auc": auc_rank(y_test, ml_scores),
        },
        "demo_heuristic_fusion": {
            **best_threshold(demo_scores), "auc": auc_rank(y_test, demo_scores),
        },
        "lexical_only": {
            **best_threshold(lex_scores), "auc": auc_rank(y_test, lex_scores),
        },
    }

    print("\nheld-out test metrics:")
    for name, m in results.items():
        if isinstance(m, dict) and "f1" in m:
            print(f"  {name:<24} acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
                  f"rec={m['recall']:.4f} f1={m['f1']:.4f} auc={m['auc']} (t={m['threshold']})")

    args.out_weights.parent.mkdir(parents=True, exist_ok=True)
    args.out_weights.write_text(json.dumps({
        "format": "defend-hc2-weights/1", "model": args.model, "type": "logistic",
        "weights": [float(x) for x in w], "bias": float(b), "threshold": 0.5,
        "trained_at_ns": time.time_ns(),
        "trained_on": {"dataset": str(args.dataset), "n_train": len(train),
                       "seed": args.seed},
        "metrics": results["embedding_logistic"],
    }))
    print(f"\nweights: {args.out_weights}")
    if args.out_metrics:
        args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
        args.out_metrics.write_text(json.dumps(results, indent=2))
        print(f"metrics: {args.out_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
