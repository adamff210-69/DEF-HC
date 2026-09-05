"""Exp-G: train-time letter-spacing augmentation (open research item 1).

A PRIORI contract (declared before any evaluation is run):

  * train base: S-Labs train only (same as Exp-A — augmentation, not
    cross-corpus mixing, is the measured variable)
  * augmentation: 25% of eligible training rows (every 4th, text <= 512
    chars) get a label-preserving letter-spaced copy (defend_hc2.augment)
  * calibration: slp-cal only; thresholds from calibration only
    (deployment criterion recall@0.95, same as Exp-A)
  * evaluation (once, on the development-test split
    development_test_previously_observed): clean rows, and the same rows
    letter-spaced; Exp-A numbers are the comparison baseline
  * predeclared success gates (reported either way):
      - clean dev-test AUC loss <= 0.01 vs Exp-A's 0.9876
      - letter-spaced dev-test AUC >= 0.75 (from 0.4382)
      - benign FPR on slp-cal at the calibrated threshold must not rise
        by more than +2 points vs Exp-A's calibrated run

If a gate fails, the numbers go into the record as-is with its gate
status — no further augmentation tuning in response to dev-test values.

Output: bench-metrics-exp-g-letteraug.json (+ scores/exp-a reference
reuse; no example dumps unless a perturbed AUC < 0.5 appears, in which
case exp-g-letter-spacing-examples.jsonl is written).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow bare run

from defend_hc2.augment import AUGMENT_EVERY, AUGMENT_MAX_CHARS, letter_spacing_augment
from defend_hc2.modeling import (
    calibrate_thresholds,
    environment_block,
    file_sha256,
    full_metric_report,
    git_commit,
    load_jsonl,
    remove_overlap,
)
from defend_hc2.perturb import letter_spacing_extreme


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("bench-data"))
    ap.add_argument("--out-dir", type=Path, default=Path("bench-out"))
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--every", type=int, default=AUGMENT_EVERY)
    ap.add_argument("--max-chars", type=int, default=AUGMENT_MAX_CHARS)
    args = ap.parse_args()

    from defend_hc2.embedder import get_sentence_transformer
    from defend_hc2.modeling import fit_classifier
    from scripts.run_experiments import evaluate, embed_texts, probs

    started = time.time_ns()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    load = lambda n: load_jsonl(args.data_dir / n)

    slp_tr, slp_cal, pi_test = (load("slp-train.jsonl"), load("slp-cal.jsonl"),
                                load("pi-test.jsonl"))
    pi_test, removed = remove_overlap(pi_test, slp_tr, slp_cal)
    print(f"dev-test overlap removal dropped {removed} examples "
          f"(clamp-tolerant canonicalization anti-leak step)")

    extra = letter_spacing_augment(slp_tr, every=args.every,
                                   max_chars=args.max_chars)
    aug_tr = slp_tr + extra
    print(f"augmentation: +{len(extra)} letter-spaced copies "
          f"(every={args.every}, max_chars={args.max_chars}) "
          f"-> {len(aug_tr)} train rows")

    print("embedding (bge, normalized, batch 256)…")
    embedder = get_sentence_transformer(args.model)
    Xtr = embed_texts(embedder, [t for t, _ in aug_tr])
    Xcal = embed_texts(embedder, [t for t, _ in slp_cal])
    Xte = embed_texts(embedder, [t for t, _ in pi_test])
    let_te = [(letter_spacing_extreme(t), y) for t, y in pi_test]
    let_cal = [(letter_spacing_extreme(t), y) for t, y in slp_cal]
    Xlet_te = embed_texts(embedder, [t for t, _ in let_te])
    Xlet_cal = embed_texts(embedder, [t for t, _ in let_cal])

    fit = fit_classifier(Xtr, [y for _, y in aug_tr], Xcal, [y for _, y in slp_cal],
                         seed=42, verbose=True)
    key = f"recall@{args.target_recall}"
    thr = calibrate_thresholds([y for _, y in slp_cal], probs(Xcal, fit))
    print(f"thresholds (cal): {thr}")

    y_te = [y for _, y in pi_test]
    rep_clean = evaluate("exp-g-letteraug-clean", fit, thr, key, Xte, y_te, out,
                         [t for t, _ in pi_test],
                         ["pi-test", "development_test_previously_observed",
                          "model:letteraug"])
    rep_let = evaluate("exp-g-letteraug-letters", fit, thr, key, Xlet_te, y_te,
                       out, [t for t, _ in let_te],
                       ["pi-test", "development_test_previously_observed",
                        "perturb:letter_spacing_extreme", "model:letteraug"])

    # calibration-side benign FPR at the calibrated threshold (gate input)
    p_cal = probs(Xcal, fit)
    t = thr[key]
    cal_benign_fpr = full_metric_report([y for _, y in slp_cal], p_cal, t)["fpr"]
    p_let_cal = probs(Xlet_cal, fit)
    let_cal_benign_fpr = full_metric_report(
        [y for _, y in slp_cal], p_let_cal, t)["fpr"]

    EXP_A_AUC = 0.9876          # Exp-A clean dev-test (this kernel, same splits)
    EXP_A_DEV_FPR = 0.0333      # Exp-A dev-test benign FPR at t(recall@0.95)
    gates = {
        "clean_dev_auc_loss<=0.01": {
            "value": rep_clean["roc_auc"], "baseline": EXP_A_AUC,
            "pass": (rep_clean["roc_auc"] or 0.0) >= EXP_A_AUC - 0.01},
        "letterspaced_dev_auc>=0.75": {
            "value": rep_let["roc_auc"], "baseline": 0.4382,
            "pass": (rep_let["roc_auc"] or 0.0) >= 0.75},
        "cal_benign_fpr_rise<=+0.02": {
            "value": cal_benign_fpr,
            "baseline_note": "Exp-A's benign FPR at the recall@0.95 "
                             "threshold was 0.0333 on the dev-test split "
                             "(bench-metrics-exp-f-zero-width.json); "
                             "Exp-G compares its CAL-signal benign FPR "
                             "against it as the reporting reference",
            "baseline": EXP_A_DEV_FPR,
            "pass": cal_benign_fpr <= EXP_A_DEV_FPR + 0.02},
        "letterspaced_cal_benign_fpr": {"value": let_cal_benign_fpr,
                                        "pass": None},  # reporting aid
    }

    summary = {
        "experiment": "exp-g letter-spacing train-time augmentation",
        "a_priori": {"every": args.every, "max_chars": args.max_chars,
                     "train_rows_base": len(slp_tr),
                     "train_rows_augmented": len(aug_tr),
                     "dev_test_overlap_removed": removed,
                     "calibration": "slp-cal only",
                     "deployment_criterion": key},
        "clean_dev_test": rep_clean, "letterspaced_dev_test": rep_let,
        "cal_benign_fpr": cal_benign_fpr,
        "letterspaced_cal_benign_fpr": let_cal_benign_fpr,
        "gates": gates,
        "label": "development_test_previously_observed",
        "environment": environment_block(),
        "git_commit": git_commit(Path(__file__).resolve().parents[1]),
        "runtime_ns": time.time_ns() - started,
        "thresholds_cal": thr,
        "file_sha256": {},
    }
    fp = out / "bench-metrics-exp-g-letteraug.json"
    fp.write_text(json.dumps(summary, indent=2))
    summary["file_sha256"] = {fp.name: file_sha256(fp)}
    fp.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {fp}")
    print("gate summary:", {k: (v.get("pass") if isinstance(v, dict) else v)
                             for k, v in gates.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
