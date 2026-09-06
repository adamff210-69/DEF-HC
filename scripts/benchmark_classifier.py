"""Benchmark the Layer-1 injection classifier on real labeled corpora
(spec Phases 5, 11, 12, 13).

Protocol guarantees:

* sklearn base model; ``C`` selected on calibration PR-AUC (Phase 3);
* the deployment threshold comes from CALIBRATION data only — target recall
  chosen a priori (Phase 5); test data is touched once, for final metrics;
* every test-derived "best" operating point is labelled
  ``ORACLE / TEST-ONLY / NOT DEPLOYABLE`` and never enters the weights file;
* baselines are threshold-calibrated with the SAME criterion on calibration
  data; explicitly-labelled DUMMY baselines (always-pos/neg) are reported
  separately and never treated as competitive defenses (Phase 12);
* full Phase-13 metrics with seeded bootstrap CIs for headline models.

Input JSONL: ``{"text": "...", "label": 0|1}`` per line (1 = injection).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow bare run

from defend_hc2.modeling import (
    assert_disjoint_roles,
    bootstrap_cis,
    calibrate_thresholds,
    environment_block,
    exact_duplicate_count,
    file_sha256,
    fit_classifier,
    full_metric_report,
    git_commit,
    load_many,
    remove_overlap,
)

ORACLE = "ORACLE / TEST-ONLY / NOT DEPLOYABLE"


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def probs(X, weights: list[float], bias: float) -> list[float]:
    return [sigmoid(sum(w * x for w, x in zip(weights, row)) + bias) for row in X]


def banner(title: str, data: list[tuple[str, int]]) -> None:
    pos = sum(y for _, y in data)
    print(f"{title}: {len(data)} rows | {pos} injection / {len(data) - pos} benign "
          f"| base rate {pos / max(1, len(data)):.4f} | exact dups {exact_duplicate_count(data)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, nargs="+", required=True)
    p.add_argument("--cal-file", type=Path, nargs="+", default=None)
    p.add_argument("--eval-file", type=Path, nargs="+", default=None)
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-class-balance", action="store_true")
    p.add_argument("--target-recall", type=float, default=0.95,
                   help="deployment criterion, chosen BEFORE seeing test")
    p.add_argument("--allow-role-overlap-debug", action="store_true")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--out-weights", type=Path, required=True)
    p.add_argument("--out-metrics", type=Path, default=None)
    p.add_argument("--out-scores", type=Path, default=None)
    args = p.parse_args()

    if not args.allow_role_overlap_debug:
        assert_disjoint_roles(dataset=args.dataset, cal=args.cal_file or [],
                              eval=args.eval_file or [])

    # ------------------------------------------------------------------ data
    print("== TRAIN SOURCES ==")
    train = load_many(args.dataset, role="train")
    cal: list[tuple[str, int]]
    if args.cal_file:
        print("== CALIBRATION SOURCES ==")
        cal = load_many(args.cal_file, role="cal")
    else:
        import random

        train = list(train)
        random.Random(args.seed).shuffle(train)
        n_cal = max(1, int(len(train) * 0.15))
        cal, train = train[:n_cal], train[n_cal:]
        print(f"== CALIBRATION == pool-split slice n={n_cal} (deployment-"
              f"matched --cal-file is preferred)")

    eval_ = None
    if args.eval_file:
        print("== TEST SOURCES (frozen) ==")
        eval_ = load_many(args.eval_file, role="test")
        eval_, removed = remove_overlap(eval_, train, cal)
        print(f"removed {removed} test rows overlapping train/cal "
              f"(normalized-text match)")
        if not eval_:
            raise SystemExit(
                "test set is EMPTY after overlap removal — the eval file "
                "duplicates training/calibration text; benchmarking it would "
                "invalidate every metric"
            )
    else:
        import random

        pool = list(train)
        random.Random(args.seed + 1).shuffle(pool)
        n_test = max(1, int(len(pool) * 0.2))
        eval_, train = pool[:n_test], pool[n_test:]
        print("== TEST == random 20% hold-out from training pool")

    banner("train", train)
    banner("calibration", cal)
    banner("test", eval_)
    y_train = [y for _, y in train]
    y_cal = [y for _, y in cal]
    y_test = [y for _, y in eval_]

    # ---------------------------------------------------------- embed + fit
    import time as _time

    import numpy as np
    from defend_hc2.embedder import device_report, get_sentence_transformer

    dev = device_report()
    print(f"\n== embedding backend ==")
    for k, v in dev.items():
        print(f"   {k}: {v}")
    if dev.get("selected_device") == "cpu":
        print("   !! running on CPU. If this machine has a GPU, the accelerator\n"
              "      is not visible to torch — embedding will take minutes to\n"
              "      tens of minutes instead of seconds.")

    model = get_sentence_transformer(args.model)
    texts = [t for t, _ in train + cal + eval_]
    bs = 512 if dev.get("selected_device") == "cuda" else 128
    print(f"\nembedding {len(texts):,} texts (batch_size={bs}, "
          f"device={dev.get('selected_device')})…")
    _t0 = _time.perf_counter()
    X = np.asarray(model.encode(
        texts, normalize_embeddings=True,
        convert_to_numpy=True, batch_size=bs, show_progress_bar=True),
        dtype=float)
    _dt = _time.perf_counter() - _t0
    print(f"embedded in {_dt:.1f}s "
          f"({len(texts) / max(_dt, 1e-9):,.0f} texts/s)")
    Xtr = X[: len(train)]
    Xcal = X[len(train): len(train) + len(cal)]
    Xte = X[len(train) + len(cal):]

    print("\n== base classifier (Phase 3) ==")
    fit = fit_classifier(Xtr, y_train, Xcal, y_cal, seed=args.seed,
                         class_balance=not args.no_class_balance, verbose=True)
    print(f"selected C={fit['selected_C']} "
          f"(cal PR-AUC {fit['selected_C_cal_pr_auc']})")

    # ------------------------------------------------- stacked meta-model
    from defend_hc2.content_risk import ContentRiskAnalyzer

    demo = ContentRiskAnalyzer(demo_mode=True)

    def meta_row(text: str, base_p: float) -> list[float]:
        lex, _ = demo.lexical_scan(text)
        struct, _ = demo._structural_features(text)
        return [1.0, float(base_p), lex, struct]

    base_cal = probs(Xcal, fit["weights"], fit["bias"])
    base_te = probs(Xte, fit["weights"], fit["bias"])
    print(f"\nbuilding meta features for {len(cal) + len(eval_):,} rows "
          f"(pure-Python lexical/structural scan, CPU-bound)…")
    _t0 = _time.perf_counter()
    Zcal = np.array([meta_row(t, p) for (t, _), p in zip(cal, base_cal)])
    Zte = np.array([meta_row(t, p) for (t, _), p in zip(eval_, base_te)])
    print(f"meta features in {_time.perf_counter() - _t0:.1f}s")
    print("\n== stacked meta-model (trained on calibration split only) ==")
    meta = fit_classifier(Zcal, y_cal, Zcal, y_cal, seed=args.seed,
                          class_balance=not args.no_class_balance)
    stack_cal = probs(Zcal, meta["weights"], meta["bias"])
    stack_te = probs(Zte, meta["weights"], meta["bias"])

    # ------------------------------------------------- calibration (cal only)
    print("\n== threshold calibration (CALIBRATION DATA ONLY) ==")
    thr_base = calibrate_thresholds(y_cal, base_cal)
    thr_stack = calibrate_thresholds(y_cal, stack_cal)
    key = f"recall@{args.target_recall}"
    print(f"base:  {thr_base}")
    print(f"stack: {thr_stack}")
    print(f"deployment criterion (declared a priori): {key}")

    # lexical / demo baselines calibrated WITH THE SAME criterion on cal
    print(f"scoring lexical + fused demo baselines on "
          f"{2 * (len(cal) + len(eval_)):,} rows…")
    _t0 = _time.perf_counter()
    lex_cal, lex_te = ([demo.lexical_scan(t)[0] for t, _ in part] for part in (cal, eval_))
    fus_cal, fus_te = ([demo.injection_score_for(t)[0] for t, _ in part] for part in (cal, eval_))
    thr_lex = calibrate_thresholds(y_cal, lex_cal)[key]
    thr_fus = calibrate_thresholds(y_cal, fus_cal)[key]
    print(f"baselines in {_time.perf_counter() - _t0:.1f}s")

    # ============================================================ metrics
    results: dict = {
        "sources": {
            "train": [str(d) for d in args.dataset],
            "calibration": [str(d) for d in args.cal_file] or ["pool-split 15%"],
            "test": [str(d) for d in args.eval_file] or ["pool-split 20%"],
        },
        "counts": {"train": len(train), "cal": len(cal), "test": len(eval_)},
        "base_rates": {
            "train": round(sum(y_train) / max(1, len(y_train)), 4),
            "cal": round(sum(y_cal) / max(1, len(y_cal)), 4),
            "test": round(sum(y_test) / max(1, len(y_test)), 4),
        },
        "seed": args.seed, "model": args.model,
        "threshold_origin": "calibration data only",
        "deployment_criterion": key,
        "selection": {k: fit[k] for k in ("selected_C", "selection_metric",
                                          "selected_C_cal_pr_auc", "C_sweep",
                                          "fold_scaler_max_abs_dev")},
        "environment": environment_block(),
        "git_commit": git_commit(Path(__file__).resolve().parents[1]),
        "timestamp_ns": time.time_ns(),
    }

    def headline(name, cal_scores, te_scores, thr_map):
        t = thr_map[key]
        rep = full_metric_report(y_test, te_scores, t)
        rep["calibrated_thresholds"] = thr_map
        rep["ci95"] = bootstrap_cis(y_test, te_scores, t, resamples=args.bootstrap, seed=args.seed)
        oracle_t = max(
            (full_metric_report(y_test, te_scores, c / 100) for c in range(0, 101)),
            key=lambda m: m["f1"],
        )
        results[name] = rep
        results[f"{name}__{ORACLE}"] = oracle_t

    headline("embedding_logistic_calibrated", base_cal, base_te, thr_base)
    headline("stacked_meta_calibrated", stack_cal, stack_te, thr_stack)

    results["baseline_lexical_calibrated"] = full_metric_report(y_test, lex_te, thr_lex)
    results["baseline_demo_fusion_calibrated"] = full_metric_report(y_test, fus_te, thr_fus)
    for name, thr, cal_s, te_s in (("lexical", thr_lex, lex_cal, lex_te),
                                   ("demo_fusion", thr_fus, fus_cal, fus_te)):
        oracle_t = max(
            (full_metric_report(y_test, te_s, c / 100) for c in range(0, 101)),
            key=lambda m: m["f1"])
        results[f"baseline_{name}__{ORACLE}"] = oracle_t

    results["dummy_always_positive__excluded_from_comparison"] = full_metric_report(
        y_test, [1.0] * len(y_test), 0.5)
    results["dummy_always_negative__excluded_from_comparison"] = full_metric_report(
        y_test, [0.0] * len(y_test), 0.5)

    # ------------------------------------------------------------ outputs
    print("\n== development-test metrics (development_test_previously_observed; calibration-derived thresholds) ==")
    for name, m in results.items():
        if isinstance(m, dict) and "f1" in m:
            tag = "  [TEST-ORACLE]" if ORACLE in name else \
                  "  [DUMMY]" if "dummy" in name else ""
            print(f"  {name:<58} bal={m['balanced_accuracy']:.4f} "
                  f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
                  f"AUC={m['roc_auc']}{tag}")

    if args.out_scores:
        args.out_scores.parent.mkdir(parents=True, exist_ok=True)
        t_deploy = thr_stack[key]
        with args.out_scores.open("w", encoding="utf-8") as fh:
            for i, ((text, g), p_base, p_stack) in enumerate(zip(eval_, base_te, stack_te)):
                pred = int(p_stack >= t_deploy)
                fh.write(json.dumps({
                    "example_id": f"test-{i:06d}", "dataset": "test",
                    "gold": g, "text": text,
                    "ml_score": round(float(p_base), 6),
                    "stacked_score": round(float(p_stack), 6),
                    "predicted": pred, "threshold": t_deploy,
                    "error": ("TP" if g and pred else "TN" if not g and not pred
                              else "FN" if g else "FP"),
                }) + "\n")
        print(f"scores: {args.out_scores}")

    args.out_weights.parent.mkdir(parents=True, exist_ok=True)
    args.out_weights.write_text(json.dumps({
        "format": "defend-hc2-weights/1", "model": args.model, "type": "logistic+standard_scaler(sklearn)",
        "weights": fit["weights"], "bias": fit["bias"], "dims": fit["dims"],
        "threshold": thr_base[key],
        "calibrations": thr_base,
        "deployment_criterion": key,
        "selection": {k: fit[k] for k in ("selected_C", "selection_metric",
                                          "selected_C_cal_pr_auc", "seed", "estimator")},
        "trained_at_ns": time.time_ns(),
        "trained_on": results["sources"] | {"counts": results["counts"]},
        "metrics_test": {"embedding_logistic_calibrated": results["embedding_logistic_calibrated"]},
        "stacked_meta": {
            "features": ["bias", "base_p", "lexical", "structural"],
            "weights": meta["weights"], "bias": meta["bias"],
            "calibrations": thr_stack,
        },
        # NO test-oracle thresholds here — production weights must never
        # contain test-derived operating points.
    }))
    print(f"weights: {args.out_weights}")
    if args.out_metrics:
        args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
        args.out_metrics.write_text(json.dumps(results, indent=2))
        print(f"metrics: {args.out_metrics}")
        print(f"run manifest sha256(metrics json): {file_sha256(args.out_metrics)[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
