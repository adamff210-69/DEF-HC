"""Experiment-matrix orchestrator (spec Phases 14, 18, 20).

Runs Exp-A .. Exp-F against data produced by prepare_benchmarks.py, saving
``bench-metrics-exp-*.json`` and ``scores-exp-*.jsonl`` under --out-dir, plus
``reports/reproducibility.json``.  Frozen-test decisions (thresholds,
policies) are taken on calibration data and then evaluated ONCE (Exp-E).

Intended for network-enabled runners (Kaggle).  All scoring logic delegates
to defend_hc2.modeling so numbers are computed by tested code paths.

Layout expected in --data-dir:
    slp-train.jsonl  slp-cal.jsonl  pi-test.jsonl
    spml-train.jsonl spml-cal.jsonl spml-test.jsonl   (full SPML schema)
    foreign-*.jsonl  (optional)

No target performance may be achieved: whatever the frozen protocol yields
on test IS the result.
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
    bootstrap_cis,
    calibrate_thresholds,
    environment_block,
    file_sha256,
    full_metric_report,
    git_commit,
    load_jsonl,
    remove_overlap,
    roc_auc,
)

BATCH = 256


def embed_texts(model, texts):
    import numpy as np

    return np.asarray(model.encode(list(texts), normalize_embeddings=True,
                                   convert_to_numpy=True, batch_size=BATCH),
                      dtype=float)


def probs(X, fit):
    return [1.0 / (1.0 + math.exp(-(sum(w * x for w, x in zip(fit["weights"], row)) + fit["bias"])))
            for row in X]


def evaluate(tag, fit, threshold_map, key, X_te, y_test, out_dir, texts, dataset_names):
    p_te = probs(X_te, fit)
    t = threshold_map[key]
    report = full_metric_report(y_test, p_te, t)
    report["ci95"] = bootstrap_cis(y_test, p_te, t, resamples=1000, seed=42)
    report["calibrated_thresholds"] = threshold_map
    (out_dir / f"bench-metrics-{tag}.json").write_text(json.dumps(report, indent=2))
    with (out_dir / f"scores-{tag}.jsonl").open("w", encoding="utf-8") as fh:
        for i, (text, g, p) in enumerate(zip(texts, y_test, p_te)):
            pred = int(p >= t)
            fh.write(json.dumps({
                "example_id": f"{tag}-{i:06d}", "dataset": dataset_names,
                "gold": g, "text": text, "ml_score": round(float(p), 6),
                "predicted": pred, "threshold": t,
                "error": ("TP" if g and pred else "TN" if not g and not pred
                          else "FN" if g else "FP"),
            }) + "\n")
    print(f"  {tag:<24} AUC={report['roc_auc']} PR={report['pr_auc']} "
          f"P={report['precision']:.4f} R={report['recall']:.4f} "
          f"bal={report['balanced_accuracy']:.4f} (t={t:.4f})")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--target-recall", type=float, default=0.95,
                    help="deployment criterion declared BEFORE test")
    ap.add_argument("--skip-bc", action="store_true",
                    help="re-run only Exp-A + Exp-F (Exp-F depends on A's fit)")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    key = f"recall@{args.target_recall}"

    from defend_hc2.embedder import get_sentence_transformer
    from defend_hc2.modeling import fit_classifier
    from defend_hc2.perturb import TRANSFORMS
    import numpy as np

    embedder = get_sentence_transformer(args.model)
    load = lambda name: load_jsonl(args.data_dir / name)
    started = time.time_ns()

    # ---- EXP-A: in-distribution ------------------------------------------
    print("\n== EXP-A in-distribution (S-Labs train/val/test) ==")
    slp_tr, slp_cal, pi_test = load("slp-train.jsonl"), load("slp-cal.jsonl"), load("pi-test.jsonl")
    Xtr = embed_texts(embedder, [t for t, _ in slp_tr])
    Xcal = embed_texts(embedder, [t for t, _ in slp_cal])
    Xte = embed_texts(embedder, [t for t, _ in pi_test])
    fit_a = fit_classifier(Xtr, [y for _, y in slp_tr], Xcal, [y for _, y in slp_cal],
                           seed=42, verbose=True)
    thr_a = calibrate_thresholds([y for _, y in slp_cal], probs(Xcal, fit_a))
    print(f"  thresholds (cal): {thr_a}")
    rep_a = evaluate("exp-a", fit_a, thr_a, key, Xte, [y for _, y in pi_test], out,
                     [t for t, _ in pi_test], ["pi-test"])

    # ---- EXP-B: zero-shot frozen model on foreign corpora ----------------
    if not args.skip_bc:
        print("\n== EXP-B zero-shot (Exp-A model + Exp-A calibration, no retraining) ==")
        for foreign in sorted(args.data_dir.glob("foreign-*.jsonl")):
            rows = load(foreign.name)
            rows, removed = remove_overlap(rows, slp_tr, slp_cal)
            print(f"  {foreign.name}: {len(rows)} rows (+{removed} overlap-removed)")
            if not rows:
                print("  skipped — empty after overlap removal")
                continue
            Xf = embed_texts(embedder, [t for t, _ in rows])
            evaluate(f"exp-b-{foreign.stem.replace('foreign-', '')}", fit_a, thr_a, key,
                     Xf, [y for _, y in rows], out, [t for t, _ in rows], [foreign.name])

    if args.skip_bc:
        print("\n(skipping Exp-B/C per --skip-bc)")
    # ---- EXP-C: mixed-source training ------------------------------------
    if not args.skip_bc:
        print("\n== EXP-C mixed-source (S-Labs train + SPML train) ==")
        spml_tr, spml_cal, spml_test = (load("spml-train.jsonl"), load("spml-cal.jsonl"),
                                        load("spml-test.jsonl"))
        mix_tr = slp_tr + spml_tr
        mix_cal_c1 = slp_cal                       # C1: deployment-matched
        mix_cal_c2 = slp_cal + spml_cal            # C2: combined
        Xmix_tr = embed_texts(embedder, [t for t, _ in mix_tr])
        y_mix_tr = [y for _, y in mix_tr]
        Xslp_cal = Xcal
        Xspml_cal = embed_texts(embedder, [t for t, _ in spml_cal])
        Xspml_te = embed_texts(embedder, [t for t, _ in spml_test])
        for variant, (Xc, yc) in {"c1": (Xslp_cal, [y for _, y in slp_cal]),
                                  "c2": (np.concatenate([Xslp_cal, Xspml_cal]),
                                         [y for _, y in slp_cal] + [y for _, y in spml_cal])}.items():
            fit_c = fit_classifier(Xmix_tr, y_mix_tr, Xc, yc, seed=42)
            thr_c = calibrate_thresholds(yc, probs(Xc, fit_c))
            print(f"  [Exp-C/{variant}] thresholds: {thr_c}")
            evaluate(f"exp-c-{variant}-slp", fit_c, thr_c, key, Xte, [y for _, y in pi_test],
                     out, [t for t, _ in pi_test], ["pi-test"])
            evaluate(f"exp-c-{variant}-spml", fit_c, thr_c, key, Xspml_te,
                     [y for _, y in spml_test], out, [t for t, _ in spml_test], ["spml-test"])
        print(f"  delta vs Exp-A on S-Labs AUC: recorded per-file "
              f"(Exp-A {rep_a['roc_auc']})")

    # ---- EXP-F: perturbation robustness on Exp-A S-Labs test -------------
    print("\n== EXP-F obfuscation robustness (held-out S-Labs test) ==")
    from defend_hc2.content_risk import ContentRiskAnalyzer, combine_signals

    lexical_layer = ContentRiskAnalyzer(demo_mode=True)
    rob = {"clean": rep_a, "recovery_note":
           "recovery = fused(raw-ML score, variant-aware lexical) AUC — "
           "threshold-free proxy for what the normalization layer restores",
           "per_transform": {}}
    for name, fn in TRANSFORMS.items():
        perturbed = [(fn(t), y) for t, y in pi_test]
        try:
            Xp = embed_texts(embedder, [t for t, _ in perturbed])
            rep_p = evaluate(f"exp-f-{name}", fit_a, thr_a, key, Xp,
                             [y for _, y in pi_test], out,
                             [t for t, _ in perturbed], ["pi-test", f"perturb:{name}"])
            p_ml = probs(Xp, fit_a)
            lex_p = [lexical_layer.lexical_scan(t)[0] for t, _ in perturbed]
            fused = [combine_signals({"injection": ml, "lexical": lx,
                                      "retrieval": None, "mismatch": None, "drift": None})
                     for ml, lx in zip(p_ml, lex_p)]
            rec_auc = roc_auc([y for _, y in pi_test], fused)
            print(f"    recovery (fused ML+variant-lexical) AUC: {rec_auc}")
            rob["per_transform"][name] = {
                "perturbed_f1": rep_p["f1"], "perturbed_recall": rep_p["recall"],
                "absolute_degradation_f1": round(rep_a["f1"] - rep_p["f1"], 4),
                "clean_auc": rep_a["roc_auc"], "perturbed_auc": rep_p["roc_auc"],
                "recovery_auc": rec_auc,
            }
        except Exception as exc:
            print(f"  transform {name} failed: {exc} (recorded, continuing)")
            rob["per_transform"][name] = {"error": str(exc)[:160]}
    (out / "bench-metrics-exp-f.json").write_text(json.dumps(rob, indent=2))

    # ---- reproducibility manifest (Phase 20) ------------------------------
    manifest = {
        "git_commit": git_commit(Path(__file__).resolve().parents[1]),
        "seed": 42, "deployment_criterion": key, "target_recall": args.target_recall,
        "model": args.model, "environment": environment_block(),
        "threshold_origin": "calibration data only",
        "file_sha256": {fp.name: file_sha256(fp) for fp in sorted(out.glob("*.json*"))},
        "runtime_ns": time.time_ns() - started,
    }
    reports = Path(args.data_dir).parent / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "reproducibility.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nreports/reproducibility.json written")
    print("note: Exp-D layer ablation runs via scripts/run_ablation.py; "
          "Exp-E policy via scripts/calibrate_policy.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
