"""Regime-matched HC-Bench policy bands: calibrated on hcbench-cal ONLY.

Reuses the same predeclared objective as scripts/calibrate_policy.py
(maximize detection precision subject to recall >= target; ties by
lowest benign FPR, then most conservative bands) on production-channel
scores of hcbench-cal rows (all four surfaces).  The chosen bands are
then evaluated ONCE on hcbench-test.  Nothing about the frozen model or
the frozen S-Labs/SPML policies changes — band provenance is identical,
only the calibration regime is regime-matched to HC-Bench traffic.

Example:
    python scripts/calibrate_hcbench_policy.py \\
        --data-dir hcbench --weights weights/bge-final.json \\
        --target-recall 0.95 --out calibrated-policy-hcbench-balanced.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow bare run

from defend_hc2.modeling import (
    environment_block,
    file_sha256,
    git_commit,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("hcbench"))
    ap.add_argument("--weights", type=Path, default=None,
                    help="omit for a heuristic-mode smoke run")
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--provenance-tag", default="hcbench-balanced")
    ap.add_argument("--fpr-budget", type=float, default=None,
                    help="switch the objective to: maximize recall subject "
                         "to benign FPR <= this value. Bounded by "
                         "construction; use when --target-recall is not "
                         "attainable on the corpus.")
    ap.add_argument("--exclude-category", nargs="*", default=(),
                    help="categories held OUT of band selection as declared "
                         "out-of-domain (e.g. harmful-content). They are "
                         "still scored and reported on the test split; only "
                         "the calibration objective ignores them. The "
                         "exclusion is recorded in the artifact.")
    ap.add_argument("--allow-infeasible", action="store_true",
                    help="write the policy even when the objective could not "
                         "be met. Off by default: an infeasible objective "
                         "yields a fallback operating point, not a solution.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-repeat-test-eval", action="store_true",
                    help="hcbench-test has already been consumed by a prior "
                         "run of this script; pass this to knowingly repeat "
                         "it (the repeat is recorded in the artifact).")
    args = ap.parse_args()

    # Real one-shot bookkeeping.  The previous `--test-once` flag was
    # `store_true, default=True` and was never read by anything, so it
    # asserted a guarantee it did not provide: hcbench-test was in fact
    # evaluated three times.  Count the passes and record the count.
    ledger_fp = args.out.parent / ".hcbench-test-evaluations.json"
    prior = (json.loads(ledger_fp.read_text()) if ledger_fp.exists()
             else {"passes": []})
    if prior["passes"] and not args.allow_repeat_test_eval:
        print(f"REFUSING: hcbench-test already evaluated "
              f"{len(prior['passes'])}x by this script:")
        for p in prior["passes"]:
            print(f"  - {p}")
        print("Pass --allow-repeat-test-eval to proceed knowingly.")
        return 2

    from defend_hc2 import DEFEND_HC2
    from defend_hc2.provenance import ToolRegistry
    from scripts.calibrate_policy import action_for, detection_metrics, select_policy
    from scripts.eval_hcbench import (
        BENCH_TOOL,
        load_split,
        per_slice_report,
        run_evaluation,
    )

    registry = ToolRegistry()
    registry.register_tool(BENCH_TOOL["name"], BENCH_TOOL["key"],
                           privileged=BENCH_TOOL["privileged"])
    demo_mode = args.weights is None
    if demo_mode:
        print("WARNING: no --weights; heuristic smoke run, NOT results.")
    system = DEFEND_HC2(db_path=":memory:", demo_mode=demo_mode,
                        weights_path=(str(args.weights) if args.weights
                                      else None),
                        tool_registry=registry, master_secret=b"S" * 32)

    cal_rows = load_split(args.data_dir / "hcbench-cal.jsonl")
    test_rows = load_split(args.data_dir / "hcbench-test.jsonl")
    print(f"scoring hcbench-cal n={len(cal_rows)} "
          f"(band selection happens ONLY on this split)…")
    y_cal = [int(r["label"]) for r in cal_rows]
    scores_cal, _ = run_evaluation(cal_rows, system, "hc-bench-policycal")

    # Band selection may deliberately ignore categories declared out of the
    # model's domain.  Those rows stay in the test report; excluding them
    # here only stops an out-of-domain category from dictating the operating
    # point for the categories the system does target.
    excluded = set(args.exclude_category or ())
    keep = [i for i, r in enumerate(cal_rows)
            if r.get("category") not in excluded]
    dropped = len(cal_rows) - len(keep)
    if excluded:
        print(f"band selection excludes {sorted(excluded)}: "
              f"{dropped} of {len(cal_rows)} cal rows held out "
              f"(still reported on test)")
        if not keep:
            print("REFUSING: every calibration row was excluded.")
            return 3
    sel_scores = [scores_cal[i] for i in keep]
    sel_y = [y_cal[i] for i in keep]

    sel = select_policy(sel_scores, sel_y,
                        target_recall=args.target_recall,
                        fpr_budget=args.fpr_budget)
    bands = sel["bands"]
    print(f"objective: {sel['objective']}")
    print(f"selected bands (sanitize/quarantine/reject): {bands}")
    print(f"calibration metrics: {sel['metrics']}")
    if not sel["feasible"]:
        print(f"\n!! {sel['note']}")
        front = sel.get("achievable_frontier") or []
        if front:
            print("\n   what IS achievable on hcbench-cal "
                  "(lowest benign FPR first):")
            print(f"   {'sanitize/quarantine/reject':30s} {'benign FPR':>11s} "
                  f"{'recall':>8s} {'precision':>10s}")
            for f in front:
                b = "/".join(str(x) for x in f["bands"])
                print(f"   {b:30s} {f['benign_fpr']:>11.4f} "
                      f"{f['recall']:>8.4f} {f['precision']:>10.4f}")
            lo = min(f["benign_fpr"] for f in front)
            print(f"\n   lowest reachable benign FPR is {lo:.4f}. "
                  f"Re-run with --fpr-budget {max(lo, 0.0):.2f} or higher.")
        if not args.allow_infeasible:
            print("REFUSING to write a policy from an unmet objective.\n"
                  "   The fallback bands above are a diagnostic, not a "
                  "calibrated operating point; anything measured under them "
                  "inherits the degeneracy.\n"
                  "   Options: --fpr-budget 0.05 for a bounded point, "
                  "--exclude-category <out-of-domain categories>, a lower "
                  "--target-recall, or --allow-infeasible to record this "
                  "point deliberately.")
            return 3
        print("   proceeding anyway (--allow-infeasible); the artifact "
              "records feasible=false.")

    print(f"evaluating hcbench-test ONCE (n={len(test_rows)})…")
    y_test = [int(r["label"]) for r in test_rows]
    scores_test, _ = run_evaluation(test_rows, system, "hc-bench-policytest")
    actions_test = [action_for(s, *bands) for s in scores_test]
    test_metrics = detection_metrics(y_test, actions_test)

    slices: dict[str, list] = defaultdict(lambda: [[], []])
    for y, s, r in zip(y_test, scores_test, test_rows):
        keys = [r["category"], f'{r["category"]}/{r["subtype"]}']
        if r["label"] == 1:
            keys.append(f'{r["category"]}/'
                        + ("lexically_invisible"
                           if r.get("lexically_invisible")
                           else "lexically_visible"))
        for key in keys:
            slices[key][0].append(y)
            slices[key][1].append(s)
    per_slice = {k: per_slice_report(v[0], v[1], bands[1], bands[0])
                 for k, v in sorted(slices.items()) if len(v[0]) >= 20}

    doc = {
        "policy": {"sanitize_at": bands[0], "quarantine_at": bands[1],
                   "reject_at": bands[2]},
        "origin": "hcbench-cal only; production channels",
        "objective": sel["objective"],
        "objective_feasible": sel["feasible"],
        "objective_note": sel["note"],
        "provenance_tag": args.provenance_tag,
        "target_recall": args.target_recall,
        "fpr_budget": args.fpr_budget,
        "calibration_excluded_categories": sorted(excluded),
        "calibration_rows_held_out": dropped,
        "calibration": {"split": "hcbench-cal", "n": len(y_cal),
                        "n_used_for_band_selection": len(sel_y),
                        "base_rate": round(sum(y_cal) / len(y_cal), 4)},
        "calibration metrics": sel["metrics"],
        "hcbench_test_metrics": test_metrics,
        "hcbench_test_pass_number": len(prior["passes"]) + 1,
        "prior_test_passes": prior["passes"],
        "per_slice_test": per_slice,
        "frozen_score_system": "weights + channels identical; bands "
                               "regime-matched only",
        "scoring_mode": ("heuristic-smoke-test-NOT-RESULTS" if demo_mode
                         else "trained-weights"),
        "weights_path": (str(args.weights) if args.weights else None),
        "environment": environment_block(),
        "git_commit": git_commit(Path(__file__).resolve().parents[1]),
    }
    args.out.write_text(json.dumps(doc, indent=2))
    digest = file_sha256(args.out)
    args.out.with_suffix(".sha256").write_text(f"{digest}  {args.out.name}\n")

    prior["passes"].append(
        {"tag": args.provenance_tag, "target_recall": args.target_recall,
         "fpr_budget": args.fpr_budget, "bands": list(bands),
         "objective_feasible": sel["feasible"],
         "excluded_categories": sorted(excluded),
         "out": str(args.out), "sha256": digest})
    ledger_fp.write_text(json.dumps(prior, indent=2))

    print(f"\ntest metrics (once): {test_metrics}")
    for cat, m in per_slice.items():
        print(f"   {cat:40s} r@q={m['recall@quarantine']} "
              f"fpr@q={m['fpr@quarantine']} fpr@s={m['fpr@sanitize']}")
    print(f"\npolicy: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
