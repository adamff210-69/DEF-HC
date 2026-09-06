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
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--provenance-tag", default="hcbench-balanced")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--test-once", action="store_true", default=True)
    args = ap.parse_args()

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
    system = DEFEND_HC2(db_path=":memory:", demo_mode=False,
                        weights_path=str(args.weights),
                        tool_registry=registry, master_secret=b"S" * 32)

    cal_rows = load_split(args.data_dir / "hcbench-cal.jsonl")
    test_rows = load_split(args.data_dir / "hcbench-test.jsonl")
    print(f"scoring hcbench-cal n={len(cal_rows)} "
          f"(band selection happens ONLY on this split)…")
    y_cal = [int(r["label"]) for r in cal_rows]
    scores_cal, _ = run_evaluation(cal_rows, system, "hc-bench-policycal")

    sel = select_policy(scores_cal, y_cal, target_recall=args.target_recall)
    bands = sel["bands"]
    print(f"selected bands (sanitize/quarantine/reject): {bands}  "
          f"[{sel['note']}]")
    print(f"calibration metrics: {sel['metrics']}")

    print(f"evaluating hcbench-test ONCE (n={len(test_rows)})…")
    y_test = [int(r["label"]) for r in test_rows]
    scores_test, _ = run_evaluation(test_rows, system, "hc-bench-policytest")
    actions_test = [action_for(s, *bands) for s in scores_test]
    test_metrics = detection_metrics(y_test, actions_test)

    slices: dict[str, list] = defaultdict(lambda: [[], []])
    for y, s, r in zip(y_test, scores_test, test_rows):
        for key in (r["category"], (r["category"], r["subtype"])):
            slices[key][0].append(y)
            slices[key][1].append(s)
    per_slice = {str(k): per_slice_report(v[0], v[1], bands[1], bands[0])
                 for k, v in sorted(slices.items(), key=lambda kv: str(kv[0]))
                 if len(v[0]) >= 20}

    doc = {
        "policy": {"sanitize_at": bands[0], "quarantine_at": bands[1],
                   "reject_at": bands[2]},
        "origin": "hcbench-cal only; bands predeclared objective "
                  "max-precision s.t. recall target; production channels",
        "provenance_tag": args.provenance_tag,
        "target_recall": args.target_recall,
        "calibration": {"split": "hcbench-cal", "n": len(y_cal),
                        "base_rate": round(sum(y_cal) / len(y_cal), 4)},
        "calibration metrics": sel["metrics"],
        "hcbench test metrics (evaluated once)": test_metrics,
        "per_slice_test": per_slice,
        "frozen_score_system": "weights + channels identical; bands "
                               "regime-matched only",
        "environment": environment_block(),
        "git_commit": git_commit(Path(__file__).resolve().parents[1]),
    }
    args.out.write_text(json.dumps(doc, indent=2))
    doc["file_sha256"] = {args.out.name: file_sha256(args.out)}
    args.out.write_text(json.dumps(doc, indent=2))

    print(f"\ntest metrics (once): {test_metrics}")
    for cat, m in per_slice.items():
        print(f"   {cat:40s} r@q={m['recall@quarantine']} "
              f"fpr@q={m['fpr@quarantine']} fpr@s={m['fpr@sanitize']}")
    print(f"\npolicy: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
