"""Aggregate every evaluation artifact into the spec's FINAL REPORT format.

Reads what exists under an artifacts directory and prints the mandated
``DEF-HC2 FINAL EVALUATION`` block; any missing experiment is reported as
``n/a`` honestly — sections are never fabricated.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow bare run


def _j(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _fmt_metrics(m: dict | None, indent: str) -> list[str]:
    if not isinstance(m, dict):
        return [f"{indent}n/a (artifact missing)"]
    keys = ("roc_auc", "pr_auc", "precision", "recall", "f1", "f2",
            "balanced_accuracy", "mcc", "fpr", "threshold")
    lines = [f"{indent}{k}: {m.get(k)}" for k in keys]
    ci = m.get("ci95")
    if ci:
        lines.append(f"{indent}95% CIs: {ci}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path, required=True,
                    help="dir containing bench-metrics-*.json, scores-*.jsonl, weights/")
    ap.add_argument("--repo", type=Path, default=Path("."))
    args = ap.parse_args()
    art = args.artifacts

    out = ["DEF-HC2 FINAL EVALUATION", "========================", ""]
    out += ["Code:"]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, cwd=str(args.repo), timeout=10).stdout.strip()
    except Exception:
        commit = "n/a"
    out += [f"    git commit: {commit}",
            "    tests: see pytest output in the run log (not parsed here)", ""]

    exp_files = sorted(art.rglob("bench-metrics-exp-*.json"))
    out += ["Experiments:"]
    for f in exp_files:
        out.append(f"  {f.stem}:")
        out += _fmt_metrics(_j(f), "      ")
    if not exp_files:
        out.append("  (no exp metrics found — run scripts/run_experiments.py)")
    out.append("")

    pol = _j(art / "calibrated-policy.json") if (art / "calibrated-policy.json").exists() else None
    out += ["Policy:"]
    if pol:
        p = pol["policy"]
        out += [f"    sanitize: {p['sanitize_at']}  quarantine: {p['quarantine_at']}  "
                f"reject: {p['reject_at']}",
                f"    origin: {pol.get('origin')}",
                f"    calibration metrics: {pol.get('calibration_metrics')}",
                f"    frozen test metrics: {pol.get('frozen_policy_test_metrics')}"]
    else:
        out.append("    n/a (run scripts/calibrate_policy.py)")
    out.append("")

    out += ["Artifacts:"]
    for pattern in ("weights/*.json", "*.jsonl", "*.png", "*.tar.gz"):
        for f in sorted(art.glob(pattern)):
            out.append(f"    {f.name} ({f.stat().st_size} B)")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
