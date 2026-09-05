"""Aggregate every evaluation artifact into the spec's FINAL REPORT format.

Reads what exists under an artifacts directory (experiments under
``--exp-dir``, default ``<artifacts>/bench-out``) and prints the mandated
``DEF-HC2 FINAL EVALUATION`` block; any missing experiment is reported as
``n/a`` honestly — sections are never fabricated.

Every metric derived from pi-test.jsonl or any SPML test split is labeled
``development_test_previously_observed`` (BUG-E): those files were
inspected during development iterations; no blind final holdout exists.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow bare run

HONESTY_PARAGRAPH = (
    "No genuinely blind final holdout currently exists. All metrics in this\n"
    "report are development/post-hoc estimates. A true final claim requires a\n"
    "dataset never previously inspected in this project."
)

_DEV_LABEL = "development_test_previously_observed"


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


def _fmt_exp_f(rob: dict, indent: str = "      ") -> list[str]:
    """Obfuscation table (BUG-D/Step 6): clean vs perturbed vs recovery,
    with recovery-aware caveats from defend_hc2.reporting — restoration
    disproves the bug hypothesis, known limitations are labeled as such,
    and WARNING is reserved for genuinely unexplained anomalies."""
    from defend_hc2.reporting import transform_caveat_lines

    clean = (rob.get("clean") or {}).get("roc_auc")
    lines = [f"{indent}clean AUC: {clean}",
             f"{indent}{'transform':<14}{'perturbed AUC':>14}{'recovery AUC':>14}"]
    for name, row in sorted((rob.get("per_transform") or {}).items()):
        if not isinstance(row, dict) or "perturbed_auc" not in row:
            lines.append(f"{indent}{name:<14}{'n/a':>14}{'n/a':>14}")
            continue
        lines.append(f"{indent}{name:<14}{row.get('perturbed_auc')!s:>14}"
                     f"{row.get('recovery_auc')!s:>14}")
        lines += transform_caveat_lines(
            name, row.get("perturbed_auc"), row.get("recovery_auc"),
            dump_name=f"exp-f-{name}-examples.jsonl", indent=indent)
    return lines


def _fmt_exp_g(g: dict, indent: str = "      ") -> list[str]:
    """Exp-G augmentation record: clean vs letter-spaced + gate ledger."""
    lines = [f"{indent}({g.get('experiment', 'exp-g')})"]
    for name, key in (("clean", "clean_dev_test"),
                      ("letter-spaced", "letterspaced_dev_test")):
        m = g.get(key) or {}
        lines.append(f"{indent}{name}: AUC={m.get('roc_auc')} "
                     f"P={m.get('precision')} R={m.get('recall')} "
                     f"FPR={m.get('fpr')}")
    for gname, gate in (g.get("gates") or {}).items():
        verdict = ("PASS" if gate.get("pass") is True
                   else "FAIL" if gate.get("pass") is False else "info")
        lines.append(f"{indent}gate {gname}: value={gate.get('value')} "
                     f"baseline={gate.get('baseline', '—')} {verdict}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path, required=True,
                    help="dir containing scores-*.jsonl, weights/, calibrated-policy.json")
    ap.add_argument("--exp-dir", type=Path, default=None,
                    help="dir containing bench-metrics-exp-*.json "
                         "(default <artifacts>/bench-out)")
    ap.add_argument("--repo", type=Path, default=Path("."))
    args = ap.parse_args()
    art = args.artifacts
    exp_dir = args.exp_dir if args.exp_dir else art / "bench-out"

    out = ["DEF-HC2 FINAL EVALUATION", "========================", ""]
    out += ["Code:"]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, cwd=str(args.repo), timeout=10).stdout.strip()
    except Exception:
        commit = "n/a"
    out += [f"    git commit: {commit}",
            "    tests: see pytest output in the run log (not parsed here)", ""]

    exp_files = sorted(exp_dir.glob("bench-metrics-exp-*.json"))
    out += [f"Experiments: (from {exp_dir})"]
    for f in exp_files:
        blob = _j(f)
        out.append(f"  {f.stem}:  [{_DEV_LABEL}]")
        if f.stem == "bench-metrics-exp-f" and isinstance(blob, dict):
            out += _fmt_exp_f(blob)
        elif "gates" in blob and "clean_dev_test" in blob:
            out += _fmt_exp_g(blob)
        else:
            out += _fmt_metrics(blob, "      ")
    if not exp_files:
        out.append("  (no exp metrics found — run scripts/run_experiments.py)")
    out.append("")

    pol_files = sorted(art.glob("calibrated-policy*.json"))
    out += ["Policy:"]
    if pol_files:
        for fp in pol_files:
            pol = _j(fp)
            if not isinstance(pol, dict) or "policy" not in pol:
                continue
            p = pol["policy"]
            eval_metrics = (pol.get("policy_eval_metrics")
                            or pol.get("frozen_policy_test_metrics"))
            out += [f"  {fp.name}:",
                    f"    sanitize: {p['sanitize_at']}  quarantine: {p['quarantine_at']}  "
                    f"reject: {p['reject_at']}",
                    f"    origin: {pol.get('origin')}",
                    f"    calibration: {pol.get('calibration')}",
                    f"    calibration metrics: {pol.get('calibration_metrics')}",
                    f"    development test metrics [{_DEV_LABEL}]: {eval_metrics}"]
    else:
        out.append("    n/a (run scripts/calibrate_policy.py)")
    out.append("")

    out += ["Labeling honesty (BUG-E):", f"    {HONESTY_PARAGRAPH}", ""]

    out += ["Artifacts:"]
    for pattern in ("weights/*.json", "*.jsonl", "*.png", "*.tar.gz"):
        for f in sorted(art.glob(pattern)):
            out.append(f"    {f.name} ({f.stat().st_size} B)")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
