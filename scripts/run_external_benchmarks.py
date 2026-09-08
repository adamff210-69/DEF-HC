"""Exp-H: named external benchmarks through the FROZEN production path.

Scores AdvBench, JBB-Behaviors (harmful + benign), and XSTest v2 with
the production analyzer + the frozen calibrated policy bands.  Zero
retraining, zero threshold changes: every number is the frozen
system applied to data it has never seen in any split.

Metrics (development_test-style label upgraded to
external_public_not_blind — the corpora are public; the project had no
prior exposure, but we do not claim benchmark blindness):

  * per-benchmark ROC AUC where both labels exist (JBB pairs, XSTest)
  * paired zero-shot AUCs: advbench vs xstest-safe / jbb-benign
  * detection recall at the frozen QUARANTINE cutoff (positives)
  * benign FPR at the frozen SANITIZE and QUARANTINE cutoffs —
    on XSTest safe rows this IS the over-refusal measurement
  * action histogram from scripts/calibrate_policy.action_for with the
    frozen bands (policy lineage, no re-derivation)

Any XSTest safe row flagged at or above QUARANTINE is dumped verbatim
(over-refusal per-example evidence, same honesty rule as Exp-F).

Example:
    python scripts/run_external_benchmarks.py \\
        --data-dir bench-data-ext --artifacts /kaggle/working \\
        --weights weights/bge-final.json \\
        --policy calibrated-policy-balanced.json --out-dir bench-out
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow bare run

from defend_hc2.modeling import environment_block, file_sha256, git_commit, roc_auc

LABEL = "external_public_not_blind"


def _load_rows(fp: Path) -> tuple[list[str], list[int], list[dict]]:
    texts, labels, metas = [], [], []
    with fp.open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            texts.append(rec["text"])
            labels.append(int(rec["label"]))
            metas.append(rec.get("meta") or {})
    return texts, labels, metas


def _flag_rates(ys: list[int], risks: list[float], cut: float) -> dict:
    pos = [r for g, r in zip(ys, risks) if g == 1]
    neg = [r for g, r in zip(ys, risks) if g == 0]
    return {
        "detection_recall": (round(sum(r >= cut for r in pos) / len(pos), 4)
                             if pos else None),
        "benign_fpr": (round(sum(r >= cut for r in neg) / len(neg), 4)
                       if neg else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("bench-data-ext"))
    ap.add_argument("--artifacts", type=Path, default=Path("."))
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("bench-out"))
    args = ap.parse_args()

    from defend_hc2 import ContentRiskAnalyzer
    from scripts.calibrate_policy import action_for
    from scripts.run_experiments import recovery_risk_for_text

    started = time.time_ns()
    pol = json.loads(args.policy.read_text())["policy"]
    sanitize, quarantine, reject = (pol["sanitize_at"], pol["quarantine_at"],
                                    pol["reject_at"])
    analyzer = ContentRiskAnalyzer(demo_mode=False, weights_path=str(args.weights))
    print(f"frozen policy bands: sanitize={sanitize} quarantine={quarantine} "
          f"reject={reject}  (from {args.policy.name})")

    scored: dict[str, dict] = {}
    for fp in sorted(args.data_dir.glob("*.jsonl")):
        bench = fp.stem
        texts, ys, metas = _load_rows(fp)
        risks = [recovery_risk_for_text(analyzer, t) for t in texts]
        actions = [action_for(r, sanitize, quarantine, reject) for r in risks]
        hist = {a: actions.count(a) for a in
                ("ALLOW", "SANITIZE_AND_ALLOW", "QUARANTINE", "REJECT")}
        entry = {
            "n": len(texts),
            "n_positive": sum(ys), "n_benign": len(ys) - sum(ys),
            "roc_auc": (round(roc_auc(ys, risks), 4)
                        if 0 < sum(ys) < len(ys) else None),
            "at_quarantine_cutoff": _flag_rates(ys, risks, quarantine),
            "at_sanitize_cutoff": _flag_rates(ys, risks, sanitize),
            "action_histogram": hist,
            "risk_mean": round(sum(risks) / len(risks), 4),
        }
        scored[bench] = entry
        # over-refusal dump: every benign row flagged at/above quarantine
        if any(g == 0 and r >= quarantine for g, r in zip(ys, risks)):
            dump = args.out_dir / (
                f"exp-h-{bench}-overrefusal-{args.policy.stem}.jsonl")
            args.out_dir.mkdir(parents=True, exist_ok=True)
            with dump.open("w", encoding="utf-8") as fh:
                for t, g, r, m in zip(texts, ys, risks, metas):
                    if g == 0 and r >= quarantine:
                        fh.write(json.dumps({"text": t[:400], "label": g,
                                             "risk": round(float(r), 6),
                                             "meta": m}) + "\n")
            entry["overrefusal_dump"] = dump.name
        print(f"{bench:<12} n={len(texts):>4} AUC={entry['roc_auc']} "
              f"recall@quarantine={entry['at_quarantine_cutoff']['detection_recall']} "
              f"benignFPR@quarantine={entry['at_quarantine_cutoff']['benign_fpr']} "
              f"benignFPR@sanitize={entry['at_sanitize_cutoff']['benign_fpr']}")

    # paired zero-shot AUCs (positive-only corpora against benign partners)
    def _pair(pos_bench: str, neg_bench: str) -> tuple[dict, float] | None:
        p = scored.get(pos_bench); n = scored.get(neg_bench)
        if not p or not n:
            return None
        texts_p, ys_p, _ = _load_rows(args.data_dir / f"{pos_bench}.jsonl")
        texts_n, ys_n, _ = _load_rows(args.data_dir / f"{neg_bench}.jsonl")
        risks = [recovery_risk_for_text(analyzer, t) for t in texts_p + texts_n]
        return {"positives": pos_bench, "benign_partner": neg_bench}, \
            round(roc_auc(ys_p + ys_n, risks), 4)
    pairs = {}
    for spec in (("advbench", "xstest"), ("advbench", "jbb-benign"),
                 ("jbb-harmful", "jbb-benign")):
        got = _pair(*spec)
        if got:
            meta, auc = got
            pairs[f"{spec[0]}__vs__{spec[1]}"] = {"roc_auc": auc, **meta}

    summary = {
        "experiment": "exp-h named external benchmarks (frozen system)",
        "label": LABEL,
        "config": {"weights": str(args.weights), "policy": str(args.policy),
                   "bands": {"sanitize_at": sanitize,
                             "quarantine_at": quarantine, "reject_at": reject}},
        "benchmarks": scored, "paired_auc": pairs,
        "environment": environment_block(),
        "git_commit": git_commit(Path(__file__).resolve().parents[1]),
        "runtime_ns": time.time_ns() - started,
        "file_sha256": {},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "bench-metrics-exp-h-external.json"
    out.write_text(json.dumps(summary, indent=2))
    summary["file_sha256"] = {out.name: file_sha256(out)}
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
