"""Policy-threshold calibration on VALIDATION data only (spec Phase 10).

Classifier calibration ≠ policy calibration: this script runs the full
``engine.process_user_message`` pipeline on calibration rows, collects every
channel value + fused risk + action, sweeps the candidate band grid with a
**predeclared objective** (default: maximize detection precision subject to
detection recall >= target, with benign FPR reported for every candidate),
selects one predeclared policy, and THEN evaluates it once on the
development-test split (development_test_previously_observed — inspected
during development; not a blind holdout).

For SPML rows the ``system_prompt`` field is honoured — one engine session
per distinct system prompt, so context-dependent channels stay honest.

Selection logic (:func:`select_policy`) is pure and unit-tested in-repo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow bare run

from defend_hc2.modeling import environment_block, git_commit

_ACTIONS = ("ALLOW", "SANITIZE_AND_ALLOW", "QUARANTINE", "REJECT")
_DEFAULT_SANITIZE = [0.15, 0.20, 0.25, 0.30]
_DEFAULT_QUARANTINE = [0.40, 0.45, 0.50, 0.55]
_DEFAULT_REJECT = [0.70, 0.75, 0.80, 0.85]


def action_for(risk: float, sanitize: float, quarantine: float, reject: float) -> str:
    if risk >= reject:
        return "REJECT"
    if risk >= quarantine:
        return "QUARANTINE"
    if risk >= sanitize:
        return "SANITIZE_AND_ALLOW"
    return "ALLOW"


def sweep_grid(san=_DEFAULT_SANITIZE, quar=_DEFAULT_QUARANTINE, rej=_DEFAULT_REJECT):
    return [(s, q, r) for s in san for q in quar for r in rej if s < q < r]


def detection_metrics(gold: list[int], actions: list[str]) -> dict:
    """Detection = action != ALLOW.  Reports per-action distribution."""
    dist = {a: 0 for a in _ACTIONS}
    for a in actions:
        dist[a] += 1
    tp = sum(g and a != "ALLOW" for g, a in zip(gold, actions))
    fp = sum((not g) and a != "ALLOW" for g, a in zip(gold, actions))
    fn = sum(g and a == "ALLOW" for g, a in zip(gold, actions))
    tn = sum((not g) and a == "ALLOW" for g, a in zip(gold, actions))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {
        "precision": round(prec, 4), "recall": round(rec, 4),
        "benign_fpr": round(fp / max(1, fp + tn), 4),
        "action_distribution": dist,
    }


def select_policy(
    risks: list[float],
    gold: list[int],
    target_recall: float = 0.95,
    grid=None,
) -> dict:
    """Predeclared objective: maximize detection PRECISION subject to
    detection recall >= target.  Ties broken by lowest benign FPR, then by
    the most conservative (highest) bands.  Degenerate policies that flag
    everything are implicitly avoided by the precision term."""
    grid = grid or sweep_grid()
    best = None
    for bands in grid:
        actions = [action_for(risk, *bands) for risk in risks]
        m = detection_metrics(gold, actions)
        if m["recall"] < target_recall - 1e-9:
            continue
        key = (m["precision"], -m["benign_fpr"], bands[2], bands[1], bands[0])
        if best is None or key > best[0]:
            best = (key, bands, m)
    if best is None:
        # fallback: lowest-FPR policy reaching highest achievable recall
        feasible = None
        for bands in grid:
            actions = [action_for(risk, *bands) for risk in risks]
            m = detection_metrics(gold, actions)
            key = (m["recall"], -m["benign_fpr"])
            if feasible is None or key > feasible[0]:
                feasible = (key, bands, m)
        _, bands, m = feasible
        return {"bands": bands, "metrics": m, "note":
                "target recall infeasible; picked highest-recall lowest-FPR policy"}
    _, bands, m = best
    return {"bands": bands, "metrics": m, "note":
            f"max precision s.t. recall >= {target_recall} (calibration data)"}


def run_rows(engine, rows: list[dict]) -> list[dict]:
    """Full-pipeline trace per row (channels + fused risk + action)."""
    from collections import defaultdict

    by_system = defaultdict(list)
    for i, row in enumerate(rows):
        by_system[row.get("system_prompt") or "You are SupportBot for Acme Corp."].append((i, row))
    out: list[dict] = [None] * len(rows)  # type: ignore[list-item]
    for system_prompt, bucket in by_system.items():
        sid = engine.create_session(system_prompt=system_prompt)["session_id"]
        for i, row in bucket:
            res = engine.process_user_message(sid, row["text"])
            comp = res.decision.component_scores
            out[i] = {
                "gold": int(row["label"]),
                "injection_score": comp.get("injection_score"),
                "lexical_score": comp.get("lexical_score"),
                "retrieval_score": comp.get("retrieval_injection_score"),
                "mismatch_score": comp.get("intent_context_mismatch_score"),
                "drift_score": comp.get("conversation_drift_score"),
                "fused_content_risk": res.decision.content_risk,
                "action": res.decision.action,
            }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("bench-data"),
                   help="layout dir used by --cal-target presets")
    p.add_argument("--cal-target", choices=["balanced", "high-recall"],
                   default="balanced",
                   help="calibration regime (FLAW-3): 'balanced' selects "
                        "slp-cal.jsonl (~50%% injection → FPR-controlled "
                        "policy for S-Labs-like traffic); 'high-recall' "
                        "selects spml-cal.jsonl (~78%% injection → "
                        "aggressive policy for high-attack-rate traffic). "
                        "Ignored when --cal-file is given explicitly.")
    p.add_argument("--cal-file", type=Path, nargs="+", default=None,
                   help="override: explicit calibration file(s); wins over "
                        "--cal-target")
    p.add_argument("--eval-file", type=Path, nargs="+", default=None)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--target-recall", type=float, default=0.95,
                   help="predeclared objective constraint")
    p.add_argument("--out", type=Path, default=Path("calibrated-policy.json"))
    args = p.parse_args()

    if args.cal_file is None:
        preset = {"balanced": "slp-cal.jsonl", "high-recall": "spml-cal.jsonl"}[
            args.cal_target]
        args.cal_file = [args.data_dir / preset]
        print(f"--cal-target {args.cal_target}: calibrating on "
              f"{args.cal_file[0]} (explicit --cal-file would override)")

    import os
    import tempfile

    from defend_hc2 import DEFEND_HC2, PolicyEngine

    def _load(fp: Path) -> list[dict]:
        out: list[dict] = []
        with fp.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    out.append({"text": str(r["text"]), "label": int(r["label"]),
                                "system_prompt": r.get("system_prompt")})
        return out

    rows_full = [row for fp in args.cal_file for row in _load(fp)]

    tmp_db = Path(tempfile.mkdtemp()) / "policy-cal.db"
    engine = DEFEND_HC2(db_path=str(tmp_db), demo_mode=False,
                        weights_path=str(args.weights))
    cal_trace = run_rows(engine, rows_full)
    cal_risks = [t["fused_content_risk"] or 0.0 for t in cal_trace]
    cal_gold = [t["gold"] for t in cal_trace]

    result = select_policy(cal_risks, cal_gold, target_recall=args.target_recall)
    bands = result["bands"]
    print(f"\nselected policy bands (sanitize/quarantine/reject): {bands}")
    print(f"calibration: {result['note']}")
    print(f"calibration metrics: {result['metrics']}")

    cal_base_rate = round(sum(cal_gold) / max(len(cal_gold), 1), 4)
    out = {
        "policy": {"sanitize_at": bands[0], "quarantine_at": bands[1], "reject_at": bands[2]},
        "origin": "calibration data only; bands predeclared before evaluation",
        "objective": f"max precision s.t. recall >= {args.target_recall}",
        "calibration": {"target": args.cal_target,
                        "files": [str(fp) for fp in args.cal_file],
                        "base_rate": cal_base_rate},
        "calibration_metrics": result["metrics"],
        "environment": environment_block(),
        "git_commit": git_commit(Path(__file__).resolve().parents[1]),
    }

    if args.eval_file:
        test_rows: list[dict] = []
        for fp in args.eval_file:
            with fp.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        test_rows.append({"text": str(r["text"]), "label": int(r["label"]),
                                          "system_prompt": r.get("system_prompt")})
        engine.policy = PolicyEngine(reject_at=bands[2], quarantine_at=bands[1],
                                     sanitize_at=bands[0])
        test_trace = run_rows(engine, test_rows)
        test_actions = [t["action"] for t in test_trace]
        # BUG-E: these evaluation files were observed during development —
        # they are development-test metrics, never a blind final holdout.
        out["policy_eval_metrics"] = detection_metrics(
            [t["gold"] for t in test_trace], test_actions)
        out["policy_eval_claim"] = (
            "development_test_previously_observed — evaluated once with "
            "predeclared bands, but the files were inspected during "
            "development; not a blind holdout")
        print(f"\npolicy on development test (evaluated ONCE, previously "
              f"observed): {out['policy_eval_metrics']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\npolicy: {args.out}")
    for suffix in ("", "-wal", "-shm"):
        stale = str(tmp_db) + suffix
        if os.path.exists(stale):
            os.remove(stale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
