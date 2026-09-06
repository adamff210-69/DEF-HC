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
_DEFAULT_SANITIZE = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
_DEFAULT_QUARANTINE = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
_DEFAULT_REJECT = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


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
    fpr_budget: float | None = None,
) -> dict:
    """Choose operating bands under a predeclared objective.

    Two objectives are supported and the choice is recorded in the result:

    ``recall-target`` (default)
        Maximize detection PRECISION subject to recall >= ``target_recall``.

    ``fpr-budget`` (when ``fpr_budget`` is not None)
        Maximize detection RECALL subject to benign FPR <= ``fpr_budget``.
        Use this when the recall target is not attainable on the corpus: it
        is bounded by construction and cannot select a flag-everything
        policy, whereas a recall-first fallback can and does.

    The returned dict always carries ``feasible``.  When it is ``False`` the
    constraint could not be met by any policy on the grid and the bands are
    a *fallback*, not a solution.  Callers must propagate that flag; a
    fallback silently reported as a solution is how a degenerate operating
    point reaches a results table.
    """
    grid = grid or sweep_grid()
    scored = [(bands, detection_metrics(gold, [action_for(r, *bands) for r in risks]))
              for bands in grid]

    def _frontier(n=10):
        """Best attainable (FPR, recall) points, so an infeasible request can
        be answered with what IS reachable instead of just 'no'."""
        seen, out = set(), []
        for b, m in sorted(scored, key=lambda x: (x[1]["benign_fpr"],
                                                  -x[1]["recall"])):
            key = (m["benign_fpr"], m["recall"])
            if key in seen:
                continue
            seen.add(key)
            out.append({"bands": list(b), "benign_fpr": m["benign_fpr"],
                        "recall": m["recall"], "precision": m["precision"]})
            if len(out) >= n:
                break
        return out

    if fpr_budget is not None:
        objective = f"max recall s.t. benign FPR <= {fpr_budget}"
        ok = [(b, m) for b, m in scored if m["benign_fpr"] <= fpr_budget + 1e-9]
        if ok:
            bands, m = max(ok, key=lambda x: (x[1]["recall"], x[1]["precision"],
                                              x[0][2], x[0][1], x[0][0]))
            return {"bands": bands, "metrics": m, "feasible": True,
                    "objective": objective,
                    "note": f"{objective} (calibration data)"}
        # No policy respects the budget: the least-bad point is the lowest
        # FPR available.  Still a fallback -- flagged as such.
        bands, m = min(scored, key=lambda x: (x[1]["benign_fpr"], -x[1]["recall"]))
        return {"bands": bands, "metrics": m, "feasible": False,
                "objective": objective,
                "achievable_frontier": _frontier(),
                "note": (f"INFEASIBLE: no policy achieves benign FPR <= "
                         f"{fpr_budget}; the lowest reachable benign FPR on "
                         f"this grid is {m['benign_fpr']} "
                         f"(recall {m['recall']})")}

    objective = f"max precision s.t. recall >= {target_recall}"
    ok = [(b, m) for b, m in scored if m["recall"] >= target_recall - 1e-9]
    if ok:
        bands, m = max(ok, key=lambda x: (x[1]["precision"], -x[1]["benign_fpr"],
                                          x[0][2], x[0][1], x[0][0]))
        return {"bands": bands, "metrics": m, "feasible": True,
                "objective": objective,
                "note": f"{objective} (calibration data)"}

    # Recall target unreachable.  The old fallback maximized recall first,
    # which drives straight to the most aggressive bands on the grid and
    # yields a flag-everything policy with a huge benign FPR -- and, because
    # every infeasible target lands on the same point, it also makes two
    # different targets produce identical policies.  Report the best
    # attainable recall for diagnosis, but mark the result infeasible.
    bands, m = max(scored, key=lambda x: (x[1]["recall"], -x[1]["benign_fpr"]))
    return {"bands": bands, "metrics": m, "feasible": False,
            "objective": objective,
            "achievable_frontier": _frontier(),
            "note": (f"INFEASIBLE: recall >= {target_recall} unreachable on "
                     f"this calibration split; best attainable recall is "
                     f"{m['recall']} at benign FPR {m['benign_fpr']}. "
                     f"Re-run with --fpr-budget for a bounded operating "
                     f"point, and/or exclude out-of-domain categories from "
                     f"band selection.")}


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
