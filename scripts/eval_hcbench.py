"""Evaluate HC-Bench (hcbench-test) under BOTH frozen policies.

Surface routing (production channels only — no score reimplementation):
  user_prompt       -> DEFEND_HC2.process_user_message(session, text)
  rag_doc           -> process_user_message(session, neutral query,
                        retrieved_docs=[{doc_id, content}])
  tool_description  -> submit_tool_result(session, registered
                        unprivileged tool, output=text)
  tool_output       -> same as tool_description

Scores: decision.content_risk (production fused).  Bands come from the
two frozen policy files; additionally a threshold is calibrated on
hcbench-cal ONLY (recall@0.95) and evaluated ONCE on hcbench-test —
never trained/selected on test, and the sealed split is untouched here.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow bare run

from defend_hc2.modeling import (
    bootstrap_cis,
    calibrate_thresholds,
    environment_block,
    file_sha256,
    git_commit,
    pr_auc,
    roc_auc,
)

NEUTRAL_QUERY = "Can you summarize this?"
BENCH_TOOL = {"name": "hc-bench-tool", "key": b"t" * 32, "privileged": False}
DEFAULT_CUTS = (0.2, 0.55)  # fallback band tuple (sanitize, quarantine)


def load_split(fp: Path) -> list[dict]:
    with fp.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def score_row(system, session: str, row: dict) -> tuple[float, str]:
    """Production-channel score; returns (content_risk, routing_proof)."""
    surface = row["surface"]
    if surface == "user_prompt":
        pr = system.process_user_message(session, row["text"])
        return pr.decision.content_risk, "process_user_message"
    if surface == "rag_doc":
        pr = system.process_user_message(
            session, NEUTRAL_QUERY,
            retrieved_docs=[{"doc_id": row["id"], "content": row["text"],
                             "source_uri": f"hcbench://{row['source']}/"
                                           f"{row['source_id']}"}])
        comp = pr.decision.component_scores or {}
        assert any(k.startswith("retrieval") and v is not None
                   for k, v in comp.items()), \
            "rag_doc row did not hit the retrieval channel"
        return pr.decision.content_risk, "retrieved_docs"
    prov, dec = system.submit_tool_result(
        session, BENCH_TOOL["name"],
        {"surface": surface, "row_id": row["id"]}, row["text"])
    assert getattr(prov, "verdict", None), "tool row missing provenance verdict"
    return dec.content_risk, f"submit_tool_result[{prov.verdict}]"


def per_slice_report(ys, scores, quarantine_cut, sanitize_cut):
    out = {}
    pos = [(y, s) for y, s in zip(ys, scores) if y == 1]
    neg = [(y, s) for y, s in zip(ys, scores) if y == 0]

    def rate(rows, cut):
        return (round(sum(s >= cut for _, s in rows) / len(rows), 4)
                if rows else None)
    out["recall@quarantine"] = rate(pos, quarantine_cut)
    out["recall@sanitize"] = rate(pos, sanitize_cut)
    out["fpr@quarantine"] = rate(neg, quarantine_cut)
    out["fpr@sanitize"] = rate(neg, sanitize_cut)
    if 0 < len(pos) < len(ys):
        out["roc_auc"] = round(roc_auc(ys, scores), 4)
        out["pr_auc"] = round(pr_auc(ys, scores), 4)
    out["n"] = len(ys)
    out["n_pos"], out["n_neg"] = len(pos), len(neg)
    return out


def run_evaluation(split_rows: list[dict], system, session: str) \
        -> tuple[list[float], list[str]]:
    try:
        system.create_session(
            "HC-Bench evaluation session (neutral assistant).",
            session_id=session)
    except Exception:
        pass  # already exists (in-memory ledger restore path)
    scores, proofs = [], []
    for row in split_rows:
        s, proof = score_row(system, session, row)
        scores.append(float(s))
        proofs.append(proof)
    return scores, proofs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("hcbench"))
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--policies", type=Path, nargs=2, required=True,
                    help="frozen policy JSONs (balanced, high-recall)")
    ap.add_argument("--out", type=Path, default=Path("bench-hcbench-metrics.json"))
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    from defend_hc2 import DEFEND_HC2
    from defend_hc2.provenance import ToolRegistry

    registry = ToolRegistry()
    registry.register_tool(BENCH_TOOL["name"], BENCH_TOOL["key"],
                           privileged=BENCH_TOOL["privileged"])
    system = DEFEND_HC2(db_path=":memory:", demo_mode=False,
                        weights_path=str(args.weights),
                        tool_registry=registry,
                        master_secret=b"S" * 32)

    eval_rows = load_split(args.data_dir / f"hcbench-{args.split}.jsonl")
    cal_rows = load_split(args.data_dir / "hcbench-cal.jsonl")
    print(f"scoring {len(eval_rows)} {args.split} rows + {len(cal_rows)} cal rows "
          f"through production channels…")
    y_eval = [int(r["label"]) for r in eval_rows]
    y_cal = [int(r["label"]) for r in cal_rows]
    scores_eval, _ = run_evaluation(eval_rows, system, "hc-bench")
    scores_cal, _ = run_evaluation(cal_rows, system, "hc-bench-cal")

    cal_thr = calibrate_thresholds(y_cal, scores_cal)["recall@0.95"]
    print(f"hcbench-cal threshold (recall@0.95): {cal_thr:.4f}")

    report = {
        "label": "hcbench_test_production_frozen_policies",
        "rule": "sealed split untouched in this pass",
        "saved_thresholds": {"hcbench_cal_recall@0.95": cal_thr},
        "evaluated_on_test_at_cal_threshold": {
            **per_slice_report(y_eval, scores_eval, cal_thr, cal_thr),
            "n": len(y_eval),
        },
        "policies": {},
        "channels": {"rag_doc": "retrieval component asserted",
                     "tool surfaces": "provenance verdict asserted"},
        "environment": environment_block(),
        "git_commit": git_commit(Path(__file__).resolve().parents[1]),
    }

    for pol_fp in args.policies:
        pol = json.loads(pol_fp.read_text())["policy"]
        name = pol_fp.stem
        san, qua = pol["sanitize_at"], pol["quarantine_at"]
        block = {"bands": {"sanitize_at": san, "quarantine_at": qua,
                           "reject_at": pol["reject_at"]},
                 "overall": per_slice_report(y_eval, scores_eval, qua, san)}
        cat_rows: dict[str, list] = defaultdict(lambda: [[], []])
        surf_rows: dict[str, list] = defaultdict(lambda: [[], []])
        for y, s, r in zip(y_eval, scores_eval, eval_rows):
            cat_rows[r["category"]][0].append(y)
            cat_rows[r["category"]][1].append(s)
            cat_rows[(r["category"], r["subtype"])][0].append(y)
            cat_rows[(r["category"], r["subtype"])][1].append(s)
            surf_rows[r["surface"]][0].append(y)
            surf_rows[r["surface"]][1].append(s)
        block["per_category"] = {
            str(k): per_slice_report(v[0], v[1], qua, san)
            for k, v in sorted(cat_rows.items(), key=lambda kv: str(kv[0]))
            if len(v[0]) >= 20}
        block["per_surface"] = {
            k: per_slice_report(v[0], v[1], qua, san)
            for k, v in sorted(surf_rows.items())}
        if 0 < sum(y_eval) < len(y_eval):
            block["ci95"] = bootstrap_cis(y_eval, scores_eval, cal_thr,
                                          resamples=1000, seed=42)
        report["policies"][name] = block
        ov = block["overall"]
        print(f"\n[{name}] test n={ov['n']} recall@q={ov['recall@quarantine']} "
              f"fpr@q={ov['fpr@quarantine']} fpr@s={ov['fpr@sanitize']} "
              f"auc={ov.get('roc_auc')}")
        for cat, m in block["per_category"].items():
            print(f"   {str(cat):40s} r@q={m['recall@quarantine']} "
                  f"fpr@q={m['fpr@quarantine']} fpr@s={m['fpr@sanitize']}")

    report["file_sha256"] = {}
    args.out.write_text(json.dumps(report, indent=2))
    report["file_sha256"] = {args.out.name: file_sha256(args.out)}
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
