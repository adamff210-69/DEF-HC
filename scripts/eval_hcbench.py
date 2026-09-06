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
        # Explicit raise, not `assert`: routing proofs must survive -O.
        if not any(k.startswith("retrieval") and v is not None
                   for k, v in comp.items()):
            raise RuntimeError(
                f"rag_doc row {row['id']!r} did not hit the retrieval "
                f"channel (components: {sorted(comp)})")
        return pr.decision.content_risk, "retrieved_docs"
    prov, dec = system.submit_tool_result(
        session, BENCH_TOOL["name"],
        {"surface": surface, "row_id": row["id"]}, row["text"])
    if not getattr(prov, "verdict", None):
        raise RuntimeError(f"tool row {row['id']!r} missing provenance verdict")
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
    # Explicit existence check rather than a blanket `except Exception`,
    # which previously masked genuine construction failures.
    if system.ledger.get_session(session) is None:
        system.create_session(
            "HC-Bench evaluation session (neutral assistant).",
            session_id=session)
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
    ap.add_argument("--split", default="test", choices=("cal", "test"),
                    help="NEVER 'sealed' — the sealed split is reachable "
                         "only through scripts/eval_sealed.py, which "
                         "enforces the one-shot guard.")
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
        # NOTE on reading these tables: `overall.fpr@*` is computed over ALL
        # negative rows.  The per_category "benign" entry covers only rows
        # whose *category* is benign — some negatives live inside attack
        # categories (e.g. deepset ships benign rows under `injection`), so
        # the two FPR figures legitimately differ.  Both are reported.
        cat_rows: dict = defaultdict(lambda: [[], []])
        sub_rows: dict = defaultdict(lambda: [[], []])
        surf_rows: dict = defaultdict(lambda: [[], []])
        diff_rows: dict = defaultdict(lambda: [[], []])
        for y, s, r in zip(y_eval, scores_eval, eval_rows):
            cat_rows[r["category"]][0].append(y)
            cat_rows[r["category"]][1].append(s)
            sub_rows[f'{r["category"]}/{r["subtype"]}'][0].append(y)
            sub_rows[f'{r["category"]}/{r["subtype"]}'][1].append(s)
            surf_rows[r["surface"]][0].append(y)
            surf_rows[r["surface"]][1].append(s)
            if r["label"] == 1:
                key = ("lexically_invisible" if r.get("lexically_invisible")
                       else "lexically_visible")
                diff_rows[f'{r["category"]}/{key}'][0].append(y)
                diff_rows[f'{r["category"]}/{key}'][1].append(s)

        def _table(d, min_n):
            """Slices >= min_n, PLUS a rollup of everything below it so no
            rows silently vanish from the report (the small-slice tail was
            previously invisible and skewed how aggregates read)."""
            big = {k: per_slice_report(v[0], v[1], qua, san)
                   for k, v in sorted(d.items()) if len(v[0]) >= min_n}
            small_y, small_s, small_k = [], [], []
            for k, v in sorted(d.items()):
                if len(v[0]) < min_n:
                    small_y += v[0]; small_s += v[1]; small_k.append(k)
            if small_y:
                big[f"__rollup_of_{len(small_k)}_slices_under_{min_n}"] = {
                    **per_slice_report(small_y, small_s, qua, san),
                    "slices": small_k}
            return big

        block["per_category"] = _table(cat_rows, 20)
        block["per_subtype"] = _table(sub_rows, 20)
        block["per_surface"] = _table(surf_rows, 1)
        block["per_category_by_lexical_visibility"] = _table(diff_rows, 20)
        if 0 < sum(y_eval) < len(y_eval):
            block["ci95"] = bootstrap_cis(y_eval, scores_eval, cal_thr,
                                          resamples=1000, seed=42)
        report["policies"][name] = block
        ov = block["overall"]
        print(f"\n[{name}] test n={ov['n']} recall@q={ov['recall@quarantine']} "
              f"fpr@q={ov['fpr@quarantine']} fpr@s={ov['fpr@sanitize']} "
              f"auc={ov.get('roc_auc')}")
        for title, key in (("per category", "per_category"),
                           ("by lexical visibility",
                            "per_category_by_lexical_visibility")):
            print(f"  -- {title} --")
            for cat, m in block[key].items():
                print(f"   {str(cat):46s} n={m['n']:<5d} "
                      f"r@q={m['recall@quarantine']} "
                      f"fpr@q={m['fpr@quarantine']} fpr@s={m['fpr@sanitize']}")

    # sha256 of the artifact is recorded in a SIDECAR file: embedding the
    # digest into the document it hashes makes the recorded value refer to
    # a file that no longer exists on disk (it was the pre-insertion form).
    args.out.write_text(json.dumps(report, indent=2))
    digest = file_sha256(args.out)
    args.out.with_suffix(".sha256").write_text(
        f"{digest}  {args.out.name}\n")
    print(f"\nwrote {args.out}")
    print(f"sha256 {digest}  -> {args.out.with_suffix('.sha256').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
