"""Baseline comparison — the table a reviewer asks for first.

DEF-HC has never been compared against a published guard model.  Its only
baselines are its own lexical/demo-fusion paths, which §0.1 reports as
collapsing onto the always-positive dummy.  "Our model 0.9851 vs two
dummies" is not a comparison.

This scores the SAME rows through published detectors and through the
DEF-HC production path, under the SAME protocol: every threshold is
calibrated on hcbench-cal only and applied once to hcbench-test.  No
baseline is disadvantaged — they get exactly the treatment DEF-HC gets.

Gated models are skipped with a verbatim reason and never substituted,
matching scripts/build_hcbench.py discipline.

    python scripts/run_baselines.py --data-dir hcbench \
        --weights weights/bge-final.json --out bench-out/bench-baselines.json

Add --hf-token to unlock the Meta Prompt Guard 2 models.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defend_hc2.modeling import (
    environment_block,
    file_sha256,
    git_commit,
    pr_auc,
    roc_auc,
)

#: Published detectors.  `gated` entries need an accepted licence + token.
#: `contamination` records overlap between the baseline's OWN training data
#: and HC-Bench sources — a baseline trained on deepset will look strong on
#: deepset rows for reasons that are not generalization.
BASELINES = {
    "protectai-v2": {
        "model_id": "protectai/deberta-v3-base-prompt-injection-v2",
        "params": "184M",
        "license": "Apache-2.0",
        "gated": False,
        "citation": "ProtectAI.com, Fine-Tuned DeBERTa-v3-base for Prompt "
                    "Injection Detection, HuggingFace, 2024",
        "publisher_note": "Model card states it targets prompt injection in "
                          "English and explicitly does NOT detect jailbreak "
                          "attacks; card advises against use on system "
                          "prompts (false positives).",
        "contamination": "Trained on combined public injection corpora; "
                         "overlap with HC-Bench deepset/gandalf lineage is "
                         "likely but not quantified by the publisher.",
    },
    "protectai-v1": {
        "model_id": "protectai/deberta-v3-base-prompt-injection",
        "params": "184M",
        "license": "Apache-2.0",
        "gated": False,
        "citation": "ProtectAI.com, Fine-Tuned DeBERTa-v3 for Prompt "
                    "Injection Detection, HuggingFace, 2023",
        "publisher_note": "Archived; superseded by v2.  Retained as a "
                          "second reference point.",
        "contamination": "As v2.",
    },
    "llama-prompt-guard-2-86m": {
        "model_id": "meta-llama/Llama-Prompt-Guard-2-86M",
        "params": "86M",
        "license": "Llama 4 Community License (gated)",
        "gated": True,
        "citation": "Meta, Llama Prompt Guard 2 Model Card, 2025",
        "publisher_note": "Publisher reports AUC .998 / recall@1%FPR 97.5% "
                          "on their own English jailbreak evaluation.",
        "contamination": "Training corpus undisclosed.",
    },
    "llama-prompt-guard-2-22m": {
        "model_id": "meta-llama/Llama-Prompt-Guard-2-22M",
        "params": "22M",
        "license": "Llama 4 Community License (gated)",
        "gated": True,
        "citation": "Meta, Llama Prompt Guard 2 Model Card, 2025",
        "publisher_note": "Smaller sibling; publisher AUC .995 English.",
        "contamination": "Training corpus undisclosed.",
    },
}

_POSITIVE = re.compile(
    r"inject|malicious|jailbreak|attack|unsafe|harmful|label_?1|^1$", re.I)


def positive_index(id2label: dict) -> int:
    for idx, name in id2label.items():
        if _POSITIVE.search(str(name)):
            return int(idx)
    return 1 if len(id2label) > 1 else 0


def score_with_hf(model_id, texts, token=None, batch_size=32, max_len=512):
    """Return (probabilities, seconds_per_text). Raises on unavailability."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    kw = {"token": token} if token else {}
    tok = AutoTokenizer.from_pretrained(model_id, **kw)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id, **kw)
    mdl.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(dev)
    pos = positive_index(mdl.config.id2label)

    probs, t0 = [], time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [t[:20000] for t in texts[i:i + batch_size]]
            enc = tok(batch, return_tensors="pt", truncation=True,
                      max_length=max_len, padding=True).to(dev)
            p = torch.softmax(mdl(**enc).logits, dim=-1)[:, pos]
            probs += p.detach().cpu().tolist()
    elapsed = time.perf_counter() - t0
    del mdl
    if dev == "cuda":
        torch.cuda.empty_cache()
    return probs, elapsed / max(1, len(texts))


def metrics_at(y, s, thr):
    pos = [v for v, g in zip(s, y) if g == 1]
    neg = [v for v, g in zip(s, y) if g == 0]
    return {
        "recall": round(sum(v >= thr for v in pos) / len(pos), 4) if pos else None,
        "benign_fpr": round(sum(v >= thr for v in neg) / len(neg), 4) if neg else None,
        "roc_auc": round(roc_auc(y, s), 4) if 0 < sum(y) < len(y) else None,
        "pr_auc": round(pr_auc(y, s), 4) if 0 < sum(y) < len(y) else None,
        "n": len(y), "n_pos": len(pos), "n_neg": len(neg),
    }


#: A threshold whose benign FPR is at or above this is not an operating
#: point, it is a shrug.  Reported as degenerate rather than quietly used.
_DEGENERATE_FPR = 0.5


def select_threshold(y, s, target_recall=None, fpr_budget=None):
    """Pick one threshold under an explicit, recorded objective.

    Mirrors ``scripts.calibrate_policy.select_policy``.  The failure this
    guards against: asking for a recall the model cannot deliver drives the
    threshold to the bottom of the score range, which "achieves" the target
    by flagging everything.  That is reported as recall 1.0 / FPR 1.0 and is
    worthless as a comparison point -- every system looks the same.
    """
    cands = sorted(set(s))
    rows = [(t, metrics_at(y, s, t)) for t in cands]

    if fpr_budget is not None:
        objective = f"max recall s.t. benign FPR <= {fpr_budget}"
        ok = [(t, m) for t, m in rows
              if (m["benign_fpr"] or 0) <= fpr_budget + 1e-9]
        if ok:
            t, m = max(ok, key=lambda x: ((x[1]["recall"] or 0), x[0]))
            return {"threshold": float(t), "objective": objective,
                    "feasible": True, "note": objective, "metrics": m}
        t, m = min(rows, key=lambda x: ((x[1]["benign_fpr"] or 0), -(x[1]["recall"] or 0)))
        return {"threshold": float(t), "objective": objective,
                "feasible": False,
                "note": (f"INFEASIBLE: no threshold reaches benign FPR <= "
                         f"{fpr_budget}; lowest available is "
                         f"{m['benign_fpr']}"),
                "metrics": m}

    objective = f"lowest benign FPR s.t. recall >= {target_recall}"
    ok = [(t, m) for t, m in rows
          if (m["recall"] or 0) >= target_recall - 1e-9]
    if not ok:                                   # cannot happen, kept honest
        t, m = max(rows, key=lambda x: (x[1]["recall"] or 0))
        return {"threshold": float(t), "objective": objective,
                "feasible": False,
                "note": (f"INFEASIBLE: recall >= {target_recall} unreachable; "
                         f"best is {m['recall']}"),
                "metrics": m}
    t, m = min(ok, key=lambda x: ((x[1]["benign_fpr"] or 0), -x[0]))
    degenerate = (m["benign_fpr"] or 0) >= _DEGENERATE_FPR
    return {
        "threshold": float(t), "objective": objective,
        "feasible": not degenerate,
        "note": (objective if not degenerate else
                 f"DEGENERATE: recall >= {target_recall} is only reachable "
                 f"by flagging {m['benign_fpr']:.0%} of benign traffic. The "
                 f"target exceeds what this model can deliver on this "
                 f"corpus; the threshold is at the bottom of its score "
                 f"range. Use --fpr-budget, and/or --exclude-category for "
                 f"classes outside the model's domain. ROC-AUC "
                 f"({m['roc_auc']}) is threshold-free and still comparable."),
        "metrics": m}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("hcbench"))
    ap.add_argument("--weights", type=Path, default=None,
                    help="DEF-HC production weights; omit to compare "
                         "baselines only")
    ap.add_argument("--hf-token", default=None,
                    help="unlocks the gated Meta Prompt Guard 2 models")
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--fpr-budget", type=float, default=None,
                    help="switch the threshold objective to: maximize recall "
                         "subject to benign FPR <= this value. Bounded by "
                         "construction. Use when the recall target is not "
                         "attainable and would otherwise select a "
                         "flag-everything threshold.")
    ap.add_argument("--exclude-category", nargs="*", default=(),
                    help="categories held OUT of threshold selection as "
                         "outside the detectors' domain (e.g. "
                         "harmful-content). Still scored and reported "
                         "per-category on test.")
    ap.add_argument("--out", type=Path,
                    default=Path("bench-out/bench-baselines.json"))
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of baseline keys to run")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from scripts.eval_hcbench import load_split

    cal = load_split(args.data_dir / "hcbench-cal.jsonl")
    test = load_split(args.data_dir / "hcbench-test.jsonl")
    # Baselines are plain text classifiers with no notion of surface, so the
    # comparison is restricted to the user_prompt surface.  Scoring a
    # retrieved document as if it were a user turn would misrepresent them.
    cal = [r for r in cal if r["surface"] == "user_prompt"]
    test = [r for r in test if r["surface"] == "user_prompt"]
    y_cal = [int(r["label"]) for r in cal]
    y_test = [int(r["label"]) for r in test]
    print(f"comparison set (user_prompt surface only): "
          f"cal n={len(cal)}  test n={len(test)}")

    excluded = set(args.exclude_category or ())
    sel_idx = [i for i, r in enumerate(cal) if r.get("category") not in excluded]
    if excluded:
        print(f"threshold selection excludes {sorted(excluded)}: "
              f"{len(cal) - len(sel_idx)} of {len(cal)} cal rows held out "
              f"(still reported per-category on test)")
    y_sel = [y_cal[i] for i in sel_idx]
    print()

    def _finish(s_cal, s_test, per_text, meta_block):
        """Threshold under the recorded objective + per-category test view."""
        sel = select_threshold([y_cal[i] for i in sel_idx],
                               [s_cal[i] for i in sel_idx],
                               target_recall=(None if args.fpr_budget is not None
                                              else args.target_recall),
                               fpr_budget=args.fpr_budget)
        thr = sel["threshold"]
        if not sel["feasible"]:
            print(f"     !! {sel['note']}")
        by_cat: dict = defaultdict(lambda: [[], []])
        for yy, ss, rr in zip(y_test, s_test, test):
            by_cat[rr["category"]][0].append(yy)
            by_cat[rr["category"]][1].append(ss)
        return {
            **meta_block,
            "threshold_source": (f"hcbench-cal, {sel['objective']}"
                                 + (f", excluding {sorted(excluded)}"
                                    if excluded else "")),
            "threshold_objective": sel["objective"],
            "threshold_objective_feasible": sel["feasible"],
            "threshold_objective_note": sel["note"],
            "threshold": round(float(thr), 6),
            "cal": metrics_at(y_cal, s_cal, thr),
            "test": metrics_at(y_test, s_test, thr),
            "test_by_category": {k: metrics_at(v[0], v[1], thr)
                                 for k, v in sorted(by_cat.items())},
            "latency_ms_per_text": round(per_text * 1000, 3),
        }

    results, skipped = {}, {}

    for key, meta in BASELINES.items():
        if args.only and key not in args.only:
            continue
        if meta["gated"] and not args.hf_token:
            reason = (f"gated: {meta['license']} — accept the licence on the "
                      f"model page and pass --hf-token")
            print(f"SKIP {key:26s} — {reason}")
            skipped[key] = {**meta, "reason": reason}
            continue
        try:
            print(f"RUN  {key:26s} {meta['model_id']}")
            s_cal, _ = score_with_hf(meta["model_id"], [r["text"] for r in cal],
                                     token=args.hf_token)
            s_test, per_text = score_with_hf(
                meta["model_id"], [r["text"] for r in test],
                token=args.hf_token)
        except Exception as exc:  # noqa: BLE001 — skip, never substitute
            reason = f"{type(exc).__name__}: {exc}"[:240]
            print(f"SKIP {key:26s} — {reason}")
            skipped[key] = {**meta, "reason": reason}
            continue
        results[key] = _finish(s_cal, s_test, per_text,
                               {k: v for k, v in meta.items() if k != "gated"})

    # ---- DEF-HC through its own production channels, same rows, same rule
    if args.weights:
        from defend_hc2 import DEFEND_HC2
        from defend_hc2.provenance import ToolRegistry
        from scripts.eval_hcbench import BENCH_TOOL, run_evaluation
        print(f"RUN  {'def-hc (this work)':26s} production channels")
        reg = ToolRegistry()
        reg.register_tool(BENCH_TOOL["name"], BENCH_TOOL["key"],
                          privileged=BENCH_TOOL["privileged"])
        system = DEFEND_HC2(db_path=":memory:", demo_mode=False,
                            weights_path=str(args.weights),
                            tool_registry=reg, master_secret=b"S" * 32)
        s_cal, _ = run_evaluation(cal, system, "base-cal")
        t0 = time.perf_counter()
        s_test, _ = run_evaluation(test, system, "base-test")
        per_text = (time.perf_counter() - t0) / max(1, len(test))
        results["def-hc"] = _finish(s_cal, s_test, per_text, {
            "model_id": "this work (L0-L5 fused, production path)",
            "params": "33M embedder + logistic head",
            "license": "see repository",
            "citation": "this work",
            "publisher_note": "Scored through the full pipeline, not the "
                              "head in isolation.",
            "contamination": "Trained on S-Labs + SPML; HC-Bench is filtered "
                             "against every previously-observed corpus.",
        })
    report = {
        "label": "baseline_comparison_public_models_not_blind",
        "protocol": "identical for every system: threshold calibrated on "
                    "hcbench-cal only at the stated target recall, applied "
                    "once to hcbench-test; user_prompt surface only, since "
                    "text-classifier baselines have no surface routing",
        "comparison_set": {"cal_n": len(cal), "test_n": len(test),
                           "surface": "user_prompt"},
        "results": results,
        "skipped": skipped,
        "reading_caution": "Published detectors were trained on public "
                           "injection corpora that overlap HC-Bench sources; "
                           "high baseline scores may reflect that overlap. "
                           "Per-model contamination notes are recorded above "
                           "and are NOT quantified.",
        "environment": environment_block(),
        "git_commit": git_commit(Path(__file__).resolve().parents[1]),
    }
    args.out.write_text(json.dumps(report, indent=2))
    digest = file_sha256(args.out)
    args.out.with_suffix(".sha256").write_text(f"{digest}  {args.out.name}\n")

    print(f"\n{'system':28s} {'params':>10s} {'AUC':>7s} {'PR-AUC':>7s} "
          f"{'recall':>7s} {'FPR':>7s} {'ms/txt':>8s}  {'':<3s}")
    print("-" * 86)
    for k, v in sorted(results.items(),
                       key=lambda kv: -(kv[1]["test"]["roc_auc"] or 0)):
        t = v["test"]
        flag = "" if v.get("threshold_objective_feasible", True) else "  <!>"
        print(f"{k:28s} {v['params']:>10s} {t['roc_auc'] or 0:>7.4f} "
              f"{t['pr_auc'] or 0:>7.4f} {t['recall'] or 0:>7.4f} "
              f"{t['benign_fpr'] or 0:>7.4f} {v['latency_ms_per_text']:>8.2f}{flag}")
    print("-" * 86)
    if any(not v.get("threshold_objective_feasible", True)
           for v in results.values()):
        print("<!> threshold objective NOT met — that row's recall/FPR is a\n"
              "    degenerate operating point, not a calibrated one. ROC-AUC\n"
              "    and PR-AUC are threshold-free and remain comparable.")
    # Per-category test view: shows which classes every detector misses.
    cats = sorted({c for v in results.values()
                   for c in v.get("test_by_category", {})})
    if cats and results:
        print(f"\nrecall by category at each system's own threshold")
        print(f"{'system':28s} " + " ".join(f"{c[:14]:>15s}" for c in cats))
        print("-" * (28 + 16 * len(cats)))
        for k, v in sorted(results.items()):
            cells = []
            for c in cats:
                m = v.get("test_by_category", {}).get(c)
                r = None if m is None else m["recall"]
                cells.append(f"{'   —':>15s}" if r is None else f"{r:>15.4f}")
            print(f"{k:28s} " + " ".join(cells))
    for k, v in skipped.items():
        print(f"SKIPPED {k}: {v['reason']}")
    print(f"\nwrote {args.out}\nsha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
