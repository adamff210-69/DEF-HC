"""Layer-ablation benchmark (spec Exp-D): quantify each content layer.

Variants (thresholds calibrated per-variant on CALIBRATION data only):

    D1: ML embedding classifier only
    D2: lexical only
    D3: ML + lexical
    D4: D3 + context/mismatch (SPML System Prompt as the context)
    D5: complete applicable fusion (D4 + drift; drift is inactive without
        history — documented as such)
    * each also evaluated without normalization variants (raw-only lexical)

Produces bench-metrics-exp-d.json with per-variant full metrics so the
complementarity claim is supported by measured deltas, not hand-waving.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow bare run

from defend_hc2.modeling import (
    calibrate_thresholds,
    full_metric_report,
    git_commit,
    environment_block,
)


def p_from_row(w, b, x):
    return 1.0 / (1.0 + math.exp(-(sum(a * v for a, v in zip(w, x)) + b)))


def load_rows(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                out.append({"text": str(r["text"]), "label": int(r["label"]),
                            "system_prompt": r.get("system_prompt")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--cal-file", type=Path, required=True)
    ap.add_argument("--eval-file", type=Path, required=True)
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--out", type=Path, default=Path("bench-metrics-exp-d.json"))
    args = ap.parse_args()

    from defend_hc2.content_risk import ContentRiskAnalyzer, combine_signals
    from defend_hc2.embedder import get_sentence_transformer

    blob = json.loads(args.weights.read_text())
    w, b = [float(x) for x in blob["weights"]], float(blob["bias"])
    analyzer = ContentRiskAnalyzer(demo_mode=False, weights_path=str(args.weights))
    model = get_sentence_transformer(blob.get("model", args.model))

    import numpy as np

    def compute_scores(rows: list[dict], normalize: bool) -> dict[str, list[float]]:
        texts = [r["text"] for r in rows]
        contexts = [[r["system_prompt"]] if r.get("system_prompt") else [] for r in rows]
        flat = [piece for text, ctx in zip(texts, contexts) for piece in [text, *ctx]]
        X = np.asarray(model.encode(flat, normalize_embeddings=True,
                                    convert_to_numpy=True, batch_size=256),
                       dtype=float)
        out = {"ml": [], "lexical": [], "mismatch": []}
        i = 0
        for text, ctx in zip(texts, contexts):
            vecs = [X[i]]
            i += 1
            for _c in ctx:
                vecs.append(X[i]); i += 1
            r = {"text": text}
            out["ml"].append(p_from_row(w, b, list(vecs[0])))
            if normalize:
                lex, _ = analyzer.lexical_scan(text)
            else:  # raw-only ablation: single-view scan, no variants
                from defend_hc2.content_risk import _LEXICAL_COMPILED

                score = 0.0
                for pattern, weight, _label in _LEXICAL_COMPILED:
                    for _m in pattern.finditer(text):
                        score += weight
                lex = min(1.0, score)
            out["lexical"].append(lex)
            if ctx:
                req = vecs[0]
                sims = [ContentRiskAnalyzer._cosine(req, v) for v in vecs[1:]]
                mm = min(1.0, max(0.0, 1.0 - (0.55 * (sum(sims) / len(sims)) + 0.45 * min(sims))))
            else:
                mm = None
            out["mismatch"].append(mm)
        return out

    def fuse(variant: str, ml, lex, mm) -> float:
        ch = {
            "D1": {"injection": ml},
            "D2": {"lexical": lex},
            "D3": {"injection": ml, "lexical": lex},
            "D4": {"injection": ml, "lexical": lex, "mismatch": mm},
            "D5": {"injection": ml, "lexical": lex, "mismatch": mm, "drift": None},
        }[variant]
        return combine_signals(ch)

    results: dict = {"weights": str(args.weights), "cal": str(args.cal_file),
                     "test": str(args.eval_file),
                     "drift_note": "drift inactive: prompt-only rows have <3 history turns",
                     "target_recall": args.target_recall,
                     "environment": environment_block(),
                     "git_commit": git_commit(Path(__file__).resolve().parents[1])}

    cal_rows = load_rows(args.cal_file)
    te_rows = load_rows(args.eval_file)
    key = f"recall@{args.target_recall}"
    for norm_flag, norm_name in ((True, "normalized"), (False, "raw-only")):
        cal_s = compute_scores(cal_rows, norm_flag)
        te_s = compute_scores(te_rows, norm_flag)
        for variant in ("D1", "D2", "D3", "D4", "D5"):
            cal_risk = [fuse(variant, ml, lx, mm)
                        for ml, lx, mm in zip(cal_s["ml"], cal_s["lexical"], cal_s["mismatch"])]
            te_risk = [fuse(variant, ml, lx, mm)
                       for ml, lx, mm in zip(te_s["ml"], te_s["lexical"], te_s["mismatch"])]
            thr = calibrate_thresholds([r["label"] for r in cal_rows], cal_risk)[key]
            rep = full_metric_report([r["label"] for r in te_rows], te_risk, thr)
            results[f"{variant}_{norm_name}"] = rep
            print(f"  {variant} ({norm_name:<9}) AUC={rep['roc_auc']} "
                  f"P={rep['precision']:.4f} R={rep['recall']:.4f} "
                  f"F1={rep['f1']:.4f} bal={rep['balanced_accuracy']:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nablation: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
