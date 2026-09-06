# Running DEFEND-HC2 on Kaggle (ML mode)

Two options:

1. **Single cell (fastest)** — paste
   [`notebooks/kaggle-all-in-one-cell.py`](../notebooks/kaggle-all-in-one-cell.py)
   into one code cell. It does everything idempotently: deps → clone branch →
   editable install (+ running-kernel import fix) → train bge weights (skips if
   present) → ML probes → full scenario matrix → checkpoint/export → in-process API.
2. **Guided notebook** — import
   [`notebooks/defend-hc2-kaggle-ml-mode.ipynb`](../notebooks/defend-hc2-kaggle-ml-mode.ipynb)
   (steps below).

## Import it

1. On kaggle.com → **Notebooks → New notebook → File → Import notebook** and
   upload the `.ipynb` (or paste the GitHub URL of the file).
2. **Right-hand panel → Notebook options:** set **Internet: ON** (required to
   clone the repo and download `BAAI/bge-small-en-v1.5` once). Accelerator
   *None* is fine; a GPU only speeds training.
3. Run all cells top to bottom.

## What the notebook does

| Cells | Content |
|---|---|
| 1–3 | environment check, clone branch `arena/01a06c45-def-hc`, install missing deps (`torch` untouched — Kaggle ships it), editable-install `defend_hc2` |
| 4 | sanity demo (heuristic mode) |
| 5 | **train the bge-small logistic classifier weights** → `/kaggle/working/weights/bge-logistic.json` (~2–4 min incl. one-time ~130 MB download; skips if present) |
| 6 | probe `p(injection)` for benign vs. injection prompts in **ML mode** |
| 7 | full L0–L5 pipeline in ML mode + all spec attack scenarios |
| 8 | export the audit trail (`audit-export.json`) + checkpoint |
| 9 | drive the FastAPI service in-process via `TestClient` |
| 10 | (optional) full 164-test suite |

All artifacts persist in `/kaggle/working` as notebook output.

## Full evaluation pipeline (spec protocol)

End-to-end, in order (each script is idempotent and prints its provenance):

```bash
cd /kaggle/working/DEF-HC
pip install -q -r requirements-ml.txt

# 1) data: official splits respected, group-aware dedup, foreign corpora
python scripts/prepare_benchmarks.py --out-dir /kaggle/working/bench-data

# 2) experiment matrix (A: in-dist, B: zero-shot, C: mixed, F: robustness)
python scripts/run_experiments.py \
    --data-dir /kaggle/working/bench-data --out-dir /kaggle/working/bench-out

# 3) final production weights: mixed training + deployment-matched calibration
python scripts/benchmark_classifier.py \
    --dataset bench-data/slp-train.jsonl bench-data/spml-train.jsonl \
    --cal-file bench-data/slp-cal.jsonl --eval-file bench-data/pi-test.jsonl \
    --target-recall 0.98 \
    --out-weights /kaggle/working/weights/bge-final.json \
    --out-metrics /kaggle/working/bench-metrics-final.json \
    --out-scores /kaggle/working/scores-final.jsonl

# 4) policy calibration (validation only; default --cal-target balanced)
#    + once-eval on development test (development_test_previously_observed)
#
# FLAW-3 / Step 5: --cal-target balanced  → slp-cal.jsonl  (~50% inj; FPR-controlled,
#   use for S-Labs-like traffic — Exp-A/Exp-C deployments; the 'balanced' test
#   asserts benign FPR <= 1% on the S-Labs development-test set)
#                  --cal-target high-recall → spml-cal.jsonl (~78% inj; aggressive,
#   use only for high-attack-rate traffic — Exp-B/Exp-D style foreign corpora)
# An explicit --cal-file always overrides the preset.
python scripts/calibrate_policy.py \
    --data-dir bench-data --cal-target balanced \
    --eval-file bench-data/slp-cal.jsonl bench-data/pi-test.jsonl \
    --weights /kaggle/working/weights/bge-final.json \
    --out /kaggle/working/calibrated-policy-balanced.json
python scripts/calibrate_policy.py \
    --data-dir bench-data --cal-target high-recall \
    --eval-file bench-data/spml-cal.jsonl bench-data/spml-test.jsonl \
    --weights /kaggle/working/weights/bge-final.json \
    --out /kaggle/working/calibrated-policy-highrecall.json

# 5) layer ablation, figures, error analysis, final report
python scripts/run_ablation.py --weights /kaggle/working/weights/bge-final.json \
    --cal-file bench-data/spml-cal.jsonl --eval-file bench-data/spml-test.jsonl
python scripts/make_figures.py --scores /kaggle/working/scores-final.jsonl \
    --out-dir /kaggle/working/figures
python scripts/error_analysis.py --scores /kaggle/working/scores-final.jsonl \
    --out /kaggle/working/reports/error-analysis.md
python scripts/final_report.py --artifacts /kaggle/working --repo .

# 6) mandated final validation
python -m pytest -q
python scripts/evaluate_complementarity.py
python scripts/run_final_demo.py --weights /kaggle/working/weights/bge-final.json \
    --db /kaggle/working/final.db --export /kaggle/working/final-export.json --check
```

## Benchmark protocol (public corpora)

For the full public-corpus evaluation — multi-dataset training, class
balancing, stacked meta-model, deployment-matched threshold calibration —
see **`docs/EVALUATION.md`** (results table, four-rule calibration doctrine,
canonical reproduction command):

```bash
python scripts/benchmark_classifier.py \
  --dataset slp-train.jsonl spml-train.jsonl \
  --eval-file pi-test.jsonl --cal-file slp-cal.jsonl \
  --target-recall 0.98 \
  --out-weights weights/bge-final.json --out-metrics bench-metrics-final.json
```

The resulting `weights/bge-final.json` is the release weights file: point the
pipeline at it with `weights_path=`, and pair it with calibrated policy bands
(`PolicyEngine(reject_at=0.80, quarantine_at=0.50, sanitize_at=0.25)`).

## Gotchas

- The implementation branch is `arena/01a06c45-def-hc`; `main` is a README stub.
- **If `import defend_hc2` fails after `pip install -e` in the same kernel**:
  editable installs register at interpreter startup, so the already-running
  kernel can't see the package. Either restart the kernel once, or add
  `import sys; sys.path.insert(0, "/kaggle/working/DEF-HC")` before importing
  (the notebook's Cell 3 already does this for you).
- Cell 5 is idempotent — delete the weights file to retrain (e.g. with your own
  `--dataset` JSONL).
- Cell 7 wipes its ledger at start, so it is fully re-runnable.
- No port forwarding needed: the API runs in-process via `TestClient`.
