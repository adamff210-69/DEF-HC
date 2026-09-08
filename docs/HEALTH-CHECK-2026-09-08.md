# Full analysis & health check — 2026-09-08

Trigger: recurring "bad results". Scope: everything runnable without a
Hugging Face connection (this sandbox blocks huggingface.co; PyPI/GitHub
work). Corpus-dependent stages are Kaggle work and listed at the end.

## 1. What was run and is GREEN

| Check | Result |
|---|---|
| `pytest -q` (full ML extra installed) | **283 passed, 2 skipped** — both skips need Kaggle artifacts (`test_recovery_gate_kaggle`, `test_signal_hardening`) |
| `python -m defend_hc2` 10-step demo | all steps clean; chain verifies; ledger tamper-blocked; checkpoint signed |
| `scripts/evaluate_complementarity.py` | **9/9**: content attacks 4/4 via L1/L3 (chain provably blind), state attacks 5/5 via chain (content provably blind) |
| `scripts/measure_overhead.py --n 500` | L1 risk 0.17–0.30 ms; full pipeline p50 1.2–1.7 ms; chain re-verify 72 µs/entry; 2.3 KB/event |
| Calibration machinery vs the documented hcbench failure | verified: the old fixed ladder reproduces the §0.10 failure (infeasible → floor 0.15/0.40/0.70 fallback); the current quantile-grid `select_policy` finds a feasible recall-0.95 point and `--auto-budget`/`--fpr-budget` behave as documented |
| Static audit (`pyflakes`) + line review of `modeling.py`, `policy.py`, `content_risk.py`, `calibrate_policy.py`, `calibrate_hcbench_policy.py`, `eval_hcbench.py`, `eval_sealed.py`, `run_baselines.py`, `benchmark_classifier.py`, `train_classifier.py` | no correctness defects found in metric math, threshold search, bootstrap CIs, one-shot guards, or anti-leakage roles |

## 2. Root causes of the recurring bad results (FIXED here)

### 2.1 The stale-branch trap (the big one)

`docs/KAGGLE.md`, `docs/EVALUATION.md` (reproduce section), and the older
notebooks (`kaggle-all-in-one-cell.py`, `defend-hc2-kaggle-ml-mode.ipynb`)
pinned **`arena/01a06c45-def-hc`** — the branch that predates every
calibration fix. Its `calibrate_hcbench_policy.py` still carries the
fixed-ladder defect that produced the documented infeasible /
benign-FPR-0.61 result, so every re-run from those docs faithfully
reproduced the bad numbers. The paper-run notebook had already hit this
("An older kernel of yours cloned the stale branch…").

**Fix:** all pins now point at `arena/01a07e57-def-hc` (current working
branch). If it prints anything else at clone time, the run is void.

### 2.2 The toy-weights trap

`kaggle-all-in-one-cell.py` trained "ML mode" weights with **no
`--dataset`**, silently using the 30 bundled seed sentences, then printed
plausible `p(injection)` probes. Every number downstream of those weights
is meaningless while looking perfectly healthy.

**Fix:** the cell now fetches S-Labs + SPML and trains via
`scripts/benchmark_classifier.py` (same protocol as the paper run); if the
corpora cannot be fetched it falls back to the seed set behind a loud
`!! SMOKE TEST — NOT RESULTS !!` banner instead of a quiet success log.

### 2.3 Stale expectations in the docs

Test counts in README/docs/notebooks said 158/164/250+3; the suite is at
**283+2**. A wrong "expected" line makes a green run look failed.

**Fix:** all counts updated to the measured values.

## 3. Known model-level limits (need Kaggle; already documented, not code bugs)

* **Exp-H domain mismatch** — frozen bands on harmful-request corpora
  (AdvBench recall 0.4365, XSTest AUC 0.5461): the system is a
  prompt-injection detector; recalibrate bands per domain with the fixed
  machinery (`--auto-budget`), never reuse foreign frozen bands.
* **Letter-spacing gap** — Exp-G augmentation weights (AUC 0.4382 → 0.9102)
  are experimental and NOT folded into `weights/bge-final.json`; folding
  them in requires a fresh calibration + development-test pass (Kaggle).
* **Spec target "benign FPR ≤ 1% at recall ≥ 0.95"** is infeasible for this
  score distribution on S-Labs; kept as strict xfail by design.

## 4. Required next run (Kaggle, in order)

1. `notebooks/kaggle-paper-run.py` Cell 1 (confirm it prints
   `arena/01a07e57-def-hc`) through Cell 10.
2. Cell 5 must report `feasible=True` for at least one policy; if both
   policies collapse to identical bands, stop and re-read the frontier.
3. `scripts/eval_sealed.py` **once**, at the very end, only after the
   policies are frozen — the sealed split is the only untouched holdout.

Nothing in this document changes any published number; it removes the
mechanisms that kept regenerating stale ones.
