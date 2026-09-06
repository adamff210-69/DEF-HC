# Kaggle run — paper evidence pass

Produces every artifact needed for the paper: HC-Bench, frozen weights,
calibrated policies, the headline evaluation with the lexical-visibility
cross-tab, the published-baseline comparison, and measured overhead.

**Notebook settings before you start**

- **Internet: ON** — required (clones the repo, downloads 5 HF datasets and
  the models). Nothing works without it.
- **Accelerator: GPU T4 x2** — optional but recommended. The baselines cell
  runs 3 transformer models over the test split; on CPU that is the slow part.
- Kaggle already ships `torch` and `transformers`. Do not reinstall them.

Run the cells in order. Each one is foreground `python -u` on purpose —
backgrounding with `nohup &` produces an empty log because of stdout buffering.

---

## Cell 1 — clone + deps

```python
import os, subprocess, sys, pathlib

REPO   = "https://github.com/adamff210-69/DEF-HC.git"
BRANCH = "arena/01a074f1-def-hc"          # <- the working branch, not main
ROOT   = "/kaggle/working/DEF-HC"

if not pathlib.Path(ROOT).exists():
    subprocess.run(["git","clone","--branch",BRANCH,"--single-branch",REPO,ROOT], check=True)
else:
    subprocess.run(["git","-C",ROOT,"fetch","origin",BRANCH], check=True)
    subprocess.run(["git","-C",ROOT,"reset","--hard",f"origin/{BRANCH}"], check=True)

os.chdir(ROOT)
sys.path.insert(0, ROOT)

# Kaggle ships torch + transformers; only add what is missing.
subprocess.run([sys.executable,"-m","pip","install","-q",
                "sentence-transformers","datasets","scikit-learn"], check=True)

print("branch :", subprocess.run(["git","branch","--show-current"],
                                 capture_output=True,text=True).stdout.strip())
print("commit :", subprocess.run(["git","rev-parse","--short","HEAD"],
                                 capture_output=True,text=True).stdout.strip())
```

Confirm it prints `arena/01a074f1-def-hc`. An older kernel of yours cloned the
stale branch `arena/01a06c45-def-hc`, which does not have any of this work.

---

## Cell 2 — fetch the source corpora (~5–10 min)

Builds `bench-data/`, which is both the training data **and** the leak-guard
corpus that HC-Bench is filtered against.

```python
!cd /kaggle/working/DEF-HC && python -u scripts/prepare_benchmarks.py \
    --out-dir /kaggle/working/bench-data
```

Downloads S-Labs, SPML, deepset, jailbreak-classification and safe-guard.
Any corpus with an incompatible schema is printed as `SKIPPED` with a reason
and simply won't participate in the leak guard.

---

## Cell 3 — train the frozen production weights (~3–6 min)

Trained on S-Labs + SPML **train** splits only. HC-Bench is never trained on.

```python
!cd /kaggle/working/DEF-HC && python -u scripts/benchmark_classifier.py \
    --dataset /kaggle/working/bench-data/slp-train.jsonl \
              /kaggle/working/bench-data/spml-train.jsonl \
    --cal-file /kaggle/working/bench-data/slp-cal.jsonl \
    --eval-file /kaggle/working/bench-data/pi-test.jsonl \
    --target-recall 0.98 \
    --out-weights /kaggle/working/weights/bge-final.json \
    --out-metrics /kaggle/working/bench-out/bench-metrics-final.json \
    --out-scores  /kaggle/working/bench-out/scores-final.jsonl
```

---

## Cell 4 — build HC-Bench

```python
!cd /kaggle/working/DEF-HC && python -u scripts/build_hcbench.py \
    --out-dir /kaggle/working/hcbench \
    --reports /kaggle/working/reports \
    --leak-guard-dir /kaggle/working/bench-data
```

**Copy these lines out of the output — they go in the paper:**

- `exact dedup removed N rows`
- `leak guard: N previously-observed rows from M files`
- `NOTE absent, so NOT guarded against: [...]` — must be short. `slp-test.jsonl`
  is expected to be absent (that split is written as `pi-test.jsonl`).
- `overlap with previously-observed corpora removed N rows`
- `lexically-invisible attacks flagged: N/M (P%)`
- the per-stratum train/cal/test/sealed table

If a whole stratum lands as `test=0`, that stratum collapsed to one template
family; the group-aware splitter refuses to split a single group across splits.

---

## Cell 5 — calibrate the two policies

Thresholds come from `hcbench-cal`. The script also records a one-shot
`hcbench-test` reading and keeps a ledger so repeat peeking is visible.

```python
!cd /kaggle/working/DEF-HC && python -u scripts/calibrate_hcbench_policy.py \
    --data-dir /kaggle/working/hcbench \
    --weights  /kaggle/working/weights/bge-final.json \
    --target-recall 0.95 --provenance-tag hcbench-balanced \
    --out /kaggle/working/reports/policy-balanced.json

!cd /kaggle/working/DEF-HC && python -u scripts/calibrate_hcbench_policy.py \
    --data-dir /kaggle/working/hcbench \
    --weights  /kaggle/working/weights/bge-final.json \
    --target-recall 0.98 --provenance-tag hcbench-high-recall \
    --out /kaggle/working/reports/policy-highrecall.json \
    --allow-repeat-test-eval
```

The second call **needs** `--allow-repeat-test-eval` — the ledger lives in the
shared `--out` directory, so the first run already marked test as read. This is
deliberate and honest: the artifact records `hcbench_test_pass_number: 2`.

---

## Cell 6 — headline evaluation

```python
!cd /kaggle/working/DEF-HC && python -u scripts/eval_hcbench.py \
    --data-dir /kaggle/working/hcbench \
    --weights  /kaggle/working/weights/bge-final.json \
    --policies /kaggle/working/reports/policy-balanced.json \
               /kaggle/working/reports/policy-highrecall.json \
    --out /kaggle/working/bench-out/bench-hcbench-metrics.json
```

**This is the money cell.** Copy the whole `-- by lexical visibility --` block.
The gap between `lexically_visible` and `lexically_invisible` inside the same
category is the circularity argument.

Sanity check: the output JSON must say `"scoring_mode": "trained-weights"`.
If it says `heuristic-smoke-test-NOT-RESULTS`, `--weights` didn't take and the
numbers are meaningless.

---

## Cell 7 — published baselines

```python
# Optional: unlocks Meta Prompt Guard 2. Requires an HF token whose account has
# accepted the Llama 4 Community License on both model pages. Without it the
# two Meta models are recorded as skipped, which is a valid result.
HF_TOKEN = ""   # e.g. from kaggle_secrets; leave "" to skip the gated models

tok = f"--hf-token {HF_TOKEN}" if HF_TOKEN else ""
!cd /kaggle/working/DEF-HC && python -u scripts/run_baselines.py \
    --data-dir /kaggle/working/hcbench \
    --weights  /kaggle/working/weights/bge-final.json \
    --target-recall 0.95 {tok} \
    --out /kaggle/working/bench-out/bench-baselines.json
```

Compares DEF-HC against `protectai-v2`, `protectai-v1`, and (token permitting)
`llama-prompt-guard-2-86m` / `-22m`, all calibrated identically on
`hcbench-cal` and scored once on `hcbench-test`, restricted to the
`user_prompt` surface.

**This is the least-tested path in the repo** — locally it has only ever run
its skip branch, because torch isn't installed there. If it throws, paste the
traceback and I'll fix it. Expect ProtectAI v2 to be competitive on raw
detection; that is fine and expected.

---

## Cell 8 — overhead

```python
!cd /kaggle/working/DEF-HC && python -u scripts/measure_overhead.py \
    --n 2000 \
    --weights /kaggle/working/weights/bge-final.json \
    --out /kaggle/working/bench-out/overhead-metrics.json
```

Note the hardware in the output — the current table in `docs/EVALUATION.md` is
heuristic-mode on 2 CPU cores, so these ML-mode numbers supersede it.

---

## Cell 9 — validation

```python
!cd /kaggle/working/DEF-HC && python -u -m pytest -q 2>&1 | tail -5
!cd /kaggle/working/DEF-HC && python -u scripts/evaluate_complementarity.py
```

Expect `250 passed, 3 skipped`.

---

## Cell 10 — collect everything for me

```python
import json, hashlib, pathlib, subprocess

OUT = pathlib.Path("/kaggle/working")
print("commit:", subprocess.run(["git","-C","/kaggle/working/DEF-HC",
      "rev-parse","HEAD"], capture_output=True, text=True).stdout.strip())

for rel in ["reports/hcbench-manifest.json",
            "reports/policy-balanced.json",
            "reports/policy-highrecall.json",
            "bench-out/bench-hcbench-metrics.json",
            "bench-out/bench-baselines.json",
            "bench-out/overhead-metrics.json"]:
    fp = OUT / rel
    if not fp.exists():
        print(f"MISSING  {rel}")
        continue
    print(f"\n{'='*70}\n### {rel}   sha256={hashlib.sha256(fp.read_bytes()).hexdigest()[:16]}\n{'='*70}")
    print(fp.read_text()[:6000])
```

Paste that output back to me. If it's too long, the three I need most are
`bench-hcbench-metrics.json`, `bench-baselines.json`, and the split/leak-guard
lines from Cell 4.

---

## If the kernel restarts

`/kaggle/working` survives a restart but the repo/venv state can get stale.
Re-run Cell 1 (it hard-resets to the branch), then continue from whichever cell
failed. Cells 2–4 are the expensive ones; if `bench-data/`, `weights/` and
`hcbench/` are still on disk you can skip straight to Cell 5.
