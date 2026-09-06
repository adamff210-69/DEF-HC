# DEF-HC — Kaggle paper evidence pass
# Pure-Python cells. Paste each block into its own Kaggle cell, run in order.
#
# Notebook settings first:
#   Internet: ON            (required — clone, datasets, models)
#   Accelerator: GPU T4 x2  (optional; speeds up Cell 7)


# ============================================================================
# %% Cell 1 — clone the working branch, install deps, define helpers
# ============================================================================
import os, sys, subprocess, pathlib, json, hashlib

REPO    = "https://github.com/adamff210-69/DEF-HC.git"
BRANCH  = "arena/01a074f1-def-hc"          # working branch, NOT main
ROOT    = "/kaggle/working/DEF-HC"
WORK    = pathlib.Path("/kaggle/working")

BENCH   = WORK / "bench-data"
HCBENCH = WORK / "hcbench"
REPORTS = WORK / "reports"
OUTDIR  = WORK / "bench-out"
WEIGHTS = WORK / "weights" / "bge-final.json"


def sh(*cmd, cwd=ROOT, check=True):
    """Run a command, streaming output live (Kaggle buffers otherwise)."""
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    if check and p.returncode != 0:
        raise SystemExit(f"\n!! FAILED (exit {p.returncode}): {' '.join(cmd)}")
    return p.returncode


def py(*args, **kw):
    return sh(sys.executable, "-u", *args, **kw)


if not pathlib.Path(ROOT, ".git").exists():
    sh("git", "clone", "--branch", BRANCH, "--single-branch", REPO, ROOT, cwd="/kaggle/working")
else:
    sh("git", "fetch", "origin", BRANCH)
    sh("git", "reset", "--hard", f"origin/{BRANCH}")

os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Kaggle already ships torch + transformers. Do not reinstall them.
sh(sys.executable, "-m", "pip", "install", "-q",
   "sentence-transformers", "datasets", "scikit-learn")

for d in (BENCH, HCBENCH, REPORTS, OUTDIR, WEIGHTS.parent):
    d.mkdir(parents=True, exist_ok=True)

print("\nbranch:", subprocess.run(["git", "branch", "--show-current"],
      capture_output=True, text=True, cwd=ROOT).stdout.strip())
print("commit:", subprocess.run(["git", "rev-parse", "--short", "HEAD"],
      capture_output=True, text=True, cwd=ROOT).stdout.strip())
# MUST print arena/01a074f1-def-hc


# ============================================================================
# %% Cell 2 — source corpora  (~5-10 min, needs internet)
# Builds bench-data/: training data AND the leak-guard corpus.
# ============================================================================
py("scripts/prepare_benchmarks.py", "--out-dir", BENCH)

print("\nfiles produced:")
for fp in sorted(BENCH.glob("*.jsonl")):
    print(f"  {fp.name:45s} {sum(1 for _ in fp.open()):>7,} rows")


# ============================================================================
# %% Cell 3 — frozen production weights  (~3-6 min)
# Trained on S-Labs + SPML *train* only. HC-Bench is never trained on.
# ============================================================================
py("scripts/benchmark_classifier.py",
   "--dataset", BENCH / "slp-train.jsonl", BENCH / "spml-train.jsonl",
   "--cal-file", BENCH / "slp-cal.jsonl",
   "--eval-file", BENCH / "pi-test.jsonl",
   "--target-recall", "0.98",
   "--out-weights", WEIGHTS,
   "--out-metrics", OUTDIR / "bench-metrics-final.json",
   "--out-scores",  OUTDIR / "scores-final.jsonl")

print("\nweights:", WEIGHTS, WEIGHTS.stat().st_size, "bytes")


# ============================================================================
# %% Cell 4 — build HC-Bench
# COPY THE OUTPUT: dedup counts, leak-guard counts, visibility flag %, splits.
# ============================================================================
py("scripts/build_hcbench.py",
   "--out-dir", HCBENCH,
   "--reports", REPORTS,
   "--leak-guard-dir", BENCH)

for fp in sorted(HCBENCH.glob("*.jsonl")):
    print(f"  {fp.name:30s} {sum(1 for _ in fp.open()):>7,} rows")
# 'slp-test.jsonl' listed as absent from the leak guard is EXPECTED
# (that split is written as pi-test.jsonl).


# ============================================================================
# %% Cell 5 — calibrate both policies on hcbench-cal
#
# Objective = max recall subject to a benign-FPR budget. This is bounded by
# construction. Do NOT go back to --target-recall 0.95/0.98 here: those are
# unreachable for this model on this benchmark, and an unmet recall target
# selects a fallback operating point that flags most benign traffic.
# The script now refuses to write such a policy unless --allow-infeasible.
#
# harmful-content is held out of BAND SELECTION as declared out-of-domain:
# the model is an injection/jailbreak detector. Those rows are still scored
# and still reported on the test split — only the calibration objective
# ignores them, and the artifact records the exclusion.
#
# The 2nd call needs --allow-repeat-test-eval: the one-shot ledger is shared
# per output dir, so the 1st run already marked test as read.
# ============================================================================
py("scripts/calibrate_hcbench_policy.py",
   "--data-dir", HCBENCH, "--weights", WEIGHTS,
   "--fpr-budget", "0.01", "--exclude-category", "harmful-content",
   "--provenance-tag", "hcbench-balanced",
   "--out", REPORTS / "policy-balanced.json")

py("scripts/calibrate_hcbench_policy.py",
   "--data-dir", HCBENCH, "--weights", WEIGHTS,
   "--fpr-budget", "0.05", "--exclude-category", "harmful-content",
   "--provenance-tag", "hcbench-high-recall",
   "--out", REPORTS / "policy-highrecall.json",
   "--allow-repeat-test-eval")

for f in ("policy-balanced", "policy-highrecall"):
    d = json.loads((REPORTS / f"{f}.json").read_text())
    print(f"\n{f}: feasible={d['objective_feasible']}  bands={d['policy']}")
    print(f"   objective: {d['objective']}")
    if not d["objective_feasible"]:
        print(f"   !! {d['objective_note']}")
# If either says feasible=False, or both print the SAME bands, stop and tell
# me before running Cell 6 — the downstream numbers would be meaningless.


# ============================================================================
# %% Cell 6 — headline evaluation   <<< the money cell
# Copy the whole '-- by lexical visibility --' block.
# ============================================================================
METRICS = OUTDIR / "bench-hcbench-metrics.json"

py("scripts/eval_hcbench.py",
   "--data-dir", HCBENCH, "--weights", WEIGHTS,
   "--policies", REPORTS / "policy-balanced.json",
                 REPORTS / "policy-highrecall.json",
   "--out", METRICS)

mode = json.loads(METRICS.read_text())["scoring_mode"]
print("\nscoring_mode:", mode)
assert mode == "trained-weights", "--weights did not take; numbers are meaningless"


# ============================================================================
# %% Cell 7 — published baselines   <<< least-tested path; paste me any traceback
# ProtectAI v1/v2 always. Meta Prompt Guard 2 only with an HF token whose
# account accepted the Llama 4 Community License on BOTH model pages.
# ============================================================================
HF_TOKEN = ""   # paste a token here, or leave "" to skip the gated Meta models
try:                                    # or pull it from Kaggle secrets
    if not HF_TOKEN:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass

args = ["scripts/run_baselines.py",
        "--data-dir", HCBENCH, "--weights", WEIGHTS,
        "--target-recall", "0.95",
        "--out", OUTDIR / "bench-baselines.json"]
if HF_TOKEN:
    args += ["--hf-token", HF_TOKEN]
    print("gated Meta models: ENABLED")
else:
    print("gated Meta models: skipped (no token) — this is a valid result")

py(*args)


# ============================================================================
# %% Cell 8 — measured overhead (supersedes the CPU heuristic table in docs)
# ============================================================================
py("scripts/measure_overhead.py",
   "--n", "2000", "--weights", WEIGHTS,
   "--out", OUTDIR / "overhead-metrics.json")


# ============================================================================
# %% Cell 9 — validation.  Expect: 250 passed, 3 skipped
# ============================================================================
py("-m", "pytest", "-q", check=False)
py("scripts/evaluate_complementarity.py")


# ============================================================================
# %% Cell 10 — dump every artifact.  PASTE THIS OUTPUT BACK.
# ============================================================================
print("commit:", subprocess.run(["git", "rev-parse", "HEAD"],
      capture_output=True, text=True, cwd=ROOT).stdout.strip())

for rel in ["reports/hcbench-manifest.json",
            "reports/policy-balanced.json",
            "reports/policy-highrecall.json",
            "bench-out/bench-hcbench-metrics.json",
            "bench-out/bench-baselines.json",
            "bench-out/overhead-metrics.json"]:
    fp = WORK / rel
    if not fp.exists():
        print(f"\nMISSING  {rel}")
        continue
    digest = hashlib.sha256(fp.read_bytes()).hexdigest()[:16]
    print(f"\n{'='*72}\n### {rel}   sha256={digest}\n{'='*72}")
    print(fp.read_text()[:6000])
