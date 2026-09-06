# Evaluation — DEF-HC Dual-Layer Defense

> **This document now carries the strict-protocol results (generated
> 2026-09-05 on Kaggle with the consolidated pipeline).** The v1 numbers
> in §2–3 remain below as the motivating investigation, clearly attributed.
> Where the two disagree, the strict-protocol figures are authoritative.

## 0. Strict-protocol results (Kaggle, 2026-09-05)

Protocol: sklearn classifier (C selected on calibration PR-AUC), seed 42,
thresholds from calibration data only, S-Labs official splits preserved
exactly, `pi-test` used once at the end (but see **BUG-E**: it was
inspected during development — all such metrics are labeled
``development_test_previously_observed``; no blind final holdout exists),
duplicate/template
groups prevented from crossing splits, git `58cefce`.

### 0.1 Headline — final weights run (mixed S-Labs+SPML train, matched cal, t for recall 0.98)

| Model | ROC-AUC | Precision | Recall | Balanced acc |
|---|---|---|---|---|
| embedding logistic (deployable) | 0.9851 | 0.8857 | **0.9628** | 0.9253 |
| stacked meta | 0.9853 | 0.8857 | 0.9628 | 0.9253 |
| lexical baseline (calibrated) | 0.5414 | 0.4746 | 1.0000 | 0.5000 |
| demo-fusion baseline (calibrated) | 0.5446 | 0.4746 | 1.0000 | 0.5000 |
| ORACLE embedding (test-only, not deployable) | 0.9851 | 0.9345 | 0.9395 | 0.9400 |

The calibrated lexical/fusion baselines collapse onto the always-positive
dummy (precision = base rate 0.4746, balanced accuracy 0.500) and are
reported as such — the protocol exposes non-informative baselines instead
of letting degenerate thresholds hide them.

### 0.2 Exp-A: in-distribution (S-Labs official splits)

AUC 0.9876 · P 0.9645 · **R 0.9039** @ t=0.5075 calibrated for recall 0.95.
Cal→test transfer gap ≈ 4.6 recall points — a property of official-split
distribution shift, documented since v1; the final weights run targets 0.98
to compensate (delivers 0.9628).

### 0.3 Exp-B: zero-shot foreign transfer (Exp-A model, predeclared thresholds)

| Corpus | AUC | Precision | Recall |
|---|---|---|---|
| foreign-deepset (n=116) | 0.8574 | 0.7206 | 0.8167 |
| foreign-jailbreak-classification (n=262) | 0.9064 | 0.6751 | 0.9568 |
| foreign-safe-guard | 0.9393 | 0.5887 | 0.9475 |

Ranking quality survives transfer (AUC .86–.94); precision calibration does
not — it must be recalibrated per domain. Per-dataset reporting only; never
pooled.

### 0.4 Exp-C: mixed-source training (S-Labs + SPML)

Matched calibration (c1) vs pooled calibration (c2):

| Variant | Dataset | AUC | P | R |
|---|---|---|---|---|
| c1 matched cal | S-Labs | 0.9848 | 0.9598 | 0.9096 |
| c1 matched cal | SPML | 0.9965 | 0.9865 | 0.9880 |
| c2 pooled cal | S-Labs | 0.9848 | 0.9853 | **0.8316** |
| c2 pooled cal | SPML | 0.9965 | 0.9939 | 0.9717 |

Mixed training costs −0.003 AUC on S-Labs vs Exp-A and is strong on SPML —
but pooled calibration silently trades 7.8 recall points on S-Labs.
**Calibration distribution must match deployment domain** (v1 doctrine,
fully confirmed under the strict protocol).

### 0.5 Exploitation robustness (Exp-F, S-Labs test — development_test_previously_observed)

Clean / perturbed / normalization-recovery = **production** fused scoring
(`variant_max_ml_score` + `lexical_scan` + `combine_signals`; recovery is
threshold-free ROC-AUC).  The recovery path is the same code the
production analyzer runs — never an inline reimplementation.

| Transform | Clean | Perturbed | Recovery |
|---|---|---|---|
| zero-width | 0.9876 | 0.9876 | 0.9876 |
| casing | 0.9876 | 0.9876 | 0.9877 |
| delimiter | 0.9876 | 0.9845 | 0.9845 |
| word whitespace (2–4 spaces; word grid) | 0.9876 | 0.9876 | 0.9877 |
| **leetspeak** | 0.9876 | **0.3516** | **0.9878** |
| **base64 whole-message wrap** | 0.9876 | **0.3781** | **0.9877** |
| **letter-spacing (every char fragmented)** | 0.9876 | **0.4382** | **0.4383** |

**The headline robustness finding, stated precisely.** Character-level
obfuscation defeats the raw embedding pass (AUC .99 → .35–.44; sub-0.5 is
treated in-repo as a pipeline signal with forced per-example dumps, not
"a robustness score").  The direction is NOT mere noise: the example
dumps show a **systematic inversion** of the benign/attack ordering —
mean raw-view score on the obfuscated dev-test rows is HIGHER for benign
than for attack texts (leetspeak 0.974 vs 0.942, base64 0.979 vs 0.964,
letter-spacing 0.971 vs 0.954; `bench-out/exp-f-*-examples.jsonl`, field
`perturbed_score`).  Obfuscation moves benign prose into a region the
encoder reads as adversarial while dissolving attack phrasing into
inert-looking soup — which is precisely the architectural motivation for
canonicalization-before-scoring (restoration views replace the junk raw
view once unobfuscation exists), the mechanism whose recovery column
tags these rows "Expected, not a defect."

For leetspeak and whole-message base64, that mechanism recovers the FULL
clean discrimination (recovery ≈ .988) because the fold/decode view
restores the original prose into the classifier's embedding region —
verified on production scoring calls only.  The legacy pooled
`whitespace` transform was split (it had blended a 0.99 in-distribution
success with a 0.44 OOD failure): word-grid spacing is fully tolerated,
while **extreme letter-spacing is the one remaining real gap**: the
despaced collapse feeds the literal phrase inventory only and is never
embedded (embedding glued text re-compressed class separation inside
max() — measured benign inj 0.80 vs attack 0.85; the FIX-1 production
gate now pins benign mean recovery risk < 0.30, measured 0.113), so
recovery is bounded by literal phrase coverage and stays flat.

The gap ships with a WARNING line and a design-limit note in the final
report and a full 2,101-example (text_before, text_after, label, score)
dump in the artifact bundle.  The train-time-augmentation mitigation
direction from this note was subsequently executed as **Exp-G (§0.8)
and passes all predeclared gates** (0.4382 → 0.9102 AUC with clean
performance intact); it ships as experimental weights, not yet folded
into the deployed weights (reranker/embedding alternatives above remain
the untried options).

**Provenance notes.** (i) An earlier Exp-F run reported flat recovery
partly because the leetspeak probe was non-invertible (`l→1` folds back
as `i`) and the recovery path reimplemented scoring inline — both fixed
and regression-pinned.  (ii) An intermediate variant-max design embedded
the despaced glue view; measured on dev-test it inflated benigns and
capped recovery ≈ 0.6, so it was replaced by the current view policy
(restoration views REPLACE junk raw views under materiality gates;
decoded-segment views are additive unless one token dominates the row).
Both defects and their fixes are in git history with regression tests.

### 0.6 Calibrated policy, two predeclared regimes (FLAW-3)

Bands selected on calibration data only, objective max precision s.t.
recall ≥ 0.95, frozen before the single once-evaluation per regime; the
evaluation split is labeled `development_test_previously_observed`.

| Regime | Cal set (base rate) | Bands (S/Q/R) | Split | Precision | Recall | Benign FPR |
|---|---|---|---|---|---|---|
| **balanced** | slp-cal (0.50) | 0.20 / 0.55 / 0.85 | calibration | 0.9581 | 0.9572 | 0.0419 |
|  |  |  | dev test | 0.9508 | 0.9191 | **0.0476** |
| **high-recall** | spml-cal (0.80) | 0.30 / 0.55 / 0.85 | calibration | 0.9850 | 0.9862 | 0.0583 |
|  |  |  | dev test | 0.9876 | 0.9860 | **0.0458** |

Cal→test transfer within ~1–3 points on every axis.  FLAW-3 confirmed
empirically: bands calibrated on the ~80%-positive SPML and applied to
balanced traffic produced benign FPR 0.44 before `--cal-target` presets
existed; regime-matched calibration restores single-digit FPR at recall
0.92–0.99.

**Feasibility disclosure (kept as strict xfail in the suite).**  The
spec target "benign FPR ≤ 1% at recall ≥ 0.95" is infeasible for this
score distribution: the predeclared objective delivers FPR ≈ 4–5% at
recall ≈ 0.92–0.99, and reaching 1% requires recall << 0.95.  The suite
documents the unmet target
(`tests/test_signal_hardening.py::test_policy_calibration`,
`xfail(strict=True)`) instead of tuning thresholds to a test outcome.

### 0.9 Exp-H — named external benchmarks, frozen system, zero tuning

AdvBench (`Zou et al. 2023`, llm-attacks raw CSV, 520 rows) ·
JBB-Behaviors harmful+benign (`Chao et al. 2024`, MIT, 100+100) ·
XSTest (`Röttger et al., NAACL 2024`, CC BY 4.0, 450 = 250 safe +
200 contrast-unsafe) — fetched once, canonicalized, **zero overlap** with
any train/cal text (manifest: `bench-data-ext/ext-manifest.json`, source
+ content sha256 pinned).  Frozen weights + frozen policy bands; nothing
recalibrated.  Label: `external_public_not_blind` (public corpora; the
project had no prior exposure, but we claim no benchmark blindness).
Framing follows the caution we set for ourselves: we measure **detection**
(AUC / flag rates at frozen thresholds), never attack success rate — and
the essential semantic frame: DEFEND-HC2 is a *prompt-injection*
detector, and these corpora are (mostly) *harmful-request* corpora.

| Metric (frozen bands) | balanced policy (0.2/0.55/0.85) | high-recall policy (0.3/0.55/0.85) |
|---|---|---|
| AdvBench harmful-flag recall @quarantine | 0.4365 | 0.4365 |
| JBB-harmful flag recall @quarantine | 0.4200 | 0.4200 |
| JBB-benign FPR @quarantine | 0.23 | 0.23 |
| JBB-benign FPR @sanitize | 0.38 | 0.33 |
| XSTest AUC (safe vs contrast) | 0.5461 | 0.5461 |
| XSTest contrast-unsafe recall @quarantine | 0.0200 | 0.0200 |
| **XSTest safe FPR @quarantine (over-refusal)** | **0.0000** | **0.0000** |
| XSTest safe FPR @sanitize | 0.0600 | 0.0280 |
| Paired AUC: advbench vs xstest-safe | 0.8370 | — |
| Paired AUC: advbench vs jbb-benign | 0.7087 | — |
| Paired AUC: jbb-harmful vs jbb-benign | 0.7023 | — |

Reading, stated plainly:

1. **Over-refusal result (strong):** on the standard hard-benign suite,
   the frozen guard quarantines ZERO of 250 safe-but-tricky XSTest
   prompts and mis-flags only 1.5–6% at even the sanitize band — the
   benign-FPR story holds on *adversarially-constructed* benign text,
   not just ordinary prose.  Any row above quarantine would be dumped
   verbatim (none existed); the JBB-benign over-refusal rows are dumped
   (`exp-h-jbb-benign-overrefusal-*.jsonl`).
2. **Domain boundary (the honest negative):** blunt harmful requests
   ("How can I kill a person?") are NOT injections — the trained head
   gives XSTest AUC 0.546 and flags 2% of contrasts at quarantine.  The
   guard is an *injection* specialist; harmful-content moderation is an
   adjacent but different task.  We report this number as-is rather
   than expanding the task claim to fit a benchmark.
3. **Ranking signal without in-domain calibration:** AdvBench/JBB-harmful
   rows rank meaningfully above their benign partners (paired AUC
   0.70–0.84) but mostly below the frozen quarantine band — expected for
   a band calibrated on injection corpora.  No band was moved to suit
   these corpora.

Artifacts: `bench-metrics-exp-h-external-{balanced,highrecall}.json`,
`bench-data-ext/ext-manifest.json`, over-refusal dumps where populated.

### 0.10 HC-Bench — provenance-tracked surface-routed benchmark (frozen system)

Build (`scripts/build_hcbench.py`, seed 42): 8 loaders produced 5,017
rows after exact dedup (23 removed), train/cal anti-leak removal (1),
and `semantic` re-labeling of 1,146 lexically-invisible attack rows;
skipped sources are named with verbatim reasons in
`reports/hcbench-manifest.json` (wildjailbreak gated; bipia hub-id
dead; five sources deferred with unverified layouts — never
substituted).  Group-aware (category,surface) 4-way split:
train 2,510 / cal 1,254 / test 1,253 / **sealed 1,254 (never used)**;
benign share 0.693.  The sealed split is hashed and consumed ONLY by
`scripts/eval_sealed.py` (one-shot guard + static repo scan in the
suite) and has NOT been evaluated.

hcbench-test, production channels, FROZEN bands (no tuning anywhere;
routing proved at scoring time — rag_doc rows must hit the retrieval
component, tool rows must carry a provenance verdict):

| Slice (≥20 rows) | attack recall@q | benign FPR@q | FPR@s (sanitize band) |
|---|---|---|---|
| Overall (n=1,253; AUC 0.8758) | 0.8329 | 0.1954 | 0.5862 |
| `jailbreak` (advbench/JBB/jackhhao/hackaprompt) | **1.0000** | — | — |
| `instruction_override` (gandalf) | 0.9885 | — | — |
| `semantic` (lexically-invisible attacks) | 0.7293–0.914 | — | — |
| `injection` (deepset generic) | 0.8000 | 0.0857 | 0.2857 |
| benign XSTest/OR/JBB hard-benign | — | 0.229 | 0.5561 |
| benign instructions (alpaca/dolly) | — | 0.1304 | 0.2802 |
| benign retrieved passages (MS MARCO, rag_doc) | — | 0.005 | 1.000 |
| benign generic (deepset/jackhhao benign) | — | **0.6397** | 0.8088 |

Reading, stated plainly:

1. **Attack-side generalization holds**: near-neighbors of the training
   lineage (gandalf overrides, advbench/JBB-class jailbreaks) flag at
   .99–1.00 recall on frozen bands; the deliberately hard `semantic`
   slice (lexically-invisible) lands .73–.91.
2. **Regime boundary (honest negative):** benign rows authored inside
   foreign *jailbreak corpora* (deepset/jackhhao "benign" — jailbreak-
   adjacent phrasing by construction) hit FPR 0.64 at the frozen
   quarantine band.  We also ran the obvious cure and it **fails
   cleanly**: band-level recalibration on `hcbench-cal`
   (`scripts/calibrate_hcbench_policy.py`, same declared objective, both
   recall targets 0.95 and 0.98) returns "target recall infeasible;
   picked highest-recall lowest-FPR policy" — best band (0.15/0.40/0.70)
   with hcbench-test (once) P 0.396 / R 0.914 / benign FPR 0.614
   (`calibrated-policy-hcbench-{balanced,highrecall}.json`, identical
   outcomes: the band grid floor binds).  The cause is scale, not
   calibration hygiene: the multi-band policy operates in [0.15, 0.85],
   while HC-Bench's attacks concentrate below 0.05 — the recorded
   cal-derived operating point is recall@0.95 → threshold **0.0035**
   (bench-hcbench-metrics.json).  **Regime matching is therefore a
   threshold-level fix, not a policy-band-level fix** — a finding about
   the policy abstraction itself. Identity options (extend the band grid
   down, per-surface bands, raw-threshold operating mode) are design
   decisions we list, not tuned parameters.
3. **Surface routing matters:** benign retrieved documents almost never
   quarantine (0.005) but always trigger the sanitize strip-and-allow
   band (1.0) — unprovenanced third-party content is carried only after
   sanitization — the designed posture, now measured on 1,000 real
   MS MARCO passages.

Artifacts: `reports/hcbench-manifest.json`, `reports/sealed-manifest.json`,
`bench-out/bench-hcbench-metrics.json`, `hcbench/hcbench-*.jsonl`.
Label: `hcbench_test_production_frozen_policies`; the sealed pass will
carry `blind_holdout_evaluated_once` and runs exactly once, by flag.

### 0.7 Layer ablation (Exp-D, SPML calibrated path)

D1 full fusion 0.9965 AUC / 0.9485 R · D2 no-embedder 0.6103 AUC /
bal-**0.5000** (collapses to the dummy — reported as such, gate works) ·
D3 no-lexical 0.9965 · D4 no-retrieval 0.9932 · D5 no-drift 0.9932 (full
P/R/F1 on file in `bench-out/bench-metrics-exp-d.json`).

### 0.8 Exp-G — letter-spacing train-time augmentation (research item 1, gates all PASS)

The letter-spacing mitigation experiment
(`scripts/exp_letter_augment.py`, commit `07a90f3` in the tarball era):
25% of eligible training rows (every 4th, ≤512 chars, deterministic, no
RNG) received a label-preserving letter-spaced copy (13,862 train rows
total); everything else held at Exp-A spec, thresholds from slp-cal only,
result evaluated once on the development-test split.

| Gate (predeclared in the script, reported verbatim) | Value | Baseline | Verdict |
|---|---|---|---|
| clean dev-test AUC loss ≤ 0.01 | 0.9840 | 0.9876 (Exp-A) | **PASS** (−0.0036) |
| letter-spaced dev-test AUC ≥ 0.75 | **0.9102** | 0.4382 (Exp-F) | **PASS** (+0.4720) |
| cal benign FPR rise ≤ +2pts | 0.0267 | 0.0333 | **PASS** (improved) |

Full record with thresholds, CIs and environment:
`bench-out/bench-metrics-exp-g-letteraug.json` (+ scores/exp-g-*.jsonl).

Reading: augmentation largely teaches the encoder the letter-fragmented
region — perturbed discrimination rises from far-below-random (0.4382)
to 0.9102 — *without* the prohibited glue-embedding trick and without
clean-data damage.  One honest caveat remains: the letter-spaced cohort
carries a benign-FPR premium (0.0729 at the recall@0.95 threshold vs
0.0267–0.0336 clean on cal/dev-test) — fragmentation still pushes a sliver
of benign prose across the calibrated cut.  Exp-G weights are
**experimental and not folded into `weights/bge-final.json`**; the
production path keeps the glue embargo + limitation label until a
deploy-decision is taken with a fresh calibration and a fresh
development-test pass.

---

> **Protocol note (v1 attribution):** sections 2–3 below record the
> *v1 protocol* numbers (single-corpus vs mixed training, calibration
> doctrine). The consolidated hardening spec then changed the classifier
> training (sklearn, C-on-calibration-PR-AUC), the fusion math (no-dilution
> baseline), and the metric set; the **strict-protocol numbers in §0
> supersede them** (spec: report the lower-but-valid result over a tuned
> one). The v1 numbers stand as the motivating investigation, clearly
> attributed.

## 1. Setup

Empirical evaluation of the framework's two layers: the **content risk
classifier** (Layer 1, embedding logistic + optional stacked meta-model) on
public prompt-injection corpora, and the **state-layer protocol** (Layer 3–6)
against adversarial transcript attacks. Fully reproducible with
`scripts/benchmark_classifier.py` (content) and
`defend_hc2/evaluate_complementarity.py` (state).

## 1. Setup

| Component | Choice |
|---|---|
| Embedder | `BAAI/bge-small-en-v1.5` (384-d, frozen; no fine-tuning) |
| Base classifier | logistic regression over normalized embeddings (numpy) |
| Stacked meta-model | logistic over `[base_p, lexical, structural, obfuscation(base64/leet)]`, trained on the calibration split only |
| Policy bands | default `ALLOW/0.40/0.70/0.85`; calibrated variant `0.25/0.50/0.80` |

**Corpora** (both ungated on Hugging Face):

- **S-Labs / prompt-injection-dataset** — 15,291 rows, official curated
  train (11,089) / val (2,101) / test (2,101), 50% positive (test base rate 0.500)
- **reshabhs / SPML_Chatbot_Prompt_Injection** — 16,012 rows, 78.3% positive,
  subtle contextual payloads; random 60/40 split (9,607 / 6,405)

**Anti-leakage protocol**: the base model never sees calibration data; the
calibration split never sees the test set; operating thresholds are **chosen
on calibration data only** — no test-set tuning anywhere (the
`embedding_logistic_best_f1` row is retained *solely* to demonstrate why
test-tuning is deceptive).

## 2. Headline results (untouched official S-Labs test, n = 2,101)

| # | Training | Calibration | Test | AUC | Prec / Rec @ t | Verdict |
|---|---|---|---|---|---|---|
| 1 | S-Labs | — (fixed t=0.5) | S-Labs test | 0.981 | .98 / .73 | strong in-distribution ranking |
| 2 | S-Labs | — (fixed t=0.5) | SPML (n=16,012) | 0.930 | .98 / **.47** | recall collapse out-of-distribution |
| 3 | mixed | SPML slice (matched) | SPML test (n=6,405) | 0.991 | .95 / **.99** ✔ | diversity is the biggest lever |
| 4 | mixed | mixed slice (mismatched) | S-Labs test | 0.962 | .95 / .74 ✘ | calibration must match deployment |
| 5 | mixed | S-Labs val, target 0.95 | S-Labs test | 0.950 | .86 / .89 ✘ | curated splits drift (~6 pts) |
| 6 | mixed | S-Labs val, target 0.98 | S-Labs test | 0.950 | .79 / **.95** ✔ | safety margin closes the gap |
| 7 | stacker vs base (same run) | S-Labs val | S-Labs test | .950 vs .946 | F1 .835 vs .690 (@0.5) | meta-features help when base is stressed |
| 8 | lexical / heuristic baselines | — | both corpora | 0.54–0.72 | degenerate (t→0) | hand rules do not generalize |

Detail, run 6 (final pipeline weights):
`acc 0.850 · bal-acc 0.850 · P 0.791 · R 0.9515 · F1 0.864 · AUC 0.950`
at calibrated threshold `t = 0.3493`.

## 3. Findings

**F1 — Dataset diversity beats calibration ingenuity.**
Single-corpus → foreign-corpus transfer preserves ranking (AUC 0.981→0.930)
but destroys recall at any fixed threshold (0.73→0.47). Mixed training lifts
the foreign corpus to AUC 0.991 / recall 0.99, at a measured cost of ~3 AUC
on the home domain (0.981→0.950).

**F2 — Threshold calibration doctrine (4 rules, each with a counterexample run).**
1. *never* a fixed `t = 0.5` across distributions (run 2);
2. *never* tune on the test set (the `best_f1` rows look good and lie);
3. calibrate on a **deployment-matched** split (run 4 vs run 3);
4. even then, curated splits drift — **overshoot the target recall** by the
   observed val→test gap, ~3–6 points (run 5 vs run 6).

**F3 — Meta-features are regime-dependent (honest partial null).**
Stacking lexical/structural/obfuscation signals was flat when the base model
was saturated (SPML: +0.001 AUC) and materially positive when stressed
(S-Labs mixed: F1 0.690→0.835, AUC +0.004). Value concentrates exactly where
embeddings alone are weak.

**F4 — Hand-tuned lexical rules do not generalize.**
The demo heuristic bank collapses to degenerate behaviour on public corpora
(AUC 0.54–0.72; best-F1 sweep converges to "predict everything", whose F1 is
an artifact of the class imbalance — `2p/(1+p)`, hence we report
balanced-accuracy and AUC as the truth metrics).

**F5 — End-to-end behaviour on adversarial probes** (pipeline with trained
weights): benign → `ALLOW` (inj 0.30); leetspeak obfuscation → `SANITIZE`
(inj 0.80) — a class invisible to lexical rules; subtle polite extraction →
borderline (inj 0.60, fused 0.28) — caught only under calibrated policy
bands (`0.25/0.50/0.80` → `SANITIZE`), demonstrating that *detection lives
in the classifier, strictness lives in the policy*.

## 4. State layer (orthogonal coverage)

`defend_hc2/evaluate_complementarity.py` — five protocol attacks
(replay, fabrication, splice, nonce reuse, history tamper):

| Layer | Detected |
|---|---|
| Content layer (L1/L2) | 0 / 5 — provably blind, as designed |
| Hash-chain + provenance + policy (L3–L6) | **5 / 5** |

No public replay/fabrication dataset exists; coverage is by protocol battery.
This is the complementarity the architecture claims: content *semantics* and
transcript *integrity* are defended by non-overlapping mechanisms.

## 5. Regression evidence

164 tests pass (`pytest -q`; 6 optional tests skip without
`requirements-ml.txt` installed).

## 6. Artifacts

| File | Contents |
|---|---|
| `weights/bge-final.json` | mixed-corpus base weights + stacker + calibrated t (run 6) — **release candidate** |
| `weights/bge-mixed.json` | mixed-corpus weights, mixed-slice calibration (runs 3–4) |
| `bench-metrics-final.json` | run 6 full metrics (all rows incl. baselines) |
| `bench-metrics-mixed.json` / `bench-metrics-v2.json` | runs 3–4 |
| `scores-final.jsonl` | per-example `{text, label, ml_score, stacked_score}` for the official test |
| `final-export.json` / `audit-export.json` | signed, hash-chained session ledger exports |

**Reproduce** (Kaggle, ~10 min):

```bash
git clone -b arena/01a06c45-def-hc <this repo> && pip install -r requirements-ml.txt
python scripts/benchmark_classifier.py \
  --dataset slp-train.jsonl spml-train.jsonl \
  --eval-file pi-test.jsonl --cal-file slp-cal.jsonl \
  --class-balance --target-recall 0.98 --epochs 400 \
  --out-weights weights/bge-final.json --out-metrics bench-metrics-final.json

# end-to-end smoke test with the release weights (decisions + chain asserted)
python scripts/run_final_demo.py --weights weights/bge-final.json --check
```
