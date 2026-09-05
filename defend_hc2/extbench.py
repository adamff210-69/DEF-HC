"""Named external benchmark ingestion (Exp-H) — pure CSV parsers.

Named, citable benchmarks scored by the FROZEN DEFEND-HC2 pipeline as
external zero-shot evaluation rows — never used for training,
calibration, threshold selection, or anything blendable into fit.

Label semantics follow the source papers:
  * AdvBench (Zou et al. 2023): all rows are harmful behaviors → label 1
  * JBB-Behaviors harmful split (Chao et al. 2024): label 1; the benign
    split is benignly-worded counterparts → label 0
  * XSTest v2 (Röttger et al., NAACL 2024): `type` starting with
    "contrast_" marks the unsafe contrast prompts → label 1; the safe
    suite rows → label 0.  (Over-refusal = safe rows flagged.)

URLs/locators are verified upstream publications; corpora are fetched
at runtime and never committed to this repository (licensing).
"""

from __future__ import annotations

import csv
import io

# --- verified locators (checked 2026-09-05) --------------------------
ADVBENCH_URL = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
                "main/data/advbench/harmful_behaviors.csv")
JBB_HARMFUL_URL = ("https://huggingface.co/datasets/JailbreakBench/"
                   "JBB-Behaviors/raw/main/data/harmful-behaviors.csv")
JBB_BENIGN_URL = ("https://huggingface.co/datasets/JailbreakBench/"
                  "JBB-Behaviors/raw/main/data/benign-behaviors.csv")
XSTEST_V2_URL = ("https://raw.githubusercontent.com/paul-rottger/"
                 "exaggerated-safety/main/xstest_v2_prompts.csv")

BENCH_META = {
    "advbench": {
        "url": ADVBENCH_URL, "license": "research (llm-attacks repo)",
        "citation": "Zou et al. 2023, Universal and Transferable "
                    "Adversarial Attacks on Aligned Language Models",
        "expected_rows": 520,
    },
    "jbb-harmful": {
        "url": JBB_HARMFUL_URL, "license": "MIT",
        "citation": "Chao et al. 2024, JailbreakBench (JBB-Behaviors)",
        "expected_rows": 100,
    },
    "jbb-benign": {
        "url": JBB_BENIGN_URL, "license": "MIT",
        "citation": "Chao et al. 2024, JailbreakBench (JBB-Behaviors)",
        "expected_rows": 100,
    },
    "xstest-v2": {
        "url": XSTEST_V2_URL, "license": "CC BY 4.0",
        "citation": "Röttger et al. 2024, XSTest (NAACL 2024)",
        "expected_rows": 450,
    },
}

Row = tuple[str, int, dict]


def parse_advbench_csv(src: str) -> list[Row]:
    """goal/target columns; every row is a harmful behavior (label 1)."""
    out: list[Row] = []
    for rec in csv.DictReader(io.StringIO(src)):
        goal = (rec.get("goal") or "").strip()
        if goal:
            out.append((goal, 1, {"target": (rec.get("target")
                                             or "").strip()[:200]}))
    return out


def parse_jbb_csv(src: str, *, split: str) -> list[Row]:
    """JBB-Behaviors (Behavior, Goal, Target, Category, Source)."""
    label = 1 if split == "harmful" else 0
    out: list[Row] = []
    for rec in csv.DictReader(io.StringIO(src)):
        goal = (rec.get("Goal") or "").strip()
        if goal:
            out.append((goal, label, {
                "behavior": (rec.get("Behavior") or "").strip(),
                "category": (rec.get("Category") or "").strip(),
                "bench_source": (rec.get("Source") or "").strip(),
            }))
    return out


def parse_xstest_csv(src: str) -> list[Row]:
    """XSTest v2: `type` prefixed with 'contrast_' = unsafe (label 1)."""
    out: list[Row] = []
    for rec in csv.DictReader(io.StringIO(src)):
        prompt = (rec.get("prompt") or "").strip()
        ptype = (rec.get("type") or "").strip()
        if prompt:
            label = 1 if ptype.startswith("contrast_") else 0
            out.append((prompt, label, {
                "xstest_type": ptype,
                "id_v2": (rec.get("id_v2") or rec.get("id_v1") or "").strip(),
            }))
    return out


PARSERS = {
    "advbench": lambda src: parse_advbench_csv(src),
    "jbb-harmful": lambda src: parse_jbb_csv(src, split="harmful"),
    "jbb-benign": lambda src: parse_jbb_csv(src, split="benign"),
    "xstest-v2": parse_xstest_csv,
}
