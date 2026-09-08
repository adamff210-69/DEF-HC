"""Error analysis over development-test score files (spec Phase 19).

Reads ``scores-*.jsonl`` artifacts, buckets FP/FN examples into documented
categories with deterministic keyword heuristics, and writes
``reports/error-analysis.md`` with aggregate counts and quoted examples.

Benchmark labels are NEVER changed here; if annotation problems are
suspected they are *reported* (original label retained in the headline
metrics, per the non-negotiable rules).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow bare run

CATEGORIES: list[tuple[str, tuple[str, ...]]] = [
    ("explicit override", ("ignore", "disregard", "forget", "override", "bypass")),
    ("prompt extraction", ("system prompt", "system-prompt", "instructions",
                            "reveal", "repeat", "verbatim", "configuration")),
    ("persona jailbreak", ("you are now", "act as", "pretend", "dan", "jailbreak",
                            "developer mode", "no restrictions")),
    ("obfuscation", ("base64", "zero", "decode", "\u200b")),
    ("benign security discussion", ("prompt injection", "jailbreak", "red team",
                                    "vulnerability", "attack", "exploit")),
    ("code/security discussion", ("code", "python", "sql", "script", "function")),
    ("role-play", ("roleplay", "role-play", "character", "story", "fiction")),
    ("indirect instruction", ("translate", "summarize", "following document",
                              "when asked", "respond with")),
]


def categorize(text: str) -> str:
    low = text.lower()
    for name, keys in CATEGORIES:
        if any(k in low for k in keys):
            return name
    return "ambiguous/unclassified"


def analyze(scores_path: Path, max_examples: int = 5) -> dict:
    rows = [json.loads(l) for l in scores_path.open(encoding="utf-8") if l.strip()]
    errors = [r for r in rows if r["error"] in ("FP", "FN")]
    by_kind: dict[str, Counter] = {"FP": Counter(), "FN": Counter()}
    examples: dict[str, dict[str, list[str]]] = {"FP": {}, "FN": {}}
    for r in errors:
        cat = categorize(r["text"])
        by_kind[r["error"]][cat] += 1
        bucket = examples[r["error"]].setdefault(cat, [])
        if len(bucket) < max_examples:
            bucket.append(r["text"][:200])
    return {
        "file": scores_path.name,
        "n": len(rows),
        "errors": len(errors),
        "by_kind": {k: dict(v) for k, v in by_kind.items()},
        "examples": examples,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="e.g. reports/error-analysis.md")
    args = ap.parse_args()

    parts = ["# Error analysis (development-test evaluations,\n"           "# development_test_previously_observed)\n",
             "Labels are those of the source benchmarks and are never "
             "relabelled; suspected annotation issues are listed, not fixed.\n"]
    for path in args.scores:
        if not path.exists():
            print(f"skip missing {path}")
            continue
        res = analyze(path)
        parts.append(f"\n## {res['file']}  (n={res['n']}, errors={res['errors']})\n")
        for kind in ("FP", "FN"):
            parts.append(f"\n### {kind}\n")
            for cat, cnt in sorted(res["by_kind"][kind].items(), key=lambda kv: -kv[1]):
                parts.append(f"- **{cat}**: {cnt}")
            if not res["by_kind"][kind]:
                parts.append("- none")
            for cat, exs in res["examples"][kind].items():
                parts.append(f"\n  _{cat}_ examples:")
                for ex in exs:
                    parts.append(f"  - `{ex}`")
    parts.append("\n\n_Note: headline metrics retain original benchmark labels; "
                 "this analysis is diagnostic only._\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
