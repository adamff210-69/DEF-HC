"""Build HC-Bench: provenance-tracked, surface-routed benchmark.

Pipeline (hard rules enforced + reported):
  loaders -> schema validation -> exact/template dedup
          -> anti-leak removal vs slp/spml train+cal
          -> 'semantic' re-labeling of lexically-invisible attack rows
          -> group/template id assignment
          -> group-aware (category,surface)-stratified 4-way split
             (train .40 / cal .20 / test .20 / sealed .20, seed 42)
          -> hcbench-{train,cal,test}.jsonl, hcbench-sealed.jsonl,
             reports/sealed-manifest.json, reports/hcbench-manifest.json

DO NOTs: no oversampled eval rows, no sealed inspection, no training on
cal/test/sealed, no silent row drops (every removal prints a count + source).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(Path(__file__).resolve().parents[1]))

from defend_hc2.content_risk import ContentRiskAnalyzer
from defend_hc2.hcbench import DEFERRED_SOURCES, LOADERS, validate_row
from defend_hc2.modeling import norm_for_dedup
from defend_hc2.splitting import (
    assert_no_group_crossing,
    group_id,
    group_stratified_split,
    normalize_key,
    template_key,
)

SPLIT_NAMES = ("train", "cal", "test", "sealed")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("hcbench"))
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument("--leak-guard-dir", type=Path, default=Path("bench-data"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)

    analyzer = ContentRiskAnalyzer(demo_mode=True)  # lexical only, no model

    all_rows: list[dict] = []
    loader_report: dict[str, dict] = {}
    for name, fn in LOADERS.items():
        try:
            rows = fn()
        except Exception as exc:  # noqa: BLE001 — skip, never substitute
            reason = f"{type(exc).__name__}: {exc}"[:220]
            print(f"SKIP {name:18s} — {reason}")
            loader_report[name] = {"status": "skipped", "reason": reason}
            continue
        all_rows += rows
        print(f"LOAD {name:18s} n={len(rows)}")
        loader_report[name] = {"status": "loaded", "n": len(rows)}
    for name, reason in DEFERRED_SOURCES.items():
        print(f"SKIP {name:18s} — deferred: {reason}")
        loader_report[name] = {"status": "deferred", "reason": reason}

    # ---- schema validation: invalid rows reported, never silently dropped
    bad = [r for r in all_rows if not validate_row(r)]
    if bad:
        print(f"DROP invalid-schema rows: {len(bad)} "
              f"(sources: {sorted({r.get('source', '?') for r in bad})})")
    rows = [r for r in all_rows if validate_row(r)]

    # ---- exact + template dedup with per-source accounting
    seen: set[str] = set()
    deduped: list[dict] = []
    dup_by_source: Counter[str] = Counter()
    for r in rows:
        key = normalize_key(r["text"])
        if key in seen:
            dup_by_source[r["source"]] += 1
            continue
        seen.add(key)
        deduped.append(r)
    print(f"exact dedup removed {sum(dup_by_source.values())} rows: "
          f"{dict(dup_by_source)}")

    # ---- anti-leak removal vs existing train/cal corpora
    from defend_hc2.modeling import load_jsonl
    guard: list[tuple[str, int]] = []
    for name in ("slp-train.jsonl", "slp-cal.jsonl",
                 "spml-train.jsonl", "spml-cal.jsonl"):
        fp = args.leak_guard_dir / name
        if fp.exists():
            guard += load_jsonl(fp)
    guard_keys = {norm_for_dedup(t) for t, _ in guard}
    kept, removed_by_source = [], Counter()
    for r in deduped:
        if norm_for_dedup(r["text"]) in guard_keys:
            removed_by_source[r["source"]] += 1
        else:
            kept.append(r)
    print(f"overlap with train/cal removed {sum(removed_by_source.values())} "
          f"rows: {dict(removed_by_source)}")

    # ---- 'semantic' category: attack rows invisible to the lexical scanner
    n_semantic = 0
    for r in kept:
        if r["label"] == 1 and analyzer.lexical_scan(r["text"])[0] == 0:
            r["category"] = "semantic"
            n_semantic += 1
    print(f"semantic re-labeled (attack, lexical_scan==0): {n_semantic}")

    # ---- group/template ids
    for r in kept:
        r["group_id"] = group_id(template_key(r["text"]))

    # ---- (category,surface)-stratified 4-way split, seed 42
    parts_named: dict[str, list[dict]] = {n: [] for n in SPLIT_NAMES}
    strata: dict[tuple[str, str], list[dict]] = {}
    for r in kept:
        strata.setdefault((r["category"], r["surface"]), []).append(r)
    for key in sorted(strata):
        parts = group_stratified_split(strata[key], "text", "label",
                                       (0.40, 0.20, 0.20, 0.20), seed=42)
        for name, part in zip(SPLIT_NAMES, parts):
            for r in part:
                r["split"] = name
            parts_named[name] += part
        print(f"  stratum {key}: " +
              ", ".join(f"{n}={len(p)}" for n, p in zip(SPLIT_NAMES, parts)))
    assert_no_group_crossing([parts_named[n] for n in SPLIT_NAMES], "text")

    # ---- benign share assert
    benign_share = sum(1 for r in kept if r["label"] == 0) / max(1, len(kept))
    print(f"benign share overall: {benign_share:.3f}")
    assert benign_share >= 0.40, f"benign share {benign_share:.3f} < 0.40"

    # ---- write splits
    for name in SPLIT_NAMES:
        fp = args.out_dir / f"hcbench-{name}.jsonl"
        with fp.open("w", encoding="utf-8") as fh:
            for r in parts_named[name]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {fp} (n={len(parts_named[name])})")

    seal_blob = json.dumps([r["id"] for r in parts_named["sealed"]]).encode()
    sealed_manifest = {
        "file": "hcbench-sealed.jsonl", "n": len(parts_named["sealed"]),
        "sha256_of_ids": hashlib.sha256(seal_blob).hexdigest(),
        "category_histogram": dict(Counter(r["category"]
                                           for r in parts_named["sealed"])),
        "rule": "never loaded anywhere except scripts/eval_sealed.py, "
                "and only with --i-understand-this-runs-once",
    }
    (args.reports / "sealed-manifest.json").write_text(
        json.dumps(sealed_manifest, indent=2))

    manifest = {
        "loaders": loader_report,
        "totals": {n: len(parts_named[n]) for n in SPLIT_NAMES},
        "benign_share": round(benign_share, 4),
        "per_category": dict(Counter(r["category"] for r in kept)),
        "per_surface": dict(Counter(r["surface"] for r in kept)),
        "dedup_removed_by_source": dict(dup_by_source),
        "overlap_removed_by_source": dict(removed_by_source),
        "semantic_relabeled": n_semantic,
        "seed": 42, "rules": ["no oversample", "no train on cal/test/sealed",
                              "sealed inspected once via eval_sealed.py only",
                              "no silent drops"],
    }
    (args.reports / "hcbench-manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"\nmanifest: {args.reports}/hcbench-manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
