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

    # ---- anti-leak removal vs EVERY previously-observed corpus.
    # Not just train/cal: any row this project has already scored or
    # inspected (the foreign transfer sets used in Exp-B, the development
    # test splits) must not reappear in hcbench-test/sealed, or the sealed
    # split cannot honestly be called a blind holdout.
    from defend_hc2.modeling import load_jsonl
    guard: list[tuple[str, int]] = []
    guard_used, guard_missing = [], []
    for name in ("slp-train.jsonl", "slp-cal.jsonl", "slp-test.jsonl",
                 "spml-train.jsonl", "spml-cal.jsonl", "spml-test.jsonl",
                 "pi-test.jsonl",
                 "foreign-deepset.jsonl",
                 "foreign-jailbreak-classification.jsonl",
                 "foreign-safe-guard.jsonl"):
        fp = args.leak_guard_dir / name
        if fp.exists():
            guard += load_jsonl(fp)
            guard_used.append(name)
        else:
            guard_missing.append(name)
    print(f"leak guard: {len(guard)} previously-observed rows "
          f"from {len(guard_used)} files")
    if guard_missing:
        print(f"  NOTE absent, so NOT guarded against: {guard_missing}")
    guard_keys = {norm_for_dedup(t) for t, _ in guard}
    kept, removed_by_source = [], Counter()
    for r in deduped:
        if norm_for_dedup(r["text"]) in guard_keys:
            removed_by_source[r["source"]] += 1
        else:
            kept.append(r)
    print(f"overlap with previously-observed corpora removed "
          f"{sum(removed_by_source.values())} rows: {dict(removed_by_source)}")

    # ---- lexical-invisibility flag (ORTHOGONAL — category is preserved).
    # This previously overwrote the row's category with "semantic", which
    # pulled ~60% of attacks out of their own class and left each attack
    # category holding only the rows the lexical channel already fires on
    # — making per-category recall circular.  Keep the difficulty axis,
    # keep the class axis, report the cross-tab.
    n_invisible = 0
    for r in kept:
        if r["label"] == 1:
            r["lexically_invisible"] = analyzer.lexical_scan(r["text"])[0] == 0
            n_invisible += int(r["lexically_invisible"])
        else:
            r["lexically_invisible"] = None
    n_attacks = sum(1 for r in kept if r["label"] == 1)
    print(f"lexically-invisible attacks flagged: {n_invisible}/{n_attacks} "
          f"({n_invisible / max(1, n_attacks):.1%}) — category preserved")

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
        "corpus_rows_after_filtering": len(kept),
        "benign_share": round(benign_share, 4),
        "per_category": dict(Counter(r["category"] for r in kept)),
        "per_surface": dict(Counter(r["surface"] for r in kept)),
        "per_surface_attacks": dict(Counter(
            r["surface"] for r in kept if r["label"] == 1)),
        "dedup_removed_by_source": dict(dup_by_source),
        "overlap_removed_by_source": dict(removed_by_source),
        "leak_guard_files_used": guard_used,
        "leak_guard_files_absent": guard_missing,
        "leak_guard_rows": len(guard),
        "lexically_invisible_attacks": n_invisible,
        "attack_rows": n_attacks,
        "seed": 42, "rules": ["no oversample", "no train on cal/test/sealed",
                              "sealed inspected once via eval_sealed.py only",
                              "no silent drops",
                              "lexical-invisibility is a flag, never a "
                              "category overwrite"],
    }
    (args.reports / "hcbench-manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"\nmanifest: {args.reports}/hcbench-manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
