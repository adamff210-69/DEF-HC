"""Reproducible benchmark data preparation (spec Phase 1).

* S-Labs official splits, respected EXACTLY (train / val / test — never merged);
* SPML: full schema retained, NFKC/template dedup, group-aware stratified
  60/20/20 split (seed 42), leakage statistics, no group crossing (asserted);
* foreign zero-shot corpora attempted with DOCUMENTED outcomes — a gated,
  unavailable, or incompatible dataset is skipped with the reason, never
  silently replaced.

Writes SHA-256 manifest lines for every generated file (Phase 20).

Runs on machines with Hugging Face network access (e.g. Kaggle); all split
logic is deterministic and unit-tested in-repo (tests/test_splitting.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow bare run

from defend_hc2.modeling import environment_block, file_sha256, git_commit
from defend_hc2.splitting import (
    assert_no_group_crossing,
    group_stratified_split,
    leakage_stats,
    normalize_key,
)

SEED = 42


def dump_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {path}  ({len(rows)} rows, sha256 {file_sha256(path)[:16]}…)")


def prepare_slabs(out_dir: Path) -> dict:
    from datasets import load_dataset

    print("== S-Labs/prompt-injection-dataset (official splits, exact) ==")
    ds = load_dataset("S-Labs/prompt-injection-dataset")
    stats = {}
    for split, fname in (("train", "slp-train.jsonl"),
                         ("validation", "slp-cal.jsonl"),
                         ("test", "pi-test.jsonl")):
        rows = [{"text": r["text"], "label": int(r["label"])} for r in ds[split]]
        stats[fname] = leakage_stats(rows, "text", "label")
        dump_jsonl(rows, out_dir / fname)
    return stats


def prepare_spml(out_dir: Path) -> dict:
    from datasets import load_dataset

    print("== reshabhs/SPML_Chatbot_Prompt_Injection (full schema retained) ==")
    ds = load_dataset("reshabhs/SPML_Chatbot_Prompt_Injection")
    split_name = list(ds.keys())[0]
    cols = {c.strip().lower().replace(" ", "_"): c for c in ds[split_name].column_names}
    text_col = next((c for k, c in cols.items() if "user" in k and "prompt" in k),
                    next((c for k, c in cols.items() if k == "text"), None))
    sys_col = next((c for k, c in cols.items() if "system" in k and "prompt" in k), None)
    label_col = next((c for k, c in cols.items()
                      if "injection" in k or k in ("label", "class")), None)
    deg_col = next((c for k, c in cols.items() if "degree" in k), None)
    src_col = next((c for k, c in cols.items() if k in ("source", "origin")), None)
    if not all((text_col, label_col)):
        raise SystemExit(f"SPML schema unrecognized: {ds[split_name].column_names}")
    print(f"  schema: text={text_col!r} system={sys_col!r} label={label_col!r} "
          f"degree={deg_col!r} source={src_col!r}")

    rows = [{
        "text": str(r[text_col]),
        "system_prompt": str(r[sys_col]) if sys_col else None,
        "label": 1 if str(r[label_col]).strip().lower() in ("1", "true", "yes", "1.0") else 0,
        "degree": str(r[deg_col]) if deg_col else None,
        "source": str(r[src_col]) if src_col else None,
    } for r in ds[split_name]]

    print("  leakage stats (pre-dedup):", leakage_stats(rows, "text", "label"))
    seen, dedup = set(), []
    for row in rows:
        key = normalize_key(row["text"])
        if key not in seen:
            seen.add(key)
            dedup.append(row)
    print(f"  exact duplicates removed: {len(rows) - len(dedup)}")

    parts = group_stratified_split(dedup, "text", "label", seed=SEED, fractions=(0.6, 0.2, 0.2))
    assert_no_group_crossing(parts, "text")
    names = ("spml-train.jsonl", "spml-cal.jsonl", "spml-test.jsonl")
    stats = {}
    for name, part in zip(names, parts):
        stats[name] = leakage_stats(part, "text", "label")
        dump_jsonl(part, out_dir / name)
    print("  leakage stats (post-split):",
          {k: v["template_duplicates"] for k, v in stats.items()})
    return stats


def prepare_foreign(out_dir: Path, train_cal_keys: set[str]) -> dict:
    """Attempt foreign zero-shot corpora; DOCUMENT every outcome."""
    from datasets import load_dataset

    candidates = {
        "safe-guard": ("xTRam1/safe-guard-prompt-injection",),
        "jailbreak-classification": ("jackhhao/jailbreak-classification",),
        "deepset": ("deepset/prompt-injections",),
    }
    stats = {}
    for name, (repo,) in candidates.items():
        print(f"== foreign zero-shot: {repo} ==")
        try:
            ds = load_dataset(repo)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {str(exc)[:160]}"
            print(f"  SKIPPED — unavailable/gated ({reason})")
            stats[name] = {"status": "skipped", "reason": reason}
            continue
        try:
            split_name = "test" if "test" in ds else list(ds.keys())[0]
            cols = {c.strip().lower().replace(" ", "_"): c
                    for c in ds[split_name].column_names}
            text_col = next((c for k, c in cols.items()
                             if k in ("text", "prompt", "user_prompt", "question")), None)
            label_col = next((c for k, c in cols.items()
                              if any(t in k for t in ("label", "class", "injection", "type"))), None)
            if not all((text_col, label_col)):
                raise ValueError(f"no usable text/label columns in {ds[split_name].column_names}")
            raw_labels = sorted({str(r[label_col]) for r in ds[split_name]})
            print(f"  raw labels observed: {raw_labels} (mapping: see manifest)")

            def to01(v) -> int:
                s = str(v).strip().lower()
                pos = ("1", "true", "yes", "injection", "jailbreak", "malicious", "unsafe", "1.0")
                neg = ("0", "false", "no", "benign", "safe", "legitimate", "0.0")
                if s in pos:
                    return 1
                if s in neg:
                    return 0
                raise ValueError(f"unmappable label {v!r}")

            rows, unmappable = [], 0
            for r in ds[split_name]:
                try:
                    rows.append({"text": str(r[text_col]), "label": to01(r[label_col])})
                except ValueError:
                    unmappable += 1
            before = len(rows)
            seen, dedup = set(), []
            for row in rows:
                key = normalize_key(row["text"])
                if key not in seen and key not in train_cal_keys:
                    seen.add(key)
                    dedup.append(row)
            removed = before - len(dedup)
            print(f"  rows kept {len(dedup)}; removed {removed} "
                  f"(exact-dup / overlap-with-train-cal); unmappable labels: {unmappable}")
            stats[name] = {**leakage_stats(dedup, "text", "label"),
                           "status": "ok", "label_mapping": "pos/neg sets in script source",
                           "removed": removed, "unmappable": unmappable}
            dump_jsonl(dedup, out_dir / f"foreign-{name}.jsonl")
        except Exception as exc:
            reason = f"{type(exc).__name__}: {str(exc)[:160]}"
            print(f"  SKIPPED — incompatible schema ({reason})")
            stats[name] = {"status": "skipped", "reason": reason}
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--skip-foreign", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": SEED, "git_commit": git_commit(Path(__file__).resolve().parents[1]),
                "environment": environment_block(), "datasets": {}}

    manifest["datasets"]["s-labs"] = prepare_slabs(args.out_dir)
    manifest["datasets"]["spml"] = prepare_spml(args.out_dir)

    if not args.skip_foreign:
        train_cal_keys: set[str] = set()
        for fname in ("slp-train.jsonl", "slp-cal.jsonl", "spml-train.jsonl", "spml-cal.jsonl"):
            fp = args.out_dir / fname
            if fp.exists():
                for line in fp.open(encoding="utf-8"):
                    train_cal_keys.add(normalize_key(json.loads(line)["text"]))
        manifest["datasets"]["foreign"] = prepare_foreign(args.out_dir, train_cal_keys)

    manifest_path = args.out_dir / "prepare-manifest.json"
    manifest["file_sha256"] = {
        fp.name: file_sha256(fp)
        for fp in sorted(args.out_dir.glob("*.jsonl"))
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
