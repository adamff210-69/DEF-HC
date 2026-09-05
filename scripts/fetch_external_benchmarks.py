"""Fetch named external benchmarks for Exp-H (idempotent, e2e-verified).

Downloads AdvBench / JBB-Behaviors / XSTest v2 (verified raw locators in
defend_hc2.extbench), normalizes to jsonl, removes any anti-leak overlap
with the train/cal corpora (canonicalized, clamp-tolerant — same guard
the S-Labs prep uses), and writes a manifest with source/content
sha256 + publisher license/citation + expected-vs-actual row counts.

These corpora are NEVER committed to git and NEVER used for training or
calibration — external zero-shot evaluation rows only
(label: external_public_not_blind).

Example:
    python scripts/fetch_external_benchmarks.py \\
        --out-dir bench-data-ext --leak-guard-dir bench-data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow bare run

from defend_hc2.extbench import BENCH_META, PARSERS
from defend_hc2.modeling import remove_overlap


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "def-hc-exp-h"})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (pinned URLs)
        return r.read().decode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("bench-data-ext"))
    ap.add_argument("--leak-guard-dir", type=Path, default=Path("bench-data"),
                    help="dir with slp-train/slp-cal/spml-train jsonl used by "
                         "the anti-leak guard")
    args = ap.parse_args()

    from defend_hc2.modeling import load_jsonl
    guard_rows: list[tuple[str, int]] = []
    for name in ("slp-train.jsonl", "slp-cal.jsonl", "spml-train.jsonl",
                 "spml-cal.jsonl"):
        fp = args.leak_guard_dir / name
        if fp.exists():
            guard_rows += load_jsonl(fp)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"label": "external_public_not_blind",
                "leak_guard_files_guarded": len(guard_rows), "files": {}}
    for bench, meta in BENCH_META.items():
        src = _fetch(meta["url"])
        rows = PARSERS[bench](src)
        expected = meta.get("expected_rows")
        status = "ok"
        if expected is not None and len(rows) != expected:
            status = f"ROW COUNT DRIFT (expected {expected}, got {len(rows)})"
            print(f"WARNING: {bench}: {status} — publisher updated the file; "
                  f"proceeding with actual rows", flush=True)
        rows, removed = remove_overlap(rows, guard_rows)
        fp = args.out_dir / f"{bench}.jsonl"
        with fp.open("w", encoding="utf-8") as fh:
            for text, label, meta_row in rows:
                fh.write(json.dumps({"text": text, "label": label,
                                     "meta": meta_row}) + "\n")
        manifest["files"][bench] = {
            **meta, "rows": len(rows), "overlap_removed": removed,
            "source_sha256": hashlib.sha256(src.encode()).hexdigest(),
            "file_sha256": hashlib.sha256(fp.read_bytes()).hexdigest(),
            "count_status": status,
        }
        print(f"{bench:<12} rows={len(rows):>4} (overlap removed {removed})  "
              f"[{status}]")
    (args.out_dir / "ext-manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"\next-manifest.json written ({args.out_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
