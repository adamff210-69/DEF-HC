"""ONE-TIME sealed-split evaluation — the project's genuine holdout.

Refuses to run unless:
  * --i-understand-this-runs-once is passed, and
  * reports/sealed-result.json does not already exist (prior run proof).

This script is the ONLY consumer of hcbench-sealed.jsonl in the repo
(the suite statically pins that).  Output label:
blind_holdout_evaluated_once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow bare run

SEALED_FILE = "hcbench-sealed.jsonl"
RESULT_FILE = "sealed-result.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--i-understand-this-runs-once", action="store_true",
                    required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("hcbench"))
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--policies", type=Path, nargs=2, required=True)
    args = ap.parse_args()

    result_fp = args.reports / RESULT_FILE
    if result_fp.exists():
        raise SystemExit(f"SEALED RESULT ALREADY EXISTS: {result_fp} — "
                         f"a second pass would bias the holdout. Refusing.")
    sealed_fp = args.data_dir / SEALED_FILE
    if not sealed_fp.exists():
        raise SystemExit(f"missing {sealed_fp} — run scripts/build_hcbench.py")

    from defend_hc2 import DEFEND_HC2, ToolRegistry
    from scripts.eval_hcbench import (
        BENCH_TOOL,
        NEUTRAL_QUERY,
        per_slice_report,
        run_evaluation,
    )
    from scripts.eval_hcbench import load_split  # sealed read happens HERE
    registry = ToolRegistry()
    registry.register_tool(BENCH_TOOL["name"], BENCH_TOOL["key"],
                           privileged=BENCH_TOOL["privileged"])
    system = DEFEND_HC2(db_path=":memory:", demo_mode=False,
                        weights_path=str(args.weights),
                        tool_registry=registry, master_secret=b"S" * 32)

    rows = load_split(sealed_fp)  # the only place the sealed file is read
    ys = [int(r["label"]) for r in rows]
    scores, _ = run_evaluation(rows, system, "hc-bench-sealed")

    results = {"label": "blind_holdout_evaluated_once",
               "n": len(rows),
               "sha256_of_sealed_file": hashlib.sha256(
                   sealed_fp.read_bytes()).hexdigest(),
               "policies": {}}
    for pol_fp in args.policies:
        pol = json.loads(pol_fp.read_text())["policy"]
        results["policies"][pol_fp.stem] = {
            "bands": pol,
            "overall": per_slice_report(
                ys, scores, pol["quarantine_at"], pol["sanitize_at"]),
        }
        print(f"[sealed once] {pol_fp.stem}: "
              f"{results['policies'][pol_fp.stem]['overall']}")

    result_fp.write_text(json.dumps(results, indent=2))
    print(f"sealed result written ONCE: {result_fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
