# Running DEFEND-HC2 on Kaggle (ML mode)

The ready-to-use notebook lives at
[`notebooks/defend-hc2-kaggle-ml-mode.ipynb`](../notebooks/defend-hc2-kaggle-ml-mode.ipynb).

## Import it

1. On kaggle.com → **Notebooks → New notebook → File → Import notebook** and
   upload the `.ipynb` (or paste the GitHub URL of the file).
2. **Right-hand panel → Notebook options:** set **Internet: ON** (required to
   clone the repo and download `BAAI/bge-small-en-v1.5` once). Accelerator
   *None* is fine; a GPU only speeds training.
3. Run all cells top to bottom.

## What the notebook does

| Cells | Content |
|---|---|
| 1–3 | environment check, clone branch `arena/01a06c45-def-hc`, install missing deps (`torch` untouched — Kaggle ships it), editable-install `defend_hc2` |
| 4 | sanity demo (heuristic mode) |
| 5 | **train the bge-small logistic classifier weights** → `/kaggle/working/weights/bge-logistic.json` (~2–4 min incl. one-time ~130 MB download; skips if present) |
| 6 | probe `p(injection)` for benign vs. injection prompts in **ML mode** |
| 7 | full L0–L5 pipeline in ML mode + all spec attack scenarios |
| 8 | export the audit trail (`audit-export.json`) + checkpoint |
| 9 | drive the FastAPI service in-process via `TestClient` |
| 10 | (optional) full 164-test suite |

All artifacts persist in `/kaggle/working` as notebook output.

## Gotchas

- The implementation branch is `arena/01a06c45-def-hc`; `main` is a README stub.
- **If `import defend_hc2` fails after `pip install -e` in the same kernel**:
  editable installs register at interpreter startup, so the already-running
  kernel can't see the package. Either restart the kernel once, or add
  `import sys; sys.path.insert(0, "/kaggle/working/DEF-HC")` before importing
  (the notebook's Cell 3 already does this for you).
- Cell 5 is idempotent — delete the weights file to retrain (e.g. with your own
  `--dataset` JSONL).
- Cell 7 wipes its ledger at start, so it is fully re-runnable.
- No port forwarding needed: the API runs in-process via `TestClient`.
