"""Process-level cached embedder loading (spec Phase 4).

A ``SentenceTransformer`` for a given model name is loaded **once per
process** and reused by every :class:`DEFEND_HC2` instance — previously each
instance re-downloaded/reconstructed the model.

Logging is quieted (progress bars, chatty HF warnings) but **errors remain
visible** — only verbosity is reduced, never exception handling.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache


def _quiet_hf_logging() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for name in ("transformers", "sentence_transformers", "huggingface_hub", "torch"):
        logging.getLogger(name).setLevel(logging.ERROR)


@lru_cache(maxsize=8)
def get_sentence_transformer(model_name: str):
    """Cached ``SentenceTransformer(model_name)`` (one load per process).

    Note for tests: the cache is keyed by model name — a test that swaps a
    fake backend for the same name must call :func:`clear_cache` first.
    """
    from defend_hc2.exceptions import EmbeddingBackendUnavailableError

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise EmbeddingBackendUnavailableError(
            "sentence-transformers is required when demo_mode=False; "
            "install the 'ml' extra (pip install -r requirements-ml.txt)"
        ) from exc
    _quiet_hf_logging()
    return SentenceTransformer(model_name)


def clear_cache() -> None:
    """Drop cached embedders (test isolation hook)."""
    get_sentence_transformer.cache_clear()
