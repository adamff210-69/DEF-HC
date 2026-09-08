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
def get_sentence_transformer(model_name: str, device: str | None = None):
    """Cached ``SentenceTransformer(model_name)`` (one load per process).

    ``device`` defaults to CUDA when torch reports it available.  This used
    to be left entirely to sentence-transformers' own auto-detection, which
    made a CPU fallback completely silent: on an accelerated machine the
    only symptom was that embedding took tens of minutes.  The resolved
    device is now returned on the object and logged by callers.

    Note for tests: the cache is keyed by (model name, device) — a test that
    swaps a fake backend for the same name must call :func:`clear_cache`.
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
    if device is None:
        device = _auto_device()
    try:
        return SentenceTransformer(model_name, device=device)
    except TypeError:
        # Older sentence-transformers (or a test double) without a `device`
        # kwarg: fall back to its own auto-detection rather than failing.
        return SentenceTransformer(model_name)


def _auto_device() -> str:
    """CUDA when it is genuinely usable, else CPU.  Never raises."""
    try:
        import torch
    except ImportError:  # pragma: no cover - environment dependent
        return "cpu"
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda"
    except Exception:  # pragma: no cover - broken driver/runtime pairing
        pass
    return "cpu"


def tune_cpu_threads() -> dict:
    """Let torch use every available core when falling back to CPU.

    torch often defaults to physical-core count or an inherited
    OMP_NUM_THREADS, which on a 4-vCPU box shows up as ~200% CPU while two
    cores sit idle.  Only widens the pool; never narrows it.
    """
    out: dict = {}
    try:
        import os

        import torch
        want = os.cpu_count() or 1
        out["cpu_count"] = want
        out["threads_before"] = torch.get_num_threads()
        if torch.get_num_threads() < want:
            torch.set_num_threads(want)
        out["threads_after"] = torch.get_num_threads()
    except ImportError:
        out["error"] = "torch not installed"
    except Exception as exc:  # pragma: no cover
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return out


def device_report() -> dict:
    """Diagnostic block: what torch can actually see.

    Printed by the training/eval scripts so a silent CPU fallback shows up
    in the log instead of only as elapsed time.  ``torch_file`` matters on
    hosted notebooks: a user-site copy shadowing the image's CUDA build is a
    common and otherwise invisible cause of CPU-only execution.
    """
    info: dict = {"selected_device": _auto_device()}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["torch_file"] = torch.__file__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = int(torch.cuda.device_count())
        info["torch_cuda_build"] = torch.version.cuda
        if info["cuda_available"]:
            info["gpu_names"] = [torch.cuda.get_device_name(i)
                                 for i in range(torch.cuda.device_count())]
        elif str(torch.__file__).startswith(("/root/.local", "/kaggle/.local",
                                             "/home/")):
            info["shadowed_install_warning"] = (
                "torch is loading from a user-site path; it may be shadowing "
                "a CUDA build shipped with the image")
    except ImportError:
        info["torch_version"] = None
    except Exception as exc:  # pragma: no cover
        info["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return info


def clear_cache() -> None:
    """Drop cached embedders (test isolation hook)."""
    get_sentence_transformer.cache_clear()
