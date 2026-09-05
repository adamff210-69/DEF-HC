"""Shared Exp-F reporting vocabulary (spec: recovery-aware caveats).

A sub-0.5 perturbed AUC used to be an unexplained anomaly and earned a
flat "pipeline bug" warning.  That template became self-contradictory
once recovery was measured on the same rows: leetspeak/base64 recover to
~0.988, which disproves the bug hypothesis by construction, and
letter-spacing fragmentation is a known, designed-for limitation.
Exactly one category must remain a WARNING — the unexplained anomaly —
so a reviewer can trust that warnings are actionable.
"""

from __future__ import annotations

KNOWN_LIMITATIONS = frozenset({"letter_spacing_extreme"})
RECOVERY_DISPROOF_THRESHOLD = 0.90


def transform_caveat_lines(
    name: str,
    perturbed_auc: float | None,
    recovery_auc: float | None,
    *,
    dump_name: str | None = None,
    indent: str = "",
) -> list[str]:
    """Recovery-aware caveat lines for one transform row.

    - perturbed >= 0.5: no caveat
    - perturbed < 0.5 but recovery >= 0.90: EXPECTED (restoration works;
      the raw obfuscated view being anti-correlated IS the finding — it
      is why canonicalization-before-scoring exists in this stack)
    - known limitation (letter-spacing): LIMITATION, with the design
      rationale and the mitigation boundary
    - anything else: the single genuine WARNING category
    """
    if not isinstance(perturbed_auc, (int, float)) or perturbed_auc >= 0.5:
        return []
    p = f"{perturbed_auc:.4f}"
    if isinstance(recovery_auc, (int, float)):
        r = f"{recovery_auc:.4f}"
    else:
        r = "n/a"
    if isinstance(recovery_auc, (int, float)) \
            and recovery_auc >= RECOVERY_DISPROOF_THRESHOLD:
        return [f"{indent}NOTE: {name} raw-view AUC {p} < 0.5 — raw obfuscated "
                f"text is out-of-distribution for the encoder and scores are "
                f"ANTI-CORRELATED (obfuscation inverts the benign/attack order); "
                f"canonical restoration recovers to {r}. Expected, not a defect."]
    if name in KNOWN_LIMITATIONS:
        return [f"{indent}LIMITATION: {name} AUC {p}, recovery {r} — "
                f"character-level fragmentation defeats subword tokenization; "
                f"'despaced' glue strings are deliberately excluded from "
                f"embedding views to avoid benign false positives. "
                f"Requires train-time augmentation or a character-level "
                f"detector."]
    dump = f" See {dump_name}." if dump_name else ""
    return [f"{indent}WARNING: {name} AUC {p} < 0.5 and recovery {r} — "
            f"unexplained; treat as a pipeline bug.{dump}"]
