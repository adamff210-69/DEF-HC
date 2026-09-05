"""Train-time letter-spacing augmentation (open research item 1).

Extreme letter-fragmentation ("i g n o r e …") destroys subword
tokenization out-of-distribution (Exp-F: perturbed AUC 0.4382,
recovery 0.4383, labeled LIMITATION).  Since embedding despaced glue is
prohibited by design (benign FPR), one sanctioned mitigation is
train-time augmentation: teach the embedding logistic region for
letter-spaced examples by adding label-preserving letter-spaced COPIES
of a deterministic subset of the TRAINING split.

Discipline (non-negotiable, this project's protocol):
  * augmentation applies to TRAIN ONLY — never calibration/test rows
  * selection is RANDOM-FREE and deterministic (index rule + length cap)
    → reproducible, seed-42 compatible
  * labels are preserved verbatim (letter-spacing is label-preserving)
  * the hyperparameters (every, max_chars) are A PRIORI constants —
    they are not searched against any evaluation outcome
"""

from __future__ import annotations

from defend_hc2.perturb import letter_spacing_extreme

Row = tuple[str, int]

# A priori constants — declared before any evaluation, frozen afterwards.
AUGMENT_EVERY: int = 4        # augment 25% of eligible training rows
AUGMENT_MAX_CHARS: int = 512  # letter-spacing triples visual length; cap cost


def letter_spacing_augment(
    train_rows: list[Row],
    *,
    every: int = AUGMENT_EVERY,
    max_chars: int = AUGMENT_MAX_CHARS,
) -> list[Row]:
    """Deterministic label-preserving letter-spaced copies of train rows.

    Returns ONLY the additional rows (caller concatenates).  Selection:
    every ``every``-th training row whose text is at most ``max_chars``.
    No RNG, no label changes, no access to anything outside ``train_rows``.
    """
    if every < 1 or max_chars < 1:
        raise ValueError("every and max_chars must be positive")
    out: list[Row] = []
    for i, (text, label) in enumerate(train_rows):
        if i % every != 0 or len(text) > max_chars:
            continue
        out.append((letter_spacing_extreme(text), label))
    return out
