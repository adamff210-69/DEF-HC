"""Leakage-safe dataset splitting (spec Phases 1, 11, 17).

Pure functions — importable and unit-testable without any dataset download:

* exact + template duplicate detection (NFKC, casefold, whitespace collapse,
  digit-run replacement for templates);
* stable SHA-256 group ids;
* group-aware, approximately stratified deterministic splitting — a duplicate
  group can never cross split boundaries;
* post-split assertion + leakage statistics.
"""

from __future__ import annotations

import hashlib
import unicodedata

_WS = " "


def normalize_key(text: str) -> str:
    """Exact-duplicate key: NFKC + casefold + whitespace collapse."""
    text = unicodedata.normalize("NFKC", text)
    return _WS.join(text.split()).casefold()


def template_key(text: str) -> str:
    """Template-duplicate key: normalize_key with digit runs collapsed."""
    key = normalize_key(text)
    out, in_digits = [], False
    for ch in key:
        if ch.isdigit():
            if not in_digits:
                out.append("0")
            in_digits = True
        else:
            out.append(ch)
            in_digits = False
    return "".join(out)


def group_id(template: str) -> str:
    """Stable SHA-256 group id for a template key (never Python hash())."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]


def build_groups(rows: list[dict], text_field: str) -> dict[str, list[int]]:
    """Map template-group id -> row indices."""
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        gid = group_id(template_key(str(row[text_field])))
        groups.setdefault(gid, []).append(i)
    return groups


def group_stratified_split(
    rows: list[dict],
    text_field: str,
    label_field: str | None = None,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> list[list[dict]]:
    """Group-aware approximately-stratified 3-way split.

    Groups are assigned whole — no duplicate/template group ever crosses a
    boundary.  Stratification is approximated by seeding group order
    deterministically and distributing groups round-robin weighted by the
    target fraction while balancing the positive-class count best-effort.
    """
    import random

    groups = sorted(build_groups(rows, text_field).items(), key=lambda kv: kv[0])
    rng = random.Random(seed)
    rng.shuffle(groups)

    def pos_of(indices: list[int]) -> int:
        if label_field is None:
            return 0
        return sum(1 for i in indices if rows[i][label_field])

    fr = [f / sum(fractions) for f in fractions]
    k = len(fr)
    buckets: list[list[int]] = [[] for _ in range(k)]
    counts = [0] * k
    pos_counts = [0] * k
    total = len(rows)
    total_pos = pos_of(list(range(total)))
    for gid, idx in groups:
        # choose the bucket furthest below its target size, tie-break by
        # the bucket furthest below its positives target
        def deficit(b: int) -> float:
            return fr[b] * total - counts[b]

        best = max(range(k), key=lambda b: (deficit(b),
                                            fr[b] * total_pos - pos_counts[b]))
        buckets[best].extend(idx)
        counts[best] += len(idx)
        pos_counts[best] += pos_of(idx)
    return [[rows[i] for i in bucket] for bucket in buckets]


def assert_no_group_crossing(parts: list[list[dict]], text_field: str) -> None:
    """Raise if any template group appears in more than one split."""
    seen: dict[str, int] = {}
    for part_i, part in enumerate(parts):
        for row in part:
            gid = group_id(template_key(str(row[text_field])))
            if gid in seen and seen[gid] != part_i:
                raise AssertionError(
                    f"template group {gid} crosses splits "
                    f"{seen[gid]} and {part_i}"
                )
            seen[gid] = part_i


def leakage_stats(rows: list[dict], text_field: str, label_field: str) -> dict:
    exact = len(rows) - len({normalize_key(str(r[text_field])) for r in rows})
    templ = len(rows) - len({template_key(str(r[text_field])) for r in rows})
    pos = sum(1 for r in rows if r[label_field])
    return {
        "rows": len(rows), "positive": pos, "base_rate": round(pos / max(1, len(rows)), 4),
        "exact_duplicates": exact, "template_duplicates": templ,
    }
