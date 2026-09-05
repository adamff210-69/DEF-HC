"""FIX-1 production gate (Kaggle-only): the restoration path must keep the
mean benign recovery risk under 0.30 and the mean attack recovery risk
over 0.85 on a fixed seed-42 sample of the development-test split.

This is the in-distribution verification that the variant view-policy
change (restoration views REPLACE junk raw views) is measured, not
assumed: benign inflation on obfuscated views (inj ~0.80) was the defect
this gate would have caught.  Skips locally — requires the Kaggle
artifacts (bench-data + Exp-A weights).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

_DATA = Path("/kaggle/working/bench-data/pi-test.jsonl")
_WEIGHTS = Path("/kaggle/working/bench-out/weights-exp-a.json")


@pytest.mark.skipif(not (_DATA.exists() and _WEIGHTS.exists()),
                    reason="requires Kaggle artifacts (bench-data + exp weights)")
def test_production_recovery_gate():
    from scripts.run_experiments import recovery_risk_for_text
    from defend_hc2.content_risk import ContentRiskAnalyzer

    rows = [json.loads(l) for l in _DATA.read_text().splitlines() if l.strip()]
    rng = random.Random(42)
    pos = rng.sample([r for r in rows if r["label"] == 1], 20)
    neg = rng.sample([r for r in rows if r["label"] == 0], 20)

    analyzer = ContentRiskAnalyzer(demo_mode=False, weights_path=_WEIGHTS)
    r_pos = [recovery_risk_for_text(analyzer, r["text"]) for r in pos]
    r_neg = [recovery_risk_for_text(analyzer, r["text"]) for r in neg]
    m_pos, m_neg = sum(r_pos) / len(r_pos), sum(r_neg) / len(r_neg)
    # measured 2026-09-05: attack 0.873, benign 0.113
    assert m_neg < 0.30, f"benign mean {m_neg:.3f} above 0.30 — unsafe view policy"
    assert m_pos > 0.85, f"attack mean {m_pos:.3f} below 0.85 — recovery regressed"
