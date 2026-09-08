"""Layer 4 tests: hard-fail matrix + exact spec thresholds/weights."""

from __future__ import annotations

import pytest

from defend_hc2.policy import PolicyEngine
from defend_hc2.results import (
    ContentRiskResult,
    DocumentProvenanceResult,
    IntegrityResult,
    ToolProvenanceResult,
)

engine = PolicyEngine()


def _content(inj=0.0, lex=0.0, rag=0.0, mm=0.0) -> ContentRiskResult:
    return ContentRiskResult(
        lexical_score=lex,
        injection_score=inj,
        retrieval_injection_score=rag,
        intent_context_mismatch_score=mm,
        content_risk=0.0,
        evidence=[],
    )


def _integrity_fail(reason: str) -> IntegrityResult:
    return IntegrityResult("FAIL", reason, "HIGH", "h" * 64, None, 1)


_PASS = IntegrityResult("PASS", "PASS", "NONE", "h" * 64, "n" * 64, 1)


class TestFusionFormula:
    def test_saturated_signals_not_diluted(self):
        # spec defect P2: five saturated channels -> risk approaches 1
        risk = engine.fused_content_risk(1.0, 1.0, 1.0, 1.0, 1.0)
        assert risk == pytest.approx(0.999, abs=1e-3)

    def test_component_combination(self):
        # spec Phase 6 baseline math, worked example:
        # channels: injection 0.5 (w 1.0), lexical 0.25 (w 0.9),
        # retrieval 1.0 (w 1.0), mismatch 0.0 (w 0.6), drift None (inactive)
        # s = [0.5, 0.225, 0.999, 0.0]; strongest 0.999
        # noisy_or = 1 - (0.5 * 0.775 * 0.001 * 1.0) = 0.9996125
        # risk = 0.999 + 0.5*(noisy_or - 0.999)
        risk = engine.fused_content_risk(0.5, 0.25, 1.0, 0.0, None)
        expected = 0.999 + 0.5 * ((1 - 0.5 * 0.775 * 0.001) - 0.999)
        assert risk == pytest.approx(expected, abs=1e-4)

    def test_single_active_channel_is_undiluted(self):
        # a maxed direct attack alone must not collapse: inj=0.958, lex=1.0,
        # no retrieval/mismatch/drift -> risk must exceed the reject band,
        # not dilute to ~0.6 as the old weighted sum did
        risk = engine.fused_content_risk(0.958, 1.0, None, None, None)
        assert risk >= 0.95


class TestThresholds:
    @pytest.mark.parametrize(
        "risk,expected",
        [
            (0.0, "ALLOW"),
            (0.39, "ALLOW"),
            (0.40, "SANITIZE_AND_ALLOW"),
            (0.64, "SANITIZE_AND_ALLOW"),
            (0.65, "QUARANTINE"),
            (0.84, "QUARANTINE"),
            (0.85, "REJECT"),
            (1.0, "REJECT"),
        ],
    )
    def test_bands(self, risk, expected):
        # drive the bands with a single injection channel; every other
        # channel inactive (None) so fused risk == clamp(risk, 0.999)
        content = _content(inj=risk, lex=0.0, rag=None, mm=None)
        decision = engine.decide(content, [_PASS], conversation_drift_score=None)
        assert decision.action == expected
        assert decision.content_risk == pytest.approx(min(0.999, risk), abs=1e-3)


class TestHardFail:
    @pytest.mark.parametrize(
        "reason",
        [
            "SEQUENCE_MISMATCH",
            "PREVIOUS_HASH_MISMATCH",
            "MAC_MISMATCH",
            "NONCE_REPLAY",
            "CROSS_SESSION_SPLICE",
            "STALE_HEAD_REPLAY",
            "SYSTEM_PROMPT_MISMATCH",
            "LOCAL_HEAD_TAMPER",
        ],
    )
    def test_integrity_failures_hard_reject(self, reason):
        decision = engine.decide(_content(), [_integrity_fail(reason)])
        assert decision.action == "REJECT"
        assert decision.hard_fail
        assert reason in decision.reasons

    def test_schema_invalid_hard_rejects(self):
        decision = engine.decide(
            _content(), [_PASS], schema_valid=False, schema_errors=["missing 'text'"]
        )
        assert decision.action == "REJECT" and decision.hard_fail

    def test_tool_provenance_invalid_hard_rejects(self):
        tool = ToolProvenanceResult(
            tool_name="files_write", input_hash="i" * 64, output_hash="o" * 64,
            privileged=True, signature_present=False, signature_valid=False,
            verdict="rejected", reason="UNSIGNED_PRIVILEGED_TOOL_OUTPUT",
        )
        decision = engine.decide(_content(), [_PASS], tools=[tool])
        assert decision.action == "REJECT" and decision.hard_fail

    def test_rejected_doc_hard_rejects(self):
        doc = DocumentProvenanceResult(
            doc_id="d1", doc_hash="h" * 64, source_uri_hash="u" * 64,
            instruction_risk=0.95, verdict="rejected", evidence=[],
        )
        decision = engine.decide(_content(rag=0.95), [_PASS], documents=[doc])
        assert decision.action == "REJECT" and decision.hard_fail

    def test_clean_pass_allows(self):
        decision = engine.decide(_content(), [_PASS])
        assert decision.action == "ALLOW" and not decision.hard_fail

    def test_suspicious_doc_noted_but_not_hard(self):
        doc = DocumentProvenanceResult(
            doc_id="d1", doc_hash="h" * 64, source_uri_hash="u" * 64,
            instruction_risk=0.5, verdict="suspicious", evidence=[],
        )
        decision = engine.decide(_content(rag=0.5), [_PASS], documents=[doc])
        assert not decision.hard_fail
        assert any("SUSPICIOUS_DOCS" in r for r in decision.reasons)

    def test_crypto_beats_low_content(self):
        """A cryptographically invalid request is rejected even at risk 0."""
        decision = engine.decide(
            _content(), [_integrity_fail("MAC_MISMATCH")]
        )
        assert decision.content_risk == 0.0
        assert decision.action == "REJECT" and decision.hard_fail
