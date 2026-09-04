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
    def test_weights_exact(self):
        risk = engine.fused_content_risk(1.0, 1.0, 1.0, 1.0, 1.0)
        assert risk == 1.0

    def test_component_combination(self):
        risk = engine.fused_content_risk(0.5, 0.25, 1.0, 0.0, 1.0)
        assert risk == pytest.approx(
            0.40 * 0.5 + 0.20 * 0.25 + 0.20 * 1.0 + 0.10 * 0.0 + 0.10 * 1.0
        )


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
        # all five components at `risk` → fused risk == risk (weights sum to 1)
        content = _content(inj=risk, lex=risk, rag=risk, mm=risk)
        decision = engine.decide(content, [_PASS], conversation_drift_score=risk)
        assert decision.action == expected
        assert decision.content_risk == pytest.approx(min(1.0, risk), abs=1e-6)


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
