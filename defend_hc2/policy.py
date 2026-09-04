"""Layer 4 — policy fusion engine.

Hard-fail (any cryptographic or structural violation — never overridden by
content scores):

* schema invalid
* sequence mismatch
* previous hash mismatch
* MAC invalid
* nonce replay
* tool provenance invalid
* cross-session splice detected

Otherwise the fused content risk (spec formula)::

    content_risk = 0.40 * injection_score
                 + 0.20 * lexical_score
                 + 0.20 * retrieval_injection_score
                 + 0.10 * intent_context_mismatch_score
                 + 0.10 * conversation_drift_score

Decision bands::

    risk >= 0.85                REJECT
    0.65 <= risk < 0.85         QUARANTINE
    0.40 <= risk < 0.65         SANITIZE_AND_ALLOW
    risk <  0.40                ALLOW
"""

from __future__ import annotations

from typing import Sequence

from defend_hc2.constants import (
    THRESHOLD_QUARANTINE,
    THRESHOLD_REJECT,
    THRESHOLD_SANITIZE,
    W_CONVERSATION_DRIFT,
    W_INJECTION,
    W_INTENT_MISMATCH,
    W_LEXICAL,
    W_RETRIEVAL_INJECTION,
)
from defend_hc2.results import (
    ContentRiskResult,
    DocumentProvenanceResult,
    IntegrityResult,
    PolicyDecision,
    ToolProvenanceResult,
)

# IntegrityResult.reason values that always hard-fail the request.
_HARD_FAIL_INTEGRITY_REASONS = {
    "SEQUENCE_MISMATCH",
    "PREVIOUS_HASH_MISMATCH",
    "MAC_MISMATCH",
    "MAC_FORGED_WITH_VALID_HEAD",
    "CHAIN_HASH_MISMATCH",
    "NONCE_REPLAY",
    "CROSS_SESSION_SPLICE",
    "STALE_HEAD_REPLAY",
    "SYSTEM_PROMPT_MISMATCH",
    "LOCAL_HEAD_TAMPER",
    "REPLAY_OR_GAP",
    "EVENT_TYPE_MISMATCH",
    "MALFORMED_HASH",
    "SEQUENCE_FUTURE_GAP",
    "UNRECORDED_EVENT",
}

# Tool provenance reasons that hard-fail.
_HARD_FAIL_TOOL_REASONS = {
    "TOOL_NOT_REGISTERED",
    "UNSIGNED_PRIVILEGED_TOOL_OUTPUT",
    "INVALID_TOOL_SIGNATURE",
}


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


class PolicyEngine:
    """Spec Layer 4."""

    def __init__(
        self,
        reject_at: float = THRESHOLD_REJECT,
        quarantine_at: float = THRESHOLD_QUARANTINE,
        sanitize_at: float = THRESHOLD_SANITIZE,
    ) -> None:
        self.reject_at = reject_at
        self.quarantine_at = quarantine_at
        self.sanitize_at = sanitize_at

    # ------------------------------------------------------------ risk math
    @staticmethod
    def fused_content_risk(
        injection_score: float,
        lexical_score: float,
        retrieval_injection_score: float,
        intent_context_mismatch_score: float,
        conversation_drift_score: float,
    ) -> float:
        risk = (
            W_INJECTION * injection_score
            + W_LEXICAL * lexical_score
            + W_RETRIEVAL_INJECTION * retrieval_injection_score
            + W_INTENT_MISMATCH * intent_context_mismatch_score
            + W_CONVERSATION_DRIFT * conversation_drift_score
        )
        return round(_clamp01(risk), 6)

    # -------------------------------------------------------------- decide
    def decide(
        self,
        content: ContentRiskResult,
        integrity_results: Sequence[IntegrityResult] = (),
        documents: Sequence[DocumentProvenanceResult] = (),
        tools: Sequence[ToolProvenanceResult] = (),
        schema_valid: bool = True,
        schema_errors: Sequence[str] = (),
        conversation_drift_score: float = 0.0,
    ) -> PolicyDecision:
        """Fuse all layer outputs into a single decision."""
        reasons: list[str] = []

        # ---- hard fail gate ------------------------------------------------
        if not schema_valid:
            reasons.append("SCHEMA_INVALID" + (f": {'; '.join(schema_errors)}" if schema_errors else ""))

        for r in integrity_results:
            if r.status == "FAIL":
                reasons.append(r.reason)

        for d in documents:
            if d.verdict == "rejected":
                reasons.append(f"RETRIEVED_DOC_REJECTED[{d.doc_id}] risk={d.instruction_risk}")

        for t in tools:
            if t.verdict == "rejected":
                reasons.append(f"TOOL_PROVENANCE_INVALID[{t.tool_name}]: {t.reason}")

        hard_fail = bool(reasons) and any(
            (
                r.startswith("SCHEMA_INVALID")
                or r.split(":")[0].split("[")[0] in _HARD_FAIL_INTEGRITY_REASONS
                or (r.startswith("TOOL_PROVENANCE_INVALID")
                    and any(hr in r for hr in _HARD_FAIL_TOOL_REASONS))
                or r.startswith("RETRIEVED_DOC_REJECTED")
            )
            for r in reasons
        )
        if hard_fail:
            return PolicyDecision(
                action="REJECT",
                content_risk=self.fused_content_risk(
                    content.injection_score,
                    content.lexical_score,
                    content.retrieval_injection_score,
                    content.intent_context_mismatch_score,
                    conversation_drift_score,
                ),
                hard_fail=True,
                reasons=reasons,
                component_scores=self._components(content, conversation_drift_score),
            )

        # ---- soft, risk-based band ----------------------------------------
        risk = self.fused_content_risk(
            content.injection_score,
            content.lexical_score,
            content.retrieval_injection_score,
            content.intent_context_mismatch_score,
            conversation_drift_score,
        )
        # Rejected (but not hard-failed) docs/tools already contribute via
        # retrieval_injection_score; rejected docs above were hard-failed.
        suspicious = [d.doc_id for d in documents if d.verdict == "suspicious"]
        if suspicious:
            reasons.append(f"SUSPICIOUS_DOCS: {suspicious}")
        unverified = [t.tool_name for t in tools if t.verdict == "unverified"]
        if unverified:
            reasons.append(f"UNVERIFIED_TOOL_OUTPUTS: {unverified}")

        if risk >= self.reject_at:
            action = "REJECT"
        elif risk >= self.quarantine_at:
            action = "QUARANTINE"
        elif risk >= self.sanitize_at:
            action = "SANITIZE_AND_ALLOW"
        else:
            action = "ALLOW"

        reasons.append(f"fused_content_risk={risk} band={action}")
        return PolicyDecision(
            action=action,  # type: ignore[arg-type]
            content_risk=risk,
            hard_fail=False,
            reasons=reasons,
            component_scores=self._components(content, conversation_drift_score),
        )

    @staticmethod
    def _components(content: ContentRiskResult, drift: float) -> dict[str, float]:
        return {
            "injection_score": content.injection_score,
            "lexical_score": content.lexical_score,
            "retrieval_injection_score": content.retrieval_injection_score,
            "intent_context_mismatch_score": content.intent_context_mismatch_score,
            "conversation_drift_score": round(_clamp01(drift), 6),
            "fused_content_risk": PolicyEngine.fused_content_risk(
                content.injection_score,
                content.lexical_score,
                content.retrieval_injection_score,
                content.intent_context_mismatch_score,
                drift,
            ),
        }
