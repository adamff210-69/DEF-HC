"""Result / record dataclasses shared across DEFEND-HC2 layers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

DecisionAction = Literal["ALLOW", "SANITIZE_AND_ALLOW", "QUARANTINE", "REJECT"]
IntegrityStatus = Literal["PASS", "FAIL"]
DocVerdict = Literal["trusted", "suspicious", "rejected"]
ToolVerdict = Literal["verified", "unverified", "rejected"]


@dataclass(slots=True)
class ContentRiskResult:
    """L1 output.

    ``retrieval_injection_score`` / ``intent_context_mismatch_score`` are
    ``None`` when the channel was not applicable (no retrieved documents) —
    absent context must never dilute active evidence (spec defect P2).
    """

    lexical_score: float
    injection_score: float
    retrieval_injection_score: float | None
    intent_context_mismatch_score: float | None
    content_risk: float
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IntegrityResult:
    """L2 output (spec exactly)."""

    status: IntegrityStatus
    reason: str
    severity: str  # "NONE" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    previous_hash: str
    new_hash: str | None
    sequence: int

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DocumentProvenanceResult:
    """L3 output for one retrieved document."""

    doc_id: str
    doc_hash: str
    source_uri_hash: str
    instruction_risk: float
    verdict: DocVerdict
    evidence: list[str] = field(default_factory=list)
    chain_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolProvenanceResult:
    """L3 output for one tool invocation result."""

    tool_name: str
    input_hash: str
    output_hash: str
    privileged: bool
    signature_present: bool
    signature_valid: bool
    verdict: ToolVerdict
    reason: str
    chain_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PolicyDecision:
    """L4 output."""

    action: DecisionAction
    content_risk: float
    hard_fail: bool
    reasons: list[str] = field(default_factory=list)
    component_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChainEntryRecord:
    """One append-only ledger row (L5)."""

    session_id: str
    sequence: int
    event_type: str
    payload: dict[str, Any]
    payload_hash: str
    previous_hash: str
    chain_hash: str
    mac: str
    timestamp_ns: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


@dataclass(slots=True)
class SessionRecord:
    session_id: str
    system_prompt_hash: str
    session_salt: str  # hex
    genesis_hash: str
    created_at_ns: int
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProcessResult:
    """End-to-end result returned by ``DEFEND_HC2.process_user_message``."""

    session_id: str
    decision: PolicyDecision
    content: ContentRiskResult
    integrity: list[IntegrityResult]
    documents: list[DocumentProvenanceResult]
    tools: list[ToolProvenanceResult]
    head_hash: str
    sequence: int
    safe_prompt: str | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "decision": self.decision.to_dict(),
            "content": self.content.to_dict(),
            "integrity": [r.to_dict() for r in self.integrity],
            "documents": [d.to_dict() for d in self.documents],
            "tools": [t.to_dict() for t in self.tools],
            "head_hash": self.head_hash,
            "sequence": self.sequence,
            "safe_prompt": self.safe_prompt,
            "notes": self.notes,
        }


@dataclass(slots=True)
class ChainVerificationReport:
    """Output of full-chain recomputation (L7 of the pipeline)."""

    session_id: str
    ok: bool
    entries_checked: int
    first_invalid_sequence: int | None
    reason: str
    checks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
