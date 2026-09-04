"""DEFEND-HC2 — dual-layer LLM security framework.

Layers
------
L0  :mod:`defend_hc2.canonicalization`  Canonicalizer
L1  :mod:`defend_hc2.content_risk`       ContentRiskAnalyzer
L2  :mod:`defend_hc2.session_chain`      SessionContinuityTracker
L3  :mod:`defend_hc2.provenance`         ProvenanceVerifier
L4  :mod:`defend_hc2.policy`             PolicyEngine
L5  :mod:`defend_hc2.ledger`             SQLiteTamperEvidentLedger
    :mod:`defend_hc2.pipeline`           DEFEND_HC2 orchestrator
"""

from defend_hc2.canonicalization import Canonicalizer
from defend_hc2.content_risk import ContentRiskAnalyzer
from defend_hc2.exceptions import (
    ChainIntegrityError,
    DEFENDHC2Error,
    LedgerError,
    NonceReplayError,
    ProvenanceError,
    SchemaValidationError,
    SessionNotFoundError,
)
from defend_hc2.ledger import SQLiteTamperEvidentLedger
from defend_hc2.pipeline import DEFEND_HC2
from defend_hc2.policy import PolicyEngine
from defend_hc2.provenance import ProvenanceVerifier
from defend_hc2.results import (
    ChainEntryRecord,
    ContentRiskResult,
    DocumentProvenanceResult,
    IntegrityResult,
    PolicyDecision,
    ProcessResult,
    ToolProvenanceResult,
)
from defend_hc2.session_chain import SessionContinuityTracker

__version__ = "2.0.0"

__all__ = [
    "Canonicalizer",
    "ChainEntryRecord",
    "ChainIntegrityError",
    "ContentRiskAnalyzer",
    "ContentRiskResult",
    "DEFENDHC2Error",
    "DEFEND_HC2",
    "DocumentProvenanceResult",
    "IntegrityResult",
    "LedgerError",
    "NonceReplayError",
    "PolicyDecision",
    "PolicyEngine",
    "ProcessResult",
    "ProvenanceError",
    "ProvenanceVerifier",
    "SchemaValidationError",
    "SessionContinuityTracker",
    "SessionNotFoundError",
    "SQLiteTamperEvidentLedger",
    "ToolProvenanceResult",
    "__version__",
]
