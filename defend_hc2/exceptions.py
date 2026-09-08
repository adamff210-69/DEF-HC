"""Exception hierarchy for DEFEND-HC2."""

from __future__ import annotations


class DEFENDHC2Error(Exception):
    """Base class for all DEFEND-HC2 errors."""


class SchemaValidationError(DEFENDHC2Error):
    """L0: request payload failed schema validation."""


class ChainIntegrityError(DEFENDHC2Error):
    """L2: hash-chain / MAC / sequence verification failed.

    Carries the machine-readable ``reason`` also used in
    :class:`defend_hc2.results.IntegrityResult`.
    """

    def __init__(self, reason: str, message: str | None = None) -> None:
        self.reason = reason
        super().__init__(message or reason)


class NonceReplayError(ChainIntegrityError):
    """L2: nonce was already consumed for this session."""

    def __init__(self, session_id: str, nonce: str) -> None:
        self.session_id = session_id
        self.nonce = nonce
        super().__init__(
            "NONCE_REPLAY",
            f"nonce {nonce!r} already used in session {session_id!r}",
        )


class SessionNotFoundError(DEFENDHC2Error):
    """Referenced session does not exist on this node."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"unknown session: {session_id!r}")


class ProvenanceError(DEFENDHC2Error):
    """L3: RAG-document or tool-output provenance verification failed."""

    def __init__(self, reason: str, message: str | None = None) -> None:
        self.reason = reason
        super().__init__(message or reason)


class LedgerError(DEFENDHC2Error):
    """L5: append-only ledger rejected a write (conflict / fork / tamper)."""

    def __init__(self, reason: str, message: str | None = None) -> None:
        self.reason = reason
        super().__init__(message or reason)


class EmbeddingBackendUnavailableError(DEFENDHC2Error):
    """L1: non-demo mode requested but sentence-transformers is not installed."""
