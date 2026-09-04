"""DEFEND_HC2 — the end-to-end orchestrator.

Pipeline (spec, "Final Improved Pipeline"):

1. Session creation      — canonicalize system prompt, salt, keyed chain, genesis
2. Request intake        — schema validation, normalization, canonical payload
3. Content analysis      — lexical + injection classifier + RAG risk + mismatch
4. Provenance            — previous-hash check, doc hashes, tool receipts
5. Policy fusion         — hard-fail crypto; risk bands for content
6. Ledger append         — user → retrieval/tool → decision → assistant events
7. Verification & export — recompute chain, find first invalid event
8. Checkpointing         — Merkle root over session heads, signed
"""

from __future__ import annotations

import secrets
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from defend_hc2.canonicalization import Canonicalizer
from defend_hc2.constants import (
    EVENT_ASSISTANT_MESSAGE,
    EVENT_CONTENT_ANALYSIS,
    EVENT_GENESIS,
    EVENT_POLICY_DECISION,
    EVENT_RETRIEVAL,
    EVENT_TOOL_OUTPUT,
    EVENT_USER_MESSAGE,
    TAG_CHECKPOINT_SIG,
)
from defend_hc2.content_risk import ContentRiskAnalyzer
from defend_hc2.exceptions import (
    DEFENDHC2Error,
    SchemaValidationError,
    SessionNotFoundError,
)
from defend_hc2.ledger import SQLiteTamperEvidentLedger, compute_merkle_root
from defend_hc2.policy import PolicyEngine
from defend_hc2.provenance import ProvenanceVerifier, ToolRegistry
from defend_hc2.results import (
    ChainEntryRecord,
    ChainVerificationReport,
    ContentRiskResult,
    DocumentProvenanceResult,
    IntegrityResult,
    PolicyDecision,
    ProcessResult,
    SessionRecord,
    ToolProvenanceResult,
)
from defend_hc2.session_chain import GenesisRecord, SessionContinuityTracker

_PROCESS_SCHEMA = {
    "session_id": str,
    "text": str,
}
_PROCESS_OPTIONAL = {
    "nonce": str,
    "claimed_previous_hash": str,
    "claimed_sequence": int,
    "client_system_prompt_hash": str,
}

_CHAT_TEMPLATE_DELIMS = (
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "</s>", "<s>",
    "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>",
)


def _sanitize_text(text: str) -> tuple[str, list[str]]:
    """Remove template-delimiter smuggling and fake role headers (L4 sanitize)."""
    notes: list[str] = []
    out = text
    for delim in _CHAT_TEMPLATE_DELIMS:
        if delim.lower() in out.lower():
            out = out.replace(delim, " ").replace(delim.lower(), " ")
            notes.append(f"stripped chat-template delimiter {delim!r}")
    import re as _re

    out2 = _re.sub(r"(?im)^\s*(system|admin|root)\s*:\s*", "", out)
    if out2 != out:
        notes.append("stripped fake role header(s)")
        out = out2
    out3 = _re.sub(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{120,}={0,2}(?![A-Za-z0-9+/=])",
                   "[encoded blob removed]", out)
    if out3 != out:
        notes.append("removed base64-like blob(s)")
        out = out3
    return out.strip(), notes


class DEFEND_HC2:
    """Top-level facade combining all DEFEND-HC2 layers.

    Parameters
    ----------
    db_path: SQLite path (``":memory:"`` for ephemeral use).
    master_secret: hex string or bytes; falls back to ``DEFEND_HC2_MASTER_SECRET``.
    demo_mode: content analyzer mode (True = deterministic heuristics,
        no model downloads).
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        master_secret: str | bytes | None = None,
        demo_mode: bool = True,
        weights_path: str | Path | None = None,
        model_name: str | None = None,
        tool_registry: ToolRegistry | None = None,
        ledger: SQLiteTamperEvidentLedger | None = None,
    ) -> None:
        if isinstance(master_secret, str):
            master_secret = bytes.fromhex(master_secret)
        if master_secret is None:
            import os

            env = os.environ.get("DEFEND_HC2_MASTER_SECRET")
            master_secret = bytes.fromhex(env) if env else secrets.token_bytes(32)
        self._master_secret: bytes = master_secret
        self.ledger = ledger or SQLiteTamperEvidentLedger(db_path)
        self.analyzer = ContentRiskAnalyzer(
            demo_mode=demo_mode,
            weights_path=weights_path,
            **({"model_name": model_name} if model_name else {}),
        )
        self.tracker = SessionContinuityTracker(master_secret=master_secret)
        self.provenance = ProvenanceVerifier(
            analyzer=self.analyzer,
            tool_registry=tool_registry or ToolRegistry(),
        )
        self.policy = PolicyEngine()
        self._lock = threading.RLock()
        self._restore_sessions_from_ledger()

    # ------------------------------------------------------- restore on boot
    def _restore_sessions_from_ledger(self) -> None:
        """Rebuild in-memory chain state from the append-only ledger.

        Every row is cryptographically re-verified during restore, so a
        tampered ledger cannot resurrect a forged head.
        """
        for record in self.ledger.list_sessions():
            try:
                self.tracker.seed_session(
                    record.session_id,
                    record.session_salt,
                    record.created_at_ns,
                    system_prompt_hash_hex=record.system_prompt_hash,
                )
                for row in self.ledger.get_entries(record.session_id):
                    nonce = None
                    self.tracker.restore_event(
                        session_id=row.session_id,
                        sequence=row.sequence,
                        event_type=row.event_type,
                        payload=row.payload,
                        chain_hash=row.chain_hash,
                        mac=row.mac,
                        previous_hash=row.previous_hash,
                        timestamp_ns=row.timestamp_ns,
                        nonce=nonce,
                    )
                for nonce in self.ledger.used_nonces(record.session_id):
                    self.tracker._state(record.session_id).used_nonces.add(nonce)
            except DEFENDHC2Error as exc:  # pragma: no cover - tampered DB
                self.ledger.record_security_event(
                    "LEDGER_RESTORE_FAILURE", "CRITICAL",
                    {"session_id": record.session_id, "error": str(exc)},
                )
                raise

    # ------------------------------------------------------------------ L1.5
    def _genesis_chain_entry(self, genesis: GenesisRecord) -> ChainEntryRecord:
        payload_hash = Canonicalizer.payload_hash(genesis.genesis_payload)
        return ChainEntryRecord(
            session_id=genesis.session_id,
            sequence=0,
            event_type=EVENT_GENESIS,
            payload=genesis.genesis_payload,
            payload_hash=payload_hash,
            previous_hash="0" * 64,
            chain_hash=genesis.genesis_hash,
            mac=genesis.genesis_mac,
            timestamp_ns=genesis.timestamp_ns,
        )

    # ------------------------------------------------------------ session io
    def create_session(
        self, system_prompt: str, session_id: str | None = None
    ) -> dict[str, Any]:
        """Step 1 of the pipeline."""
        normalized_prompt = Canonicalizer.normalize_text(system_prompt)
        if not normalized_prompt.strip():
            raise SchemaValidationError("system_prompt must be non-empty")
        sid = session_id or f"sess-{secrets.token_hex(8)}"
        with self._lock:
            if self.ledger.get_session(sid) is not None:
                raise DEFENDHC2Error(f"session {sid!r} already exists")
            genesis = self.tracker.create_session(sid, normalized_prompt)
            self.ledger.create_session(
                SessionRecord(
                    session_id=sid,
                    system_prompt_hash=genesis.system_prompt_hash,
                    session_salt=genesis.session_salt,
                    genesis_hash=genesis.genesis_hash,
                    created_at_ns=genesis.timestamp_ns,
                )
            )
            self.ledger.append_chain_entry(self._genesis_chain_entry(genesis))
        return {
            "session_id": sid,
            "system_prompt_hash": genesis.system_prompt_hash,
            "genesis_hash": genesis.genesis_hash,
            "created_at_ns": genesis.timestamp_ns,
            "head_hash": genesis.genesis_hash,
            "next_sequence": 1,
        }

    def _ensure_session(self, session_id: str) -> None:
        if self.tracker.has_session(session_id):
            return
        if self.ledger.get_session(session_id) is None:
            raise SessionNotFoundError(session_id)
        self._restore_sessions_from_ledger()

    def _append_and_persist(
        self, session_id: str, event_type: str, payload: dict, nonce: str | None = None,
        **append_kwargs: Any,
    ) -> tuple[IntegrityResult, ChainEntryRecord | None]:
        """Append to the in-memory chain, then persist (or roll back)."""
        result, event = self.tracker.append_event(
            session_id, event_type, payload, nonce=nonce, **append_kwargs
        )
        if not result.passed or event is None:
            return result, None
        record = ChainEntryRecord(
            session_id=event.session_id,
            sequence=event.sequence,
            event_type=event.event_type,
            payload=event.payload,
            payload_hash=event.payload_hash,
            previous_hash=event.previous_hash,
            chain_hash=event.chain_hash,
            mac=event.mac,
            timestamp_ns=event.timestamp_ns,
        )
        self.ledger.append_chain_entry(record, nonce=nonce)
        return result, record

    # ------------------------------------------------------------- main flow
    def process_user_message(
        self,
        session_id: str,
        text: str,
        retrieved_docs: Sequence[dict[str, Any]] | None = None,
        history: Sequence[str] | None = None,
        nonce: str | None = None,
        claimed_previous_hash: str | None = None,
        claimed_sequence: int | None = None,
        client_system_prompt_hash: str | None = None,
        assistant_response: str | None = None,
    ) -> ProcessResult:
        """Steps 2-6 for one user turn.

        ``retrieved_docs``: ``[{"doc_id": str, "content": str,
        "source_uri": str}, ...]`` — the *fetched* contents, treated as
        untrusted input.
        """
        with self._lock:
            self._ensure_session(session_id)

            # ---- L0: schema validation + canonicalization -------------------
            schema_errors: list[str] = []
            try:
                Canonicalizer.validate_schema(
                    {"session_id": session_id, "text": text},
                    _PROCESS_SCHEMA,
                )
            except SchemaValidationError as exc:
                schema_errors.append(str(exc))
            normalized_text = Canonicalizer.normalize_text(text) if isinstance(text, str) else ""

            # ---- L2: append the user event FIRST so claimed_previous_hash is
            #      checked against the pre-turn head. -------------------------
            user_payload = {
                "role": "user",
                "text": normalized_text,
                "intake": {
                    "schema_valid": not schema_errors,
                    "nonce_present": nonce is not None,
                },
            }
            user_integrity, user_record = self._append_and_persist(
                session_id,
                EVENT_USER_MESSAGE,
                user_payload,
                nonce=nonce,
                claimed_previous_hash=claimed_previous_hash,
                claimed_sequence=claimed_sequence,
                client_system_prompt_hash=client_system_prompt_hash,
            )
            integrity_results = [user_integrity]

            # ---- L3: RAG document provenance --------------------------------
            documents: list[DocumentProvenanceResult] = []
            doc_records: list[ChainEntryRecord] = []
            doc_texts: list[str] = []
            for raw_doc in retrieved_docs or []:
                try:
                    Canonicalizer.validate_schema(
                        raw_doc,
                        {"doc_id": str, "content": str, "source_uri": str},
                    )
                except SchemaValidationError as exc:
                    schema_errors.append(f"retrieved_doc: {exc}")
                    continue
                doc_result = self.provenance.verify_document(
                    session_id=session_id,
                    doc_id=raw_doc["doc_id"],
                    content=raw_doc["content"],
                    source_uri=raw_doc["source_uri"],
                )
                documents.append(doc_result)
                doc_texts.append(raw_doc["content"])
                if user_record is not None:  # chain is live — bind the retrieval
                    _r, rec = self._append_and_persist(
                        session_id,
                        EVENT_RETRIEVAL,
                        {
                            "doc_id": doc_result.doc_id,
                            "doc_hash": doc_result.doc_hash,
                            "source_uri_hash": doc_result.source_uri_hash,
                            "instruction_risk": doc_result.instruction_risk,
                            "verdict": doc_result.verdict,
                        },
                    )
                    if rec is not None:
                        doc_records.append(rec)
                        doc_result.chain_hash = rec.chain_hash
                        integrity_results.append(_r)

            # ---- L1: content risk -------------------------------------------
            content = self.analyzer.analyze(
                normalized_text, retrieved_docs=doc_texts
            )
            drift, drift_ev = self.analyzer.conversation_drift_score(
                list(history or []), normalized_text
            )
            if drift_ev:
                content.evidence.extend(f"drift: {e}" for e in drift_ev)

            if user_record is not None:
                _r, _rec = self._append_and_persist(
                    session_id,
                    EVENT_CONTENT_ANALYSIS,
                    {
                        "content": content.to_dict(),
                        "conversation_drift_score": drift,
                    },
                )
                if _r is not None:
                    integrity_results.append(_r)

            # ---- L4: policy fusion ------------------------------------------
            decision = self.policy.decide(
                content,
                integrity_results=[r for r in integrity_results if r.status == "FAIL"],
                documents=documents,
                schema_valid=not schema_errors,
                schema_errors=schema_errors,
                conversation_drift_score=drift,
            )

            # Every decision is recorded as a chain event (spec).
            safe_prompt: str | None = None
            notes: list[str] = []
            if decision.action == "ALLOW":
                safe_prompt = normalized_text
            elif decision.action == "SANITIZE_AND_ALLOW":
                safe_prompt, notes = _sanitize_text(normalized_text)
                kept = [d.doc_id for d in documents if d.verdict != "rejected"]
                if len(kept) != len(documents):
                    notes.append(f"dropped rejected docs; kept {kept}")

            if user_record is not None:
                _r, _rec = self._append_and_persist(
                    session_id,
                    EVENT_POLICY_DECISION,
                    {
                        "decision": decision.to_dict(),
                        "user_chain_hash": user_record.chain_hash,
                        "doc_chain_hashes": [r.chain_hash for r in doc_records],
                    },
                )
                if _r is not None:
                    integrity_results.append(_r)
            if decision.action in {"REJECT", "QUARANTINE"}:
                self.ledger.record_security_event(
                    f"POLICY_{decision.action}",
                    "HIGH" if decision.action == "REJECT" else "MEDIUM",
                    {
                        "reasons": decision.reasons,
                        "content_risk": decision.content_risk,
                    },
                    session_id=session_id,
                )

            # Optionally record the assistant turn it would be safe to keep.
            if assistant_response is not None and decision.action in {
                "ALLOW",
                "SANITIZE_AND_ALLOW",
            }:
                self.record_assistant_message(session_id, assistant_response)

            head = self.tracker.head_hash_hex(session_id)
            seq = self.tracker.next_sequence(session_id) - 1
            return ProcessResult(
                session_id=session_id,
                decision=decision,
                content=content,
                integrity=integrity_results,
                documents=documents,
                tools=[],
                head_hash=head,
                sequence=seq,
                safe_prompt=safe_prompt,
                notes=notes,
            )

    # ------------------------------------------------------------- tool path
    def submit_tool_result(
        self,
        session_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: dict[str, Any] | str,
        signature: str | None = None,
        nonce: str | None = None,
    ) -> tuple[ToolProvenanceResult, PolicyDecision]:
        """L3 tool provenance + policy; binds valid outputs into the chain."""
        with self._lock:
            self._ensure_session(session_id)
            head = self.tracker.head_hash_hex(session_id)
            tool_result = self.provenance.verify_tool_output(
                session_id=session_id,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=tool_output,
                session_head=head,
                signature=signature,
            )

            bound_record = None
            integrity: list[IntegrityResult] = []
            if tool_result.verdict != "rejected":
                _r, bound_record = self._append_and_persist(
                    session_id,
                    EVENT_TOOL_OUTPUT,
                    {
                        "tool_name": tool_name,
                        "input_hash": tool_result.input_hash,
                        "output_hash": tool_result.output_hash,
                        "privileged": tool_result.privileged,
                        "signature_valid": tool_result.signature_valid,
                        "session_head": head,
                    },
                    nonce=nonce,
                )
                integrity.append(_r)

            tool_text = (
                tool_output
                if isinstance(tool_output, str)
                else Canonicalizer.canonical_json(tool_output)
            )
            if not tool_text.strip():
                tool_text = "[empty tool output]"
            content = self.analyzer.analyze(tool_text)
            decision = self.policy.decide(
                content,
                integrity_results=[r for r in integrity if r.status == "FAIL"],
                tools=[tool_result],
            )
            _r, _rec = self._append_and_persist(
                session_id,
                EVENT_POLICY_DECISION,
                {
                    "decision": decision.to_dict(),
                    "tool": tool_result.to_dict(),
                    "tool_chain_hash": bound_record.chain_hash if bound_record else None,
                },
            )
            if tool_result.verdict == "rejected":
                self.ledger.record_security_event(
                    "TOOL_PROVENANCE_REJECTED", "CRITICAL",
                    tool_result.to_dict(), session_id=session_id,
                )
            return tool_result, decision

    # ------------------------------------------------------ assistant record
    def record_assistant_message(
        self, session_id: str, text: str
    ) -> ChainEntryRecord:
        """Record a *server-observed* assistant turn on the chain."""
        with self._lock:
            self._ensure_session(session_id)
            result, record = self._append_and_persist(
                session_id,
                EVENT_ASSISTANT_MESSAGE,
                {"role": "assistant", "text": Canonicalizer.normalize_text(text)},
            )
            if record is None:
                raise DEFENDHC2Error(
                    f"assistant event rejected: {result.reason}"
                )
            return record

    # ------------------------------------------- stateless-history verify
    def verify_presented_event(
        self, session_id: str, event: dict[str, Any]
    ) -> IntegrityResult:
        """Verify one client-presented event (stateless transcript tail).

        ``event`` keys: sequence, previous_hash, event_type, payload,
        chain_hash, mac, timestamp_ns.
        """
        with self._lock:
            self._ensure_session(session_id)
            return self.tracker.verify_presented_event(
                session_id=session_id,
                sequence=int(event["sequence"]),
                previous_hash=str(event["previous_hash"]),
                event_type=str(event["event_type"]),
                payload=dict(event["payload"]),
                chain_hash=str(event["chain_hash"]),
                mac=str(event["mac"]),
                timestamp_ns=int(event["timestamp_ns"]),
            )

    def verify_presented_history(
        self, session_id: str, events: Sequence[dict[str, Any]]
    ) -> list[IntegrityResult]:
        return [self.verify_presented_event(session_id, e) for e in events]

    # ----------------------------------------------------- chain verify (L7)
    def verify_session(self, session_id: str) -> ChainVerificationReport:
        """Recompute the whole chain from the ledger; find the 1st bad event."""
        record = self.ledger.get_session(session_id)
        if record is None:
            raise SessionNotFoundError(session_id)
        entries = self.ledger.get_entries(session_id)

        # fresh throwaway tracker — no mutation of live state
        fresh = SessionContinuityTracker(master_secret=self._master_secret)
        fresh.seed_session(
            session_id,
            record.session_salt,
            record.created_at_ns,
            system_prompt_hash_hex=record.system_prompt_hash,
        )
        checks: list[dict[str, Any]] = []
        ok = True
        first_invalid: int | None = None
        reason = "OK"

        for entry in entries:
            if entry.sequence == 0:
                genesis_ok = (
                    entry.chain_hash == record.genesis_hash
                    and entry.previous_hash == "0" * 64
                )
                checks.append({
                    "sequence": 0,
                    "kind": "genesis",
                    "ok": genesis_ok,
                })
                if not genesis_ok and ok:
                    ok, first_invalid, reason = False, 0, "GENESIS_MISMATCH"
                continue
            try:
                prev_head = fresh.head_hash_hex(session_id)
                fresh.restore_event(
                    session_id=entry.session_id,
                    sequence=entry.sequence,
                    event_type=entry.event_type,
                    payload=entry.payload,
                    chain_hash=entry.chain_hash,
                    mac=entry.mac,
                    previous_hash=entry.previous_hash,
                    timestamp_ns=entry.timestamp_ns,
                )
                checks.append({
                    "sequence": entry.sequence,
                    "kind": entry.event_type,
                    "previous_hash_ok": entry.previous_hash == prev_head,
                    "mac_ok": True,
                    "ok": True,
                })
                if entry.previous_hash != prev_head and ok:
                    ok, first_invalid, reason = (
                        False, entry.sequence, "PREVIOUS_HASH_MISMATCH"
                    )
            except DEFENDHC2Error as exc:
                if ok:
                    ok, first_invalid, reason = False, entry.sequence, str(exc)
                checks.append({
                    "sequence": entry.sequence,
                    "kind": entry.event_type,
                    "ok": False,
                    "error": str(exc),
                })

        return ChainVerificationReport(
            session_id=session_id,
            ok=ok,
            entries_checked=len(entries),
            first_invalid_sequence=first_invalid,
            reason=reason,
            checks=checks,
        )

    # ---------------------------------------------------------------- export
    def export_session(self, session_id: str) -> dict[str, Any]:
        record = self.ledger.get_session(session_id)
        if record is None:
            raise SessionNotFoundError(session_id)
        entries = self.ledger.get_entries(session_id)
        report = self.verify_session(session_id)
        return {
            "session": record.to_dict(),
            "entries": [e.to_dict() for e in entries],
            "security_events": self.ledger.security_events(session_id=session_id),
            "verification": report.to_dict(),
            "exported_at_ns": time.time_ns(),
            "format": "defend-hc2-export/1",
        }

    # ------------------------------------------------------------ checkpoint
    def create_checkpoint(self) -> dict[str, Any]:
        """Merkle root over all session heads, MAC-signed (spec step 8)."""
        with self._lock:
            heads = self.ledger.session_heads()
            root = compute_merkle_root(heads)
            signature = Canonicalizer.hmac_sha3_256_hex(
                self._master_secret, bytes.fromhex(root), tag=TAG_CHECKPOINT_SIG
            )
            checkpoint_id = self.ledger.append_checkpoint(root, heads, signature)
            return {
                "checkpoint_id": checkpoint_id,
                "merkle_root": root,
                "session_heads": heads,
                "signature": signature,
                "sessions": len(heads),
            }

    # ------------------------------------------------------------- utilities
    def head(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            self._ensure_session(session_id)
            return {
                "session_id": session_id,
                "head_hash": self.tracker.head_hash_hex(session_id),
                "next_sequence": self.tracker.next_sequence(session_id),
            }

    def close(self) -> None:
        self.ledger.close()
