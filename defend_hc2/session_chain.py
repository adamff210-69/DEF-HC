"""Layer 2 — cryptographic session-continuity verification.

Implements the spec's keyed hash chain with forward key evolution:

Genesis (new session)::

    session_salt        = secrets.token_bytes(32)
    system_prompt_hash  = SHA3-256(canonical system prompt)
    K_0   = HMAC-SHA3-256(master_secret, session_id || session_salt)
    H_0   = SHA3-256("DEFEND-HC2-GENESIS" || session_id || system_prompt_hash
                     || session_salt || timestamp_ns)
    MAC_0 = HMAC-SHA3-256(K_0, H_0)

Event extension::

    payload_hash = SHA3-256(canonical_payload)
    H_t   = SHA3-256("DEFEND-HC2-EVENT" || session_id || sequence || H_{t-1}
                     || event_type || payload_hash || system_prompt_hash
                     || timestamp_ns)
    MAC_t = HMAC-SHA3-256(K_t, canonical_payload || H_t)
    K_{t+1} = SHA3-256("DEFEND-HC2-KEY-EVOLVE" || K_t || H_t)

Every ``||`` above is the length-prefixed framing of
:meth:`Canonicalizer.frame`.  Key evolution gives *forward secrecy of
verification state*: leaking the current key does not let anyone recompute
past keys, and no key can be rolled backwards.

The tracker validates, before any append:

* session identity,
* sequence monotonicity,
* ``claimed_previous_hash`` against the local head,
* nonce freshness,
* the bound system-prompt hash,
* a self-check of the local current head (in-memory tamper detection),

and can *non-destructively* verify events presented by a (stateless) client,
which is how fabricated assistant messages, stale-head replays and
cross-session splices are caught.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass, field

from defend_hc2.canonicalization import Canonicalizer, ct_equal
from defend_hc2.constants import (
    EVENT_GENESIS,
    SESSION_SALT_BYTES,
    TAG_EVENT,
    TAG_GENESIS,
    TAG_KEY_EVOLVE,
    TAG_SESSION_KEY,
)
from defend_hc2.exceptions import DEFENDHC2Error, SessionNotFoundError
from defend_hc2.results import IntegrityResult

int_struct = lambda seq: seq.to_bytes(8, "big")  # noqa: E731 - tiny helper


@dataclass(slots=True)
class GenesisRecord:
    session_id: str
    session_salt: str          # hex
    system_prompt_hash: str    # hex
    genesis_hash: str          # H_0, hex
    genesis_mac: str           # MAC_0, hex
    timestamp_ns: int
    genesis_payload: dict

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "session_salt": self.session_salt,
            "system_prompt_hash": self.system_prompt_hash,
            "genesis_hash": self.genesis_hash,
            "genesis_mac": self.genesis_mac,
            "timestamp_ns": self.timestamp_ns,
            "genesis_payload": self.genesis_payload,
        }


@dataclass(slots=True)
class AppendedEvent:
    session_id: str
    sequence: int
    event_type: str
    payload: dict
    payload_hash: str
    previous_hash: str
    chain_hash: str
    mac: str
    timestamp_ns: int

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "payload": self.payload,
            "payload_hash": self.payload_hash,
            "previous_hash": self.previous_hash,
            "chain_hash": self.chain_hash,
            "mac": self.mac,
            "timestamp_ns": self.timestamp_ns,
        }


@dataclass(slots=True)
class _SessionState:
    session_id: str
    salt: bytes
    system_prompt_hash: bytes
    head_hash: bytes
    current_key: bytes
    next_sequence: int
    created_at_ns: int
    # verification support: per-sequence (H_t, K_t, MAC_t, event_type)
    history: dict[int, tuple[bytes, bytes, bytes, str]] = field(default_factory=dict)
    used_nonces: set[str] = field(default_factory=set)
    last_timestamp_ns: int = 0
    # material for the head self-check (MAC recomputation)
    last_payload_canonical: bytes = b""


class SessionContinuityTracker:
    """Keyed hash-chain tracker (spec: Layer 2).

    Holding ``master_secret`` is what separates *verification* from
    *forgery*: clients present events; only the server can produce or check
    ``MAC_t``.
    """

    def __init__(self, master_secret: bytes | None = None) -> None:
        if master_secret is None:
            import os

            env = os.environ.get("DEFEND_HC2_MASTER_SECRET")
            master_secret = bytes.fromhex(env) if env else secrets.token_bytes(32)
        if len(master_secret) < 16:
            raise DEFENDHC2Error("master_secret must be at least 16 bytes")
        self._master_secret = master_secret
        self._sessions: dict[str, _SessionState] = {}
        # global index: chain hash -> session id (cross-session splice sensor)
        self._hash_owner: dict[bytes, str] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _system_prompt_hash(system_prompt: str) -> bytes:
        canonical = Canonicalizer.normalize_text(system_prompt).encode("utf-8")
        return hashlib.sha3_256(canonical).digest()

    def _derive_k0(self, session_id: str, salt: bytes) -> bytes:
        return Canonicalizer.hmac_sha3_256(
            self._master_secret,
            session_id.encode("utf-8"),
            salt,
            tag=TAG_SESSION_KEY,
        )

    @staticmethod
    def _genesis_hash(
        session_id: str, sp_hash: bytes, salt: bytes, ts_ns: int
    ) -> bytes:
        return Canonicalizer.sha3_256(
            session_id.encode("utf-8"),
            sp_hash,
            salt,
            int_struct(ts_ns),
            tag=TAG_GENESIS,
        )

    @staticmethod
    def _event_hash(
        session_id: str,
        sequence: int,
        prev_hash: bytes,
        event_type: str,
        payload_hash: bytes,
        sp_hash: bytes,
        ts_ns: int,
    ) -> bytes:
        return Canonicalizer.sha3_256(
            session_id.encode("utf-8"),
            int_struct(sequence),
            prev_hash,
            event_type.encode("utf-8"),
            payload_hash,
            sp_hash,
            int_struct(ts_ns),
            tag=TAG_EVENT,
        )

    @staticmethod
    def _event_mac(key: bytes, canonical_payload: bytes, h_t: bytes) -> bytes:
        return Canonicalizer.hmac_sha3_256(key, canonical_payload, h_t)

    @staticmethod
    def _evolve_key(key: bytes, h_t: bytes) -> bytes:
        return Canonicalizer.sha3_256(key, h_t, tag=TAG_KEY_EVOLVE)

    # -------------------------------------------------------------- genesis
    def create_session(
        self, session_id: str, system_prompt: str, timestamp_ns: int | None = None
    ) -> GenesisRecord:
        with self._lock:
            if session_id in self._sessions:
                raise DEFENDHC2Error(f"session {session_id!r} already exists on this node")
            ts = timestamp_ns if timestamp_ns is not None else time.time_ns()
            salt = secrets.token_bytes(SESSION_SALT_BYTES)
            sp_hash = self._system_prompt_hash(system_prompt)
            k0 = self._derive_k0(session_id, salt)
            h0 = self._genesis_hash(session_id, sp_hash, salt, ts)
            mac0 = Canonicalizer.hmac_sha3_256(k0, h0, tag=TAG_GENESIS)

            genesis_payload = {
                "kind": EVENT_GENESIS,
                "session_id": session_id,
                "system_prompt_hash": sp_hash.hex(),
                "session_salt": salt.hex(),
                "timestamp_ns": ts,
            }
            genesis_payload_canonical = Canonicalizer.canonical_bytes(genesis_payload)

            self._sessions[session_id] = _SessionState(
                session_id=session_id,
                salt=salt,
                system_prompt_hash=sp_hash,
                head_hash=h0,
                current_key=self._evolve_key(k0, h0),  # K_1
                next_sequence=1,
                created_at_ns=ts,
                history={0: (h0, k0, mac0, EVENT_GENESIS)},
                last_timestamp_ns=ts,
                last_payload_canonical=genesis_payload_canonical,
            )
            self._hash_owner[h0] = session_id
            return GenesisRecord(
                session_id=session_id,
                session_salt=salt.hex(),
                system_prompt_hash=sp_hash.hex(),
                genesis_hash=h0.hex(),
                genesis_mac=mac0.hex(),
                timestamp_ns=ts,
                genesis_payload=genesis_payload,
            )

    # ------------------------------------------------------------ accessors
    def _state(self, session_id: str) -> _SessionState:
        state = self._sessions.get(session_id)
        if state is None:
            raise SessionNotFoundError(session_id)
        return state

    def has_session(self, session_id: str) -> bool:
        return session_id in self._sessions

    def head_hash_hex(self, session_id: str) -> str:
        with self._lock:
            return self._state(session_id).head_hash.hex()

    def system_prompt_hash_hex(self, session_id: str) -> str:
        with self._lock:
            return self._state(session_id).system_prompt_hash.hex()

    def next_sequence(self, session_id: str) -> int:
        with self._lock:
            return self._state(session_id).next_sequence

    def session_salt_hex(self, session_id: str) -> str:
        with self._lock:
            return self._state(session_id).salt.hex()

    def created_at_ns(self, session_id: str) -> int:
        with self._lock:
            return self._state(session_id).created_at_ns

    def seed_session(
        self,
        session_id: str,
        salt_hex: str,
        timestamp_ns: int,
        system_prompt: str | None = None,
        system_prompt_hash_hex: str | None = None,
    ) -> GenesisRecord:
        """Recreate a session tracker from persisted genesis material.

        Used when the pipeline reloads sessions from the ledger after a
        restart; deterministic because salt + timestamp are persisted.
        Either the plaintext system prompt *or* its hash may be supplied
        (only the hash ever touches the ledger).
        """
        with self._lock:
            if session_id in self._sessions:
                self._sessions.pop(session_id)
            ts = timestamp_ns
            salt = bytes.fromhex(salt_hex)
            if system_prompt is not None:
                sp_hash = self._system_prompt_hash(system_prompt)
            elif system_prompt_hash_hex is not None:
                sp_hash = bytes.fromhex(system_prompt_hash_hex)
            else:
                raise DEFENDHC2Error(
                    "seed_session requires system_prompt or system_prompt_hash_hex"
                )
            k0 = self._derive_k0(session_id, salt)
            h0 = self._genesis_hash(session_id, sp_hash, salt, ts)
            mac0 = Canonicalizer.hmac_sha3_256(k0, h0, tag=TAG_GENESIS)
            genesis_payload = {
                "kind": EVENT_GENESIS,
                "session_id": session_id,
                "system_prompt_hash": sp_hash.hex(),
                "session_salt": salt.hex(),
                "timestamp_ns": ts,
            }
            self._sessions[session_id] = _SessionState(
                session_id=session_id,
                salt=salt,
                system_prompt_hash=sp_hash,
                head_hash=h0,
                current_key=self._evolve_key(k0, h0),
                next_sequence=1,
                created_at_ns=ts,
                history={0: (h0, k0, mac0, EVENT_GENESIS)},
                last_timestamp_ns=ts,
                last_payload_canonical=Canonicalizer.canonical_bytes(genesis_payload),
            )
            self._hash_owner[h0] = session_id
            return GenesisRecord(
                session_id=session_id,
                session_salt=salt.hex(),
                system_prompt_hash=sp_hash.hex(),
                genesis_hash=h0.hex(),
                genesis_mac=mac0.hex(),
                timestamp_ns=ts,
                genesis_payload=genesis_payload,
            )

    # ---------------------------------------------------------- validation
    def _validate_append(
        self,
        state: _SessionState,
        claimed_previous_hash: str | None,
        claimed_sequence: int | None,
        nonce: str | None,
        client_system_prompt_hash: str | None,
    ) -> IntegrityResult | None:
        sid = state.session_id
        head_hex = state.head_hash.hex()
        seq = state.next_sequence

        # system-prompt binding
        if client_system_prompt_hash is not None and not ct_equal(
            client_system_prompt_hash.lower(), state.system_prompt_hash.hex()
        ):
            return IntegrityResult(
                "FAIL", "SYSTEM_PROMPT_MISMATCH", "HIGH", head_hex, None, seq
            )
        # nonce freshness
        if nonce is not None and nonce in state.used_nonces:
            return IntegrityResult(
                "FAIL", "NONCE_REPLAY", "HIGH", head_hex, None, seq
            )
        # claimed previous hash against the local head (checked before the
        # sequence claim: the hash claim is the cryptographically specific
        # signal — a replayed old head is positively identified as stale,
        # spliced, or wrong, whereas a bare sequence gap is ambiguous)
        if claimed_previous_hash is not None:
            claimed = claimed_previous_hash.lower()
            try:
                claimed_b = bytes.fromhex(claimed)
            except ValueError:
                return IntegrityResult(
                    "FAIL", "MALFORMED_HASH", "HIGH", head_hex, None, seq
                )
            if ct_equal(claimed, head_hex):
                pass
            elif claimed_b in {v[0] for v in state.history.values()}:
                return IntegrityResult(
                    "FAIL", "STALE_HEAD_REPLAY", "HIGH", head_hex, None, seq
                )
            else:
                owner = self._hash_owner.get(claimed_b)
                reason = (
                    "CROSS_SESSION_SPLICE"
                    if (owner is not None and owner != sid)
                    else "PREVIOUS_HASH_MISMATCH"
                )
                severity = "CRITICAL" if reason == "CROSS_SESSION_SPLICE" else "HIGH"
                return IntegrityResult("FAIL", reason, severity, head_hex, None, seq)
        # sequence monotonicity (when the caller asserts one)
        if claimed_sequence is not None and claimed_sequence != seq:
            return IntegrityResult(
                "FAIL", "SEQUENCE_MISMATCH", "HIGH", head_hex, None, seq
            )
        # self-check: local head consistency (in-memory tamper sensor).
        # Genesis (sequence 0) is MAC-ed as MAC_0 = HMAC(K_0, H_0) with no
        # payload, so the payload-inclusive check applies from sequence 1 on.
        if state.last_payload_canonical and state.next_sequence >= 2:
            h_last, k_last, mac_last, _etype = state.history[state.next_sequence - 1]
            recomputed = self._event_mac(k_last, state.last_payload_canonical, h_last)
            if not hmac.compare_digest(recomputed, mac_last):
                return IntegrityResult(
                    "FAIL", "LOCAL_HEAD_TAMPER", "CRITICAL", head_hex, None, seq
                )
        return None

    # -------------------------------------------------------------- append
    def append_event(
        self,
        session_id: str,
        event_type: str,
        payload: dict,
        nonce: str | None = None,
        claimed_previous_hash: str | None = None,
        claimed_sequence: int | None = None,
        client_system_prompt_hash: str | None = None,
        timestamp_ns: int | None = None,
    ) -> tuple[IntegrityResult, AppendedEvent | None]:
        """Validate and append a new event to ``session_id``'s chain."""
        with self._lock:
            state = self._state(session_id)
            failure = self._validate_append(
                state,
                claimed_previous_hash,
                claimed_sequence,
                nonce,
                client_system_prompt_hash,
            )
            if failure is not None:
                return failure, None

            ts = timestamp_ns if timestamp_ns is not None else time.time_ns()
            normalized = Canonicalizer.normalize_obj(payload)
            canonical_payload = Canonicalizer.canonical_bytes(normalized)
            payload_hash = Canonicalizer.payload_hash(normalized)

            seq = state.next_sequence
            prev = state.head_hash
            h_t = self._event_hash(
                session_id,
                seq,
                prev,
                event_type,
                bytes.fromhex(payload_hash),
                state.system_prompt_hash,
                ts,
            )
            k_t = state.current_key
            mac_t = self._event_mac(k_t, canonical_payload, h_t)

            state.history[seq] = (h_t, k_t, mac_t, event_type)
            state.head_hash = h_t
            state.current_key = self._evolve_key(k_t, h_t)
            state.next_sequence = seq + 1
            state.last_timestamp_ns = ts
            state.last_payload_canonical = canonical_payload
            if nonce is not None:
                state.used_nonces.add(nonce)
            self._hash_owner[h_t] = session_id

            event = AppendedEvent(
                session_id=session_id,
                sequence=seq,
                event_type=event_type,
                payload=normalized,
                payload_hash=payload_hash,
                previous_hash=prev.hex(),
                chain_hash=h_t.hex(),
                mac=mac_t.hex(),
                timestamp_ns=ts,
            )
            return IntegrityResult(
                "PASS", "PASS", "NONE", prev.hex(), h_t.hex(), seq
            ), event

    # ----------------------------------------- non-destructive verification
    def verify_presented_event(
        self,
        session_id: str,
        sequence: int,
        previous_hash: str,
        event_type: str,
        payload: dict,
        chain_hash: str,
        mac: str,
        timestamp_ns: int,
    ) -> IntegrityResult:
        """Verify a client-presented historical event *without* mutating state.

        This is how stateless clients prove their transcript tail: fabricated
        assistant messages fail the MAC check, stale-head replays fail the
        sequence/position check, and cross-session splices fail the MAC and
        are positively identified when the hash belongs to another session.
        """
        with self._lock:
            state = self._state(session_id)
            head_hex = state.head_hash.hex()

            try:
                chain_hash_b = bytes.fromhex(chain_hash)
            except ValueError:
                return IntegrityResult(
                    "FAIL", "MALFORMED_HASH", "HIGH", head_hex, None, sequence
                )

            owner = self._hash_owner.get(chain_hash_b)
            if owner is not None and owner != session_id:
                return IntegrityResult(
                    "FAIL", "CROSS_SESSION_SPLICE", "CRITICAL", head_hex, None, sequence
                )

            try:
                previous_hash_b = bytes.fromhex(previous_hash.lower())
                mac_s = mac.lower()
            except ValueError:
                return IntegrityResult(
                    "FAIL", "MALFORMED_HASH", "HIGH", head_hex, None, sequence
                )

            canonical_payload = Canonicalizer.canonical_bytes(payload)
            recorded = state.history.get(sequence)
            if recorded is None:
                if sequence < state.next_sequence:
                    return IntegrityResult(
                        "FAIL", "REPLAY_OR_GAP", "HIGH", head_hex, None, sequence
                    )
                if sequence > state.next_sequence:
                    return IntegrityResult(
                        "FAIL", "SEQUENCE_FUTURE_GAP", "HIGH", head_hex, None, sequence
                    )
                # sequence == next: a claimed *new* event (e.g. a fabricated
                # assistant message).  Verify it speculatively against the
                # live head/key — without mutating state.
                if not ct_equal(previous_hash.lower(), head_hex):
                    return IntegrityResult(
                        "FAIL", "PREVIOUS_HASH_MISMATCH", "HIGH", head_hex, None, sequence
                    )
                payload_hash = Canonicalizer.payload_hash(payload)
                h_spec = self._event_hash(
                    session_id, sequence, previous_hash_b, event_type,
                    bytes.fromhex(payload_hash), state.system_prompt_hash,
                    timestamp_ns,
                )
                if not ct_equal(h_spec.hex(), chain_hash.lower()):
                    return IntegrityResult(
                        "FAIL", "CHAIN_HASH_MISMATCH", "CRITICAL", head_hex, None, sequence
                    )
                mac_spec = self._event_mac(state.current_key, canonical_payload, h_spec)
                if not hmac.compare_digest(mac_spec.hex(), mac_s):
                    return IntegrityResult(
                        "FAIL", "MAC_MISMATCH", "CRITICAL", head_hex, None, sequence
                    )
                # Cryptographically valid but never recorded by this node.
                # Treat conservative-fail: only server-appended events count.
                return IntegrityResult(
                    "FAIL", "UNRECORDED_EVENT", "MEDIUM", head_hex, None, sequence
                )

            h_t, k_t, mac_t, etype_recorded = recorded
            if not ct_equal(chain_hash.lower(), h_t.hex()):
                return IntegrityResult(
                    "FAIL", "CHAIN_HASH_MISMATCH", "CRITICAL", head_hex, None, sequence
                )
            if etype_recorded != event_type:
                return IntegrityResult(
                    "FAIL", "EVENT_TYPE_MISMATCH", "HIGH", head_hex, None, sequence
                )
            recomputed_mac = self._event_mac(k_t, canonical_payload, h_t)
            if not hmac.compare_digest(recomputed_mac.hex(), mac_s):
                return IntegrityResult(
                    "FAIL", "MAC_MISMATCH", "CRITICAL", head_hex, None, sequence
                )
            if not hmac.compare_digest(recomputed_mac, mac_t):
                return IntegrityResult(
                    "FAIL", "MAC_FORGED_WITH_VALID_HEAD", "CRITICAL", head_hex, None, sequence
                )
            if sequence < state.next_sequence - 1:
                # cryptographically valid event, but presented out of position
                return IntegrityResult(
                    "FAIL", "STALE_HEAD_REPLAY", "HIGH", head_hex, None, sequence
                )
            return IntegrityResult(
                "PASS", "PASS", "NONE", previous_hash.lower(), chain_hash.lower(), sequence
            )

    # ------------------------------------------------------- restore/replay
    def restore_event(
        self,
        session_id: str,
        sequence: int,
        event_type: str,
        payload: dict,
        chain_hash: str,
        mac: str,
        previous_hash: str,
        timestamp_ns: int,
        nonce: str | None = None,
    ) -> None:
        """Fast-forward an in-memory tracker from trusted ledger rows.

        The rows are verified (MAC recomputation) before being accepted,
        so a tampered ledger row cannot resurrect state.
        """
        with self._lock:
            state = self._state(session_id)
            if sequence == 0:
                return  # genesis already installed by create/seed_session
            if sequence != state.next_sequence:
                raise DEFENDHC2Error(
                    f"restore gap for {session_id!r}: have {state.next_sequence}, "
                    f"got row {sequence}"
                )
            if not ct_equal(previous_hash.lower(), state.head_hash.hex()):
                raise DEFENDHC2Error(
                    f"restore linkage break for {session_id!r} at sequence {sequence}"
                )
            normalized = Canonicalizer.normalize_obj(payload)
            canonical_payload = Canonicalizer.canonical_bytes(normalized)
            payload_hash = Canonicalizer.payload_hash(normalized)
            h_t = self._event_hash(
                session_id,
                sequence,
                state.head_hash,
                event_type,
                bytes.fromhex(payload_hash),
                state.system_prompt_hash,
                timestamp_ns,
            )
            if not ct_equal(h_t.hex(), chain_hash.lower()):
                raise DEFENDHC2Error(
                    f"restore hash mismatch for {session_id!r} at sequence {sequence}"
                )
            k_t = state.current_key
            recomputed_mac = self._event_mac(k_t, canonical_payload, h_t)
            if not hmac.compare_digest(recomputed_mac.hex(), mac.lower()):
                raise DEFENDHC2Error(
                    f"restore MAC mismatch for {session_id!r} at sequence {sequence}"
                )
            state.history[sequence] = (h_t, k_t, recomputed_mac, event_type)
            state.head_hash = h_t
            state.current_key = self._evolve_key(k_t, h_t)
            state.next_sequence = sequence + 1
            state.last_timestamp_ns = timestamp_ns
            state.last_payload_canonical = canonical_payload
            if nonce is not None:
                state.used_nonces.add(nonce)
            self._hash_owner[h_t] = session_id


def new_master_secret() -> str:
    """Generate a fresh master secret (hex) — put it in DEFEND_HC2_MASTER_SECRET."""
    return secrets.token_hex(32)
