"""Layer 5 — append-only, tamper-evident SQLite ledger.

* WAL mode for crash-safe concurrent reads.
* Append-only enforced *inside the database*: ``BEFORE UPDATE`` /
  ``BEFORE DELETE`` triggers on ``chain_entries``, ``ledger_checkpoints``
  and ``security_events`` abort any mutation, so even direct SQL access
  (or a compromised code path inside this process) cannot rewrite history.
* Integrity constraints:
  ``UNIQUE(session_id, sequence)``, ``UNIQUE(session_id, chain_hash)``,
  ``UNIQUE(session_id, nonce)``.
* All appends run inside ``BEGIN IMMEDIATE`` transactions and re-check the
  chain head under the write lock, so two writers can never fork a chain.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from defend_hc2.canonicalization import Canonicalizer
from defend_hc2.constants import (
    TAG_CHECKPOINT_LEAF,
    TAG_CHECKPOINT_NODE,
    TAG_CHECKPOINT_ROOT,
)
from defend_hc2.exceptions import LedgerError, NonceReplayError
from defend_hc2.results import ChainEntryRecord, SessionRecord

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = FULL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
    session_id          TEXT PRIMARY KEY,
    system_prompt_hash  TEXT NOT NULL,
    session_salt        TEXT NOT NULL,
    genesis_hash        TEXT NOT NULL,
    created_at_ns       INTEGER NOT NULL,
    status              TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS chain_entries (
    session_id      TEXT    NOT NULL REFERENCES sessions(session_id),
    sequence        INTEGER NOT NULL CHECK (sequence >= 0),
    event_type      TEXT    NOT NULL,
    payload_json    TEXT    NOT NULL,
    payload_hash    TEXT    NOT NULL,
    previous_hash   TEXT    NOT NULL,
    chain_hash      TEXT    NOT NULL,
    mac             TEXT    NOT NULL,
    timestamp_ns    INTEGER NOT NULL,
    UNIQUE (session_id, sequence),
    UNIQUE (session_id, chain_hash)
);

CREATE TABLE IF NOT EXISTS used_nonces (
    session_id  TEXT    NOT NULL REFERENCES sessions(session_id),
    nonce       TEXT    NOT NULL,
    sequence    INTEGER NOT NULL,
    recorded_at_ns INTEGER NOT NULL,
    UNIQUE (session_id, nonce)
);

CREATE TABLE IF NOT EXISTS ledger_checkpoints (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    merkle_root         TEXT NOT NULL,
    session_heads_json  TEXT NOT NULL,
    signature           TEXT NOT NULL,
    created_at_ns       INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS security_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT,
    event_type      TEXT NOT NULL,
    severity        TEXT NOT NULL,
    details_json    TEXT NOT NULL,
    created_at_ns   INTEGER NOT NULL
);

-- ------------------------------------------------------------ append-only
CREATE TRIGGER IF NOT EXISTS chain_entries_no_update
BEFORE UPDATE ON chain_entries
BEGIN
    SELECT RAISE(ABORT, 'append-only: chain_entries cannot be updated');
END;

CREATE TRIGGER IF NOT EXISTS chain_entries_no_delete
BEFORE DELETE ON chain_entries
BEGIN
    SELECT RAISE(ABORT, 'append-only: chain_entries cannot be deleted');
END;

CREATE TRIGGER IF NOT EXISTS ledger_checkpoints_no_update
BEFORE UPDATE ON ledger_checkpoints
BEGIN
    SELECT RAISE(ABORT, 'append-only: ledger_checkpoints cannot be updated');
END;

CREATE TRIGGER IF NOT EXISTS ledger_checkpoints_no_delete
BEFORE DELETE ON ledger_checkpoints
BEGIN
    SELECT RAISE(ABORT, 'append-only: ledger_checkpoints cannot be deleted');
END;

CREATE TRIGGER IF NOT EXISTS security_events_no_update
BEFORE UPDATE ON security_events
BEGIN
    SELECT RAISE(ABORT, 'append-only: security_events cannot be updated');
END;

CREATE TRIGGER IF NOT EXISTS security_events_no_delete
BEFORE DELETE ON security_events
BEGIN
    SELECT RAISE(ABORT, 'append-only: security_events cannot be deleted');
END;

CREATE INDEX IF NOT EXISTS idx_chain_session_seq
    ON chain_entries (session_id, sequence);
CREATE INDEX IF NOT EXISTS idx_security_events_session
    ON security_events (session_id);
"""


def compute_merkle_root(
    session_heads: dict[str, str],
) -> str:
    """Merkle root over ``{session_id: head_hash}`` (spec: checkpointing)."""
    leaves = []
    for session_id in sorted(session_heads):
        leaf = Canonicalizer.sha3_256(
            session_id.encode("utf-8"),
            bytes.fromhex(session_heads[session_id]),
            tag=TAG_CHECKPOINT_LEAF,
        )
        leaves.append(leaf)
    if not leaves:
        leaves = [Canonicalizer.sha3_256(b"empty", tag=TAG_CHECKPOINT_LEAF)]
    level = leaves
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])  # duplicate odd leaf
        level = [
            Canonicalizer.sha3_256(level[i], level[i + 1], tag=TAG_CHECKPOINT_NODE)
            for i in range(0, len(level), 2)
        ]
    return Canonicalizer.sha3_256(level[0], tag=TAG_CHECKPOINT_ROOT).hex()


class SQLiteTamperEvidentLedger:
    """Spec Layer 5."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)

    # -------------------------------------------------------------- sessions
    def create_session(self, record: SessionRecord) -> None:
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO sessions (session_id, system_prompt_hash,"
                    " session_salt, genesis_hash, created_at_ns, status)"
                    " VALUES (?,?,?,?,?,?)",
                    (
                        record.session_id,
                        record.system_prompt_hash,
                        record.session_salt,
                        record.genesis_hash,
                        record.created_at_ns,
                        record.status,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise LedgerError("SESSION_EXISTS", str(exc)) from exc

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        return SessionRecord(
            session_id=row["session_id"],
            system_prompt_hash=row["system_prompt_hash"],
            session_salt=row["session_salt"],
            genesis_hash=row["genesis_hash"],
            created_at_ns=row["created_at_ns"],
            status=row["status"],
        )

    def list_sessions(self) -> list[SessionRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sessions ORDER BY created_at_ns"
            ).fetchall()
        return [
            SessionRecord(
                session_id=r["session_id"],
                system_prompt_hash=r["system_prompt_hash"],
                session_salt=r["session_salt"],
                genesis_hash=r["genesis_hash"],
                created_at_ns=r["created_at_ns"],
                status=r["status"],
            )
            for r in rows
        ]

    # ----------------------------------------------------------------- chain
    def append_chain_entry(
        self, entry: ChainEntryRecord, nonce: str | None = None
    ) -> None:
        """Append one entry; re-verify the head under the write lock.

        ``BEGIN IMMEDIATE`` acquires the reserved write lock up-front, so no
        other writer can interleave and fork the chain between the head
        check and the insert.
        """
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE")
                head = cur.execute(
                    "SELECT chain_hash, sequence FROM chain_entries"
                    " WHERE session_id = ? ORDER BY sequence DESC LIMIT 1",
                    (entry.session_id,),
                ).fetchone()
                expected_prev = head["chain_hash"] if head else None
                expected_seq = (head["sequence"] + 1) if head else 0
                if expected_prev is not None and entry.previous_hash != expected_prev:
                    cur.execute("ROLLBACK")
                    raise LedgerError(
                        "CHAIN_FORK",
                        f"previous_hash {entry.previous_hash[:16]}… != ledger head "
                        f"{expected_prev[:16]}… (concurrent write or stale state)",
                    )
                if entry.sequence != expected_seq:
                    cur.execute("ROLLBACK")
                    raise LedgerError(
                        "SEQUENCE_GAP",
                        f"entry.sequence={entry.sequence} but ledger expects {expected_seq}",
                    )
                if nonce is not None:
                    try:
                        cur.execute(
                            "INSERT INTO used_nonces (session_id, nonce, sequence,"
                            " recorded_at_ns) VALUES (?,?,?,?)",
                            (entry.session_id, nonce, entry.sequence, time.time_ns()),
                        )
                    except sqlite3.IntegrityError:
                        cur.execute("ROLLBACK")
                        raise NonceReplayError(entry.session_id, nonce) from None
                cur.execute(
                    "INSERT INTO chain_entries (session_id, sequence, event_type,"
                    " payload_json, payload_hash, previous_hash, chain_hash, mac,"
                    " timestamp_ns) VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        entry.session_id,
                        entry.sequence,
                        entry.event_type,
                        Canonicalizer.canonical_json(entry.payload),
                        entry.payload_hash,
                        entry.previous_hash,
                        entry.chain_hash,
                        entry.mac,
                        entry.timestamp_ns,
                    ),
                )
                cur.execute("COMMIT")
            except LedgerError:
                raise
            except NonceReplayError:
                raise
            except sqlite3.IntegrityError as exc:
                cur.execute("ROLLBACK")
                raise LedgerError("CONSTRAINT_VIOLATION", str(exc)) from exc
            except sqlite3.Error as exc:  # pragma: no cover - defensive
                cur.execute("ROLLBACK")
                raise LedgerError("SQLITE_ERROR", str(exc)) from exc

    def head(self, session_id: str) -> tuple[str, int] | None:
        """Current (chain_hash, sequence) of the session, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT chain_hash, sequence FROM chain_entries"
                " WHERE session_id = ? ORDER BY sequence DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return (row["chain_hash"], row["sequence"]) if row else None

    def get_entries(self, session_id: str) -> list[ChainEntryRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chain_entries WHERE session_id = ?"
                " ORDER BY sequence ASC",
                (session_id,),
            ).fetchall()
        return [
            ChainEntryRecord(
                session_id=r["session_id"],
                sequence=r["sequence"],
                event_type=r["event_type"],
                payload=json.loads(r["payload_json"]),
                payload_hash=r["payload_hash"],
                previous_hash=r["previous_hash"],
                chain_hash=r["chain_hash"],
                mac=r["mac"],
                timestamp_ns=r["timestamp_ns"],
            )
            for r in rows
        ]

    def entry_at(self, session_id: str, sequence: int) -> ChainEntryRecord | None:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM chain_entries WHERE session_id = ? AND sequence = ?",
                (session_id, sequence),
            ).fetchone()
        if r is None:
            return None
        return ChainEntryRecord(
            session_id=r["session_id"],
            sequence=r["sequence"],
            event_type=r["event_type"],
            payload=json.loads(r["payload_json"]),
            payload_hash=r["payload_hash"],
            previous_hash=r["previous_hash"],
            chain_hash=r["chain_hash"],
            mac=r["mac"],
            timestamp_ns=r["timestamp_ns"],
        )

    def session_heads(self) -> dict[str, str]:
        """``{session_id: current chain head}`` for checkpointing."""
        heads: dict[str, str] = {}
        with self._lock:
            rows = self._conn.execute(
                "SELECT c.session_id, c.chain_hash FROM chain_entries c"
                " JOIN (SELECT session_id, MAX(sequence) AS maxseq"
                "       FROM chain_entries GROUP BY session_id) m"
                "   ON c.session_id = m.session_id AND c.sequence = m.maxseq"
            ).fetchall()
        for r in rows:
            heads[r["session_id"]] = r["chain_hash"]
        return heads

    def nonce_seen(self, session_id: str, nonce: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM used_nonces WHERE session_id = ? AND nonce = ?",
                (session_id, nonce),
            ).fetchone()
        return row is not None

    def used_nonces(self, session_id: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT nonce FROM used_nonces WHERE session_id = ? ORDER BY recorded_at_ns",
                (session_id,),
            ).fetchall()
        return [r["nonce"] for r in rows]

    # ------------------------------------------------------------ checkpoints
    def append_checkpoint(
        self,
        merkle_root: str,
        session_heads: dict[str, str],
        signature: str,
        created_at_ns: int | None = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO ledger_checkpoints (merkle_root, session_heads_json,"
                " signature, created_at_ns) VALUES (?,?,?,?)",
                (
                    merkle_root,
                    Canonicalizer.canonical_json(session_heads),
                    signature,
                    created_at_ns if created_at_ns is not None else time.time_ns(),
                ),
            )
            return int(cur.lastrowid)

    def list_checkpoints(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM ledger_checkpoints ORDER BY id"
            ).fetchall()
        return [
            {
                "id": r["id"],
                "merkle_root": r["merkle_root"],
                "session_heads": json.loads(r["session_heads_json"]),
                "signature": r["signature"],
                "created_at_ns": r["created_at_ns"],
            }
            for r in rows
        ]

    # ------------------------------------------------------- security events
    def record_security_event(
        self,
        event_type: str,
        severity: str,
        details: dict[str, Any],
        session_id: str | None = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO security_events (session_id, event_type, severity,"
                " details_json, created_at_ns) VALUES (?,?,?,?,?)",
                (
                    session_id,
                    event_type,
                    severity,
                    Canonicalizer.canonical_json(details),
                    time.time_ns(),
                ),
            )
            return int(cur.lastrowid)

    def security_events(self, session_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM security_events"
        params: tuple = ()
        if session_id is not None:
            sql += " WHERE session_id = ?"
            params = (session_id,)
        sql += " ORDER BY id"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "event_type": r["event_type"],
                "severity": r["severity"],
                "details": json.loads(r["details_json"]),
                "created_at_ns": r["created_at_ns"],
            }
            for r in rows
        ]

    # ----------------------------------------------------------------- misc
    def raw_execute(self, sql: str, params: Iterable = ()) -> int:
        """Escape hatch used by tests/demos to show triggers blocking tampering.

        Returns the number of rows affected; raises ``sqlite3.IntegrityError``
        when an append-only trigger aborts the statement.
        """
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            return cur.rowcount

    def integrity_pragma(self) -> str:
        with self._lock:
            row = self._conn.execute("PRAGMA journal_mode").fetchone()
            return str(row[0]).upper()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SQLiteTamperEvidentLedger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
