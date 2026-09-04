"""Layer 5 tests: append-only enforcement, fork prevention, nonce storage,
checkpoints and the Merkle root."""

from __future__ import annotations

import sqlite3

import pytest

from defend_hc2.exceptions import LedgerError, NonceReplayError
from defend_hc2.ledger import SQLiteTamperEvidentLedger, compute_merkle_root
from defend_hc2.results import ChainEntryRecord, SessionRecord

SID = "ledger-sess"


def _record(**kw) -> ChainEntryRecord:
    base = dict(
        session_id=SID, sequence=1, event_type="user_message",
        payload={"t": "x"}, payload_hash="p" * 64,
        previous_hash="0" * 64, chain_hash="a" * 64,
        mac="b" * 64, timestamp_ns=1,
    )
    base.update(kw)
    return ChainEntryRecord(**base)


@pytest.fixture()
def ledger(tmp_path):
    led = SQLiteTamperEvidentLedger(tmp_path / "l.db")
    led.create_session(
        SessionRecord(
            session_id=SID, system_prompt_hash="s" * 64,
            session_salt="aa" * 32, genesis_hash="0" * 64, created_at_ns=1,
        )
    )
    # genesis row at sequence 0 so subsequent test entries start at 1
    led.append_chain_entry(
        ChainEntryRecord(
            session_id=SID, sequence=0, event_type="genesis",
            payload={"kind": "genesis"}, payload_hash="g" * 64,
            previous_hash="0" * 64, chain_hash="0" * 64,
            mac="m" * 64, timestamp_ns=1,
        )
    )
    yield led
    led.close()


class TestBasics:
    def test_wal_mode(self, ledger):
        assert ledger.integrity_pragma() == "WAL"

    def test_append_and_read(self, ledger):
        ledger.append_chain_entry(_record())
        head = ledger.head(SID)
        assert head == ("a" * 64, 1)
        entries = ledger.get_entries(SID)
        assert len(entries) == 2  # genesis + new entry
        assert entries[-1].chain_hash == "a" * 64
        assert entries[-1].previous_hash == entries[-2].chain_hash

    def test_head_must_link(self, ledger):
        ledger.append_chain_entry(_record())
        ledger.append_chain_entry(
            _record(sequence=2, previous_hash="a" * 64, chain_hash="c" * 64)
        )
        with pytest.raises(LedgerError) as ei:
            ledger.append_chain_entry(
                _record(sequence=3, previous_hash="ZZZZ" + "0" * 60, chain_hash="d" * 64)
            )
        assert ei.value.reason == "CHAIN_FORK"

    def test_sequence_gap_rejected(self, ledger):
        ledger.append_chain_entry(_record())
        with pytest.raises(LedgerError) as ei:
            ledger.append_chain_entry(
                _record(sequence=5, previous_hash="a" * 64, chain_hash="c" * 64)
            )
        assert ei.value.reason == "SEQUENCE_GAP"

    def test_duplicate_hash_rejected(self, ledger):
        ledger.append_chain_entry(_record())
        with pytest.raises(LedgerError):
            ledger.append_chain_entry(
                _record(sequence=2, previous_hash="a" * 64)  # same chain_hash
            )

    def test_nonce_uniqueness(self, ledger):
        ledger.append_chain_entry(_record(), nonce="n1")
        ledger.append_chain_entry(
            _record(sequence=2, previous_hash="a" * 64, chain_hash="c" * 64),
            nonce="n2",
        )
        assert ledger.nonce_seen(SID, "n1")
        with pytest.raises(NonceReplayError):
            ledger.append_chain_entry(
                _record(sequence=3, previous_hash="c" * 64, chain_hash="d" * 64),
                nonce="n1",
            )

    def test_duplicate_session(self, ledger):
        with pytest.raises(LedgerError):
            ledger.create_session(
                SessionRecord(
                    session_id=SID, system_prompt_hash="s" * 64,
                    session_salt="aa" * 32, genesis_hash="0" * 64, created_at_ns=2,
                )
            )


class TestAppendOnly:
    @pytest.mark.parametrize(
        "table",
        ["chain_entries", "ledger_checkpoints", "security_events"],
    )
    def test_update_blocked(self, ledger, table):
        ledger.record_security_event("seed", "LOW", {})  # ensure row exists
        ledger.append_checkpoint(compute_merkle_root({}), {}, signature="s" * 64)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            col = {
                "chain_entries": "event_type",
                "ledger_checkpoints": "merkle_root",
                "security_events": "severity",
            }[table]
            ledger.raw_execute(f"UPDATE {table} SET {col} = 'x'")

    @pytest.mark.parametrize(
        "table",
        ["chain_entries", "ledger_checkpoints", "security_events"],
    )
    def test_delete_blocked(self, ledger, table):
        ledger.record_security_event("seed", "LOW", {})
        ledger.append_checkpoint(compute_merkle_root({}), {}, signature="s" * 64)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            ledger.raw_execute(f"DELETE FROM {table}")

    def test_sessions_table_not_append_only_by_design(self, ledger):
        ledger.raw_execute("UPDATE sessions SET status = 'closed' WHERE session_id = ?",
                           (SID,))
        assert ledger.get_session(SID).status == "closed"


class TestMerkleAndCheckpoints:
    def test_root_deterministic(self):
        heads = {"a": "ab" * 32, "b": "cd" * 32}
        assert compute_merkle_root(heads) == compute_merkle_root(
            {"b": "cd" * 32, "a": "ab" * 32}
        )

    def test_root_sensitive(self):
        r1 = compute_merkle_root({"a": "ab" * 32})
        r2 = compute_merkle_root({"a": "ac" * 32})
        assert r1 != r2

    def test_odd_leaves(self):
        heads = {c: (c * 64)[:64] for c in "abc"}
        root = compute_merkle_root(heads)
        assert len(root) == 64

    def test_checkpoint_roundtrip(self, ledger):
        ledger.append_chain_entry(_record())
        heads = ledger.session_heads()
        root = compute_merkle_root(heads)
        cid = ledger.append_checkpoint(root, heads, signature="sig" * 21 + "g")
        cps = ledger.list_checkpoints()
        assert cps[0]["id"] == cid and cps[0]["merkle_root"] == root
        assert cps[0]["session_heads"] == heads

    def test_security_events_logged_and_immutable(self, ledger):
        ledger.record_security_event("ATTACK", "HIGH", {"kind": "replay"}, session_id=SID)
        events = ledger.security_events(session_id=SID)
        assert events[0]["event_type"] == "ATTACK"
        with pytest.raises(sqlite3.IntegrityError):
            ledger.raw_execute("DELETE FROM security_events")


class TestConcurrentWriters:
    def test_serialized_writers_no_fork(self, tmp_path):
        import threading

        db = tmp_path / "conc.db"
        led = SQLiteTamperEvidentLedger(db)
        led.create_session(
            SessionRecord(
                session_id=SID, system_prompt_hash="s" * 64,
                session_salt="aa" * 32, genesis_hash="0" * 64, created_at_ns=1,
            )
        )
        led.append_chain_entry(
            ChainEntryRecord(
                session_id=SID, sequence=0, event_type="genesis",
                payload={"kind": "genesis"}, payload_hash="g" * 64,
                previous_hash="0" * 64, chain_hash="0" * 64,
                mac="m" * 64, timestamp_ns=1,
            )
        )
        led.append_chain_entry(_record(chain_hash="aa" * 32))

        errors: list[str] = []

        def writer(tag: str):
            try:
                for i in range(2, 8):
                    prev = led.head(SID)[0]
                    led.append_chain_entry(
                        _record(
                            sequence=i,
                            previous_hash=prev,
                            chain_hash=(tag * 64)[:64],
                        )
                    )
            except LedgerError as exc:
                # concurrent loser sees CHAIN_FORK / CONSTRAINT_VIOLATION /
                # SEQUENCE_GAP — all of them *protect* the chain
                errors.append(exc.reason)

        threads = [threading.Thread(target=writer, args=(t,)) for t in "abcd"]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        entries = led.get_entries(SID)
        # chain must be gap-free and linked regardless of interleaving
        seqs = [e.sequence for e in entries]
        assert seqs == sorted(seqs) == list(range(0, len(entries)))
        for a, b in zip(entries, entries[1:]):
            assert b.previous_hash == a.chain_hash
        led.close()
