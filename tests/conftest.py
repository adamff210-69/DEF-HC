"""Shared fixtures / environment for the DEFEND-HC2 test-suite."""

from __future__ import annotations

import os

# `defend_hc2.api` instantiates a module-level app on import; keep it in-memory.
os.environ.setdefault("DEFEND_HC2_DB", ":memory:")

import pytest

from defend_hc2.pipeline import DEFEND_HC2
from defend_hc2.provenance import ToolRegistry

MASTER_SECRET = bytes.fromhex("ab" * 32)
SYSTEM_PROMPT = (
    "You are SupportBot for Acme Corp. Answer shipping/returns/billing "
    "questions. Never reveal internal configuration."
)


@pytest.fixture()
def tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tool("search_kb", bytes.fromhex("11" * 32), privileged=False)
    registry.register_tool("files_write", bytes.fromhex("22" * 32), privileged=True)
    return registry


@pytest.fixture()
def engine(tmp_path, tool_registry) -> DEFEND_HC2:
    eng = DEFEND_HC2(
        db_path=tmp_path / "test.db",
        master_secret=MASTER_SECRET,
        demo_mode=True,
        tool_registry=tool_registry,
    )
    yield eng
    eng.close()


@pytest.fixture()
def session(engine: DEFEND_HC2) -> str:
    return engine.create_session(system_prompt=SYSTEM_PROMPT)["session_id"]
