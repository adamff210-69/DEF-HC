"""FastAPI surface for DEFEND-HC2.

Endpoints (spec)::

    POST /session
    POST /process
    POST /tool-result
    GET  /verify/{session_id}
    GET  /export/{session_id}
    POST /checkpoint

plus ``GET /health`` and ``POST /assistant-message`` (recording the
server-observed assistant turn on the chain).
"""

from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from defend_hc2.exceptions import (
    ChainIntegrityError,
    DEFENDHC2Error,
    NonceReplayError,
    SchemaValidationError,
    SessionNotFoundError,
)
from defend_hc2.pipeline import DEFEND_HC2

# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    system_prompt: str = Field(min_length=1)
    session_id: Optional[str] = None


class RetrievedDoc(BaseModel):
    doc_id: str
    content: str
    source_uri: str


class ProcessRequest(BaseModel):
    session_id: str
    text: str
    retrieved_docs: list[RetrievedDoc] = Field(default_factory=list)
    history: list[str] = Field(default_factory=list)
    nonce: Optional[str] = None
    claimed_previous_hash: Optional[str] = None
    claimed_sequence: Optional[int] = None
    client_system_prompt_hash: Optional[str] = None
    assistant_response: Optional[str] = None


class ToolResultRequest(BaseModel):
    session_id: str
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_output: Any
    signature: Optional[str] = None
    nonce: Optional[str] = None


class AssistantMessageRequest(BaseModel):
    session_id: str
    text: str


class PresentedEventRequest(BaseModel):
    session_id: str
    events: list[dict[str, Any]]


# --------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------


def create_app(
    db_path: str | None = None,
    master_secret: str | None = None,
    demo_mode: bool | None = None,
    weights_path: str | None = None,
    engine: DEFEND_HC2 | None = None,
) -> FastAPI:
    if engine is None:
        engine = DEFEND_HC2(
            db_path=db_path or os.environ.get("DEFEND_HC2_DB", "defend_hc2.db"),
            master_secret=master_secret or os.environ.get("DEFEND_HC2_MASTER_SECRET"),
            demo_mode=(
                demo_mode
                if demo_mode is not None
                else os.environ.get("DEFEND_HC2_DEMO_MODE", "1") != "0"
            ),
            weights_path=weights_path or os.environ.get("DEFEND_HC2_WEIGHTS"),
        )

    app = FastAPI(
        title="DEFEND-HC2",
        version="2.0.0",
        description="Dual-layer LLM security: content risk + cryptographic "
        "session-continuity enforcement with an append-only audit ledger.",
    )
    app.state.engine = engine

    @app.exception_handler(SessionNotFoundError)
    async def _session_not_found(_: Request, exc: SessionNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"error": "SESSION_NOT_FOUND", "detail": str(exc)})

    @app.exception_handler(NonceReplayError)
    async def _nonce_replay(_: Request, exc: NonceReplayError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"error": "NONCE_REPLAY", "detail": str(exc)})

    @app.exception_handler(SchemaValidationError)
    async def _schema_invalid(_: Request, exc: SchemaValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"error": "SCHEMA_INVALID", "detail": str(exc)})

    @app.exception_handler(ChainIntegrityError)
    async def _chain_error(_: Request, exc: ChainIntegrityError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"error": exc.reason, "detail": str(exc)})

    @app.exception_handler(DEFENDHC2Error)
    async def _generic(_: Request, exc: DEFENDHC2Error) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": type(exc).__name__, "detail": str(exc)})

    # ------------------------------------------------------------- endpoints
    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "demo_mode": engine.analyzer.demo_mode,
            "journal_mode": engine.ledger.integrity_pragma(),
        }

    @app.post("/session", status_code=201)
    def create_session(req: CreateSessionRequest) -> dict[str, Any]:
        return engine.create_session(
            system_prompt=req.system_prompt, session_id=req.session_id
        )

    @app.post("/process")
    def process(req: ProcessRequest) -> dict[str, Any]:
        result = engine.process_user_message(
            session_id=req.session_id,
            text=req.text,
            retrieved_docs=[d.model_dump() for d in req.retrieved_docs],
            history=req.history,
            nonce=req.nonce,
            claimed_previous_hash=req.claimed_previous_hash,
            claimed_sequence=req.claimed_sequence,
            client_system_prompt_hash=req.client_system_prompt_hash,
            assistant_response=req.assistant_response,
        )
        return result.to_dict()

    @app.post("/tool-result")
    def tool_result(req: ToolResultRequest) -> dict[str, Any]:
        provenance, decision = engine.submit_tool_result(
            session_id=req.session_id,
            tool_name=req.tool_name,
            tool_input=req.tool_input,
            tool_output=req.tool_output,
            signature=req.signature,
            nonce=req.nonce,
        )
        return {"provenance": provenance.to_dict(), "decision": decision.to_dict()}

    @app.post("/assistant-message", status_code=201)
    def assistant_message(req: AssistantMessageRequest) -> dict[str, Any]:
        record = engine.record_assistant_message(req.session_id, req.text)
        return record.to_dict()

    @app.post("/verify-presented")
    def verify_presented(req: PresentedEventRequest) -> dict[str, Any]:
        results = engine.verify_presented_history(req.session_id, req.events)
        return {
            "results": [r.to_dict() for r in results],
            "all_passed": all(r.passed for r in results),
        }

    @app.get("/verify/{session_id}")
    def verify(session_id: str) -> dict[str, Any]:
        return engine.verify_session(session_id).to_dict()

    @app.get("/export/{session_id}")
    def export(session_id: str) -> dict[str, Any]:
        return engine.export_session(session_id)

    @app.post("/checkpoint")
    def checkpoint() -> dict[str, Any]:
        return engine.create_checkpoint()

    @app.get("/head/{session_id}")
    def head(session_id: str) -> dict[str, Any]:
        return engine.head(session_id)

    return app


app = create_app()


def main() -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "defend_hc2.api:app",
        host=os.environ.get("DEFEND_HC2_HOST", "0.0.0.0"),
        port=int(os.environ.get("DEFEND_HC2_PORT", "8200")),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
