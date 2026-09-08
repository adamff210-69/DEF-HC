"""Layer 3 — RAG and tool-output provenance verification.

Every piece of *context* the model will see is treated as an
integrity-sensitive object:

Retrieved documents
    * content hash + source-URI hash are computed over canonical bytes,
    * the document is scanned for instruction-like content
      (indirect prompt injection),
    * the retrieval is bound into the session hash chain,
    * the document is marked ``trusted`` / ``suspicious`` / ``rejected``.

Tool outputs
    * input and output hashes are computed,
    * an optional (mandatory for privileged tools) HMAC receipt is verified
      against the tool's provisioned key,
    * the output is bound to the session chain at the current head —
      fabricated or unsigned privileged outputs are rejected.
"""

from __future__ import annotations

import hmac
import os
from typing import Any

from defend_hc2.canonicalization import Canonicalizer
from defend_hc2.constants import (
    DOC_SUSPICIOUS_MAX,
    DOC_TRUSTED_MAX,
    TAG_DOC,
    TAG_DOC_URI,
    TAG_TOOL_INPUT,
    TAG_TOOL_OUTPUT,
    TAG_TOOL_SIG,
)
from defend_hc2.content_risk import ContentRiskAnalyzer
from defend_hc2.results import DocumentProvenanceResult, ToolProvenanceResult


class ToolRegistry:
    """Provisioned tool keys + privilege flags (prototype keystore).

    Keys are provisioned out-of-band; verification uses
    ``hmac.compare_digest``.  In production this would be backed by a KMS
    and asymmetric signatures (e.g. Ed25519 receipts).
    """

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}

    def register_tool(
        self, name: str, key: bytes, privileged: bool = False
    ) -> None:
        self._tools[name] = {"key": key, "privileged": privileged}

    def is_registered(self, name: str) -> bool:
        return name in self._tools

    def is_privileged(self, name: str) -> bool:
        tool = self._tools.get(name)
        return bool(tool and tool["privileged"])

    def key_for(self, name: str) -> bytes | None:
        tool = self._tools.get(name)
        return tool["key"] if tool else None

    def sign_for(self, name: str, *parts: bytes) -> str:
        """Dev helper: produce a valid receipt for tests/demos (server-side)."""
        key = self.key_for(name)
        if key is None:
            raise KeyError(f"unknown tool {name!r}")
        return Canonicalizer.hmac_sha3_256_hex(key, *parts, tag=TAG_TOOL_SIG)

    @classmethod
    def from_env(cls, specs: str | None = None) -> "ToolRegistry":
        """Build from env: ``DEFEND_HC2_TOOLS='search:<hexkey>:0,files_write:<hexkey>:1'``."""
        registry = cls()
        spec = specs if specs is not None else os.environ.get("DEFEND_HC2_TOOLS", "")
        for item in filter(None, (s.strip() for s in spec.split(","))):
            name, keyhex, priv = item.split(":")
            registry.register_tool(name, bytes.fromhex(keyhex), priv == "1")
        return registry


class ProvenanceVerifier:
    """Spec Layer 3."""

    def __init__(
        self,
        analyzer: ContentRiskAnalyzer | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.analyzer = analyzer or ContentRiskAnalyzer(demo_mode=True)
        self.registry = tool_registry or ToolRegistry()

    # ------------------------------------------------------------ documents
    @staticmethod
    def document_content_hash(content: str | dict) -> str:
        canonical = (
            Canonicalizer.normalize_text(content)
            if isinstance(content, str)
            else Canonicalizer.normalize_obj(content)
        )
        if isinstance(canonical, str):
            data = canonical.encode("utf-8")
        else:
            data = Canonicalizer.canonical_bytes(canonical)
        return Canonicalizer.sha3_256_hex(data, tag=TAG_DOC)

    @staticmethod
    def source_uri_hash(source_uri: str) -> str:
        return Canonicalizer.sha3_256_hex(
            Canonicalizer.normalize_text(source_uri).encode("utf-8"),
            tag=TAG_DOC_URI,
        )

    def verify_document(
        self,
        session_id: str,
        doc_id: str,
        content: str,
        source_uri: str,
        metadata: dict | None = None,
    ) -> DocumentProvenanceResult:
        """Hash + indirect-injection analysis for one retrieved document."""
        doc_hash = self.document_content_hash(content)
        uri_hash = self.source_uri_hash(source_uri)
        risk, evidence = self.analyzer.analyze_document(content)

        if risk < DOC_TRUSTED_MAX:
            verdict = "trusted"
        elif risk < DOC_SUSPICIOUS_MAX:
            verdict = "suspicious"
        else:
            verdict = "rejected"

        return DocumentProvenanceResult(
            doc_id=doc_id,
            doc_hash=doc_hash,
            source_uri_hash=uri_hash,
            instruction_risk=round(risk, 6),
            verdict=verdict,
            evidence=evidence,
        )

    # ---------------------------------------------------------------- tools
    @staticmethod
    def tool_input_hash(tool_input: dict) -> str:
        return Canonicalizer.sha3_256_hex(
            Canonicalizer.canonical_bytes(tool_input), tag=TAG_TOOL_INPUT
        )

    @staticmethod
    def tool_output_hash(tool_output: dict | str) -> str:
        canonical = Canonicalizer.normalize_obj(tool_output)
        data = (
            canonical.encode("utf-8")
            if isinstance(canonical, str)
            else Canonicalizer.canonical_bytes(canonical)
        )
        return Canonicalizer.sha3_256_hex(data, tag=TAG_TOOL_OUTPUT)

    @staticmethod
    def expected_tool_signature(
        key: bytes,
        session_id: str,
        tool_name: str,
        input_hash: str,
        output_hash: str,
        session_head: str,
    ) -> str:
        """Receipt = HMAC(tool_key, session_id || tool || in || out || head)."""
        return Canonicalizer.hmac_sha3_256_hex(
            key,
            session_id.encode("utf-8"),
            tool_name.encode("utf-8"),
            bytes.fromhex(input_hash),
            bytes.fromhex(output_hash),
            bytes.fromhex(session_head),
            tag=TAG_TOOL_SIG,
        )

    def verify_tool_output(
        self,
        session_id: str,
        tool_name: str,
        tool_input: dict,
        tool_output: dict | str,
        session_head: str,
        signature: str | None = None,
    ) -> ToolProvenanceResult:
        """Verify a tool output before it enters the conversation.

        Privileged tools *must* present a valid receipt bound to the current
        session head; unsigned privileged outputs are rejected as fabricated.
        Unprivileged tools may be unsigned but are then marked
        ``unverified`` (handled by policy).
        """
        input_hash = self.tool_input_hash(tool_input)
        output_hash = self.tool_output_hash(tool_output)

        registry = self.registry
        registered = registry.is_registered(tool_name)
        privileged = registry.is_privileged(tool_name)

        signature_present = signature is not None
        signature_valid = False
        if signature_present:
            key = registry.key_for(tool_name)
            if key is not None:
                expected = self.expected_tool_signature(
                    key, session_id, tool_name, input_hash, output_hash, session_head
                )
                signature_valid = hmac.compare_digest(expected, signature.lower())

        if not registered:
            return ToolProvenanceResult(
                tool_name, input_hash, output_hash, privileged=True,
                signature_present=signature_present, signature_valid=False,
                verdict="rejected", reason="TOOL_NOT_REGISTERED",
            )
        if privileged and not signature_present:
            return ToolProvenanceResult(
                tool_name, input_hash, output_hash, privileged=True,
                signature_present=False, signature_valid=False,
                verdict="rejected", reason="UNSIGNED_PRIVILEGED_TOOL_OUTPUT",
            )
        if signature_present and not signature_valid:
            return ToolProvenanceResult(
                tool_name, input_hash, output_hash, privileged=privileged,
                signature_present=True, signature_valid=False,
                verdict="rejected", reason="INVALID_TOOL_SIGNATURE",
            )
        if privileged and signature_valid:
            verdict, reason = "verified", "PRIVILEGED_SIGNED"
        elif signature_valid:
            verdict, reason = "verified", "SIGNED"
        else:
            verdict, reason = "unverified", "UNSIGNED_UNPRIVILEGED"
        return ToolProvenanceResult(
            tool_name, input_hash, output_hash, privileged=privileged,
            signature_present=signature_present, signature_valid=signature_valid,
            verdict=verdict, reason=reason,
        )
