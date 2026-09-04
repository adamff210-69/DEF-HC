"""Protocol constants for DEFEND-HC2.

All domain-separation tags are UTF-8 encoded and *length-prefixed* when fed
into a hash or MAC (see :func:`defend_hc2.canonicalization.Canonicalizer.frame`),
so concatenation fields can never be reinterpreted (``H("ab"||"c") !=
H("a"||"bc")`` under framing).
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Domain-separation tags (spec, Layer 0)
# --------------------------------------------------------------------------
TAG_GENESIS = b"DEFEND-HC2-GENESIS"
TAG_EVENT = b"DEFEND-HC2-EVENT"
TAG_PAYLOAD = b"DEFEND-HC2-PAYLOAD"
TAG_KEY_EVOLVE = b"DEFEND-HC2-KEY-EVOLVE"

# Additional tags used by layers 3-5.  Every distinct hashing context in the
# system gets its own tag so no two contexts can ever collide.
TAG_DOC = b"DEFEND-HC2-DOC"
TAG_DOC_URI = b"DEFEND-HC2-DOC-URI"
TAG_TOOL_INPUT = b"DEFEND-HC2-TOOL-INPUT"
TAG_TOOL_OUTPUT = b"DEFEND-HC2-TOOL-OUTPUT"
TAG_TOOL_SIG = b"DEFEND-HC2-TOOL-SIG"
TAG_CHECKPOINT_LEAF = b"DEFEND-HC2-CHECKPOINT-LEAF"
TAG_CHECKPOINT_NODE = b"DEFEND-HC2-CHECKPOINT-NODE"
TAG_CHECKPOINT_ROOT = b"DEFEND-HC2-CHECKPOINT-ROOT"
TAG_CHECKPOINT_SIG = b"DEFEND-HC2-CHECKPOINT-SIG"
TAG_SESSION_KEY = b"DEFEND-HC2-SESSION-KEY"

# --------------------------------------------------------------------------
# Signal fusion (spec Layer 4) — PREDEFINED BASELINE weights, *not* tuned on
# any evaluation data.  Inactive channels are ``None`` upstream and simply
# absent here: fusion = strongest + 0.5 * (noisy_or - strongest) over active
# channels only, so a strong direct attack is never diluted by missing
# context (defect P2).
# --------------------------------------------------------------------------
FUSION_WEIGHTS = {
    "injection": 1.0,
    "lexical": 0.9,
    "retrieval": 1.0,
    "mismatch": 0.6,
    "drift": 0.3,
}

# Legacy weights kept for provenance only (benchmark metrics recorded them);
# no code path uses them anymore.
W_INJECTION = 0.40
W_LEXICAL = 0.20
W_RETRIEVAL_INJECTION = 0.20
W_INTENT_MISMATCH = 0.10
W_CONVERSATION_DRIFT = 0.10

THRESHOLD_REJECT = 0.85
THRESHOLD_QUARANTINE = 0.65
THRESHOLD_SANITIZE = 0.40

# Provenance verdict thresholds for retrieved documents.
DOC_TRUSTED_MAX = 0.40
DOC_SUSPICIOUS_MAX = 0.75

# Key / salt sizes.
SESSION_SALT_BYTES = 32
KEY_BYTES = 32  # SHA3-256 / HMAC-SHA3-256 output size.

# Event types recorded on the chain.
EVENT_GENESIS = "genesis"
EVENT_USER_MESSAGE = "user_message"
EVENT_ASSISTANT_MESSAGE = "assistant_message"
EVENT_RETRIEVAL = "retrieval"
EVENT_TOOL_OUTPUT = "tool_output"
EVENT_POLICY_DECISION = "policy_decision"
EVENT_CONTENT_ANALYSIS = "content_analysis"
EVENT_CHECKPOINT = "checkpoint"

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
