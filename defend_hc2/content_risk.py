"""Layer 1 — content risk analysis.

Combines four deterministic signals:

1. **Lexical detector** — weighted pattern bank for injection/jailbreak syntax.
2. **Embedding classifier** — logistic layer over ``BAAI/bge-small-en-v1.5``
   embeddings, with trained weights loaded from disk
   (``demo_mode=False``).  In ``demo_mode=True`` the classifier output is a
   deterministic heuristic fusion of the lexical and structural features
   instead (no randomness anywhere).
3. **RAG document instruction detector** — imperative / role-assertion /
   encoded-payload cues in retrieved documents (the indirect-injection
   surface).
4. **Intent/context mismatch detector** — semantic (embedding cosine) or
   lexical-overlap disagreement between the user's stated request and the
   retrieved context.

sentence-transformers is imported lazily: ``demo_mode=True`` works in a
pure-stdlib environment.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from defend_hc2 import normalize as _norm
from defend_hc2.canonicalization import Canonicalizer
from defend_hc2.embedder import get_sentence_transformer
from defend_hc2.constants import (
    DEFAULT_EMBEDDING_MODEL,
    FUSION_WEIGHTS,
)
from defend_hc2.exceptions import (
    EmbeddingBackendUnavailableError,
    SchemaValidationError,
)
from defend_hc2.results import ContentRiskResult


def dedup_evidence(evidence: Sequence[str]) -> list[str]:
    """Deterministic dedup (first-seen order, casefold keys — spec P6).

    ``casefold`` rather than ``hash()`` so identity never depends on hash
    randomization.
    """
    seen: set[str] = set()
    out: list[str] = []
    for item in evidence:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def combine_signals(channels: dict[str, float | None]) -> float:
    """Predefined-baseline fusion (spec Phase 6).

    ``None`` channels are inactive — absent context does not dilute active
    evidence.  For active channels ``s_i = clamp(w_i * v_i, 0, 0.999)`` and
    ``risk = strongest + 0.5 * (noisy_or - strongest)``, clamped to [0, 1].
    """
    active = [
        min(max(FUSION_WEIGHTS[name] * value, 0.0), 0.999)
        for name, value in channels.items()
        if value is not None
    ]
    if not active:
        return 0.0
    strongest = max(active)
    prod = 1.0
    for s_i in active:
        prod *= 1.0 - s_i
    noisy_or = 1.0 - prod
    return round(_clamp01(strongest + 0.5 * (noisy_or - strongest)), 6)

# ==========================================================================
# Lexical pattern bank.  (pattern, weight, human-readable evidence label)
# Weights are additive; lexical_score = min(1, sum of matched weights).
# ==========================================================================
_LEXICAL_PATTERNS: list[tuple[str, float, str]] = [
    # --- direct instruction override ------------------------------------
    (r"\bignore\s+(all|any|the|your|every|previous|prior|above|earlier)\b[^.\n]{0,40}\b(instructions?|prompts?|rules?|guidelines?|constraints?)", 0.55, "instruction override"),
    (r"\bdisregard\b[^.\n]{0,40}\b(instructions?|rules?|guidelines?|prompts?|above|previous)", 0.55, "instruction override"),
    (r"\bforget\b[^.\n]{0,30}\b(everything|instructions?|training|rules?|prompts?)", 0.50, "memory-wipe directive"),
    (r"\bdo\s+not\s+follow\b[^.\n]{0,40}\b(instructions?|rules?)", 0.45, "instruction override"),
    (r"\bnew\s+(instructions?|directives?|rules?)\b[^.\n]{0,30}\b(override|replace|supersede|follow)\b", 0.45, "instruction replacement"),
    (r"\boverride\s+(your\s+)?(system|previous|safety)\b", 0.45, "explicit safety override"),
    # --- persona / role hijack -------------------------------------------
    (r"\byou\s+are\s+now\b", 0.40, "persona hijack"),
    (r"\bact\s+as\b[^.\n]{0,40}\b(unrestricted|uncensored|evil|dan|no\s+filter|amoral)", 0.50, "persona hijack"),
    (r"\bpretend\s+(to\s+be|you\s+are|you'?re)\b", 0.35, "persona hijack"),
    (r"\bdan\b[^.\n]{0,25}\b(mode|jailbreak)?\b", 0.35, "DAN jailbreak token"),
    (r"\bdo\s+anything\s+now\b", 0.50, "DAN jailbreak token"),
    (r"\bjailbreak\b", 0.45, "jailbreak token"),
    (r"\bdeveloper\s+mode\b[^.\n]{0,30}\b(enabled|on|activate)", 0.45, "developer-mode jailbreak"),
    (r"\b(no|without)\s+(ethical|safety|content)\s+(guidelines?|restrictions?|filters?|limitations?)", 0.45, "safety-policy removal"),
    (r"\bbypass\b[^.\n]{0,30}\b(safety|filter|restriction|guardrail|policy)", 0.50, "guardrail bypass"),
    # --- secret exfiltration ---------------------------------------------
    (r"\b(reveal|show|print|display|output|repeat|leak|exfiltrate|disclose)\b[^.\n]{0,30}\b(system\s+prompt|instructions?|api\s+key|secret|password|token|credentials?|config)", 0.55, "secret/prompt exfiltration"),
    (r"\byour\s+(original\s+)?(system\s+prompt|initial\s+instructions?|hidden\s+instructions?)", 0.40, "system-prompt probing"),
    (r"\bwhat\s+(are|were)\s+your\s+(instructions?|rules?|directives?)\b", 0.35, "system-prompt probing"),
    # --- chat-template / delimiter injection ------------------------------
    (r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>", 0.55, "chat-template delimiter injection"),
    (r"\[/?INST\]|<<\s*/?\s*SYS\s*>>", 0.55, "chat-template delimiter injection"),
    (r"<\s*/?\s*system\s*>|^\s*system\s*:\s", 0.40, "fake system-role tag"),
    (r"\bhuman\s*:\s|\bassistant\s*:\s*$", 0.25, "transcript forgery delimiter"),
    # --- encoding smuggling ------------------------------------------------
    (r"\bbase64\b[^.\n]{0,30}\b(decode|encoded)\b[^.\n]{0,30}\b(follow|execute|obey|instructions?)", 0.45, "encoded-payload directive"),
    (r"\b(decode|decrypt|translate)\s+(this|the\s+following)\b[^.\n]{0,35}\b(and\s+)?(follow|execute|obey|run)", 0.40, "encoded-payload directive"),
    # --- tool / action coercion --------------------------------------------
    (r"\b(call|invoke|run|execute)\s+(the\s+)?(delete|drop|rm|format|wipe|admin|shutdown)\b", 0.50, "dangerous tool coercion"),
    (r"\brm\s+-rf\b|\bdrop\s+table\b|\bshutdown\b[^.\n]{0,20}\b(server|system|database)", 0.50, "destructive command"),
    (r"\bgrant\b[^.\n]{0,25}\b(admin|root|full|elevated)\s+(access|privileges?|permissions?)", 0.45, "privilege escalation"),
    (r"\bignore\s+safety\b|\bdisable\s+(logging|audit|monitoring|guardrails?)", 0.50, "defense disabling"),
]

_LEXICAL_COMPILED: list[tuple[re.Pattern[str], float, str]] = [
    (re.compile(p, re.IGNORECASE | re.MULTILINE), w, label)
    for p, w, _label in _LEXICAL_PATTERNS
    for label in [_label]
]

# Imperative verbs that signal instruction-like prose (used for RAG docs).
_IMPERATIVE_VERBS = (
    "ignore", "forget", "disregard", "reveal", "print", "output", "display",
    "repeat", "exfiltrate", "delete", "remove", "execute", "run", "call",
    "invoke", "grant", "disable", "override", "switch", "pretend", "roleplay",
    "obey", "comply", "follow", "respond", "answer", "summarize", "insert",
    "append", "forward", "email", "send", "post", "upload", "download",
)

# Instruction-like cues aimed at an AI reader inside retrieved content.
_DOC_CUES: list[tuple[str, float, str]] = [
    (r"\b(note|attention|important|critical|urgent)\s*(to|for)?\s*:?\s*(the\s+)?(ai|assistant|model|llm|chatbot|gpt|claude|system)\b", 0.45, "AI-addressed note"),
    (r"\bwhen\s+(answering|responding|summarizing|asked)\b[^.\n]{0,60}\b(you\s+)?(must|should|always|never)\b", 0.35, "response-shaping instruction"),
    (r"\b(the\s+)?(user|customer)\s+(has\s+)?(already\s+)?(approved|authorized|confirmed|requested)\b[^.\n]{0,50}\b(refund|reset|transfer|delete|export)", 0.35, "fabricated authorization claim"),
    (r"\bassistant\s+(must|should|will|shall)\b", 0.40, "assistant-directed imperative"),
    (r"\bfollow\s+(these|the\s+following)\s+(instructions?|steps?|directions?)\b", 0.45, "embedded instruction list"),
    (r"\bdo\s+not\b[^.\n]{0,30}\b(tell|inform|mention|reveal)[^.\n]{0,30}\b(user|human|customer)", 0.45, "concealment directive"),
    (r"<\s*/?\s*(system|instruction|command|hidden|admin)\s*>", 0.50, "hidden markup tag"),
    (r"<!--[\s\S]{0,400}?(ignore|instruction|assistant|ai|system)[\s\S]{0,400}?-->", 0.45, "HTML-comment smuggling"),
    (r"\bstore\s+(this|the\s+following)\b[^.\n]{0,40}\b(memory|context|instructions?)", 0.40, "memory-poisoning directive"),
]
_DOC_CUES_COMPILED: list[tuple[re.Pattern[str], float, str]] = [
    (re.compile(p, re.IGNORECASE | re.MULTILINE), w, label) for p, w, label in _DOC_CUES
]

_BASE64ISH_RE = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{120,}={0,2}(?![A-Za-z0-9+/=])")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_CHAT_TEMPLATE_DELIMS = (
    "<|im_start|>", "<|im_end|>", "</s>", "<s>", "[INST]", "[/INST]", "<<SYS>>",
)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _stem(token: str) -> str:
    """Ultra-light stemmer: kills the worst inflection mismatches
    ('returns' vs 'return') without external dependencies."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _token_set(text: str) -> set[str]:
    return {_stem(t) for t in _TOKEN_RE.findall(text.lower())}


class ContentRiskAnalyzer:
    """Layer 1 content analyzer (spec: ``demo_mode`` switches the backend).

    Parameters
    ----------
    demo_mode:
        ``True``  — lexical + structural heuristics only (default; no model
        downloads, fully deterministic).
        ``False`` — load the sentence-transformer model and the trained
        classifier weights from ``weights_path``.
    weights_path:
        JSON file produced by ``scripts/train_classifier.py``.  Required
        when ``demo_mode=False``.
    model_name:
        Sentence-embedding model (spec: ``BAAI/bge-small-en-v1.5``).
    """

    def __init__(
        self,
        demo_mode: bool = True,
        weights_path: str | Path | None = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.demo_mode = demo_mode
        self.model_name = model_name
        self._model: Any = None
        self._clf_weights: list[float] | None = None
        self._clf_bias: float = 0.0
        self._clf_meta: dict[str, Any] = {}

        if not demo_mode:
            self._load_embedding_backend(weights_path)

    # ------------------------------------------------------------------ ml
    def _load_embedding_backend(self, weights_path: str | Path | None) -> None:
        # Check the weights file first so the actionable error wins even on
        # machines without the ML extra installed.
        path = Path(weights_path) if weights_path else None
        if path is None or not path.exists():
            raise FileNotFoundError(
                "trained classifier weights not found: "
                f"{weights_path!r}. Train them with scripts/train_classifier.py "
                "or use demo_mode=True."
            )

        blob = json.loads(path.read_text(encoding="utf-8"))
        self._clf_weights = [float(w) for w in blob["weights"]]
        self._clf_bias = float(blob["bias"])
        self._clf_meta = {k: v for k, v in blob.items() if k not in {"weights"}}
        self.model_name = blob.get("model", self.model_name)
        self._model = get_sentence_transformer(self.model_name)

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        """L2-normalized embeddings; requires the embedding backend."""
        if self._model is None:  # pragma: no cover - guarded by callers
            raise EmbeddingBackendUnavailableError("embedding backend not loaded")
        import numpy as np  # noqa: PLC0415

        emb = self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(emb, dtype=float).tolist()

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        num = sum(x * y for x, y in zip(a, b))
        da = math.sqrt(sum(x * x for x in a)) or 1.0
        db = math.sqrt(sum(x * x for x in b)) or 1.0
        return num / (da * db)

    # ------------------------------------------------------------- lexical
    @staticmethod
    def lexical_scan(text: str) -> tuple[float, list[str]]:
        """Weighted keyword/regex scan over conservative variants.

        ``lexical_score = max(score(variant))`` across ``raw / normalized /
        folded / b64_i`` views (spec Phase 2).  Evidence from a non-raw
        variant is tagged, e.g. ``instruction override [folded]: ...``.
        """
        best = -1.0
        best_ev: list[str] = []
        for tag, variant in _norm.variants(text).items():
            score = 0.0
            evidence: list[str] = []
            seen: set[str] = set()
            for pattern, weight, label in _LEXICAL_COMPILED:
                for match in pattern.finditer(variant):
                    score += weight
                    body = f"{match.group(0)[:60]!r}"
                    key = f"{label}: {body}"
                    if key not in seen:
                        seen.add(key)
                        evidence.append(key if tag == "raw"
                                       else f"{label} [{tag}]: {body}")
            if score > best:  # strict: ties keep the earliest (raw-most) view
                best, best_ev = score, evidence
        return _clamp01(max(best, 0.0)), best_ev

    # ---------------------------------------------- structural heuristics
    def _structural_features(self, text: str) -> tuple[float, list[str]]:
        """Deterministic non-ML signals: delimiters, imperative density,
        second-person directives, fake role headers, encoded blobs."""
        score = 0.0
        evidence: list[str] = []

        delims = [d for d in _CHAT_TEMPLATE_DELIMS if d.lower() in text.lower()]
        if delims:
            score += 0.35
            evidence.append(f"chat-template delimiters present: {sorted(set(delims))}")

        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        if sentences:
            imperative_hits = 0
            for sent in sentences:
                first = (_TOKEN_RE.findall(sent.lower()) or [""])[0]
                if first in _IMPERATIVE_VERBS:
                    imperative_hits += 1
            density = imperative_hits / len(sentences)
            if density >= 0.34 and imperative_hits >= 2:
                score += 0.30
                evidence.append(
                    f"high imperative density ({imperative_hits}/{len(sentences)} sentences)"
                )

        if re.search(r"\byou\s+(will|must|shall|are\s+required\s+to)\b", text, re.I):
            score += 0.20
            evidence.append("second-person directive ('you will/must')")

        if re.search(r"^\s*(system|admin|root)\s*:\s", text, re.I | re.M):
            score += 0.30
            evidence.append("fake role header at line start")

        if _BASE64ISH_RE.search(text):
            score += 0.15
            evidence.append("long base64-like blob (possible encoded payload)")

        return _clamp01(score), evidence

    # -------------------------------------------- document/RAG heuristics
    def analyze_document(self, text: str) -> tuple[float, list[str]]:
        """Instruction-likeness score for one retrieved document."""
        text = Canonicalizer.normalize_text(text)
        score = 0.0
        evidence: list[str] = []
        for pattern, weight, label in _DOC_CUES_COMPILED:
            if pattern.search(text):
                score += weight
                evidence.append(f"doc cue [{label}]: {pattern.search(text).group(0)[:70]!r}")  # type: ignore[union-attr]
        lex, lex_ev = self.lexical_scan(text)
        if lex > 0.0:
            score += 0.6 * lex
            evidence.extend(f"doc lexical: {e}" for e in lex_ev)
        struct, struct_ev = self._structural_features(text)
        if struct > 0.0:
            score += 0.5 * struct
            evidence.extend(f"doc structural: {e}" for e in struct_ev)
        return _clamp01(score), evidence

    # ------------------------------------------------------------ injection
    def injection_score_for(self, text: str) -> tuple[float, list[str]]:
        """Prompt-injection probability for a single text.

        demo_mode=False → logistic layer over bge embeddings (trained
        weights loaded from disk).  demo_mode=True → deterministic fusion
        of lexical + structural heuristics.
        """
        text = Canonicalizer.normalize_text(text)
        lex, lex_ev = self.lexical_scan(text)
        struct, struct_ev = self._structural_features(text)
        evidence = lex_ev + struct_ev

        if not self.demo_mode and self._model is not None and self._clf_weights:
            # ML analysis evaluates the raw and the most-processed variant
            # (folded when present, else normalized); combined via max —
            # a single view scoring an attack raises the attack.
            views = [text]
            vs = _norm.variants(text)
            for tag in ("folded", "normalized"):
                if tag in vs and vs[tag] not in views:
                    views.append(vs[tag])
                    break
            probs = []
            for vec in self._embed(views):
                z = sum(w * x for w, x in zip(self._clf_weights, vec)) + self._clf_bias
                probs.append(1.0 / (1.0 + math.exp(-z)))
            ml = max(probs)
            # The heuristic surface still contributes — fusion, not override.
            score = _clamp01(0.75 * ml + 0.25 * max(lex, struct))
            evidence.append(
                f"embedding classifier p(injection)={ml:.3f} "
                f"over {len(views)} view(s)"
            )
            return score, evidence

        # demo_mode: monotone, deterministic fusion.  lex saturates fast;
        # structural pushes borderline cases over decision thresholds.
        fused = 1.0 - (1.0 - lex) * (1.0 - 0.8 * struct)  # noisy-OR
        return _clamp01(fused), evidence

    # ----------------------------------------------- intent/context match
    def mismatch_score(
        self, user_request: str, contexts: Iterable[str]
    ) -> tuple[float | None, list[str]]:
        """Intent/context mismatch; ``(None, [])`` when no context exists —
        an absent channel must not dilute fusion (spec defect P2)."""
        contexts = [c for c in contexts if c and c.strip()]
        if not contexts:
            return None, []
        evidence: list[str] = []

        if not self.demo_mode and self._model is not None:
            vecs = self._embed([user_request, *contexts])
            req, doc_vecs = vecs[0], vecs[1:]
            sims = [self._cosine(req, dv) for dv in doc_vecs]
            min_sim = min(sims)
            mean_sim = sum(sims) / len(sims)
            score = _clamp01(1.0 - (0.55 * mean_sim + 0.45 * min_sim))
            evidence.append(
                f"embedding similarity: min={min_sim:.3f} mean={mean_sim:.3f} "
                f"over {len(sims)} context doc(s)"
            )
            return score, evidence

        # demo_mode: token-overlap disagreement + instruction-like docs.
        req_tokens = _token_set(user_request)
        if not req_tokens:
            return 0.0, []
        overlaps = []
        for doc in contexts:
            doc_tokens = _token_set(doc)
            if not doc_tokens:
                overlaps.append(0.0)
                continue
            jaccard = len(req_tokens & doc_tokens) / max(1, len(req_tokens | doc_tokens))
            containment = len(req_tokens & doc_tokens) / max(1, len(req_tokens))
            overlaps.append(max(jaccard, 0.5 * containment))
        best = max(overlaps)
        instr_risks = [self.analyze_document(d)[0] for d in contexts]
        max_instr = max(instr_risks, default=0.0)
        # Mismatch = topical distance, amplified by instruction-likeness.
        # A benign on-topic doc scores low; an off-topic *instruction-laden*
        # doc (the classic indirect-injection signature) scores near 1.
        distance = 1.0 - best
        score = _clamp01(distance * (0.25 + 0.75 * max_instr))
        evidence.append(
            f"token overlap best={best:.3f}; max doc instruction-risk={max_instr:.3f}"
        )
        return score, evidence

    # ------------------------------------------------------ conversation
    def conversation_drift_score(
        self, history: Sequence[str], current: str
    ) -> tuple[float | None, list[str]]:
        """Topic drift of ``current`` relative to earlier user messages.

        ``(None, [])`` — channel **inactive** — when history is insufficient
        (fewer than 3 prior turns, spec defect P7); weak context must not
        fabricate risk.  Constants are predefined, not tuned on benchmarks.
        """
        history = [h for h in history if h and h.strip()]
        if len(history) < 3 or not current.strip():
            return None, []

        if not self.demo_mode and self._model is not None:
            vecs = self._embed([current, *history])
            cur, past = vecs[0], vecs[1:]
            mean_past = [sum(col) / len(col) for col in zip(*past)]
            sim = self._cosine(cur, mean_past)
        else:
            cur_tokens = _token_set(current)
            if not cur_tokens:
                return None, []
            sims = []
            for h in history:
                h_tokens = _token_set(h)
                sims.append(len(cur_tokens & h_tokens) / max(1, len(cur_tokens | h_tokens)))
            sim = sum(sims) / len(sims)

        gated = max(0.0, (0.55 - sim) / 0.55) * min(1.0, len(history) / 6)
        return (round(_clamp01(gated), 6),
                [f"drift similarity-to-history={sim:.3f} over {len(history)} turn(s)"])

    # --------------------------------------------------------------- fuse
    def analyze_user_message(self, text: str) -> tuple[float, float, list[str]]:
        """(lexical_score, injection_score, evidence) for a user message."""
        text = Canonicalizer.normalize_text(text)
        if not isinstance(text, str) or not text.strip():
            raise SchemaValidationError("user message must be non-empty text")
        lex, _lex_ev = self.lexical_scan(text)
        inj, inj_ev = self.injection_score_for(text)
        # inj_ev already contains the lexical + structural evidence; the old
        # lex_ev + inj_ev concatenation duplicated every entry (spec P6).
        return lex, inj, dedup_evidence(inj_ev)

    def fuse(
        self,
        lexical_score: float,
        injection_score: float,
        retrieval_injection_score: float | None,
        intent_context_mismatch_score: float | None,
        evidence: Sequence[str],
    ) -> ContentRiskResult:
        """L1 fusion over *active* channels only (spec Phase 6; ``None`` =
        channel not applicable).  Drift folds in later at L4."""
        content_risk = combine_signals({
            "injection": injection_score,
            "lexical": lexical_score,
            "retrieval": retrieval_injection_score,
            "mismatch": intent_context_mismatch_score,
        })

        def _r(x: float | None) -> float | None:
            return None if x is None else round(_clamp01(x), 6)

        return ContentRiskResult(
            lexical_score=round(_clamp01(lexical_score), 6),
            injection_score=round(_clamp01(injection_score), 6),
            retrieval_injection_score=_r(retrieval_injection_score),
            intent_context_mismatch_score=_r(intent_context_mismatch_score),
            content_risk=content_risk,
            evidence=dedup_evidence(evidence),
        )

    # -------------------------------------------------------- top-level api
    def analyze(
        self,
        user_text: str,
        retrieved_docs: Sequence[str] | None = None,
        history: Sequence[str] | None = None,
    ) -> ContentRiskResult:
        """Full L1 pass for one request."""
        lex, inj, evidence = self.analyze_user_message(user_text)
        docs = list(retrieved_docs or [])

        doc_risks = []
        for i, doc in enumerate(docs):
            risk, doc_ev = self.analyze_document(doc)
            doc_risks.append(risk)
            evidence.extend(f"doc[{i}] {e}" for e in doc_ev)
        # no documents -> channel inactive (None), not a zero score
        retrieval_injection = max(doc_risks) if doc_risks else None

        mismatch, mm_ev = self.mismatch_score(user_text, docs)
        evidence.extend(mm_ev)

        return self.fuse(lex, inj, retrieval_injection, mismatch, evidence)
