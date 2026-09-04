# DEFEND-HC2

**Dual-layer LLM security framework** — semantic prompt-injection detection
*combined with* cryptographic session-continuity enforcement, RAG/tool
provenance verification, risk-based policy fusion, and an append-only
tamper-evident audit ledger for stateless LLM applications.

```
Request
  ↓
L0: Canonicalization and Schema Validation
  ↓
L1: Content Risk Analysis
  ↓
L2: Cryptographic Session-Continuity Verification
  ↓
L3: RAG and Tool Provenance Verification
  ↓
L4: Policy Fusion Engine
  ↓
L5: Append-Only Tamper-Evident Ledger
  ↓
LLM Call or Rejection
```

DEFEND-HC2 defends against two **orthogonal** attack classes:

| Class | Examples | Detected by |
|---|---|---|
| **A. Content-level** | direct prompt injection, jailbreaks, indirect injection in retrieved docs, malicious tool outputs | L1 + L3 (lexical/embedding analysis, provenance) |
| **B. State-level** | history modification, message deletion/reordering, fabricated assistant messages, fabricated tool results, cross-session splicing, replay of old chain heads | L2 (+ L5) — content classifiers **cannot** see these |

That complementarity is the point of the framework (see *Research framing*
below).

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # core (stdlib crypto + FastAPI + tests)

# 1. run the required demonstration (all 9 scenarios, ~1 second)
python -m defend_hc2

# 2. run the test suite (158 tests)
pytest

# 3. run the API
python -m defend_hc2.api                 # http://0.0.0.0:8200 (+/docs)
```

A fresh master secret for production use:

```bash
python -c "from defend_hc2.cli import keygen; keygen()"   # then:
export DEFEND_HC2_MASTER_SECRET=<hex>
```

### The one-minute demo

`python -m defend_hc2` demonstrates, end-to-end:

1. create a valid session (genesis event on the chain)
2. benign prompt → `ALLOW`
3. direct injection → risk `0.70` → `QUARANTINE`
4. retrieved doc with hidden HTML-comment injection → doc risk `1.0` → hard-fail `REJECT`
5. replay of an old chain head → `STALE_HEAD_REPLAY` (high severity)
6. fabricated assistant message → `CHAIN_HASH_MISMATCH`, and with correct hash algebra → `MAC_MISMATCH` (critical)
7. cross-session splice → `CROSS_SESSION_SPLICE` (critical)
8. full-chain verification → every hash & MAC recomputed clean
9. manual SQLite `UPDATE`/`DELETE` → aborted by append-only triggers
10. checkpoint → Merkle root over session heads, HMAC-signed

---

## Architecture

### L0 — `Canonicalizer` (`canonicalization.py`)

* Unicode **NFKC** normalization; strips invalid control characters **and**
  zero-width / bidi-override code points used to hide instructions.
* Deterministic JSON: `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`;
  all hashes over canonical UTF-8 bytes.
* Domain-separation tags (`DEFEND-HC2-GENESIS`, `DEFEND-HC2-EVENT`,
  `DEFEND-HC2-PAYLOAD`, `DEFEND-HC2-KEY-EVOLVE`, …) and **length-prefixed
  framing** for every `a || b` in the spec, so fields can never be
  re-interpreted (`H("ab"||"c") ≠ H("a"||"bc")`).

### L1 — `ContentRiskAnalyzer` (`content_risk.py`)

Four deterministic signals: a weighted lexical pattern bank, an injection
classifier, a RAG-document instruction detector, and an intent/context
mismatch detector.

| Mode | Behaviour |
|---|---|
| `demo_mode=True` (default) | lexical + structural heuristics only. No downloads, no randomness (scores are stable across runs). |
| `demo_mode=False` | loads `BAAI/bge-small-en-v1.5` via sentence-transformers **and trained logistic weights from disk** (`scripts/train_classifier.py`). The file is loaded, never guessed — no random scores anywhere. |

```bash
pip install -r requirements-ml.txt
python scripts/train_classifier.py --out defend_hc2/weights/bge-logistic.json
export DEFEND_HC2_WEIGHTS=defend_hc2/weights/bge-logistic.json
export DEFEND_HC2_DEMO_MODE=0
python -m defend_hc2.api
```

### L2 — `SessionContinuityTracker` (`session_chain.py`)

Keyed hash chain with **forward key evolution** (`SHA3-256` + `HMAC-SHA3-256`):

```
session_salt       = secrets.token_bytes(32)
system_prompt_hash = SHA3-256(canonical system prompt)
K_0   = HMAC-SHA3-256(master_secret, session_id || session_salt)
H_0   = SHA3-256("DEFEND-HC2-GENESIS" || session_id || system_prompt_hash
                 || session_salt || timestamp_ns)
MAC_0 = HMAC-SHA3-256(K_0, H_0)

payload_hash = SHA3-256(canonical_payload)
H_t   = SHA3-256("DEFEND-HC2-EVENT" || session_id || sequence || H_{t-1}
                 || event_type || payload_hash || system_prompt_hash || timestamp_ns)
MAC_t = HMAC-SHA3-256(K_t, canonical_payload || H_t)
K_{t+1} = SHA3-256("DEFEND-HC2-KEY-EVOLVE" || K_t || H_t)
```

Before every append it validates: session identity, sequence monotonicity,
`claimed_previous_hash`, nonce freshness, the bound system-prompt hash, and a
local head self-check. `verify_presented_event()` verifies *client-presented*
history **non-destructively** — this is how fabricated assistant turns,
stale-head replays and cross-session splices are caught in stateless
deployments.

### L3 — `ProvenanceVerifier` (`provenance.py`)

* Documents: content hash + source-URI hash, instruction-content analysis,
  verdict `trusted / suspicious / rejected`, retrieval event bound to chain.
* Tool outputs: input/output hashes; HMAC receipts bound to
  `session_id || tool || hashes || current chain head`. **Unsigned privileged
  outputs are rejected as fabricated**; receipts can't be replayed into another
  session or another chain position.

### L4 — `PolicyEngine` (`policy.py`)

Hard-fails on: schema invalid, sequence mismatch, previous-hash mismatch,
invalid MAC, nonce replay, invalid tool provenance, cross-session splice.
Otherwise:

```
content_risk = 0.40·injection + 0.20·lexical + 0.20·retrieval_injection
             + 0.10·intent_mismatch + 0.10·conversation_drift
```

| risk | decision |
|---|---|
| `≥ 0.85` | `REJECT` |
| `0.65 – 0.85` | `QUARANTINE` |
| `0.40 – 0.65` | `SANITIZE_AND_ALLOW` |
| `< 0.40` | `ALLOW` |

Every decision is recorded as a chain event.

### L5 — `SQLiteTamperEvidentLedger` (`ledger.py`)

* WAL mode; tables `sessions`, `chain_entries`, `used_nonces`,
  `ledger_checkpoints`, `security_events`.
* `BEFORE UPDATE`/`BEFORE DELETE` triggers make `chain_entries`,
  `ledger_checkpoints`, `security_events` **append-only inside the database** —
  even compromised in-process code can't rewrite history.
* `UNIQUE(session_id, sequence)`, `UNIQUE(session_id, chain_hash)`,
  `UNIQUE(session_id, nonce)`.
* `BEGIN IMMEDIATE` + in-transaction head re-check → concurrent writers
  cannot fork a chain (covered by a threading test).

### Orchestrator — `DEFEND_HC2` (`pipeline.py`)

Chains all layers; on boot it **rebuilds in-memory state from the ledger,
re-verifying every MAC**, so a tampered database cannot resurrect forged
state. Provides `verify_session` (independent recomputation with a throwaway
tracker → first invalid event), `export_session` (audit trail JSON), and
`create_checkpoint` (Merkle root over all session heads, HMAC-signed —
anchor externally in production).

## HTTP API

| Method & path | Purpose |
|---|---|
| `POST /session` | create session → genesis (canonicalize prompt, salt, K₀, H₀) |
| `POST /process` | full pipeline for one user turn (text, optional `retrieved_docs`, `history`, `nonce`, `claimed_previous_hash`, `claimed_sequence`, `client_system_prompt_hash`) |
| `POST /tool-result` | tool-output provenance + policy |
| `POST /assistant-message` | record the server-observed assistant turn on the chain |
| `POST /verify-presented` | verify stateless client-presented history events |
| `GET /verify/{session_id}` | independent chain recomputation → first invalid event |
| `GET /export/{session_id}` | full audit trail as JSON |
| `POST /checkpoint` | signed Merkle root over session heads |
| `GET /head/{session_id}`, `GET /health` | introspection |

Config: `DEFEND_HC2_DB`, `DEFEND_HC2_MASTER_SECRET`, `DEFEND_HC2_DEMO_MODE`,
`DEFEND_HC2_WEIGHTS`, `DEFEND_HC2_TOOLS` (`name:hexkey:priv,...`).

## Repository layout

```
defend_hc2/
  canonicalization.py   # L0
  content_risk.py       # L1
  session_chain.py      # L2
  provenance.py         # L3
  policy.py             # L4
  ledger.py             # L5
  pipeline.py           # DEFEND_HC2 orchestrator
  api.py                # FastAPI service
  demo.py / __main__.py # required demonstration (`python -m defend_hc2`)
  constants.py exceptions.py results.py cli.py
scripts/train_classifier.py          # trains the L1 embedding classifier weights
scripts/evaluate_complementarity.py  # layer-coverage attack matrix evaluation
scripts/ci/test.yml                  # CI workflow template — copy to .github/workflows/
notebooks/defend-hc2-kaggle-ml-mode.ipynb  # ready-to-upload Kaggle notebook (ML mode)
docs/QUICKSTART.md docs/KAGGLE.md    # step-by-step guides
tests/                               # 164 tests across all layers + attacks + API
```

## Research framing

Proposed contribution statement:

> This paper proposes DEFEND-HC2, a dual-layer defense framework for LLM
> applications that combines semantic prompt-injection detection with
> cryptographic session-continuity enforcement. Unlike prior approaches that
> focus primarily on content classification or post-hoc audit logging,
> DEFEND-HC2 treats conversation state, retrieved context, and tool outputs
> as integrity-sensitive security objects. Each event is canonicalized,
> hash-bound to a session-specific chain, authenticated with keyed message
> authentication, and recorded in an append-only ledger. The framework detects
> both content-level attacks, such as jailbreaks and indirect prompt
> injection, and state-level attacks, such as replay, transcript rewriting,
> assistant-message fabrication, and cross-session splicing.

**On novelty (be honest):** hash chains, append-only logs, embedding
classifiers, prompt-injection detection, and SQLite audit logging are each
established. The contribution is the *integration*: online prompt-injection
detection fused with **request-time** cryptographic session-continuity
enforcement for stateless LLM applications — plus an evaluation showing the
two layers detect **complementary** attack classes (content classifiers are
blind to replay/fabrication; the chain is blind to payload semantics).

**Layer-complementarity evaluation** — `python
scripts/evaluate_complementarity.py` runs 9 attacks and prints the coverage
matrix. On this build:

```
attack scenario                                  A/B  content  chain  policy
-----------------------------------------------------------------------------
Direct prompt injection (content)                A    FLAG     -      SANITIZE_AND_ALLOW
Jailbreak w/ history drift (content)             A    FLAG     -      QUARANTINE
Indirect injection in RAG doc (content)          A    FLAG     -      REJECT
Malicious unregistered tool output (content)     A    FLAG     -      REJECT
Replay of old chain head (state)                 B    -        FLAG   REJECT
Fabricated assistant message (state)             B    -        FLAG   (verifier)
Cross-session transcript splice (state)          B    -        FLAG   (verifier)
Nonce replay (state)                             B    -        FLAG   REJECT
History tamper: forged prior turn (state)        B    -        FLAG   (verifier)
-----------------------------------------------------------------------------
content attacks:  L1/L3 4/4, chain 0/4 (chain is blind — as expected)
state attacks:    L1/L3 0/5, chain 5/5 (content is blind — as expected)
```

**Suggested further evaluation:** false-positive rate of the embedding
classifier vs. the lexical baseline on a public injection corpus (train via
`scripts/train_classifier.py --dataset`), and overhead measurements (append
cost is ~two SHA3-256 + one HMAC-SHA3-256 per event; ledger commits <1 ms
locally).

## Threat model & production notes

* **In scope:** stateless LLM apps whose clients present conversation
  transcripts; RAG and tool-using agents; auditability requirements.
* **Key custody:** the node master secret gates *all* session keys; tool keys
  are provisioned out-of-band. Use a KMS and Ed25519 envelopes in production;
  HMAC here keeps the prototype dependency-free.
* **Trust boundary:** the tracker trusts nothing on the wire — claimed heads,
  nonces, system-prompt bindings, presented histories and tool receipts are
  all re-verified against server-held secret state on every request.
* Forward key evolution bounds damage from a leaked verification key; the
  Merkle checkpoints give an external anchoring point for the ledger.
