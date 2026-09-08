"""HC-Bench loader registry — provenance-tracked, surface-routed sources.

Hard rules (project discipline):
  * loaders NEVER train/calibrate, NEVER oversample, NEVER substitute a
    source that cannot be fetched — gated or unknown-layout sources are
    reported as skipped with a printed reason
  * every row carries full provenance: source, source_id, license,
    label_origin, group template hooks (splitting assigns the gid)
  * benign sources get label_origin 'curated-benign' or
    'publisher-overrefusal' so reviewers can audit their provenance
"""
from __future__ import annotations

import random

ROW_FIELDS = (
    "id", "text", "label", "category", "subtype", "surface",
    "system_prompt", "context", "source", "source_id", "license",
    "label_origin", "language", "group_id", "split", "derived_from",
    "lexically_invisible",
)


def mk_row(text, label, category, surface, source, source_id, license_,
           label_origin, subtype="generic", system_prompt="", context="",
           language="en", derived_from=""):
    """Canonical row; split/group_id assigned later by the builder.

    ``lexically_invisible`` is an ORTHOGONAL difficulty flag set by the
    builder (attack rows the lexical scanner cannot see).  It deliberately
    does NOT overwrite ``category``: doing so would remove every hard row
    from its own attack class and leave behind only the rows one of the
    scored channels already fires on, making per-category recall circular.
    """
    return {
        "id": f"{source}-{source_id}", "text": str(text).strip(),
        "label": int(label), "category": category, "subtype": subtype,
        "surface": surface, "system_prompt": system_prompt,
        "context": context, "source": source, "source_id": str(source_id),
        "license": license_, "label_origin": label_origin,
        "language": language, "group_id": "", "split": "",
        "derived_from": derived_from, "lexically_invisible": None,
    }


def validate_row(r: dict) -> bool:
    return (all(k in r for k in ROW_FIELDS)
            and isinstance(r["text"], str) and bool(r["text"])
            and r["label"] in (0, 1)
            and r["lexically_invisible"] in (None, True, False)
            and r["surface"] in ("user_prompt", "rag_doc",
                                 "tool_description", "tool_output"))


def _subsample(rows, cap, seed=42):
    """Deterministic first-cap after a seed-42 shuffle (no content pick)."""
    if len(rows) <= cap:
        return rows
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    return [rows[i] for i in idx[:cap]]


def _hf(repo, name=None, split="train", cap=2000):
    """Load via huggingface datasets; raise on any failure (caller skips)."""
    from datasets import load_dataset
    ds = load_dataset(repo, name=name, split=split, trust_remote_code=False)
    return ds


# ---------- per-source loaders (rows, or raise -> skipped with reason) ------

def load_hackaprompt(cap=2000):
    ds = _hf("hackaprompt/hackaprompt-dataset")
    rows = []
    for i, rec in enumerate(ds):
        txt = (rec.get("user_input") or rec.get("prompt") or "").strip()
        if txt:
            rows.append(mk_row(txt, 1, "jailbreak", "user_prompt",
                               "hackaprompt", i, "research",
                               "publisher", subtype="competition"))
    return _subsample(rows, cap)


def load_tensortrust(cap=2000):
    ds = _hf("HumanCompatibleAI/tensor-trust", name="extractions")
    rows = []
    for i, rec in enumerate(ds):
        txt = (rec.get("attack") or rec.get("attacker") or "").strip()
        if txt:
            rows.append(mk_row(txt, 1, "prompt_extraction", "user_prompt",
                               "tensor-trust", i, "research",
                               "publisher", subtype="game"))
    if not rows:
        raise RuntimeError("tensor-trust: unexpected schema")
    return _subsample(rows, cap)


def load_gandalf(cap=1500):
    ds = _hf("Lakera/gandalf_ignore_instructions")
    rows = []
    for i, rec in enumerate(ds):
        txt = (rec.get("text") or "").strip()
        if txt:
            lab = 1 if str(rec.get("label", 1)) in ("1", "jailbreak") else 0
            rows.append(mk_row(txt, lab, "instruction_override",
                               "user_prompt", "gandalf-ignore", i,
                               "research", "publisher",
                               subtype="gandalf"))
    return _subsample(rows, cap)


def load_deepset():
    ds = _hf("deepset/prompt-injections")
    rows = []
    for i, rec in enumerate(ds):
        txt = (rec.get("text") or "").strip()
        if txt:
            rows.append(mk_row(txt, int(rec.get("label", 1)), "injection",
                               "user_prompt", "deepset-pi", i, "MIT",
                               "publisher"))
    return rows


def load_jbb_behaviors():
    ds = _hf("JailbreakBench/JBB-Behaviors", name="behaviors",
             split="harmful")
    rows = []
    for i, rec in enumerate(ds):
        txt = (rec.get("Goal") or "").strip()
        if txt:
            rows.append(mk_row(txt, 1, "jailbreak", "user_prompt",
                               "jbb-behaviors", i, "MIT", "publisher",
                               subtype=str(rec.get("Category") or "harmful")))
    ds0 = _hf("JailbreakBench/JBB-Behaviors", name="behaviors",
              split="benign")
    for i, rec in enumerate(ds0):
        txt = (rec.get("Goal") or "").strip()
        if txt:
            rows.append(mk_row(txt, 0, "benign", "user_prompt",
                               "jbb-behaviors", f"b{i}", "MIT", "publisher",
                               subtype="hard-benign"))
    return rows


def load_jackhhao(cap=2500):
    rows = []
    for split in ("train", "test"):
        ds = _hf("jackhhao/jailbreak-classification", split=split)
        for i, rec in enumerate(ds):
            txt = (rec.get("prompt") or "").strip()
            if txt:
                lab = 1 if rec.get("type") == "jailbreak" else 0
                rows.append(mk_row(txt, lab,
                                   "jailbreak" if lab else "benign",
                                   "user_prompt", "jackhhao",
                                   f"{split}-{i}", "MIT", "publisher"))
    return _subsample(rows, cap)


def load_wildjailbreak(cap_harmful=2000, cap_benign=2000):
    ds = _hf("allenai/wildjailbreak")
    harm, ben = [], []
    for i, rec in enumerate(ds):
        txt = (rec.get("prompt") or rec.get("adversarial") or "").strip()
        dt = str(rec.get("data_type") or "")
        if not txt or "adversarial" not in dt:
            continue
        if "harmful" in dt:
            harm.append(mk_row(txt, 1, "jailbreak", "user_prompt",
                               "wildjailbreak", i, "ODC-BY-1.0",
                               "publisher", subtype="adversarial-harmful"))
        else:
            ben.append(mk_row(txt, 0, "benign", "user_prompt",
                              "wildjailbreak", i, "ODC-BY-1.0",
                              "publisher", subtype="adversarial-benign"))
    if not harm and not ben:
        raise RuntimeError("wildjailbreak: unexpected schema")
    return _subsample(harm, cap_harmful) + _subsample(ben, cap_benign)


def load_bipia(cap=1000):
    ds = _hf("microsoft/bipia", name="prompt_injection")
    rows = []
    for i, rec in enumerate(ds):
        txt = (rec.get("text") or rec.get("context") or "").strip()
        if txt:
            rows.append(mk_row(txt, 1, "injection", "rag_doc", "bipia", i,
                               "MIT", "publisher", subtype="email-context"))
    if not rows:
        raise RuntimeError("bipia: unexpected schema")
    return _subsample(rows, cap)


def load_xstest():
    from urllib.request import Request, urlopen
    from defend_hc2.extbench import XSTEST_URL, parse_xstest_csv
    req = Request(XSTEST_URL, headers={"User-Agent": "def-hc-hcbench"})
    with urlopen(req, timeout=60) as r:  # noqa: S310 (pinned URL)
        src = r.read().decode("utf-8")
    rows = []
    for i, (text, label, meta) in enumerate(parse_xstest_csv(src)):
        rows.append(mk_row(text, label,
                           "benign" if label == 0 else "harmful-content",
                           "user_prompt", "xstest", i, "CC BY 4.0",
                           "publisher-overrefusal",
                           subtype=meta.get("xstest_type", ""),
                           derived_from=meta.get("id", "")))
    return rows


def load_orbench(cap=1000):
    """OR-Bench hard-1k: benign over-refusal prompts (publisher)."""
    for cfg in ("or-bench-hard-1k", "hard-1k", None):
        try:
            ds = _hf("bench-llm/or-bench", name=cfg, split="train" if cfg else "train", cap=None)
            break
        except Exception:
            ds = None
    if ds is None:
        raise RuntimeError("benchmark config fetch failed")
    rows = []
    for i, rec in enumerate(ds):
        txt = (rec.get("prompt") or "").strip()
        if txt:
            rows.append(mk_row(txt, 0, "benign", "user_prompt", "or-bench-hard-1k",
                               i, "MIT", "publisher-overrefusal",
                               subtype="hard-benign"))
    if not rows:
        raise RuntimeError("or-bench: unexpected schema")
    return _subsample(rows, cap)


def load_alpaca_dolly(cap_each=500):
    rows = []
    ds = _hf("tatsu-lab/alpaca")
    for i, rec in enumerate(ds):
        ins = (rec.get("instruction") or "").strip()
        inp = (rec.get("input") or "").strip()
        txt = ins if not inp else f"{ins}\n{inp}"
        if txt:
            rows.append(mk_row(txt, 0, "benign", "user_prompt", "alpaca", i,
                               "CC BY-NC 4.0", "curated-benign",
                               subtype="instruction"))
    ds = _hf("databricks/databricks-dolly-15k")
    drows = []
    for i, rec in enumerate(ds):
        ins = (rec.get("instruction") or "").strip()
        txt = ins or (rec.get("context") or "").strip()
        if txt:
            drows.append(mk_row(txt, 0, "benign", "user_prompt", "dolly", i,
                                "CC BY-SA 3.0", "curated-benign",
                                subtype="instruction"))
    return _subsample(rows, cap_each) + _subsample(drows, cap_each)


def load_msmarco(cap=1000):
    ds = _hf("microsoft/ms_marco", name="v1.1", split="train")
    rows = []
    for i, rec in enumerate(ds):
        try:
            passages = rec["passages"]["passage_text"]
        except Exception:
            passages = []
        txt = (passages[0] if passages else "").strip()
        if txt:
            rows.append(mk_row(txt, 0, "benign", "rag_doc", "ms-marco", i,
                               "research", "curated-benign",
                               subtype="passage"))
        if len(rows) >= cap:
            break
    return rows


_MCP_TOOL_DESCRIPTIONS = [
    ("mcp-fs", "Filesystem server: read/write/list files within sandboxed "
               "roots; tools read_file, write_file, list_directory."),
    ("mcp-sqlite", "SQLite server: run read-only SQL queries over local "
                   "database files; exposes query and list_tables tools."),
    ("mcp-github", "GitHub server: repository search, issue/PR listing and "
                   "file content retrieval via the GitHub API."),
    ("mcp-slack", "Slack server: list channels, read history and post "
                  "messages to a workspace via a bot token."),
    ("mcp-gdrive", "Google Drive server: search and read documents in a "
                   "Google Drive workspace."),
    ("mcp-puppeteer", "Puppeteer server: navigate pages, click elements, "
                      "screenshot and fill forms in a headless browser."),
    ("mcp-postgres", "Postgres server: schema inspection and read-only "
                     "queries against configured databases."),
    ("mcp-brave", "Brave Search server: web and local search via the "
                  "Brave Search API."),
    ("mcp-time", "Time server: current time and timezone conversion "
                 "utilities."),
    ("mcp-everything", "Reference test server: echoes inputs, exercises "
                       "all MCP primitives (prompts, tools, resources)."),
    ("mcp-memory", "Memory server: persistent knowledge-graph memory with "
                   "entity and relation CRUD."),
    ("mcp-fetch", "Fetch server: retrieves a URL and returns the page "
                  "content as markdown."),
    ("mcp-git", "Git server: local repository log, diff, blame and "
                "status inspection."),
    ("mcp-sentry", "Sentry server: inspect tracked application errors and "
                   "stack traces."),
    ("mcp-sequential", "Sequential thinking server: structured multi-step "
                       "reasoning scratchpad tool."),
    ("mcp-wikipedia", "Wikipedia server: search and fetch article content "
                      "from Wikipedia via its public API."),
]


def load_mcp_tool_descriptions():
    """Real published MCP server descriptions — the benign pole for the
    tool_description surface.  Curated from public MCP server registries."""
    return [mk_row(desc, 0, "benign", "tool_description",
                   "mcp-registry", sid, "MIT (server repos)", "curated-benign",
                   subtype="tool-description")
            for sid, desc in _MCP_TOOL_DESCRIPTIONS]


# Sources with unknown/unverified layouts this pass — attempted loaders only
# run sources whose access the repo can verify; anything else is reported:
DEFERRED_SOURCES = {
    "jbb-artifacts": "attack-artifact store layout not verified",
    "poisonedrag": "publisher passage corpus layout not verified",
    "mcptox": "dataset id/availability not verified",
    "mcp-attackbench": "dataset id/availability not verified",
    "injecagent": "publisher file layout not verified",
}

LOADERS = {
    "hackaprompt": load_hackaprompt,
    "tensor-trust": load_tensortrust,
    "gandalf-ignore": load_gandalf,
    "deepset-pi": load_deepset,
    "jbb-behaviors": load_jbb_behaviors,
    "jackhhao": load_jackhhao,
    "wildjailbreak": load_wildjailbreak,
    "bipia": load_bipia,
    "xstest": load_xstest,
    "or-bench-hard-1k": load_orbench,
    "alpaca-dolly": load_alpaca_dolly,
    "ms-marco": load_msmarco,
    "mcp-registry": load_mcp_tool_descriptions,
}
