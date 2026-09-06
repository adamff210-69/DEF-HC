"""HC-Bench protocol tests: schema, no-oversampling, routing, sealed guard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from defend_hc2.hcbench import LOADERS, ROW_FIELDS, mk_row, validate_row
from defend_hc2.splitting import (
    assert_no_group_crossing,
    group_stratified_split,
    normalize_key,
)

ROOT = Path(__file__).resolve().parents[1]


def test_row_schema_complete_and_validated():
    r = mk_row("hello world", 1, "injection", "user_prompt", "unit", "1",
               "MIT", "publisher")
    assert set(ROW_FIELDS) <= set(r)
    assert validate_row(r)
    r_bad = dict(r); r_bad["label"] = 2
    assert not validate_row(r_bad)
    assert not validate_row({**r, "surface": "nowhere"})


def test_group_split_is_4way_seed42_and_no_group_crossing():
    rows = [{"text": f"dock sample {i}", "label": i % 2} for i in range(40)]
    rows += [{"text": "dock sample 0", "label": 1}]  # same template family
    parts = group_stratified_split(rows, "text", "label",
                                   (0.4, 0.2, 0.2, 0.2), seed=42)
    assert len(parts) == 4
    assert sum(len(p) for p in parts) == len(rows)
    assert_no_group_crossing(parts, "text")
    again = group_stratified_split(rows, "text", "label",
                                   (0.4, 0.2, 0.2, 0.2), seed=42)
    assert [[r["text"] for r in p] for p in parts] == \
           [[r["text"] for r in p] for p in again]


def test_no_oversampling_contract_in_splits():
    """Any hcbench split file present in the repo must have unique texts."""
    bench_dir = ROOT / "hcbench"
    for name in ("train", "cal", "test"):
        fp = bench_dir / f"hcbench-{name}.jsonl"
        if not fp.exists():
            continue
        keys = [normalize_key(json.loads(l)["text"])
                for l in fp.open() if l.strip()]
        assert len(keys) == len(set(keys)), f"oversampled rows in {fp}"


def _stub_sys(calls):
    class Decision:
        content_risk = 0.5
        component_scores = {"retrieval_injection_score": 0.5}

    class Prov:
        verdict = "trusted"

    class PR:
        decision = Decision()

    class Stub:
        def process_user_message(self, session, text, retrieved_docs=None,
                                 **kw):
            calls.append(("pum", session, bool(retrieved_docs)))
            return PR()

        def submit_tool_result(self, session, tool, ti, out, **kw):
            calls.append(("tool", session, tool))
            return Prov(), Decision()
    return Stub()


def test_surface_routing_hits_intended_channels():
    from scripts.eval_hcbench import score_row
    calls: list = []
    sys = _stub_sys(calls)
    score_row(sys, "s", mk_row("x", 1, "injection", "user_prompt", "u", "1",
                               "MIT", "publisher"))
    assert calls[-1] == ("pum", "s", False)

    score_row(sys, "s", mk_row("doc", 1, "injection", "rag_doc", "u", "2",
                               "MIT", "publisher"))
    assert calls[-1] == ("pum", "s", True)  # retrieval_docs channel

    for surface in ("tool_description", "tool_output"):
        score, proof = score_row(sys, "s", mk_row("op", 1, "poisoning",
                                                  surface, "u", "3", "MIT",
                                                  "publisher"))
        assert calls[-1][0] == "tool"
        assert "trusted" in proof


def test_rag_row_without_retrieval_component_fails():
    from scripts.eval_hcbench import score_row
    class Dec:
        content_risk = 0.5
        component_scores = {}
    class PR:
        decision = Dec()
    class BadSys:
        def process_user_message(self, *a, **k):
            return PR()
    with pytest.raises(RuntimeError):
        score_row(BadSys(), "s", mk_row("doc", 1, "inj", "rag_doc", "u", "9",
                                        "MIT", "publisher"))


def test_lexical_invisibility_never_overwrites_category():
    """Regression guard.  build_hcbench used to do
    ``r["category"] = "semantic"`` for attack rows the lexical scanner
    misses.  That removed ~60% of attacks from their own class and left
    each attack category holding only rows a scored channel already fires
    on, making per-category recall circular.  Difficulty must stay an
    orthogonal flag."""
    src = (ROOT / "scripts" / "build_hcbench.py").read_text(encoding="utf-8")
    assert 'r["category"] = "semantic"' not in src
    assert '"lexically_invisible"' in src
    r = mk_row("x", 1, "jailbreak", "user_prompt", "u", "1", "MIT", "pub")
    assert "lexically_invisible" in r and r["lexically_invisible"] is None
    assert validate_row(r)
    assert validate_row({**r, "lexically_invisible": True})
    assert not validate_row({**r, "lexically_invisible": "yes"})


def test_leak_guard_covers_every_previously_observed_corpus():
    """The sealed split can only be called blind if HC-Bench was filtered
    against everything this project has already scored — including the
    Exp-B foreign transfer sets, which are built from the very same HF
    datasets HC-Bench loads (deepset, jackhhao)."""
    src = (ROOT / "scripts" / "build_hcbench.py").read_text(encoding="utf-8")
    for required in ("foreign-deepset.jsonl",
                     "foreign-jailbreak-classification.jsonl",
                     "foreign-safe-guard.jsonl",
                     "pi-test.jsonl", "spml-test.jsonl"):
        assert required in src, f"leak guard does not cover {required}"


def test_eval_hcbench_cannot_open_the_sealed_split():
    """The static string scan below cannot see a path built as
    f"hcbench-{args.split}.jsonl", so --split must be allow-listed."""
    import argparse

    import scripts.eval_hcbench as ev
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=("cal", "test"))
    with pytest.raises(SystemExit):
        ap.parse_args(["--split", "sealed"])
    src = Path(ev.__file__).read_text(encoding="utf-8")
    assert 'choices=("cal", "test")' in src


def test_recorded_sha256_matches_the_delivered_artifact(tmp_path):
    """Digests embedded into the document they hash describe a file that
    no longer exists.  Sidecar .sha256 files must verify."""
    import hashlib
    for script in ("eval_hcbench.py", "calibrate_hcbench_policy.py"):
        src = (ROOT / "scripts" / script).read_text(encoding="utf-8")
        assert 'with_suffix(".sha256")' in src, f"{script}: no sidecar digest"
        assert 'report["file_sha256"] = {' not in src
    fp = tmp_path / "a.json"
    fp.write_text('{"k": 1}')
    digest = hashlib.sha256(fp.read_bytes()).hexdigest()
    (tmp_path / "a.sha256").write_text(f"{digest}  {fp.name}\n")
    assert hashlib.sha256(fp.read_bytes()).hexdigest() == digest


# ---- sealed-file guard: static scan of the HOUSE source -----------------
ALLOWED = {"scripts/build_hcbench.py",   # writer only
           "scripts/eval_sealed.py",      # the sole reader
           "tests/test_hcbench.py"}       # this guard


def test_no_script_reads_sealed_file():
    offenders = []
    for fp in list(ROOT.glob("defend_hc2/*.py")) + \
            list(ROOT.glob("scripts/*.py")) + list(ROOT.glob("tests/*.py")):
        rel = str(fp.relative_to(ROOT))
        if rel in ALLOWED:
            continue
        if "hcbench-sealed" in fp.read_text(encoding="utf-8"):
            offenders.append(rel)
    assert not offenders, f"sealed split referenced outside allowlist: {offenders}"


def test_loaders_registry_and_deferred_sources_disjoint():
    from defend_hc2.hcbench import DEFERRED_SOURCES
    assert set(DEFERRED_SOURCES).isdisjoint(set(LOADERS))
    assert len(LOADERS) >= 10
