"""final_report must locate experiment artifacts in --exp-dir and render
Exp-F obfuscation tables honestly (spec Step 6c, BUG-D)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_final_report():
    spec = importlib.util.spec_from_file_location(
        "final_report_under_test",
        Path(__file__).resolve().parents[1] / "scripts" / "final_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


final_report = _load_final_report()


def _run(capsys, tmp_path, exp_payloads, monkeypatch):
    art = tmp_path / "artifacts"
    exp = tmp_path / "expdir"
    art.mkdir()
    exp.mkdir()
    for name, payload in exp_payloads.items():
        (exp / f"{name}.json").write_text(json.dumps(payload))
    monkeypatch.setattr(sys, "argv",
                        ["final_report.py", "--artifacts", str(art),
                         "--exp-dir", str(exp), "--repo", "."])
    assert final_report.main() == 0
    return capsys.readouterr().out


def test_report_contains_exp_auc(capsys, tmp_path, monkeypatch):
    out = _run(capsys, tmp_path, {"bench-metrics-exp-a": {"roc_auc": 0.4231,
                                                          "recall": 0.88}},
               monkeypatch)
    assert "0.4231" in out
    assert "no exp metrics found" not in out


def test_exp_f_obfuscation_table_and_anticorrelation_warning(capsys, tmp_path,
                                                             monkeypatch):
    payload = {"clean": {"roc_auc": 0.99},
               "per_transform": {"mystery_transform": {"perturbed_auc": 0.3312,
                                                       "recovery_auc": 0.38}}}
    out = _run(capsys, tmp_path, {"bench-metrics-exp-f": payload}, monkeypatch)
    assert "0.3312" in out
    assert "WARNING" in out and "unexplained" in out and "pipeline bug" in out


def test_exp_f_recovery_disproves_bug_hypothesis(capsys, tmp_path, monkeypatch):
    """Sub-0.5 perturbed + high recovery = EXPECTED note, never 'pipeline bug';
    known limitation gets its own labeled line; no WARNING at all."""
    payload = {"clean": {"roc_auc": 0.99},
               "per_transform": {
                   "leetspeak": {"perturbed_auc": 0.3516, "recovery_auc": 0.9878},
                   "letter_spacing_extreme": {"perturbed_auc": 0.4382,
                                              "recovery_auc": 0.4383},
               }}
    out = _run(capsys, tmp_path, {"bench-metrics-exp-f": payload}, monkeypatch)
    assert "NOTE" in out and "ANTI-CORRELATED" in out
    assert "LIMITATION" in out and "character-level fragmentation" in out
    assert "WARNING" not in out, out


def test_empty_exp_dir_stays_honest(capsys, tmp_path, monkeypatch):
    out = _run(capsys, tmp_path, {}, monkeypatch)
    assert "no exp metrics found" in out
