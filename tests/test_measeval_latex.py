"""Rung-1 tests for analysis/measeval_latex.py on hand-built frames and configs."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from analysis import measeval_latex as ml  # noqa: E402


def _results(rows: dict[str, tuple]) -> pd.DataFrame:
    """rows: id -> (Quantity F, Unit F, MeasuredEntity F)."""
    return pd.DataFrame.from_dict(rows, orient="index", columns=["Quantity", "Unit", "MeasuredEntity"])


def _spec():
    return {
        "decimals": 1, "caption": "cap", "label": "tab:x",
        "blocks": [
            {"rows": [{"label": "A", "id": "a"}]},
            {"rows": [{"label": "B", "id": "b"}, {"label": "C", "id": "c"}]},
        ],
    }


def test_table_by_inspection():
    res = _results({"a": (0.5, 0.25, 0.1), "b": (0.75, 0.25, 0.05), "c": (0.125, 0.2, 0.3), "unused": (1.0, 1.0, 1.0)})
    text = ml.build_table(_spec(), res)
    assert "Method & Quantity & Unit & Entity \\\\" in text
    assert "\\begin{tabular}{lccc}" in text
    # Best per column among the rows shown ("unused" is not shown); the Unit tie bolds both.
    assert "A & 50.0 & \\textbf{25.0} & 10.0 \\\\" in text
    assert "B & \\textbf{75.0} & \\textbf{25.0} & 5.0 \\\\" in text
    assert "C & 12.5 & 20.0 & \\textbf{30.0} \\\\" in text
    assert text.count("\\midrule") == 2
    assert text.index("\nA &") < text.rindex("\\midrule") < text.index("\nB &")


def test_format_f_rejects_out_of_range_and_nan():
    assert ml.format_f(0.7142, decimals=2) == "71.42"
    for bad in (float("nan"), -0.1, 1.5, True):
        with pytest.raises(ValueError):
            ml.format_f(bad, decimals=1)


def test_build_table_missing_id_raises():
    with pytest.raises(KeyError):
        ml.build_table(_spec(), _results({"a": (0.5, 0.5, 0.5), "b": (0.5, 0.5, 0.5)}))


# ---------------------------------------------------------------------------
# load_spec / load_results against written configs and CSVs
# ---------------------------------------------------------------------------

_IDS = ["exp-a", "exp-b"]


def _write_configs(tmp_path, monkeypatch, csv_rows, experiment_ids=_IDS, spec_rows=None):
    """Write an evaluation config + its CSV and a latex spec config; point the loaders at tmp_path."""
    cfg_dir = tmp_path / "analysis" / "analysis-configs" / "measeval"
    cfg_dir.mkdir(parents=True)
    csv_path = tmp_path / "eval.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    eval_cfg = {"id": "eval-01", "project": "scholarlm", "description": "d", "seed": 0,
                "params": {"ground_truth_file": "x", "experiment_ids": experiment_ids,
                           "measeval_evaluation": {"dev": False, "output": str(csv_path)}}}
    (cfg_dir / "eval-01.yaml").write_text(yaml.safe_dump(eval_cfg))
    spec_cfg = {"id": "latex-01", "project": "scholarlm", "description": "d", "seed": 0,
                "params": {"measeval_latex": {
                    "analysis_config": "eval-01", "decimals": 1, "output": str(tmp_path / "t.tex"),
                    "caption": "cap", "label": "tab:x",
                    "blocks": [{"rows": spec_rows or [{"label": "A", "id": "exp-a"}]}]}}}
    spec_path = cfg_dir / "latex-01.yaml"
    spec_path.write_text(yaml.safe_dump(spec_cfg))
    monkeypatch.setattr(ml, "analysis_config_path", lambda t, i: cfg_dir / f"{i}.yaml")
    return spec_path


def _csv_rows(ids=_IDS, f=0.5, dev=False, config_id="eval-01", dataset="measeval"):
    return [{"experiment_id": i, "dataset": dataset, "dev": dev, "annot_type": t,
             "analysis_config_id": config_id, "f_measure": f}
            for i in ids for t in ("Quantity", "Unit", "MeasuredEntity")]


def _load(tmp_path, monkeypatch, **kw):
    spec_path = _write_configs(tmp_path, monkeypatch, **kw)
    spec = ml.load_spec(spec_path)
    return spec, ml.load_results(spec["analysis_config"])


def test_load_results_round_trip(tmp_path, monkeypatch):
    spec, res = _load(tmp_path, monkeypatch, csv_rows=_csv_rows())
    assert list(res.columns) == ["Quantity", "Unit", "MeasuredEntity"]
    assert sorted(res.index) == _IDS
    assert "A & \\textbf{50.0} & \\textbf{50.0} & \\textbf{50.0} \\\\" in ml.build_table(spec, res)


@pytest.mark.parametrize("csv_rows, match", [
    (_csv_rows(ids=["exp-a"]), "stale"),
    (_csv_rows(ids=["exp-a", "exp-b", "exp-c"]), "stale"),
    (_csv_rows(dev=True), "dev-mode"),
    (_csv_rows(config_id="other"), "analysis_config_id"),
    (_csv_rows(dataset="pond"), "dataset"),
    (_csv_rows()[:-1], "no F-measure"),
    (_csv_rows() + _csv_rows()[:1], "duplicate"),
    (_csv_rows(f=float("nan")), "no F-measure"),
])
def test_load_results_rejects_bad_csv(tmp_path, monkeypatch, csv_rows, match):
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, monkeypatch, csv_rows=csv_rows)


def test_load_spec_rejects_reused_id(tmp_path, monkeypatch):
    rows = [{"label": "A", "id": "exp-a"}, {"label": "A again", "id": "exp-a"}]
    spec_path = _write_configs(tmp_path, monkeypatch, csv_rows=_csv_rows(), spec_rows=rows)
    with pytest.raises(ValueError, match="more than one row"):
        ml.load_spec(spec_path)


def test_load_spec_rejects_row_without_id(tmp_path, monkeypatch):
    spec_path = _write_configs(tmp_path, monkeypatch, csv_rows=_csv_rows(), spec_rows=[{"label": "A"}])
    with pytest.raises(ValueError, match="exactly keys"):
        ml.load_spec(spec_path)
