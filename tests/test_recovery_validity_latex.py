"""Rung-1 tests for analysis/recovery_validity_latex.py on hand-built frames."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from analysis import recovery_validity_latex as rvl  # noqa: E402


def _frame(rows: dict[str, tuple]) -> pd.DataFrame:
    """rows: id -> (rec, rec_lo, rec_hi, val, val_lo, val_hi)."""
    recs = [
        dict(experiment_id=i, dataset="pond", analysis_config_id="cfg",
             recovery=r[0], recovery_ci_lo=r[1], recovery_ci_hi=r[2],
             validity=r[3], validity_ci_lo=r[4], validity_ci_hi=r[5])
        for i, r in rows.items()
    ]
    return pd.DataFrame(recs).set_index("experiment_id")


def _spec(ci_format="pm"):
    return {
        "datasets": {"PLW": {}, "NF": {}},
        "decimals": 1, "ci_format": ci_format, "caption": "cap", "label": "tab:x",
        "blocks": [
            {"rows": [{"label": "Direct A", "ids": {"PLW": "a", "NF": None}}]},
            {"rows": [{"label": "B", "ids": {"PLW": "b", "NF": "c"}}]},
        ],
    }


def _results():
    return {
        "PLW": _frame({"a": (0.5, 0.4, 0.6, 0.9, 0.8, 1.0), "b": (0.25, 0.2, 0.3, 0.5, 0.4, 0.6)}),
        "NF": _frame({"c": (0.1, 0.0, 0.2, 0.75, 0.7, 0.8)}),
    }


def test_pm_table_by_inspection():
    text = rvl.build_table(_spec("pm"), _results())
    assert "Method & PLW & NF \\\\" in text
    assert "Direct A & (\\textbf{50.0} $\\pm$ 10.0, \\textbf{90.0} $\\pm$ 10.0) & -- \\\\" in text
    assert "B & (25.0 $\\pm$ 5.0, 50.0 $\\pm$ 10.0) & (\\textbf{10.0} $\\pm$ 10.0, \\textbf{75.0} $\\pm$ 5.0) \\\\" in text
    # header rule + exactly one rule between the two blocks, no title rows
    assert text.count("\\midrule") == 2
    assert text.index("Direct A") < text.rindex("\\midrule") < text.index("\nB &")
    assert "multicolumn" not in text


def test_interval_format():
    assert rvl.format_estimate(0.388, 0.28, 0.498, decimals=1, ci_format="interval") == "38.8 [28.0, 49.8]"


def test_point_outside_interval_raises():
    with pytest.raises(ValueError, match="outside its interval"):
        rvl.format_estimate(0.9, 0.1, 0.2, decimals=1, ci_format="pm")


def test_nan_validity_raises():
    res = _results()
    res["PLW"].loc["a", "validity"] = float("nan")
    with pytest.raises(ValueError, match="a: .*validity|a: .*point"):
        rvl.build_table(_spec(), res)


def test_unknown_id_raises():
    spec = _spec()
    spec["blocks"][1]["rows"][0]["ids"]["PLW"] = "nope"
    with pytest.raises(KeyError, match="nope"):
        rvl.build_table(spec, _results())


def test_pre_any_edge_csv_without_matching_column_raises(tmp_path, monkeypatch):
    # A CSV written before 2026-10-07 has `recovery` = max-weight matching count
    # and no recovery_max_weight_matching column: refuse it rather than render it.
    import yaml
    from analysis.common import config as ac
    monkeypatch.setattr(ac, "ANALYSIS_CONFIGS_ROOT", tmp_path)
    csv_path = tmp_path / "out.csv"
    cfg_id = "2026-10-05-test-rv-01"
    (tmp_path / "recovery-validity").mkdir()
    with open(tmp_path / "recovery-validity" / f"{cfg_id}.yaml", "w") as f:
        yaml.safe_dump({
            "id": cfg_id, "project": "scholarlm", "description": "test", "seed": 342,
            "params": {"experiment_ids": ["a"], "recovery_validity": {"output": str(csv_path)}},
        }, f)
    old = dict(experiment_id="a", dataset="pond", analysis_config_id=cfg_id,
               recovery=0.5, recovery_ci_lo=0.4, recovery_ci_hi=0.6, recovery_any_edge=0.7,
               validity=0.9, validity_ci_lo=0.8, validity_ci_hi=1.0)
    pd.DataFrame([old]).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="recovery_max_weight_matching"):
        rvl.load_results("PLW", "pond", cfg_id)
    pd.DataFrame([{**old, "recovery_max_weight_matching": 0.5}]).to_csv(csv_path, index=False)
    assert rvl.load_results("PLW", "pond", cfg_id).loc["a", "recovery"] == 0.5
