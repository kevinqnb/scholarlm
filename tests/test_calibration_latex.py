"""Rung 1: analysis/calibration_latex.py on hand-built metrics frames."""
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import calibration_latex as cl

REAL_CFG = "2026-10-04-calibration-v3-gemma27b-qwen-2.5-7b-v3-01"
DS = ["pond", "nfix", "supermat"]
SPEC = {"datasets": {"PLW": "pond", "NF": "nfix", "SM": "supermat"}, "decimals": 3, "ci_format": "pm",
        "label_prefix": "tab:t", "calibration_config": REAL_CFG, "output_dir": "x",
        "labels": "llm_matching"}


def _row(setting, kind, tr, te, smece):
    r = {"Dataset type": setting, "Judge model": "j", "Train dataset": tr, "Test dataset": te, "Type": kind,
         "N": 1234, "Label rate": 0.5, "Platt N": float("nan") if setting == "syn" else 100,
         "Accuracy": 0.9, "Precision": 0.8, "Recall": 0.7, "F1": 0.75, "AUROC": 0.95, "Validity": 0.8,
         "ECE": 0.05, "ECE_lo": 0.04, "ECE_hi": 0.07, "ECE_em": 0.06, "ECE_em_lo": 0.05, "ECE_em_hi": 0.08,
         "RMSCE_db": 0.07, "RMSCE_db_lo": 0.06, "RMSCE_db_hi": 0.09,
         "SmECE": smece, "SmECE_lo": smece - 0.01, "SmECE_hi": smece + 0.03}
    return r


def _frames(**override):
    out = {}
    for kind in ("NTP", "Probe"):
        rows = [_row(s, kind, tr, te, 0.1 + 0.01 * DS.index(tr) + (0.2 if kind == "NTP" else 0))
                for s in ("syn", "real") for tr in DS for te in DS]
        out[kind] = pd.DataFrame(rows)
    return out


def test_format_known_answer():
    # point 0.05077, interval [0.04197, 0.05956] -> half-width 0.0088 -> "0.051 $\pm$ 0.009"
    assert cl.format_estimate(0.05076895907228567, 0.04197373579428747, 0.05956418235028388,
                              decimals=3, ci_format="pm") == "0.051 $\\pm$ 0.009"
    assert cl.format_estimate(0.5, 0.4, 0.7, decimals=1, ci_format="interval", bold=True) == "\\textbf{0.5} [0.4, 0.7]"


def test_negative_lower_bound_allowed_not_clipped():
    assert cl.format_estimate(0.005, -0.004, 0.014, decimals=3, ci_format="interval") == "0.005 [-0.004, 0.014]"


def test_point_outside_interval_raises():
    with pytest.raises(ValueError, match="outside its interval"):
        cl.format_estimate(0.2, 0.3, 0.4, decimals=3, ci_format="pm")


def test_nan_raises():
    with pytest.raises(ValueError, match="not finite"):
        cl.format_estimate(float("nan"), 0.0, 0.1, decimals=3, ci_format="pm")
    with pytest.raises(ValueError, match="not finite"):
        cl.format_point(float("nan"), decimals=3)


def test_smece_table_structure_and_bold():
    tex = cl.build_smece_table(SPEC, _frames(), "real", "j", 100)
    lines = tex.splitlines()
    assert sum(l == "\\midrule" for l in lines) == 2  # one before each block
    assert tex.count("Probe (") == 3 and tex.count("NTP (") == 3
    # Probe train=pond (0.100) is the column-min for test pond only through ties across rows:
    assert tex.count("\\textbf{0.100}") == 3  # Probe (PLW) bold in every column
    assert "\\textbf{0.110}" not in tex
    assert "\\textbf{0.100} $\\pm$ 0.020" in tex  # half-width of [-0.01, +0.03] is 0.02


def test_smece_row_order_is_ntp_then_probe():
    tex = cl.build_smece_table(SPEC, _frames(), "syn", "j", 100)
    assert tex.index("NTP (PLW)") < tex.index("NTP (SM)") < tex.index("Probe (PLW)") < tex.index("Probe (SM)")


def test_classification_rejects_validity_mismatch():
    f = _frames()
    f["Probe"].loc[0, "Validity"] = 0.1
    with pytest.raises(ValueError, match="Validity"):
        cl.build_classification_table(SPEC, f, "syn", "j", 100)


def test_classification_nan_precision_inconsistent_raises():
    f = _frames()  # Recall 0.7, Validity 0.8: NaN precision is not explained by "no predicted positives"
    f["NTP"].loc[0, "Precision"] = float("nan")
    with pytest.raises(ValueError, match="Validity"):
        cl.build_classification_table(SPEC, f, "syn", "j", 100)


def test_classification_no_predicted_positives_renders_marker():
    f = _frames()
    f["NTP"].loc[0, ["Precision", "F1", "Recall", "Validity"]] = [float("nan"), float("nan"), 0.0, 0.0]
    tex = cl.build_classification_table(SPEC, f, "syn", "j", 100)
    assert "0.900 & -- & 0.000 & -- & 0.950" in tex
    assert "undefined" in tex and "nan" not in tex.lower()


def test_classification_nan_accuracy_still_raises():
    f = _frames()
    f["NTP"].loc[0, "Accuracy"] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        cl.build_classification_table(SPEC, f, "syn", "j", 100)


def test_classification_and_variants_have_18_rows_two_blocks():
    for fn in (cl.build_classification_table, cl.build_variants_table):
        tex = fn(SPEC, _frames(), "real", "j", 100)
        assert sum(l.startswith(("NTP &", "Probe &")) for l in tex.splitlines()) == 18
        assert tex.count("\\midrule") == 2


def test_variants_plugin_ece_point_outside_interval_renders_exact_interval():
    f = _frames()
    f["NTP"].loc[3, ["ECE", "ECE_lo", "ECE_hi"]] = [0.0071, 0.0083, 0.0489]
    tex = cl.build_variants_table(SPEC, f, "syn", "j", 100)
    assert "0.007 [0.008, 0.049]" in tex


def test_variants_rmsce_point_outside_interval_still_raises():
    f = _frames()
    f["NTP"].loc[3, "RMSCE_db"] = 0.5
    with pytest.raises(ValueError, match="outside its interval"):
        cl.build_variants_table(SPEC, f, "syn", "j", 100)


@pytest.fixture
def staged(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "cfgs"
    cfg_dir.mkdir()
    shutil.copy(cl.ANALYSIS_CONFIGS_ROOT / f"{REAL_CFG}.yaml", cfg_dir)
    monkeypatch.setattr(cl, "ANALYSIS_CONFIGS_ROOT", cfg_dir)
    monkeypatch.setattr(cl, "_REPO_ROOT", tmp_path)
    res = tmp_path / "analysis" / "results" / "calibration" / REAL_CFG
    res.mkdir(parents=True)
    for kind, df in _frames().items():
        df.to_csv(res / f"metrics_{kind.lower()}.csv", index=False)
    return res


def test_load_metrics_ok(staged):
    frames, judge = cl.load_metrics(SPEC)
    assert judge == "j" and set(frames) == {"NTP", "Probe"}


def test_load_metrics_missing_column(staged):
    p = staged / "metrics_ntp.csv"
    pd.read_csv(p).drop(columns=["SmECE_lo"]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="SmECE_lo"):
        cl.load_metrics(SPEC)


def test_load_metrics_wrong_platt_n(staged):
    p = staged / "metrics_probe.csv"
    df = pd.read_csv(p)
    df.loc[df["Dataset type"] == "real", "Platt N"] = 50
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="platt_n"):
        cl.load_metrics(SPEC)


def test_load_metrics_dropped_row(staged):
    p = staged / "metrics_probe.csv"
    pd.read_csv(p).iloc[1:].to_csv(p, index=False)
    with pytest.raises(ValueError, match="cover exactly"):
        cl.load_metrics(SPEC)


def test_load_metrics_mixed_judges(staged):
    p = staged / "metrics_probe.csv"
    df = pd.read_csv(p)
    df["Judge model"] = "other"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="mix judge models"):
        cl.load_metrics(SPEC)


def test_real_caption_states_n_pattern_from_data():
    # Fixture N is 1234 everywhere -> constant down each column.
    tex = cl.build_smece_table(SPEC, _frames(), "real", "j", 100)
    assert "$N$ is the same down each column" in tex and "differs" not in tex
    f = _frames()
    f["Probe"].loc[(f["Probe"]["Dataset type"] == "real") & (f["Probe"]["Train dataset"] == "nfix"), "N"] = 999
    tex = cl.build_smece_table(SPEC, f, "real", "j", 100)
    assert "$N$ differs down a column" in tex and "same down" not in tex


def test_real_caption_names_label_source():
    llm = cl.build_smece_table(SPEC, _frames(), "real", "j", 100)
    human = cl.build_smece_table({**SPEC, "labels": "human_validated"}, _frames(), "real", "j", 100)
    assert "human validity labels" not in llm and "Platt-scaled on 100 labelled rows" in llm
    assert "scored against human validity labels" in human and "100 LLM+matching-labelled rows" in human
    # Synthetic captions do not depend on the label source.
    assert (cl.build_smece_table(SPEC, _frames(), "syn", "j", 100)
            == cl.build_smece_table({**SPEC, "labels": "human_validated"}, _frames(), "syn", "j", 100))


def test_wrong_loader_for_config_raises(staged):
    # A v3 (llm_matching) config fails the validated loader: it has nfix and no validation_sha256.
    with pytest.raises(ValueError):
        cl.load_metrics({**SPEC, "labels": "human_validated"})


def test_load_spec_rejects_unknown_labels(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("id: c\nproject: scholarlm\ndescription: x\nseed: 0\nparams:\n  calibration_latex:\n"
                 "    calibration_config: a\n    labels: human\n    datasets: {PLW: pond}\n    decimals: 3\n"
                 "    ci_format: pm\n    output_dir: x\n    label_prefix: t\n")
    with pytest.raises(ValueError, match="labels must be one of"):
        cl.load_spec(p)
