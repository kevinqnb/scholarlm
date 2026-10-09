"""Rung 1: analysis/calibration_latex.py on hand-built metrics frames."""
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import calibration_latex as cl

REAL_CFG = "2026-01-01-calibration-v4-test-01"
DS = ["pond", "nfix", "supermat"]
SPEC = {"datasets": {"PLW": "pond", "NF": "nfix", "SM": "supermat"}, "decimals": 3,
        "label_prefix": "tab:t", "calibration_config": REAL_CFG, "output_dir": "x",
        "labels": "llm_matching"}


NAN = float("nan")


def _cal(recalibration="intercept_fit", source="sample", **over):
    """``params`` of a v4 calibration config, as the table builders and CSV checks read it."""
    block = {"use_matching_labels": True, "pi_te_estimate": 0.4 if source == "manual" else None}
    params = {"n_boot": 2000, "recalibration": recalibration, "fit_source": source,
              "fit_n": 100 if source == "sample" else None, "fit_seed": 7 if source == "sample" else None,
              "datasets": {ds: dict(block) for ds in DS}}
    params.update(over)
    return params


CAL = _cal()


def _row(setting, kind, tr, te, smece, cal=None):
    cal = cal or CAL
    syn = setting == "syn"
    manual = cal["fit_source"] == "manual"
    r = {"Dataset type": setting, "Judge model": "j", "Train dataset": tr, "Test dataset": te, "Type": kind,
         "N": 1234, "Label rate": 0.5,
         "Recalibration": None if syn else cal["recalibration"], "Fit source": None if syn else cal["fit_source"],
         "Fit n": NAN if syn else (0 if manual else cal["fit_n"]),
         "Fit seed": NAN if syn or manual else cal["fit_seed"],
         "pi_te": NAN if syn else (0.4 if manual else 0.55), "Doc resamples": cal["n_boot"],
         "Accuracy": 0.9, "Precision": 0.8, "Recall": 0.7, "F1": 0.75, "AUROC": 0.95, "Validity": 0.8,
         "ECE": 0.05, "ECE_lo": 0.04, "ECE_hi": 0.07, "ECE_em": 0.06, "ECE_em_lo": 0.05, "ECE_em_hi": 0.08,
         "RMSCE_db": 0.07, "RMSCE_db_lo": 0.06, "RMSCE_db_hi": 0.09,
         "SmECE": smece, "SmECE_lo": smece - 0.01, "SmECE_hi": smece + 0.03}
    return r


def _frames(cal=None):
    out = {}
    for kind in ("NTP", "Probe"):
        rows = [_row(s, kind, tr, te, 0.1 + 0.01 * DS.index(tr) + (0.2 if kind == "NTP" else 0), cal)
                for s in ("syn", "real") for tr in DS for te in DS]
        out[kind] = pd.DataFrame(rows)
    return out


def test_format_known_answer():
    assert cl.format_estimate(0.05076895907228567, 0.04197373579428747, 0.05956418235028388,
                              decimals=3) == "0.051 [0.042, 0.060]"
    assert cl.format_estimate(0.5, 0.4, 0.7, decimals=1, bold=True) == "\\textbf{0.5} [0.4, 0.7]"


def test_negative_lower_bound_allowed_not_clipped():
    assert cl.format_estimate(0.005, -0.004, 0.014, decimals=3) == "0.005 [-0.004, 0.014]"


def test_point_outside_percentile_interval_renders_exact_interval():
    # Resampled |gap| statistics sit above the un-resampled point near perfect calibration.
    assert cl.format_estimate(0.2, 0.3, 0.4, decimals=3) == "0.200 [0.300, 0.400]"


def test_inverted_interval_raises():
    with pytest.raises(ValueError, match="lo > hi"):
        cl.format_estimate(0.2, 0.4, 0.3, decimals=3)


def test_nan_raises():
    with pytest.raises(ValueError, match="not finite"):
        cl.format_estimate(float("nan"), 0.0, 0.1, decimals=3)
    with pytest.raises(ValueError, match="not finite"):
        cl.format_point(float("nan"), decimals=3)


def test_smece_table_structure_and_bold():
    tex = cl.build_smece_table(SPEC, _frames(), "real", "j", CAL)
    lines = tex.splitlines()
    assert sum(l == "\\midrule" for l in lines) == 2  # one before each block
    assert tex.count("Probe (") == 3 and tex.count("NTP (") == 3
    # Probe train=pond (0.100) is the column-min for test pond only through ties across rows:
    assert tex.count("\\textbf{0.100}") == 3  # Probe (PLW) bold in every column
    assert "\\textbf{0.110}" not in tex
    assert "\\textbf{0.100} [0.090, 0.130]" in tex


def test_smece_row_order_is_ntp_then_probe():
    tex = cl.build_smece_table(SPEC, _frames(), "syn", "j", CAL)
    assert tex.index("NTP (PLW)") < tex.index("NTP (SM)") < tex.index("Probe (PLW)") < tex.index("Probe (SM)")


def test_classification_rejects_validity_mismatch():
    f = _frames()
    f["Probe"].loc[0, "Validity"] = 0.1
    with pytest.raises(ValueError, match="Validity"):
        cl.build_classification_table(SPEC, f, "syn", "j", CAL)


def test_classification_nan_precision_inconsistent_raises():
    f = _frames()  # Recall 0.7, Validity 0.8: NaN precision is not explained by "no predicted positives"
    f["NTP"].loc[0, "Precision"] = float("nan")
    with pytest.raises(ValueError, match="Validity"):
        cl.build_classification_table(SPEC, f, "syn", "j", CAL)


def test_classification_no_predicted_positives_renders_marker():
    f = _frames()
    f["NTP"].loc[0, ["Precision", "F1", "Recall", "Validity"]] = [float("nan"), float("nan"), 0.0, 0.0]
    tex = cl.build_classification_table(SPEC, f, "syn", "j", CAL)
    assert "0.900 & -- & 0.000 & -- & 0.950" in tex
    assert "undefined" in tex and "nan" not in tex.lower()


def test_classification_nan_accuracy_still_raises():
    f = _frames()
    f["NTP"].loc[0, "Accuracy"] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        cl.build_classification_table(SPEC, f, "syn", "j", CAL)


def test_classification_and_variants_have_18_rows_two_blocks():
    for fn in (cl.build_classification_table, cl.build_variants_table):
        tex = fn(SPEC, _frames(), "real", "j", CAL)
        assert sum(l.startswith(("NTP &", "Probe &")) for l in tex.splitlines()) == 18
        assert tex.count("\\midrule") == 2


def test_variants_plugin_ece_point_outside_interval_renders_exact_interval():
    f = _frames()
    f["NTP"].loc[3, ["ECE", "ECE_lo", "ECE_hi"]] = [0.0071, 0.0083, 0.0489]
    tex = cl.build_variants_table(SPEC, f, "syn", "j", CAL)
    assert "0.007 [0.008, 0.049]" in tex


def test_variants_rmsce_point_outside_interval_renders_exact_interval():
    f = _frames()
    f["NTP"].loc[3, ["RMSCE_db", "RMSCE_db_lo", "RMSCE_db_hi"]] = [0.5, 0.06, 0.09]
    tex = cl.build_variants_table(SPEC, f, "syn", "j", CAL)
    assert "0.500 [0.060, 0.090]" in tex


def _v4_config(gt, cal=None):
    cal = cal or CAL
    block = {"extraction_id": "e", "judge_interp_id": "j", "judge_combine_id": "c", "ground_truth_file": str(gt),
             "synthetic_probe_config": "p", "use_matching_labels": True, "pi_te_estimate": None,
             "syn_test_ids": {"primary": "tp", "diag": "td"}}
    return {"id": REAL_CFG, "project": "scholarlm", "description": "t", "seed": 0,
            "params": {"probe_type": "head", "probe_variant": "platt", "syn_split": "primary",
                       "n_boot": cal["n_boot"], "recalibration": cal["recalibration"], "fit_source": cal["fit_source"],
                       "fit_n": cal["fit_n"], "fit_seed": cal["fit_seed"],
                       "datasets": {ds: {**block, "pi_te_estimate": cal["datasets"][ds]["pi_te_estimate"]} for ds in DS}}}


@pytest.fixture
def staged(tmp_path, monkeypatch):
    import yaml
    from analysis.common import config as ac
    cfg_dir = tmp_path / "cfgs"
    gt = tmp_path / "gt.json"
    gt.write_text("[]")
    # The same v4 config filed under both types: the validated loader must still reject it.
    for typ in ("calibration", "calibration-validated"):
        (cfg_dir / typ).mkdir(parents=True)
        (cfg_dir / typ / f"{REAL_CFG}.yaml").write_text(yaml.safe_dump(_v4_config(gt)))
    monkeypatch.setattr(ac, "ANALYSIS_CONFIGS_ROOT", cfg_dir)
    monkeypatch.setattr(ac, "ANALYSIS_RESULTS_ROOT", tmp_path / "results")
    res = tmp_path / "results" / "calibration" / REAL_CFG
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


def _rewrite(path, fn):
    df = pd.read_csv(path)
    fn(df)
    df.to_csv(path, index=False)


@pytest.mark.parametrize("col, value", [
    ("Fit n", 50), ("Fit seed", 8), ("Recalibration", "platt_fit"), ("Fit source", "oracle"), ("Doc resamples", 10),
])
def test_load_metrics_real_rows_must_match_config(staged, col, value):
    _rewrite(staged / "metrics_probe.csv", lambda df: df.__setitem__(
        col, df[col].where(df["Dataset type"] == "syn", value) if col != "Doc resamples" else value))
    with pytest.raises(ValueError, match=col):
        cl.load_metrics(SPEC)


def test_load_metrics_syn_rows_must_not_be_recalibrated(staged):
    _rewrite(staged / "metrics_ntp.csv", lambda df: df.__setitem__(
        "Fit n", df["Fit n"].where(df["Dataset type"] == "real", 100)))
    with pytest.raises(ValueError, match="syn rows"):
        cl.load_metrics(SPEC)


def test_load_metrics_rejects_old_platt_n_schema(staged):
    p = staged / "metrics_probe.csv"
    pd.read_csv(p).drop(columns=["Fit source", "pi_te"]).assign(**{"Platt N": 100}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing column"):
        cl.load_metrics(SPEC)


def _restage(staged, cal, frames=None):
    """Replace the staged config and metrics CSVs with ones built for ``cal``."""
    import yaml
    from analysis.common import config as ac
    (ac.ANALYSIS_CONFIGS_ROOT / "calibration" / f"{REAL_CFG}.yaml").write_text(
        yaml.safe_dump(_v4_config(staged.parents[2] / "gt.json", cal)))
    for kind, df in (frames or _frames(cal)).items():
        df.to_csv(staged / f"metrics_{kind.lower()}.csv", index=False)


@pytest.mark.parametrize("recalibration, source", [("prior_shift", "manual"), ("platt_fit", "sample"), ("prior_shift", "sample")])
def test_load_metrics_accepts_every_method_and_source(staged, recalibration, source):
    _restage(staged, _cal(recalibration, source))
    frames, judge = cl.load_metrics(SPEC)
    assert judge == "j" and set(frames) == {"NTP", "Probe"}


def test_manual_pi_te_must_match_config(staged):
    cal = _cal("prior_shift", "manual")
    frames = _frames(cal)
    frames["NTP"].loc[frames["NTP"]["Dataset type"] == "real", "pi_te"] = 0.41
    _restage(staged, cal, frames)
    with pytest.raises(ValueError, match="pi_te"):
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
    tex = cl.build_smece_table(SPEC, _frames(), "real", "j", CAL)
    assert "$N$ is the same down each column" in tex and "differs" not in tex
    f = _frames()
    f["Probe"].loc[(f["Probe"]["Dataset type"] == "real") & (f["Probe"]["Train dataset"] == "nfix"), "N"] = 999
    tex = cl.build_smece_table(SPEC, f, "real", "j", CAL)
    assert "$N$ differs down a column" in tex and "same down" not in tex


def test_real_caption_names_label_source():
    llm = cl.build_smece_table(SPEC, _frames(), "real", "j", CAL)
    human = cl.build_smece_table({**SPEC, "labels": "human_validated"}, _frames(), "real", "j", CAL)
    assert "human validity labels" not in llm and "Scored against judge-or-match validity labels" in llm
    assert "Scored against human validity labels on the validated rows" in human
    # The fit rows are LLM+matching-labelled either way.
    assert "100 judge-or-match-labelled rows" in llm and "100 judge-or-match-labelled rows" in human
    # Synthetic captions do not depend on the label source.
    assert (cl.build_smece_table(SPEC, _frames(), "syn", "j", CAL)
            == cl.build_smece_table({**SPEC, "labels": "human_validated"}, _frames(), "syn", "j", CAL))


def test_wrong_loader_for_config_raises(staged):
    # A calibration (llm_matching) config fails the validated loader: it has nfix and no validation_sha256.
    with pytest.raises(ValueError):
        cl.load_metrics({**SPEC, "labels": "human_validated"})


def test_load_spec_rejects_unknown_labels(tmp_path):
    p = tmp_path / "calibration" / "c.yaml"
    p.parent.mkdir()
    p.write_text("id: c\nproject: scholarlm\ndescription: x\nseed: 0\nparams:\n  calibration_latex:\n"
                 "    calibration_config: a\n    labels: human\n    datasets: {PLW: pond}\n    decimals: 3\n"
                 "    output_dir: x\n    label_prefix: t\n")
    with pytest.raises(ValueError, match="labels must be one of"):
        cl.load_spec(p)


_SPEC_YAML = ("id: c\nproject: scholarlm\ndescription: x\nseed: 0\nparams:\n  calibration_latex:\n"
              "    calibration_config: a\n    labels: {labels}\n    datasets: {{PLW: pond}}\n    decimals: 3\n"
              "    output_dir: x\n    label_prefix: t\n")


@pytest.mark.parametrize("typ, labels", [("calibration", "human_validated"), ("calibration-validated", "llm_matching")])
def test_load_spec_rejects_labels_from_other_type(tmp_path, typ, labels):
    p = tmp_path / typ / "c.yaml"
    p.parent.mkdir()
    p.write_text(_SPEC_YAML.format(labels=labels))
    with pytest.raises(ValueError, match="config lives in"):
        cl.load_spec(p)


def test_load_spec_rejects_other_type_dir(tmp_path):
    p = tmp_path / "recovery-validity" / "c.yaml"
    p.parent.mkdir()
    p.write_text(_SPEC_YAML.format(labels="llm_matching"))
    with pytest.raises(ValueError, match="must live in one of"):
        cl.load_spec(p)


def test_caption_describes_the_document_bootstrap():
    real = cl.build_smece_table(SPEC, _frames(), "real", "j", CAL)
    syn = cl.build_smece_table(SPEC, _frames(), "syn", "j", CAL)
    for tex in (real, syn):
        assert "2000 document-level bootstrap resamples" in tex and "un-resampled value" in tex
        assert "refit" not in tex and "averaged" not in tex  # v3's fit-sample cross is gone
    assert "no recalibration" in syn and "Recalibrated" not in syn


@pytest.mark.parametrize("cal, phrases", [
    (_cal("intercept_fit", "sample"), ["single intercept shift (slope 1)", "100 judge-or-match-labelled rows drawn once (seed 7)"]),
    (_cal("platt_fit", "sample"), ["Platt scaling (slope and intercept)", "100 judge-or-match-labelled rows drawn once (seed 7)"]),
    (_cal("prior_shift", "sample"), ["prior-shift correction", "100 judge-or-match-labelled rows drawn once (seed 7)"]),
    (_cal("prior_shift", "manual"), ["prior-shift correction", "assumed valid rate per test dataset"]),
    (_cal("intercept_fit", "oracle"), ["evaluated rows themselves (oracle fit"]),
])
def test_real_caption_states_the_recalibration(cal, phrases):
    tex = cl.build_smece_table(SPEC, _frames(cal), "real", "j", cal)
    for ph in phrases:
        assert ph in tex
    assert "Platt-scaled" not in tex


def test_caption_label_source_follows_use_matching_labels():
    cal = _cal()
    for block in cal["datasets"].values():
        block["use_matching_labels"] = False
    tex = cl.build_smece_table(SPEC, _frames(cal), "real", "j", cal)
    assert "Scored against judge validity labels" in tex and "100 judge-labelled rows" in tex
    cal["datasets"]["pond"]["use_matching_labels"] = True  # mixed: no single honest caption
    with pytest.raises(ValueError, match="mix use_matching_labels"):
        cl.build_smece_table(SPEC, _frames(cal), "real", "j", cal)


def test_load_metrics_oracle_fits_on_all_evaluated_rows(staged):
    cal = _cal("intercept_fit", "oracle")
    frames = _frames(cal)
    for df in frames.values():
        df.loc[df["Dataset type"] == "real", "Fit n"] = df.loc[df["Dataset type"] == "real", "N"]
    _restage(staged, cal, frames)
    cl.load_metrics(SPEC)
    frames["NTP"].loc[frames["NTP"]["Dataset type"] == "real", "Fit n"] = 100
    _restage(staged, cal, frames)
    with pytest.raises(ValueError, match="oracle rows"):
        cl.load_metrics(SPEC)


def test_load_metrics_human_validated_end_to_end(tmp_path, monkeypatch):
    # Oracle is rejected by the validated loader, so a validated spec runs on a sample config.
    import hashlib
    import json
    import yaml
    from analysis.common import config as ac
    ds2 = ["pond", "supermat"]
    spec = {**SPEC, "labels": "human_validated", "datasets": {"PLW": "pond", "SM": "supermat"}}
    gt = tmp_path / "gt.json"
    gt.write_text("[]")
    vdir = tmp_path / "validations"
    vdir.mkdir()
    cfg = _v4_config(gt)
    cfg["params"]["datasets"] = {ds: cfg["params"]["datasets"][ds] for ds in ds2}
    for ds in ds2:
        (vdir / f"{ds}.json").write_text(json.dumps({"dataset": ds}))
        cfg["params"]["datasets"][ds]["validation_sha256"] = hashlib.sha256((vdir / f"{ds}.json").read_bytes()).hexdigest()
    monkeypatch.setenv(ac.VALIDATIONS_ENV, str(vdir))
    monkeypatch.setattr(ac, "ANALYSIS_CONFIGS_ROOT", tmp_path / "cfgs")
    monkeypatch.setattr(ac, "ANALYSIS_RESULTS_ROOT", tmp_path / "results")
    (tmp_path / "cfgs" / "calibration-validated").mkdir(parents=True)
    (tmp_path / "cfgs" / "calibration-validated" / f"{REAL_CFG}.yaml").write_text(yaml.safe_dump(cfg))
    res = tmp_path / "results" / "calibration-validated" / REAL_CFG
    res.mkdir(parents=True)
    for kind, df in _frames().items():
        df[df["Train dataset"].isin(ds2) & df["Test dataset"].isin(ds2)].to_csv(res / f"metrics_{kind.lower()}.csv", index=False)
    frames, judge = cl.load_metrics(spec)
    assert judge == "j" and len(frames["Probe"]) == 8
    tex = cl.build_smece_table(spec, frames, "real", "j", cl.load_calibration_config(spec)["params"])
    assert "Scored against human validity labels on the validated rows" in tex and "& SM \\\\" in tex
