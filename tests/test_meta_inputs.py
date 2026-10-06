"""Unit tests for analysis/meta_inputs.py on tiny hand-built fixtures.

The fixture mirrors the real hazard: postprocessed.json expands a list-valued
datapoint into several rows that all carry their parent's measurement_id, while the
scores live one-per-datapoint (final.json). Expected outputs below are checkable by
inspection.
"""
import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from analysis.meta_inputs import attach_scores, load_meta_config, numeric_point_value, stored_prediction_rows

SCORE_COLS = ["ntp_prob", "probe_prob"]


def scored():
    return pd.DataFrame({
        "measurement_id": [10, 11, 12],
        "document_id": ["a", "a", "b"],
        "attribute": ["tn", "ph", "tn"],
        "ntp_prob": [0.1, 0.2, 0.3],
        "probe_prob": [0.9, 0.8, 0.7],
    })


def rows():
    # measurement_id 11 was list-expanded into two child rows (4.0, 5.0); 10 dropped by dedup.
    return pd.DataFrame({
        "measurement_id": [11, 11, 12],
        "document_id": ["a", "a", "b"],
        "attribute": ["ph", "ph", "tn"],
        "point_value": [4.0, 5.0, 7.0],
    })


def test_children_inherit_parent_scores_and_order_is_kept():
    out = attach_scores(rows(), scored(), SCORE_COLS)
    assert len(out) == 3
    assert out["point_value"].tolist() == [4.0, 5.0, 7.0]
    assert out["ntp_prob"].tolist() == [0.2, 0.2, 0.3]
    assert out["probe_prob"].tolist() == [0.8, 0.8, 0.7]
    assert list(out.columns) == ["measurement_id", "document_id", "attribute", "point_value", *SCORE_COLS]


def test_identity_join_on_final_rows():
    s = scored()
    out = attach_scores(s[["measurement_id", "document_id", "attribute"]], s, SCORE_COLS)
    assert out[SCORE_COLS].equals(s[SCORE_COLS])


def test_duplicate_scored_id_raises():
    s = pd.concat([scored(), scored().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        attach_scores(rows(), s, SCORE_COLS)


def test_row_without_scored_datapoint_raises():
    r = rows()
    r.loc[2, "measurement_id"] = 99
    with pytest.raises(ValueError, match="no scored datapoint"):
        attach_scores(r, scored(), SCORE_COLS)


def test_document_or_attribute_mismatch_raises():
    r = rows()
    r.loc[0, "document_id"] = "zzz"
    with pytest.raises(ValueError, match="document_id"):
        attach_scores(r, scored(), SCORE_COLS)
    r = rows()
    r.loc[2, "attribute"] = "tp"
    with pytest.raises(ValueError, match="attribute"):
        attach_scores(r, scored(), SCORE_COLS)


def test_nan_score_raises():
    s = scored()
    s.loc[1, "probe_prob"] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        attach_scores(rows(), s, SCORE_COLS)


# ── config validation ────────────────────────────────────────────────────────

GOOD = {
    "id": "t", "project": "scholarlm", "description": "d", "seed": 0,
    "params": {"meta": {
        "calibration_config_id": "cal", "rows": "final",
        "deduplication_config_id": None, "n_boot": 10, "qq_attributes": ["tn"],
    }},
}


def write(tmp_path, cfg):
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_good_config_loads(tmp_path):
    assert load_meta_config(write(tmp_path, GOOD))["params"]["meta"]["rows"] == "final"


@pytest.mark.parametrize("mutate", [
    lambda m: m.pop("n_boot"),                                   # missing key: no default
    lambda m: m.update(extra=1),                                 # stray key
    lambda m: m.update(rows="both"),                             # bad choice
    lambda m: m.update(deduplication_config_id="x"),             # id given for rows=final
    lambda m: m.update(rows="deduplicated"),                     # rows=deduplicated without id
    lambda m: m.update(n_boot=True),                             # bool is not an int
])
def test_bad_config_raises(tmp_path, mutate):
    cfg = copy.deepcopy(GOOD)
    mutate(cfg["params"]["meta"])
    with pytest.raises((ValueError, KeyError)):  # get_section raises KeyError for missing/stray keys
        load_meta_config(write(tmp_path, cfg))


# ── stored_prediction_rows ───────────────────────────────────────────────────

def final():
    return pd.DataFrame({"measurement_id": [0, 1, 2, 3], "document_id": ["a", "t", "b", "t"],
                         "attribute": ["tn", "tn", "ph", "ph"]})


def cell(n=2):
    return {"probe_probs": np.array([0.1, 0.2][:n]), "ntp_probs": np.array([0.3, 0.4][:n]),
            "labels": np.array([True, False][:n])}


def test_stored_rows_map_in_final_order_skipping_training_docs():
    out = stored_prediction_rows(final(), {"t"}, cell())
    assert out["measurement_id"].tolist() == [0, 2]
    assert out["probe_prob"].tolist() == [0.1, 0.2]
    assert out["ntp_prob"].tolist() == [0.3, 0.4]


def test_stored_rows_length_mismatch_raises():
    with pytest.raises(ValueError, match="does not match"):
        stored_prediction_rows(final(), {"t"}, cell(n=1))
    with pytest.raises(ValueError, match="does not match"):
        stored_prediction_rows(final(), set(), cell())


def test_stored_rows_nonfinite_raises():
    c = cell()
    c["probe_probs"][0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        stored_prediction_rows(final(), {"t"}, c)


# ── numeric_point_value ──────────────────────────────────────────────────────

def test_numeric_point_value_known_answers():
    pv = pd.Series(["0.26", 1.5, "1.5 × 10^8", None, "pH", "1 m²", "ca. 3"], index=[5, 6, 7, 8, 9, 10, 11])
    out = numeric_point_value(pv)
    assert out.index.tolist() == pv.index.tolist()
    assert out.iloc[:3].tolist() == [0.26, 1.5, 1.5e8]
    assert out.iloc[3:].isna().all()
