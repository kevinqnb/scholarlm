"""Unit tests for analysis/common/meta_inputs.py on tiny hand-built fixtures.

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

from analysis.common.meta_inputs import (
    attach_scores, dedup_rows_with_scores, load_meta_v2_config, numeric_point_value, row_provenance,
    stored_prediction_rows,
)

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
    "params": {"meta_v2": {
        "calibration_config_id": "cal", "calibration_version": "v4", "rows": "final",
        "deduplication_config_id": None, "confidence": None, "n_boot": 10, "reference": "valid",
        "ecosystems": ["pond"], "attributes": ["tn", "tp"], "qq_attributes": ["tn"],
        "thresholds": [0.0, 0.5], "min_n": 5, "n_shuffle_samples": 10, "outlier_adjust": False, "w1_curve_scale": "raw",
        "threshold_mode": "value",
    }},
}


def write(tmp_path, cfg):
    p = tmp_path / "meta" / "t.yaml"
    p.parent.mkdir(exist_ok=True)
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_postprocessed_and_deduplicated_configs_load(tmp_path):
    for rows, dd, conf in [("postprocessed", None, None), ("deduplicated", "x", "center"),
                           ("deduplicated", "x", "cluster_mean")]:
        cfg = copy.deepcopy(GOOD)
        cfg["params"]["meta_v2"].update(rows=rows, deduplication_config_id=dd, confidence=conf)
        assert load_meta_v2_config(write(tmp_path, cfg))["params"]["meta_v2"]["confidence"] == conf


def test_good_config_loads(tmp_path):
    assert load_meta_v2_config(write(tmp_path, GOOD))["params"]["meta_v2"]["rows"] == "final"


@pytest.mark.parametrize("mutate", [
    lambda m: m.pop("n_boot"),                                   # missing key: no default
    lambda m: m.update(extra=1),                                 # stray key
    lambda m: m.update(rows="both"),                             # bad choice
    lambda m: m.update(deduplication_config_id="x"),             # id given for rows=final
    lambda m: m.update(rows="deduplicated"),                     # rows=deduplicated without id
    lambda m: m.update(n_boot=True),                             # bool is not an int
    lambda m: m.update(confidence="center"),                     # confidence given for rows=final
    lambda m: m.update(rows="deduplicated", deduplication_config_id="x"),                       # no confidence
    lambda m: m.update(rows="deduplicated", deduplication_config_id="x", confidence="median"),  # bad choice
    lambda m: m.pop("reference"),                                # reference has no default
    lambda m: m.update(reference="gt"),                          # bad choice
    lambda m: m.pop("ecosystems"),                               # cell subset has no default
    lambda m: m.pop("attributes"),
    lambda m: m.update(ecosystems=[]),                           # empty subset
    lambda m: m.update(attributes=["tn", "tn"]),                 # duplicate
    lambda m: m.update(ecosystems="pond"),                       # not a list
    lambda m: m.update(qq_attributes=["ph"]),                    # qq attribute outside attributes
    lambda m: m.update(calibration_version="v3"),                # v3 calibrations no longer read
])
def test_bad_config_raises(tmp_path, mutate):
    cfg = copy.deepcopy(GOOD)
    mutate(cfg["params"]["meta_v2"])
    with pytest.raises((ValueError, KeyError)):  # get_section raises KeyError for missing/stray keys
        load_meta_v2_config(write(tmp_path, cfg))


# ── stored_prediction_rows ───────────────────────────────────────────────────

def final():
    return pd.DataFrame({"measurement_id": [0, 1, 2, 3], "document_id": ["a", "t", "b", "t"],
                         "attribute": ["tn", "tn", "ph", "ph"]})


FINAL_SHA = "abc"
CAL_ID = "cal-1"


def cell(final_order=(0, 2)):
    """A real cell as calibration stores it: ids t(raining doc) rows 1 and 3 excluded."""
    f = final().set_index("measurement_id")
    ids = list(final_order)
    return {"probe_probs": np.array([0.1, 0.2]), "ntp_probs": np.array([0.3, 0.4]),
            "labels": np.array([True, False]), "measurement_ids": np.array(ids),
            "document_ids": f.loc[ids, "document_id"].tolist(), "attributes": f.loc[ids, "attribute"].tolist(),
            "final_sha256": FINAL_SHA, "combined_sha256": "def", "calibration_config_id": CAL_ID, "seed": 0,
            "platt_measurement_ids": np.array([1]), "excluded_documents": ["t"]}


def stored(c=None, fin=None, docs=("t",), sha=FINAL_SHA, cal=CAL_ID):
    return stored_prediction_rows(final() if fin is None else fin, set(docs), cell() if c is None else c, sha, cal)


def test_stored_rows_join_by_id_known_answer():
    out = stored()
    assert out["measurement_id"].tolist() == [0, 2]
    assert out["probe_prob"].tolist() == [0.1, 0.2]
    assert out["ntp_prob"].tolist() == [0.3, 0.4]
    assert out["label"].tolist() == [True, False]


def test_stored_rows_scores_follow_ids_not_position():
    # final.json reordered: positional pairing would swap the scores; id pairing must not.
    fin = final().iloc[[2, 1, 0, 3]].reset_index(drop=True)
    out = stored(fin=fin)
    assert dict(zip(out["measurement_id"], out["probe_prob"])) == {0: 0.1, 2: 0.2}


@pytest.mark.parametrize("mutate,match", [
    (lambda c: c.pop("measurement_ids"), "predates row provenance"),
    (lambda c: c.update(final_sha256="zzz"), "sha256 mismatch"),
    (lambda c: c.update(calibration_config_id="other"), "asked for"),
    (lambda c: c.update(excluded_documents=["t", "a"]), "excluded documents"),
    (lambda c: c.update(measurement_ids=np.array([0, 0])), "not unique"),
    (lambda c: c.update(measurement_ids=np.array([0, 3])), "not the final.json ids"),
    (lambda c: c.update(document_ids=["a", "a"]), "disagree with final.json on 'document_id'"),
    (lambda c: c.update(attributes=["tn", "tn"]), "disagree with final.json on 'attribute'"),
    (lambda c: c.update(probe_probs=np.array([0.1])), "disagree in length"),
    (lambda c: c.update(platt_measurement_ids=np.array([0])), "Platt sample overlaps"),
    (lambda c: c["probe_probs"].__setitem__(0, np.nan), "non-finite"),
])
def test_stored_rows_raise(mutate, match):
    c = cell()
    mutate(c)
    with pytest.raises(ValueError, match=match):
        stored(c)


def test_stored_rows_wrong_excluded_docs_raises():
    with pytest.raises(ValueError, match="excluded documents"):
        stored(docs=())


def test_stored_rows_changed_final_raises():
    fin = final().iloc[[0, 1, 2]]  # final.json lost id 3 (a training-doc row): ids outside excluded still match
    assert stored(fin=fin)["measurement_id"].tolist() == [0, 2]
    fin = final().iloc[[0, 1, 3]]  # lost id 2: stored id 2 no longer exists
    with pytest.raises(ValueError, match="not the final.json ids"):
        stored(fin=fin)


def test_real_cell_provenance_known_answer(tmp_path):
    from analysis.common.prediction_store import real_cell_provenance
    f, c = tmp_path / "final.json", tmp_path / "combined.json"
    f.write_text("[]"); c.write_text("[1]")
    out = real_cell_provenance(final(), np.array([0, 2]), np.array([1]), {"t"}, f, c, CAL_ID, 0)
    assert out["measurement_ids"].tolist() == [0, 2] and out["document_ids"] == ["a", "b"]
    assert out["platt_measurement_ids"].tolist() == [1] and out["excluded_documents"] == ["t"]
    assert out["final_sha256"] != out["combined_sha256"]
    with pytest.raises(AssertionError, match="overlap"):
        real_cell_provenance(final(), np.array([0, 1]), np.array([1]), set(), f, c, CAL_ID, 0)
    with pytest.raises(AssertionError, match="excluded document"):
        real_cell_provenance(final(), np.array([0, 1]), np.array([2]), {"t"}, f, c, CAL_ID, 0)


# ── numeric_point_value ──────────────────────────────────────────────────────

def test_numeric_point_value_known_answers():
    pv = pd.Series(["0.26", 1.5, "1.5 × 10^8", None, "pH", "1 m²", "ca. 3"], index=[5, 6, 7, 8, 9, 10, 11])
    out = numeric_point_value(pv)
    assert out.index.tolist() == pv.index.tolist()
    assert out.iloc[:3].tolist() == [0.26, 1.5, 1.5e8]
    assert out.iloc[3:].isna().all()


# ── row_provenance / dedup_rows_with_scores ──────────────────────────────────
# Postprocessed fixture (row: measurement_id, document, point_value):
#   0: 0 a 1.0 | 1: 1 a 4.0 (child 0 of parent 1) | 2: 1 a 5.0 (child 1) | 3: 2 a 4.0 (plain,
#   duplicates row 1) | 4: 3 b 7.0 | 5: 4 t 9.0 (probe-training document, never scored)
# Clusters: {0} {1,3} (center 3) {2} {4} {5}.

def post():
    return pd.DataFrame({
        "measurement_id": [0, 1, 1, 2, 3, 4],
        "document_id": ["a", "a", "a", "a", "b", "t"],
        "attribute": ["tn", "ph", "ph", "ph", "tn", "tn"],
        "value": ["1.0", "4 and 5", "4 and 5", "4.0", "7.0", "9.0"],
        "point_value": [1.0, 4.0, 5.0, 4.0, 7.0, 9.0],
        "list_values": [None] * 6,
    })


def clusters():
    return pd.DataFrame({
        "row": range(6), "measurement_id": [0, 1, 1, 2, 3, 4],
        "cluster_id": [0, 1, 2, 1, 4, 5], "center_row": [0, 3, 2, 3, 4, 5],
        "is_center": [True, False, True, True, True, True], "cluster_size": [1, 2, 1, 2, 1, 1],
        "mean_w": 1.0,
    })


def scored3():
    return pd.DataFrame({
        "measurement_id": [0, 1, 2, 3], "document_id": ["a", "a", "a", "b"],
        "attribute": ["tn", "ph", "ph", "tn"],
        "judgement_combined": [True, False, True, False],
        "ntp_prob": [0.1, 0.2, 0.3, 0.4], "probe_prob": [0.9, 0.8, 0.7, 0.6],
    })


def kept():
    return post().iloc[[0, 2, 3, 4, 5]].reset_index(drop=True)[
        ["measurement_id", "document_id", "attribute", "point_value"]]


SC = ["judgement_combined", "ntp_prob", "probe_prob"]
MC = ["ntp_prob", "probe_prob"]


def dedup(conf, **over):
    a = dict(kept=kept(), post=row_provenance(post()), clusters=clusters(), scored=scored3(),
             score_cols=SC, mean_cols=MC, confidence=conf, excluded_docs={"t"})
    a.update(over)
    return dedup_rows_with_scores(**a)


def test_row_provenance_known_answer():
    out = row_provenance(post())
    assert out["row"].tolist() == [0, 1, 2, 3, 4, 5]
    assert out["n_siblings"].tolist() == [1, 2, 2, 1, 1, 1]
    assert out["list_index"].tolist() == [-1, 0, 1, -1, -1, -1]


def test_row_provenance_noncontiguous_siblings_raise():
    p = post()
    p["measurement_id"] = [0, 1, 7, 1, 3, 4]   # id 1 at rows 1 and 3, split by row 2
    with pytest.raises(ValueError, match="contiguous"):
        row_provenance(p)


def test_row_provenance_siblings_disagree_raises():
    p = post()
    p.loc[2, "attribute"] = "tp"
    with pytest.raises(ValueError, match="attribute"):
        row_provenance(p)


def test_row_provenance_non_int_id_raises():
    p = post()
    p["measurement_id"] = p["measurement_id"].astype(float)
    with pytest.raises(ValueError, match="int"):
        row_provenance(p)


def test_dedup_center_confidence_is_the_centers_parents():
    out = dedup("center")
    assert out["row"].tolist() == [0, 2, 3, 4]          # row 5 is a training document
    assert out["point_value"].tolist() == [1.0, 5.0, 4.0, 7.0]
    assert out["probe_prob"].tolist() == [0.9, 0.8, 0.7, 0.6]   # row 2 is a child of parent 1
    assert out["ntp_prob"].tolist() == [0.1, 0.2, 0.3, 0.4]
    assert out["judgement_combined"].tolist() == [True, False, True, False]
    assert out["n_siblings"].tolist() == [1, 2, 1, 1] and out["list_index"].tolist() == [-1, 1, -1, -1]


def test_dedup_cluster_mean_averages_members_but_not_the_judge_label():
    out = dedup("cluster_mean")
    # cluster {1,3} = parents 1 (probe .8, ntp .2) and 2 (probe .7, ntp .3); center is row 3.
    assert out["row"].tolist() == [0, 2, 3, 4]
    assert out["probe_prob"].tolist() == pytest.approx([0.9, 0.8, 0.75, 0.6])
    assert out["ntp_prob"].tolist() == pytest.approx([0.1, 0.2, 0.25, 0.4])
    assert out["judgement_combined"].tolist() == [True, False, True, False]  # center's own


def test_dedup_kept_row_not_matching_its_center_raises():
    k = kept()
    k.loc[2, "point_value"] = 99.0
    with pytest.raises(ValueError, match="point_value"):
        dedup("center", kept=k)


def test_dedup_clusters_for_other_rows_raise():
    c = clusters()
    c.loc[1, "measurement_id"] = 9
    with pytest.raises(ValueError, match="does not describe"):
        dedup("center", clusters=c)


def test_dedup_cluster_spanning_documents_raises():
    c = clusters()
    c.loc[4, "cluster_id"] = 1  # row 4 (document b) joins cluster {1,3}
    c.loc[4, "is_center"] = False
    with pytest.raises(ValueError):
        dedup("center", clusters=c)


def test_dedup_unscored_parent_raises():
    with pytest.raises(ValueError, match="no scored datapoint"):
        dedup("center", scored=scored3().iloc[[0, 1, 3]])


def test_dedup_bad_confidence_raises():
    with pytest.raises(ValueError, match="confidence"):
        dedup("median")
