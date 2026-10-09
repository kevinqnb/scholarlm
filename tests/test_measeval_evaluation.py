"""Unit tests for analysis/measeval_evaluation.py's span-recovery heuristic and TSV export.

Rung 1 of the staged-gate ladder (see CLAUDE.md): hand-built and known-answer
fixtures only, no model calls, no invocation of the real measeval-eval.py
subprocess (that's rung 2/3, run directly in-session against a real extraction
result -- see notes/scholarlm/builds/ for this module's build note).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import analysis.measeval_evaluation as measeval_evaluation
from analysis.measeval_evaluation import (
    Coverage,
    _correct_for_calibration,
    build_submission_tsv,
    load_doc_text,
    pick_entity,
    pick_unit,
    resolve_entity_span,
    resolve_quantity_span,
)


def test_locate_span_disambiguates_by_proximity_to_quantity():
    # "STC" appears twice; the correct one is the occurrence nearest the
    # resolved Quantity span (25 C, in the second sentence), not the first.
    text = "STC is a control site. Later, STC was measured at 25 C under load."
    quantity = resolve_quantity_span(text, "25", "C")
    assert quantity is not None
    qstart, qend, qtext = quantity
    assert qtext == "25 C"

    entity = resolve_entity_span(text, "STC", anchor=(qstart, qend))
    assert entity is not None
    estart, eend, etext = entity
    assert etext == "STC"
    # The second "STC", not the first (index 0).
    assert estart == text.index("STC", 1)
    assert estart != 0


def test_locate_span_leftmost_when_quantity_itself_ambiguous():
    text = "25 C was recorded at dawn. 25 C was recorded again at dusk."
    quantity = resolve_quantity_span(text, "25", "C")
    assert quantity is not None
    qstart, _, _ = quantity
    assert qstart == 0  # leftmost tie-break, documented as arbitrary


def test_resolve_quantity_span_falls_back_from_concatenation_to_value_alone():
    # "43.57% ± 4.35%" appears alone in the text; "43.57% ± 4.35% %" (the
    # concatenation with a redundant unit) does not.
    text = "The cells contained 43.57% ± 4.35% of the population."
    quantity = resolve_quantity_span(text, "43.57% ± 4.35%", "%")
    assert quantity is not None
    _, _, qtext = quantity
    assert qtext == "43.57% ± 4.35%"


def test_resolve_quantity_span_none_when_unlocatable():
    text = "No numbers appear in this sentence at all."
    assert resolve_quantity_span(text, "25", "C") is None


def test_resolve_entity_span_none_for_nullish_name():
    text = "25 C was recorded."
    for nullish_name in (None, "None", ""):
        assert resolve_entity_span(text, nullish_name, anchor=(0, 4)) is None


def test_known_answer_against_real_gold_example():
    """data/measeval/README.md's own worked example, S0927024813002961-1085:
    Quantity "25 °C" @ 324-329, MeasuredEntity "STC" @ 180-183."""
    doc_text = load_doc_text("S0927024813002961-1085")
    quantity = resolve_quantity_span(doc_text, "25", "°C")
    assert quantity == (324, 329, "25 °C")
    entity = resolve_entity_span(doc_text, "STC", anchor=(324, 329))
    assert entity == (180, 183, "STC")


def test_build_submission_tsv_text_matches_doc_text_exactly(tmp_path):
    """Vladiate's LengthValidator requires text == doc_text[start:end] exactly."""
    df = pd.DataFrame([
        {"document_id": "S0927024813002961-1085", "name": "STC", "value": "25", "units": "°C"},
    ])
    coverage = build_submission_tsv(df, tmp_path)
    assert coverage.quantity_located == 1
    assert coverage.entity_located == 1

    out_file = tmp_path / "S0927024813002961-1085.tsv"
    assert out_file.exists()
    written = pd.read_csv(out_file, sep="\t", keep_default_na=False)
    doc_text = load_doc_text("S0927024813002961-1085")

    for _, row in written.iterrows():
        start, end = int(row["startOffset"]), int(row["endOffset"])
        assert row["text"] == doc_text[start:end]
        assert len(row["text"]) == end - start

    quantity_row = written[written["annotType"] == "Quantity"].iloc[0]
    assert json.loads(quantity_row["other"]) == {"unit": "°C"}

    entity_row = written[written["annotType"] == "MeasuredEntity"].iloc[0]
    entity_other = json.loads(entity_row["other"])
    assert entity_other == {"HasQuantity": quantity_row["annotId"]}


def test_build_submission_tsv_drops_unlocatable_records_without_guessing(tmp_path):
    df = pd.DataFrame([
        {"document_id": "S0927024813002961-1085", "name": "nonexistent entity phrase",
         "value": "999999", "units": "furlongs"},
    ])
    coverage = build_submission_tsv(df, tmp_path)
    assert coverage.quantity_dropped_unlocatable == 1
    assert coverage.quantity_located == 0
    assert not (tmp_path / "S0927024813002961-1085.tsv").exists()


def test_correct_for_calibration_subtracts_known_placeholder_contribution():
    # Quantity: raw tp=5 (includes the calibration doc's 1 guaranteed match),
    # fp=2, fn=1 -> n_total_raw=8. em_mean=0.6 -> em_sum=4.8; f1_mean=0.7 -> f1_sum=5.6.
    raw = {
        "Quantity": {
            "True positives (matching rows)": 5.0,
            "False positives (submission only)": 2.0,
            "False negatives (gold only)": 1.0,
            "Exact Match Score": 0.6,
            "F1 (Overlap) Score": 0.7,
        },
        "Unit": {
            "True positives (matching rows)": 1.0,
            "False positives (submission only)": 0.0,
            "False negatives (gold only)": 0.0,
            "Exact Match Score": 1.0,
            "F1 (Overlap) Score": 1.0,
        },
        "MeasuredEntity": {
            "True positives (matching rows)": 3.0,
            "False positives (submission only)": 4.0,
            "False negatives (gold only)": 2.0,
            "Exact Match Score": 0.5,
            "F1 (Overlap) Score": 0.55,
        },
    }
    corrected = _correct_for_calibration(raw)

    q = corrected["Quantity"]
    assert q["True positives (matching rows)"] == pytest.approx(4.0)
    assert q["False positives (submission only)"] == pytest.approx(2.0)
    assert q["False negatives (gold only)"] == pytest.approx(1.0)
    assert q["Precision"] == pytest.approx(4 / 6)
    assert q["Recall"] == pytest.approx(4 / 5)
    assert q["Exact Match Score"] == pytest.approx((0.6 * 8 - 1.0) / 7)
    assert q["F1 (Overlap) Score"] == pytest.approx((0.7 * 8 - 1.0) / 7)

    # Unit's only match WAS the calibration doc's -- corrected tp is 0.
    u = corrected["Unit"]
    assert u["True positives (matching rows)"] == pytest.approx(0.0)

    # MeasuredEntity is untouched -- the calibration doc emits no entity row.
    assert corrected["MeasuredEntity"] == raw["MeasuredEntity"]


def test_coverage_as_dict_reports_all_fields():
    cov = Coverage(total_records=5, quantity_located=3, entity_located=2)
    d = cov.as_dict()
    assert d["total_records"] == 5
    assert d["quantity_located"] == 3
    assert d["entity_located"] == 2


# ---------------------------------------------------------------------------
# Span merging: one Quantity row per distinct span
# ---------------------------------------------------------------------------

def test_pick_entity_fuzzy_variants_outvote_exact_majority():
    # "water" is the exact-string majority (2 votes), but the three "soil sample"
    # variants support each other. Summed fuzz.ratio/100 to the other four texts:
    #   water 1.685, soil sample 2.303, soil samples 2.242, the soil sample 2.061.
    texts = ["water", "water", "soil sample", "soil samples", "the soil sample"]
    idx, tied = pick_entity(texts)
    assert texts[idx] == "soil sample"
    assert not tied


def test_pick_entity_exact_majority_wins_without_variants():
    texts = ["STC", "graphene oxide", "graphene oxide"]
    idx, tied = pick_entity(texts)
    assert idx == 1  # first of the two identical winners
    assert not tied


def test_pick_entity_tie_goes_to_earliest_and_is_flagged():
    # Two distinct texts score the same similarity to each other.
    idx, tied = pick_entity(["alpha", "omega"])
    assert idx == 0
    assert tied


def test_pick_entity_identical_texts_are_not_a_tie():
    assert pick_entity(["STC", "STC"]) == (0, False)
    assert pick_entity(["STC"]) == (0, False)


def test_pick_unit_majority_then_earliest():
    assert pick_unit(["g", "kg", "kg"]) == ("kg", False)
    assert pick_unit([None, "g"]) == (None, True)      # null is a vote for "no unit"
    assert pick_unit(["g", None, None]) == (None, False)


_MERGE_TEXT = "The water held 5 g. The soil sample held 5 g, the soil samples held 7 g."


def _merge_df(records):
    return pd.DataFrame([{"document_id": "doc", **r} for r in records])


def test_build_submission_tsv_merges_records_on_one_span(tmp_path, monkeypatch):
    monkeypatch.setattr(measeval_evaluation, "load_doc_text", lambda _id: _MERGE_TEXT)
    df = _merge_df([
        # All three "5 g" records land on the leftmost "5 g" (offsets 15-18).
        {"name": "water", "value": "5", "units": "g"},
        {"name": "soil sample", "value": "5", "units": "g"},
        {"name": "soil sample", "value": "5", "units": "g"},
        # A different value is a different span and keeps its own annotSet.
        {"name": "soil samples", "value": "7", "units": "g"},
    ])
    cov = build_submission_tsv(df, tmp_path)
    t = pd.read_csv(tmp_path / "doc.tsv", sep="\t", keep_default_na=False)

    quantities = t[t.annotType == "Quantity"]
    assert list(quantities.text) == ["5 g", "7 g"]
    assert list(quantities.startOffset) == [15, _MERGE_TEXT.index("7 g")]
    assert list(quantities.annotSet) == [1, 2]

    entities = t[t.annotType == "MeasuredEntity"].set_index("annotSet")
    # Summed fuzz.ratio/100 to the other two: water 0.25 + 0.25 = 0.50;
    # each "soil sample" 0.25 + 1.00 = 1.25 -> "soil sample".
    assert entities.loc[1, "text"] == "soil sample"
    assert entities.loc[1, "startOffset"] == _MERGE_TEXT.index("soil sample")
    assert json.loads(entities.loc[1, "other"]) == {"HasQuantity": "T1-1"}
    assert entities.loc[2, "text"] == "soil samples"

    assert cov.quantity_located == 4
    assert cov.quantity_rows_written == 2
    assert cov.quantity_records_merged == 2
    assert cov.entity_conflict_spans == 1
    assert cov.entity_tie_spans == 0


def test_build_submission_tsv_merges_by_exact_offsets_only(tmp_path, monkeypatch):
    # The "g" record matches "5 g" (15-18); the "kg" records find no "5 kg"/"5kg" and
    # fall back to bare "5" (15-16). Overlapping but unequal spans are not merged.
    monkeypatch.setattr(measeval_evaluation, "load_doc_text", lambda _id: _MERGE_TEXT)
    df = _merge_df([
        {"name": "water", "value": "5", "units": "g"},
        {"name": "water", "value": "5", "units": "kg"},
        {"name": "water", "value": "5", "units": "kg"},
    ])
    cov = build_submission_tsv(df, tmp_path)
    t = pd.read_csv(tmp_path / "doc.tsv", sep="\t", keep_default_na=False)
    q = t[t.annotType == "Quantity"]
    assert list(zip(q.startOffset, q.endOffset)) == [(15, 18), (15, 16)]
    assert [json.loads(o) for o in q.other] == [{"unit": "g"}, {"unit": "kg"}]
    assert cov.quantity_rows_written == 2
    assert cov.quantity_records_merged == 1
    assert cov.unit_conflict_spans == 0


def test_build_submission_tsv_unit_conflict_on_one_span(tmp_path, monkeypatch):
    # Bare-value fallback for both: "5 mg" and "5 kg" don't occur, so both land on "5".
    monkeypatch.setattr(measeval_evaluation, "load_doc_text", lambda _id: "Sample A weighed 5.")
    df = _merge_df([
        {"name": "Sample A", "value": "5", "units": "mg"},
        {"name": "Sample A", "value": "5", "units": "kg"},
        {"name": "Sample A", "value": "5", "units": "kg"},
    ])
    cov = build_submission_tsv(df, tmp_path)
    t = pd.read_csv(tmp_path / "doc.tsv", sep="\t", keep_default_na=False)
    q = t[t.annotType == "Quantity"]
    assert len(q) == 1
    assert json.loads(q.iloc[0]["other"]) == {"unit": "kg"}
    assert cov.unit_conflict_spans == 1
    assert cov.unit_tie_spans == 0


def test_build_submission_tsv_unlocated_entities_do_not_vote(tmp_path, monkeypatch):
    monkeypatch.setattr(measeval_evaluation, "load_doc_text", lambda _id: _MERGE_TEXT)
    df = _merge_df([
        {"name": "not in the text", "value": "5", "units": "g"},
        {"name": "not in the text", "value": "5", "units": "g"},
        {"name": None, "value": "5", "units": "g"},
        {"name": "water", "value": "5", "units": "g"},
        # A span whose only record has no located entity gets a Quantity row only.
        {"name": "also not in the text", "value": "7", "units": "g"},
    ])
    cov = build_submission_tsv(df, tmp_path)
    t = pd.read_csv(tmp_path / "doc.tsv", sep="\t", keep_default_na=False)
    entities = t[t.annotType == "MeasuredEntity"]
    assert list(entities.text) == ["water"]
    assert list(entities.annotSet) == [1]
    assert (t.annotType == "Quantity").sum() == 2
    assert cov.entity_dropped_unlocatable == 3
    assert cov.entity_dropped_no_name == 1
    assert cov.entity_located == 1
    assert cov.entity_none_spans == 1
