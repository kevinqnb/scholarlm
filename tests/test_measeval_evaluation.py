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

from analysis.measeval_evaluation import (
    Coverage,
    _correct_for_calibration,
    build_submission_tsv,
    load_doc_text,
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
