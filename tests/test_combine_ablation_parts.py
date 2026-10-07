"""Unit tests for experiments/combine_ablation_parts.combine_records.

Hand-built fixture: 5 papers a..e (full order), split into part 0 = {b, e} and
part 1 = {a, c, d}. Inside each part, ``doc_<i>_...`` ids are indexed by position
*in that part*, as a real split run would write them.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
from combine_ablation_parts import combine_records

FULL = ["a", "b", "c", "d", "e"]


def _rec(doc, local_i, j, mid, source="text"):
    return {"document_id": doc, "entity_id": f"doc_{local_i}_entity_{j}", "value": f"{doc}{j}",
            "source": [source], "measurement_id": mid}


def _parts():
    return [
        # each part: text block then table block, as fit() writes it
        {"papers": ["b", "e"], "records": [_rec("b", 0, 0, 0), _rec("b", 0, 1, 1), _rec("e", 1, 0, 2, "table")]},
        {"papers": ["a", "c", "d"], "records": [_rec("a", 0, 0, 0), _rec("d", 2, 0, 1), _rec("c", 1, 0, 2, "table"),
                                                _rec("d", 2, 1, 3, "table")]},
    ]


def test_merge_orders_text_block_then_table_block_by_paper_remaps_entity_ids_and_renumbers():
    out = combine_records(FULL, _parts())
    assert [(r["document_id"], r["entity_id"], r["source"][0], r["measurement_id"]) for r in out] == [
        ("a", "doc_0_entity_0", "text", 0),
        ("b", "doc_1_entity_0", "text", 1),
        ("b", "doc_1_entity_1", "text", 2),
        ("d", "doc_3_entity_0", "text", 3),
        ("c", "doc_2_entity_0", "table", 4),
        ("d", "doc_3_entity_1", "table", 5),
        ("e", "doc_4_entity_0", "table", 6),
    ]
    assert [r["value"] for r in out] == ["a0", "b0", "b1", "d0", "c0", "d1", "e0"]  # payload untouched


def test_mixed_or_unknown_source_raises():
    p = _parts()
    p[0]["records"][0]["source"] = ["text", "table"]
    with pytest.raises(ValueError, match="all-text or all-table"):
        combine_records(FULL, p)


def test_overlapping_or_incomplete_parts_raise():
    p = _parts()
    p[1]["papers"] = ["a", "b", "c", "d"]  # b claimed twice, e missing
    with pytest.raises(ValueError, match="overlap"):
        combine_records(FULL, p)
    p = _parts()
    p[1]["papers"] = ["a", "c"]  # d, missing from the union
    with pytest.raises(ValueError, match="do not cover"):
        combine_records(FULL, p)


def test_entity_id_pointing_at_the_wrong_local_document_raises():
    p = _parts()
    p[0]["records"][2]["entity_id"] = "doc_0_entity_0"  # an "e" record claiming local doc 0 (= b)
    with pytest.raises(ValueError, match="points at local document 0"):
        combine_records(FULL, p)


def test_record_outside_its_part_raises():
    p = _parts()
    p[0]["records"][0]["document_id"] = "a"
    with pytest.raises(ValueError, match="outside the part"):
        combine_records(FULL, p)
