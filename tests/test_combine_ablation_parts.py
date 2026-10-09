"""Unit tests for experiments/combine_ablation_parts.combine_records.

Hand-built fixture: 5 papers a..e (full order), split into part 0 = {b, e} and
part 1 = {a, c, d}. Inside each part, ``doc_<i>_...`` ids are indexed by position
*in that part*, as a real split run would write them. Each (entity_id, attribute)
is one _deduplicate() group; there are no event fields in the fixture.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
from combine_ablation_parts import combine_records

FULL = ["a", "b", "c", "d", "e"]


def _rec(doc, local_i, j, mid, source="text"):
    return {"document_id": doc, "entity_id": f"doc_{local_i}_entity_{j}", "attribute": "x",
            "value": f"{doc}{j}{source[0]}", "source": [source], "measurement_id": mid}


def _parts():
    return [
        # each part: groups ordered by (block of first record, document), as fit() writes it
        {"papers": ["b", "e"], "records": [_rec("b", 0, 0, 0), _rec("b", 0, 1, 1), _rec("e", 1, 0, 2, "table")]},
        {"papers": ["a", "c", "d"], "records": [_rec("a", 0, 0, 0), _rec("d", 2, 0, 1), _rec("c", 1, 0, 2, "table"),
                                                _rec("d", 2, 1, 3, "table")]},
    ]


def _combine(parts):
    return combine_records(FULL, parts, ())


def test_merge_orders_groups_by_block_then_paper_remaps_entity_ids_and_renumbers():
    out = _combine(_parts())
    assert [(r["document_id"], r["entity_id"], r["source"][0], r["measurement_id"]) for r in out] == [
        ("a", "doc_0_entity_0", "text", 0),
        ("b", "doc_1_entity_0", "text", 1),
        ("b", "doc_1_entity_1", "text", 2),
        ("d", "doc_3_entity_0", "text", 3),
        ("c", "doc_2_entity_0", "table", 4),
        ("d", "doc_3_entity_1", "table", 5),
        ("e", "doc_4_entity_0", "table", 6),
    ]
    assert [r["value"] for r in out] == ["a0t", "b0t", "b1t", "d0t", "c0t", "d1t", "e0t"]  # payload untouched


def test_a_groups_later_table_record_stays_inside_its_group():
    # _deduplicate keeps a group contiguous: b's entity-0 group = text record then a table record.
    p = _parts()
    p[0]["records"].insert(1, _rec("b", 0, 0, 9, "table"))  # same (entity_id, attribute) as the text record before it
    out = _combine(p)
    assert [(r["entity_id"], r["source"][0]) for r in out][1:4] == [
        ("doc_1_entity_0", "text"), ("doc_1_entity_0", "table"), ("doc_1_entity_1", "text")]


def test_unknown_source_raises():
    p = _parts()
    p[0]["records"][0]["source"] = ["pdf"]
    with pytest.raises(ValueError, match="expected entries text/table"):
        _combine(p)


def test_part_not_in_fit_order_raises():
    p = _parts()
    p[1]["records"][1], p[1]["records"][2] = p[1]["records"][2], p[1]["records"][1]  # table group before text group
    with pytest.raises(ValueError, match="not ordered"):
        _combine(p)


def test_reopened_group_raises():
    p = _parts()
    p[0]["records"].append(_rec("b", 0, 0, 3, "table"))  # b's entity-0 group again, after e's
    with pytest.raises(ValueError, match="reopens an earlier group"):
        _combine(p)


def test_overlapping_or_incomplete_parts_raise():
    p = _parts()
    p[1]["papers"] = ["a", "b", "c", "d"]  # b claimed twice, e missing
    with pytest.raises(ValueError, match="overlap"):
        _combine(p)
    p = _parts()
    p[1]["papers"] = ["a", "c"]  # d, missing from the union
    with pytest.raises(ValueError, match="do not cover"):
        _combine(p)


def test_entity_id_pointing_at_the_wrong_local_document_raises():
    p = _parts()
    p[0]["records"][2]["entity_id"] = "doc_0_entity_0"  # an "e" record claiming local doc 0 (= b)
    with pytest.raises(ValueError, match="points at local document 0"):
        _combine(p)


def test_record_outside_its_part_raises():
    p = _parts()
    p[0]["records"][0]["document_id"] = "a"
    with pytest.raises(ValueError, match="outside the part"):
        _combine(p)
