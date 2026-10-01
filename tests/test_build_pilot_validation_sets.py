"""Model-free unit tests for data/validation/build_pilot_validation_sets.py (top-up phase)."""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "data" / "validation" / "build_pilot_validation_sets.py"
_spec = importlib.util.spec_from_file_location("build_pilot_validation_sets", _SCRIPT)
bp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bp  # @dataclass needs the module registered
_spec.loader.exec_module(bp)


def _rows(doc: str, n: int, start: int) -> list[dict]:
    return [{"document_id": doc, "measurement_id": start + i, "page_number": [0]} for i in range(n)]


def test_top_up_reaches_target_and_skips_taken_and_non_test_papers():
    working = _rows("taken", 5, 0) + _rows("train", 5, 100)
    for i in range(6):
        working += _rows(f"p{i}", 3, 1000 + 10 * i)
    test_ids = {"taken", "p0", "p1", "p2", "p3", "p4", "p5"}
    docs, rows = bp._top_up(working, taken_docs={"taken"}, test_ids=test_ids, total=bp.TARGET_MEASUREMENTS - 4)
    assert len(rows) >= 4 and len(docs) == 2  # 3 + 3 rows, last paper overshoots
    assert {r["document_id"] for r in rows} == set(docs)
    assert not set(docs) & {"taken", "train"}
    # paper by paper, sorted by measurement_id within each paper
    for d in docs:
        mids = [r["measurement_id"] for r in rows if r["document_id"] == d]
        assert mids == sorted(mids)


def test_top_up_is_deterministic():
    working = [r for i in range(10) for r in _rows(f"p{i}", 4, 100 * i)]
    test_ids = {f"p{i}" for i in range(10)}
    a = bp._top_up(working, taken_docs=set(), test_ids=test_ids, total=90)
    b = bp._top_up(working, taken_docs=set(), test_ids=test_ids, total=90)
    assert a == b
    candidates = sorted(test_ids)
    random.Random(bp.SEED).shuffle(candidates)
    assert a[0][0] == candidates[0]  # first drawn paper is the first of the seeded shuffle


def test_top_up_fails_loudly_when_candidates_run_out():
    working = _rows("p0", 2, 0)
    with pytest.raises(AssertionError, match="exhausting"):
        bp._top_up(working, taken_docs=set(), test_ids={"p0"}, total=0)


def test_normalize_page_number_shapes():
    f = lambda pn: bp._normalize_page_number(pn, method_key="m", measurement_id=1)
    assert f(None) == [bp.NO_PROVENANCE_PAGE]
    assert f(3) == [3]
    assert f([1, 2]) == [1, 2]
    with pytest.raises(AssertionError):
        f(-1)
    with pytest.raises(AssertionError):
        f([])
