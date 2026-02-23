"""Tests for scholarlm.utils subpackage — no GPU or API keys required."""

import json
import pytest
import numpy as np
import pandas as pd

from scholarlm.utils.fs import get_filenames_in_directory, get_foldernames_in_directory
from scholarlm.utils.tables import add_row_names, table_extract
from scholarlm.utils.data import (
    _try_coerce_value,
    load_and_process_results,
    match_datasets,
    matching_precision_recall,
)


# ── fs ────────────────────────────────────────────────────────────────────────


class TestGetFilenamesInDirectory:
    def test_returns_filenames(self, tmp_path):
        (tmp_path / "a.txt").touch()
        (tmp_path / "b.txt").touch()
        assert sorted(get_filenames_in_directory(str(tmp_path))) == ["a.txt", "b.txt"]

    def test_ignores_specified_files(self, tmp_path):
        (tmp_path / "a.txt").touch()
        (tmp_path / "b.txt").touch()
        assert get_filenames_in_directory(str(tmp_path), ignore=["a.txt"]) == ["b.txt"]

    def test_excludes_subdirectories(self, tmp_path):
        (tmp_path / "file.txt").touch()
        (tmp_path / "subdir").mkdir()
        assert get_filenames_in_directory(str(tmp_path)) == ["file.txt"]

    def test_missing_directory_returns_error_string(self):
        result = get_filenames_in_directory("/nonexistent/path")
        assert isinstance(result, str) and "Error" in result


class TestGetFoldernamesInDirectory:
    def test_returns_foldernames(self, tmp_path):
        (tmp_path / "dir1").mkdir()
        (tmp_path / "dir2").mkdir()
        assert sorted(get_foldernames_in_directory(str(tmp_path))) == ["dir1", "dir2"]

    def test_ignores_specified_folders(self, tmp_path):
        (tmp_path / "dir1").mkdir()
        (tmp_path / "dir2").mkdir()
        assert get_foldernames_in_directory(str(tmp_path), ignore=["dir1"]) == ["dir2"]

    def test_excludes_files(self, tmp_path):
        (tmp_path / "file.txt").touch()
        (tmp_path / "subdir").mkdir()
        assert get_foldernames_in_directory(str(tmp_path)) == ["subdir"]

    def test_missing_directory_returns_error_string(self):
        result = get_foldernames_in_directory("/nonexistent/path")
        assert isinstance(result, str) and "Error" in result


# ── tables ────────────────────────────────────────────────────────────────────


class TestAddRowNames:
    def test_adds_row_indices_to_data_rows(self):
        html = (
            "<table>"
            "<tr><th>Name</th><th>Value</th></tr>"
            "<tr><td>alpha</td><td>1</td></tr>"
            "<tr><td>beta</td><td>2</td></tr>"
            "</table>"
        )
        result = add_row_names(html)
        assert "row 1" in result
        assert "row 2" in result

    def test_header_row_not_numbered(self):
        html = (
            "<table>"
            "<tr><th>Name</th><th>Value</th></tr>"
            "<tr><td>alpha</td><td>1</td></tr>"
            "</table>"
        )
        result = add_row_names(html)
        assert "row 0" not in result

    def test_no_table_returns_input_unchanged(self):
        html = "<p>no table here</p>"
        assert add_row_names(html) == html


class TestTableExtract:
    def test_extracts_correct_value(self):
        df = pd.DataFrame({"Name": ["Alice", "Bob"], "Score": [90, 85]})
        assert table_extract(df, row_indices=["Alice"], column_indices=["Score"]) == 90

    def test_second_row(self):
        df = pd.DataFrame({"Name": ["Alice", "Bob"], "Score": [90, 85]})
        assert table_extract(df, row_indices=["Bob"], column_indices=["Score"]) == 85


# ── data ──────────────────────────────────────────────────────────────────────


class TestTryCoerceValue:
    def test_float_from_string(self):
        assert _try_coerce_value("3.14", float) == pytest.approx(3.14)

    def test_int_from_float_string(self):
        assert _try_coerce_value("42.0", int) == 42

    def test_strips_commas(self):
        assert _try_coerce_value("1,234", float) == pytest.approx(1234.0)

    def test_handles_unicode_minus(self):
        assert _try_coerce_value("−5.0", float) == pytest.approx(-5.0)

    def test_returns_original_on_failure(self):
        assert _try_coerce_value("not_a_number", float) == "not_a_number"

    def test_none_returns_none(self):
        assert _try_coerce_value(None, float) is None

    def test_already_correct_type_unchanged(self):
        assert _try_coerce_value(3.14, float) == pytest.approx(3.14)


class TestLoadAndProcessResults:
    def test_basic_loading(self, tmp_path):
        records = [
            {"attribute": "ph", "value": 7.5, "units": None},
            {"attribute": "depth", "value": 2.0, "units": "m"},
        ]
        p = tmp_path / "results.json"
        p.write_text(json.dumps(records))

        df = load_and_process_results(str(p), unit_conversion_table={"depth": {"m": 1.0}})
        assert len(df) == 2
        assert "processed_value" in df.columns

    def test_unit_conversion(self, tmp_path):
        records = [{"attribute": "depth", "value": 100.0, "units": "cm"}]
        p = tmp_path / "results.json"
        p.write_text(json.dumps(records))

        df = load_and_process_results(str(p), unit_conversion_table={"depth": {"cm": 0.01}})
        assert df["processed_value"].iloc[0] == pytest.approx(1.0)

    def test_drop_attrs(self, tmp_path):
        records = [
            {"attribute": "ph", "value": 7.0, "units": None},
            {"attribute": "depth", "value": 2.0, "units": "m"},
        ]
        p = tmp_path / "results.json"
        p.write_text(json.dumps(records))

        df = load_and_process_results(str(p), unit_conversion_table={}, drop_attrs=["ph"])
        assert len(df) == 1
        assert df["attribute"].iloc[0] == "depth"

    def test_drops_null_values(self, tmp_path):
        records = [
            {"attribute": "ph", "value": None, "units": None},
            {"attribute": "ph", "value": 7.0, "units": None},
        ]
        p = tmp_path / "results.json"
        p.write_text(json.dumps(records))

        df = load_and_process_results(str(p), unit_conversion_table={})
        assert len(df) == 1


class TestMatchDatasets:
    @pytest.fixture
    def sample_dfs(self):
        left = pd.DataFrame({
            "title": ["paper_a", "paper_b", "paper_c"],
            "measurement": ["ph", "depth", "ph"],
            "value": [7.0, 2.0, 6.5],
        })
        right = pd.DataFrame({
            "title": ["paper_a", "paper_b", "paper_d"],
            "measurement": ["ph", "depth", "ph"],
            "value": [7.0, 2.0, 8.0],
        })
        return left, right

    def test_finds_exact_matches(self, sample_dfs):
        left, right = sample_dfs
        matching, edges, weights = match_datasets(
            left, right,
            strict_matching={"title": "title", "measurement": "measurement", "value": "value"},
        )
        assert len(matching) == 2

    def test_matching_is_one_to_one(self, sample_dfs):
        left, right = sample_dfs
        matching, _, _ = match_datasets(
            left, right,
            strict_matching={"title": "title", "measurement": "measurement", "value": "value"},
        )
        left_indices = [m[0] for m in matching]
        right_indices = [m[1] for m in matching]
        assert len(left_indices) == len(set(left_indices))
        assert len(right_indices) == len(set(right_indices))

    def test_no_matches_returns_empty(self):
        left = pd.DataFrame({"title": ["x"], "value": [1.0]})
        right = pd.DataFrame({"title": ["y"], "value": [2.0]})
        matching, edges, weights = match_datasets(left, right, strict_matching={"title": "title"})
        assert matching == []

    def test_empty_strict_matching_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError):
            match_datasets(df, df, strict_matching={})

    def test_missing_column_raises(self):
        left = pd.DataFrame({"a": [1]})
        right = pd.DataFrame({"b": [1]})
        with pytest.raises(KeyError):
            match_datasets(left, right, strict_matching={"nonexistent": "b"})

    def test_fuzzy_matching_scores_edges(self):
        left = pd.DataFrame({"title": ["paper_a"], "name": ["Alpha Lake"]})
        right = pd.DataFrame({"title": ["paper_a"], "name": ["Alpha Pond"]})
        matching, edges, weights = match_datasets(
            left, right,
            strict_matching={"title": "title"},
            fuzzy_matching={"name": "name"},
        )
        assert len(matching) == 1
        assert 0.0 < weights[0] < 1.0


class TestMatchingPrecisionRecall:
    def test_perfect_match(self):
        df = pd.DataFrame({"title": ["a", "b"], "value": [1.0, 2.0]})
        recall, precision = matching_precision_recall(
            df, df.copy(),
            strict_matching={"title": "title", "value": "value"},
        )
        assert recall == pytest.approx(1.0)
        assert precision == pytest.approx(1.0)

    def test_no_match(self):
        left = pd.DataFrame({"title": ["a"], "value": [1.0]})
        right = pd.DataFrame({"title": ["b"], "value": [2.0]})
        recall, precision = matching_precision_recall(
            left, right, strict_matching={"title": "title"},
        )
        assert recall == pytest.approx(0.0)
        assert precision == pytest.approx(0.0)

    def test_partial_recall(self):
        left = pd.DataFrame({"title": ["a", "b"], "value": [1.0, 2.0]})
        right = pd.DataFrame({"title": ["a"], "value": [1.0]})
        recall, precision = matching_precision_recall(
            left, right,
            strict_matching={"title": "title", "value": "value"},
        )
        assert recall == pytest.approx(0.5)
        assert precision == pytest.approx(1.0)
