"""Clean an extraction's final.json into postprocessed.json before matching.

Four steps per record, none of which invents a value:
1. Qualifier fill: parse ``value`` into point_value/lower/upper/... only if the
   model left every shape field blank. Partial attempts are left untouched.
2. Unit standardization: rewrite a unit only to a variant that appears in the
   ground truth's own units for that attribute.
3. Provenance: wrap the six PROVENANCE_FIELDS as lists so deduplication can merge them.
4. List expansion: split a row with ``list_values`` into one row per entry, so each
   value can match on its own. Children keep the parent's measurement_id and
   inherit its judge label.

match_cache.py and recovery_validity.py read postprocessed.json when it exists.
Numbers from postprocessed.json are not comparable to ones from final.json.

Usage
-----
    python analysis/postprocessing.py <experiment_id> [...] --ground-truth-file <path>
    python analysis/postprocessing.py --config analysis/analysis-configs/recovery-validity/<id>.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))
sys.path.insert(0, str(_REPO_ROOT))

from scholarlm.utils import parsing
from analysis.common.config import get_ground_truth_path, load_analysis_config
from analysis.common.loaders import load_ground_truth_file
import utils as paths

# Shape fields that must all be blank before qualifiers are filled.
_SHAPE_FIELDS = ("point_value", "lower", "upper", "list_values", "tolerance", "standard_deviation")

# Where in the paper a record came from; one list entry per merged source.
PROVENANCE_FIELDS = ("page_number", "table_number", "row_index", "column_index", "source", "context")


def _is_numeric_string(u: str) -> bool:
    """Whether a string parses as a float.

    Args:
        u: String to test.

    Returns:
        True if ``float(u)`` succeeds.
    """
    try:
        float(u)
    except ValueError:
        return False
    return True


def canonical_units_by_attribute(ground_truth_df: pd.DataFrame) -> dict:
    """Units each attribute actually uses in the ground truth: the standardization targets.

    Read from the ground truth (not attribute_info_dict, which can drift) because
    that is what strict matching compares against. Numeric-looking unit strings are
    ground-truth noise; they are dropped and printed.

    Args:
        ground_truth_df: Ground-truth rows with ``attribute`` and ``units``.

    Returns:
        ``{attribute: frozenset of unit strings}``.
    """
    out: dict = {}
    for attribute, group in ground_truth_df.groupby("attribute"):
        units = set()
        numeric_noise = set()
        for u in group["units"]:
            if u is None or (isinstance(u, float) and pd.isna(u)):
                continue
            (numeric_noise if _is_numeric_string(u) else units).add(u)
        if numeric_noise:
            print(
                f"canonical_units_by_attribute: dropped {len(numeric_noise)} "
                f"numeric-looking 'units' value(s) for attribute {attribute!r} "
                f"(ground-truth data-quality noise, not real units): "
                f"{sorted(numeric_noise)}"
            )
        out[attribute] = frozenset(units)
    return out


def _is_blank(value) -> bool:
    """Whether a field was left unpopulated by the model.

    Args:
        value: Field value.

    Returns:
        True for None, ``[]`` or a whitespace-only string.
    """
    if value is None or value == []:
        return True
    return isinstance(value, str) and value.strip() == ""


def _qualifiers_unfilled(record: dict) -> bool:
    """Whether the model left every shape field and ``qualifiers`` blank.

    Args:
        record: Extraction record.

    Returns:
        True if all ``_SHAPE_FIELDS`` are blank and ``qualifiers`` is None or [].
    """
    if any(not _is_blank(record.get(field)) for field in _SHAPE_FIELDS):
        return False
    qualifiers = record.get("qualifiers")
    return qualifiers is None or qualifiers == []


def postprocess_record(record: dict, *, canonical_units: dict) -> tuple:
    """Fill unattempted qualifier fields and standardize units for one record.

    Args:
        record: Extraction record (not modified).
        canonical_units: Output of ``canonical_units_by_attribute``.

    Returns:
        ``(new_record, changed)``; ``changed`` lists ``"qualifiers"`` and/or ``"units"``.
    """
    record = dict(record)
    changed: list = []

    attribute = record.get("attribute")
    canonical = canonical_units.get(attribute, frozenset())

    value = record.get("value")
    if _qualifiers_unfilled(record) and value is not None:
        text, detected_units = parsing.split_value_and_unit_suffix(
            str(value), record.get("units"), canonical,
        )
        # Numeric (not str) shape fields, matching the ground truth's convention.
        shape = parsing.parse_quantity_shape_numeric(text)
        if shape["qualifiers"] is not None:
            for field in parsing.QUALIFIER_FIELDS:
                record[field] = shape[field]
            changed.append("qualifiers")
            if detected_units is not None and record.get("units") is None:
                record["units"] = detected_units
                changed.append("units")

    new_units, units_changed = parsing.standardize_units(record.get("units"), canonical)
    if units_changed:
        record["units"] = new_units
        if "units" not in changed:
            changed.append("units")

    return record, changed


def normalize_provenance(record: dict) -> tuple:
    """Wrap each non-list PROVENANCE_FIELDS value in a list (absent -> ``[None]``).

    Args:
        record: Extraction record (not modified).

    Returns:
        ``(new_record, wrapped)``; ``wrapped`` names the fields that were wrapped.
    """
    record = dict(record)
    wrapped = []
    for field in PROVENANCE_FIELDS:
        value = record.get(field)
        if isinstance(value, list):
            continue
        record[field] = [value]
        wrapped.append(field)
    return record, wrapped


def provenance_lengths_equal(record: dict) -> bool:
    """Whether all provenance lists have equal length (required to merge the row).

    Args:
        record: Record after ``normalize_provenance``.

    Returns:
        True if every PROVENANCE_FIELDS list has the same length.
    """
    return len({len(record[field]) for field in PROVENANCE_FIELDS}) == 1


def expand_list_values(record: dict) -> list:
    """Split a record with non-empty ``list_values`` into one record per entry.

    Each child gets ``point_value`` = the entry (as float if parseable, else the raw
    string) and ``list_values`` = None; all other fields are copied.

    Args:
        record: Extraction record.

    Returns:
        List of child records, or ``[record]`` if there is nothing to expand.
    """
    list_values = record.get("list_values")
    if not isinstance(list_values, list) or not list_values:
        return [record]
    expanded = []
    for entry in list_values:
        new_record = dict(record)
        try:
            new_record["point_value"] = parsing.to_float(entry)
        except ValueError:
            new_record["point_value"] = entry
        new_record["list_values"] = None
        expanded.append(new_record)
    return expanded


def postprocess_experiment(experiment_id: str, ground_truth_path: Path) -> Path:
    """Write postprocessed.json for one experiment.

    Args:
        experiment_id: Extraction experiment id.
        ground_truth_path: Ground truth defining each attribute's canonical units.

    Returns:
        Path of the written postprocessed.json.

    Raises:
        FileNotFoundError: No final.json.
        AssertionError: A step changed a row's document_id, attribute or
            measurement_id, or rows were lost.
    """
    result_dir = paths.find_result_dir(experiment_id)
    final_path = result_dir / "final.json"
    if not final_path.exists():
        raise FileNotFoundError(f"{experiment_id}: no final.json at {final_path}")

    with open(final_path) as f:
        records = json.load(f)

    ground_truth_df = load_ground_truth_file(ground_truth_path)
    canonical = canonical_units_by_attribute(ground_truth_df)

    n_qualifiers_filled = 0
    n_units_changed = 0
    n_wrapped_by_field = dict.fromkeys(PROVENANCE_FIELDS, 0)
    n_unequal_provenance = 0
    n_rows_list_expanded = 0
    n_extra_rows_from_expansion = 0
    new_records = []
    for record in records:
        new_record, changed = postprocess_record(record, canonical_units=canonical)
        for key in ("document_id", "attribute", "measurement_id"):
            if new_record.get(key) != record.get(key):
                raise AssertionError(
                    f"{experiment_id}: postprocess_record changed {key!r} -- "
                    f"it must only ever touch qualifier/units fields"
                )
        new_record, wrapped = normalize_provenance(new_record)
        for field in wrapped:
            n_wrapped_by_field[field] += 1
        n_unequal_provenance += not provenance_lengths_equal(new_record)

        expanded = expand_list_values(new_record)
        if len(expanded) > 1:
            n_rows_list_expanded += 1
            n_extra_rows_from_expansion += len(expanded) - 1
            for e in expanded:
                for key in ("document_id", "attribute", "measurement_id"):
                    if e.get(key) != record.get(key):
                        raise AssertionError(
                            f"{experiment_id}: expand_list_values changed {key!r} -- "
                            f"it must only ever set point_value"
                        )
        new_records.extend(expanded)
        n_qualifiers_filled += "qualifiers" in changed
        n_units_changed += "units" in changed

    if len(new_records) < len(records):
        raise AssertionError(
            f"{experiment_id}: postprocessing lost rows: {len(records)} -> "
            f"{len(new_records)} -- expand_list_values must never shrink"
        )

    out_path = result_dir / "postprocessed.json"
    with open(out_path, "w") as f:
        json.dump(new_records, f, indent=2)

    print(
        f"{experiment_id}: {len(records)} rows -> {len(new_records)} rows -- "
        f"qualifier fields filled: {n_qualifiers_filled}, units standardized: "
        f"{n_units_changed}, {n_rows_list_expanded} list-value row(s) expanded "
        f"into {n_extra_rows_from_expansion} extra row(s) -> {out_path}"
    )
    print(
        f"{experiment_id}: provenance fields wrapped into lists (rows per field, of {len(records)}): "
        f"{n_wrapped_by_field}; {n_unequal_provenance} row(s) with unequal provenance list lengths "
        f"(deduplication cannot merge those)"
    )
    return out_path


def main() -> None:
    """CLI: postprocess ids given directly or via ``--config`` (exactly one)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_ids", nargs="*", help="Experiment ids to postprocess.")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="analysis-configs/recovery-validity/<id>.yaml providing params.experiment_ids and "
             "params.ground_truth_file -- mutually exclusive with passing "
             "experiment_ids/--ground-truth-file directly.",
    )
    parser.add_argument(
        "--ground-truth-file", type=Path, default=None,
        help="Ground-truth CSV/JSON whose (attribute, units) pairs define the "
             "standardization target. Required when passing experiment_ids "
             "directly; ignored (read from params.ground_truth_file instead) "
             "with --config.",
    )
    args = parser.parse_args()

    if bool(args.config) == bool(args.experiment_ids):
        parser.error("pass experiment_ids directly, or --config, not both/neither")
    if args.config and args.ground_truth_file is not None:
        parser.error(
            "--ground-truth-file is ignored with --config -- set "
            "params.ground_truth_file in the analysis config instead"
        )
    if args.experiment_ids and args.ground_truth_file is None:
        parser.error("--ground-truth-file is required when passing experiment_ids directly")

    if args.config:
        cfg = load_analysis_config(args.config, "recovery-validity")
        experiment_ids = cfg["params"]["experiment_ids"]
        ground_truth_path = get_ground_truth_path(cfg)
    else:
        experiment_ids = args.experiment_ids
        ground_truth_path = args.ground_truth_file
        if not ground_truth_path.is_absolute():
            ground_truth_path = _REPO_ROOT / ground_truth_path
        if not ground_truth_path.exists():
            parser.error(f"--ground-truth-file {ground_truth_path} does not exist")

    for experiment_id in experiment_ids:
        postprocess_experiment(experiment_id, ground_truth_path)


if __name__ == "__main__":
    main()
