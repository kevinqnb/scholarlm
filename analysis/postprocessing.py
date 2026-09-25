"""Postprocess extraction/baseline `final.json` output before matching:
best-effort qualifier-field fill for rows a model never attempted (baselines
struggle with this -- see src/scholarlm/utils/parsing.parse_quantity_shape),
and unit-string standardization against this dataset's own ground-truth unit
vocabulary (see src/scholarlm/utils/parsing.standardize_units).

Writes `postprocessed.json` next to each experiment id's `final.json` --
analysis/match_cache.py and analysis/recovery_validity.py prefer it over
final.json when present, falling back (with a warning) to final.json
otherwise. Deduplication is a separate, later development session -- not
handled here.

A third step, list-value expansion (see expand_list_values), turns any row
whose `list_values` ends up non-empty (whether the model wrote it directly,
or this script's own qualifier fill produced it from a comma-separated
`value`) into one row per list entry, point_value set to that entry and
every other field copied verbatim -- so each reported value gets its own
shot at matching a ground-truth row, instead of the whole list being
unmatchable as a single row. This is the one step that changes row count:
match_cache.py's matching is purely positional within postprocessed.json and
is unaffected (more extraction rows only ever means more match candidates),
but analysis/recovery_validity.py's validity/judge_combine path joins by
`measurement_id`, which list-expanded rows duplicate -- for an id with any
list rows, that join's own row-count/measurement_id checks
(load_validity_labels) will correctly refuse rather than silently
misattribute a judgement to the wrong expanded row. Recovery is unaffected;
validity for such an id needs the judge re-run against the expanded rows.

Both postprocessing steps are best-effort and NEVER invent a value they
aren't confident about:
- Qualifier fields (point_value/lower/upper/list_values/tolerance/
  standard_deviation/qualifiers) are filled only when a row's six shape
  fields are ALL null/absent AND qualifiers is null/[] -- i.e. the model
  never attempted them at all (chatextract, old nuextract/gliner baselines).
  A row where the model wrote something -- even something wrong, like
  GLiNER's stray non-null `upper` on an otherwise-unfilled row -- is left
  completely untouched; arbitrating a partial/garbled model attempt is not
  this script's job.
- Units are only ever replaced with a string that is a genuine member of
  this attribute's own ground-truth unit vocabulary -- built from the
  ground truth file itself (not attribute_info_dict, which can name a unit
  the ground truth never actually uses, or omit one it does -- see
  canonical_units_by_attribute). A unit string this script doesn't
  recognize as a notational variant of something already in the ground
  truth is left exactly as extracted.

This changes which rows survive analysis/match_cache.py's strict
"units"/"point_value" match -- any recovery/validity number computed
against final.json (from before this script existed) is not comparable to
one computed against postprocessed.json for the same experiment id.

Usage
-----
    python analysis/postprocessing.py <experiment_id> [<experiment_id> ...] \\
        --ground-truth-file <path>
    python analysis/postprocessing.py --config analysis/analysis-configs/<id>.yaml
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
from analysis.analysis_config import get_ground_truth_path, load_analysis_config
from analysis.loaders import load_ground_truth_file
import utils as paths

# A row's six shape fields -- `_qualifiers_unfilled` requires every one of
# these null/absent before this script will touch qualifier fields at all.
_SHAPE_FIELDS = ("point_value", "lower", "upper", "list_values", "tolerance", "standard_deviation")


def _is_numeric_string(u: str) -> bool:
    try:
        float(u)
    except ValueError:
        return False
    return True


def canonical_units_by_attribute(ground_truth_df: pd.DataFrame) -> dict:
    """{attribute: frozenset of every distinct non-null `units` value this
    attribute has in the ground truth} -- what analysis/match_cache.py's
    strict "units" match actually compares against, so this is the only
    safe standardization target. Deliberately NOT each dataset's own
    attribute_info_dict: that dict can drift from the ground truth it
    describes (e.g. pond's `attribute_info_dict` lists "x10^-6 km^2" for
    surface_area, but the ground truth itself uses "x10^-6 m^2") -- reading
    the ground truth directly can't have that mismatch.

    A numeric-looking `units` string (e.g. pond's surface_area has ~50 rows
    whose `units` is a bare number like "0.75") is a pre-existing ground-truth
    data-quality artifact, not a real unit -- included in the canonical set,
    split_value_and_unit_suffix's suffix search could "recover" a second
    number out of a value like "7.0 8.00" as if it were this attribute's
    unit. Filtered out here, printed so a real occurrence isn't silently lost.
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
    """None, [] (langextract's empty list_values), or "" (whitespace-only
    string -- observed across several baselines' point_value/lower/upper/
    tolerance/standard_deviation, hundreds of rows repo-wide) all mean "the
    model never actually populated this field", not "the model wrote an
    empty value on purpose"."""
    if value is None or value == []:
        return True
    return isinstance(value, str) and value.strip() == ""


def _qualifiers_unfilled(record: dict) -> bool:
    if any(not _is_blank(record.get(field)) for field in _SHAPE_FIELDS):
        return False
    qualifiers = record.get("qualifiers")
    return qualifiers is None or qualifiers == []


def postprocess_record(record: dict, *, canonical_units: dict) -> tuple:
    """Postprocess one extraction record. Returns (new record, list of field
    names it changed -- "qualifiers" and/or "units")."""
    record = dict(record)
    changed: list = []

    attribute = record.get("attribute")
    canonical = canonical_units.get(attribute, frozenset())

    value = record.get("value")
    if _qualifiers_unfilled(record) and value is not None:
        text, detected_units = parsing.split_value_and_unit_suffix(
            str(value), record.get("units"), canonical,
        )
        # point_value/lower/upper as float, not str: matches pond/nfix ground
        # truth's own point_value convention, and means match_cache.py's
        # numeric_coerce (originally patched on to coerce the real pipeline's
        # own str-typed point_value output at match time) has nothing left to
        # do for rows postprocessing fills -- they're already numeric.
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


def expand_list_values(record: dict) -> list:
    """If record["list_values"] is a non-empty list, return one copy of
    record per entry -- each with point_value set to that entry, converted
    to float (parsing.to_float; matching postprocess_record's own
    point_value-as-float convention) where that entry is a plain float or
    sci-notation string. An entry that isn't (e.g. a list containing the
    rare compact "value(uncertainty)" notation, "2.05(5)") is kept as its
    raw string rather than dropped or raising -- match_cache.py's own
    numeric_coerce will correctly leave it unmatched rather than this
    function guessing at it. Every other field, list_values/qualifiers
    included, is copied unchanged.

    Otherwise returns [record] unchanged -- the overwhelmingly common case.
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
        expanded.append(new_record)
    return expanded


def postprocess_experiment(experiment_id: str, ground_truth_path: Path) -> Path:
    """Build postprocessed.json for one experiment id. Returns its path.

    Raises:
        FileNotFoundError: no final.json for this id.
        AssertionError: postprocess_record or expand_list_values changed a
            row's document_id/attribute, or expand_list_values changed a
            row's measurement_id -- neither should ever happen (see their
            own docstrings); checked here rather than trusted, since a
            match_cache.pkl's cached edges are positions into this same row
            order/identity (see match_cache.py's module docstring).
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
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_ids", nargs="*", help="Experiment ids to postprocess.")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="analysis-configs/<id>.yaml providing params.experiment_ids and "
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
        cfg = load_analysis_config(args.config)
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
