"""
Probe dataset creation for the nfix (aquatic dinitrogen fixation) dataset.

Samples approximately 50% of ground-truth data by paper, labelling those rows
as valid (label="valid"), then creates two invalid counterparts per valid record
(2:1 invalid:valid ratio).

Subset 1 — swap invalids (one per valid record):
  change_value  -- swap value with a same-paper record measuring a different
                   entity or attribute; falls back to cross-paper if needed.
                   Ensures the swapped value is numerically distinct.
  change_entity -- swap judge_entity_fields values with a same-paper record
                   whose entity differs; falls back to cross-paper if needed.
  change_units  -- replace the units field with a different valid unit for the
                   same attribute type (value is kept unchanged, making the
                   stated unit incorrect relative to the document).

Subset 2 — noise invalids (~half the valid set):
  noise_value   -- add Gaussian noise (std = 30% of |value|) to the numeric
                   value, preserving original decimal precision.
  noise_entity  -- replace the entity name with a fabricated site name drawn
                   from a fixed list; used when value is non-numeric or randomly
                   chosen alongside noise_value.

Subset 3 — OCR-table invalids (~half the valid set):
  table_value   -- replace the value with a numeric cell drawn at random from
                   an HTML table in the paper's OCR text file.  Falls back to
                   noise_value / noise_entity when no suitable table value is
                   found.

nfix has only one generic attribute in the GT ("nfix_rate"); the specific
sub-type (nfix_rate_mass / nfix_rate_areal / nfix_rate_volumetric) is inferred
from the units string.  Attribute swapping is therefore not applicable.

The nfix paper_filter (exclude figure/supplement/archive/author papers) is
applied when building document_id values, matching the extraction pipeline.

Output
------
    data/nfix/probe_dataset.json   (final.json-compatible format, plus
                                   "label" and "modification_type" fields)

Usage
-----
Run from the repo root:

    python data/nfix/create_probe_dataset.py [--seed N]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent        # data/nfix/
REPO_ROOT = BASE.parent.parent      # repo root
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from configs.nfix import CONFIG

_JUDGE_ENTITY_FIELDS: list[str] = CONFIG.judge_entity_fields  # ["name", "abbreviations", "site_type"]
_ATTR_DICT: dict = CONFIG.attribute_info_dict
_PAPER_FILTER = CONFIG.paper_filter
_OCR_DIR = BASE / "ocr_output_raw"
_GT_FILE = BASE / "ground_truth.csv"
_META_FILE = BASE / "directory.json"
_OUTPUT_FILE = BASE / "probe_dataset.json"

DEFAULT_SEED = 42

_MADE_UP_NAMES: list[str] = [
    "Seahaven Bay", "Rockpoint Estuary", "Saltmarsh Station A", "Tidal Flat B",
    "Coastal Site Bravo", "Southern Inlet C", "Mangrove Point D",
    "Intertidal Zone E", "Seagrass Bed F", "Mudflat Station G",
    "Coastal Wetland H", "Harbour Inlet I", "Peninsula Tidal Zone",
    "Sheltered Bay J", "Exposed Shore K", "Estuarine Channel L",
    "Fringe Mangrove M", "Upper Saltmarsh N", "Low Intertidal O",
    "High Intertidal P", "Supralittoral Q", "Backbarrier Lagoon R",
    "Open Coast S", "Reef Flat T", "Headland Site U",
    "Bayside Flat V", "Eastern Saltflat W", "Western Cove X",
    "Northern Transect Y", "Southern Platform Z", "Central Mat Site",
    "Outer Estuary AA", "Inner Lagoon BB", "Transition Zone CC",
    "Carbonate Platform DD", "Hypersaline Pond EE",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean(val) -> object:
    """Convert float NaN / pandas NA to None for JSON serialization."""
    if val is None:
        return None
    try:
        if math.isnan(float(val)):
            return None
    except (TypeError, ValueError):
        pass
    return val


def _entity_key(record: dict) -> tuple:
    """Tuple identifying the entity by judge_entity_fields (for equality checks)."""
    return tuple(record.get(f) for f in _JUDGE_ENTITY_FIELDS)


def _extract_table_values(ocr_path: Path) -> list[str]:
    """Return plain numeric cell values from all HTML tables in an OCR file."""
    text = ocr_path.read_text(encoding="utf-8", errors="replace")
    values: list[str] = []
    for block in re.findall(r"<table[^>]*>(.*?)</table>", text, re.DOTALL):
        for raw in re.findall(r"<td[^>]*>(.*?)</td>", block, re.DOTALL):
            cell = re.sub(r"<[^>]+>", "", raw).strip()
            cell = re.sub(r"[−–−]", "-", cell)
            try:
                float(cell)
                values.append(cell)
            except ValueError:
                pass
    return values


def _extract_text_values(ocr_path: Path) -> list[str]:
    """Return unique numeric tokens found anywhere in an OCR text file.

    Used as a fallback when a paper has no HTML tables with plain numbers.
    """
    text = ocr_path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[−–−]", "-", text)
    seen: set[str] = set()
    values: list[str] = []
    for token in re.split(r"\s+", text):
        tok = token.strip(".,;:()[]{}\"'`")
        if tok and tok not in seen:
            try:
                float(tok)
                values.append(tok)
                seen.add(tok)
            except ValueError:
                pass
    return values


def _format_noisy_value(original_str: str, new_val: float) -> str:
    """Format new_val with the same decimal precision as original_str."""
    if "." in original_str:
        n_dec = len(original_str.split(".")[-1])
        return f"{new_val:.{n_dec}f}"
    return str(int(round(new_val)))


def _format_date(row: pd.Series) -> str | None:
    """Format nfix date from year / month / day GT columns."""
    year = _clean(row.get("year"))
    month = _clean(row.get("month"))
    day = _clean(row.get("day"))
    if year is None:
        return None
    y = str(int(year))
    if month is None:
        return y
    m = f"{int(month):02d}"
    if day is None:
        return f"{m}-{y}"
    d = f"{int(day):02d}"
    return f"{d}-{m}-{y}"


def _classify_attribute(units_str: str | None) -> str:
    """Infer nfix_rate_* sub-type from a units string using token-level matching.

    Returns "nfix_rate" (unclassified) if the units don't match any known pattern.
    """
    if not units_str:
        return "nfix_rate"
    u = units_str.lower()
    padded = f" {u} "
    if " g-1 " in padded or " kg-1 " in padded:
        return "nfix_rate_mass"
    if " m-2 " in padded or " cm-2 " in padded:
        return "nfix_rate_areal"
    if any(f" {tok} " in padded for tok in ["l-1", "m-3", "ml-1", "cm-3"]):
        return "nfix_rate_volumetric"
    return "nfix_rate"


def _unit_candidates(attribute: str, current_units: str) -> list[str]:
    """Return alternative canonical units for attribute, excluding current_units."""
    all_units = _ATTR_DICT.get(attribute, {}).get("units", [])
    return [u for u in all_units if u.lower() != current_units.lower()]


# ---------------------------------------------------------------------------
# Document-ID and metadata map (with paper_filter applied)
# ---------------------------------------------------------------------------


def build_document_id_map() -> tuple[dict[str, int], dict[str, dict]]:
    """Return (paper_code → document_id, paper_code → metadata) mappings.

    Applies the nfix paper_filter before computing document_ids, exactly
    mirroring the ordering used by the extraction pipeline.
    """
    with open(_META_FILE) as f:
        paper_info: dict[str, dict] = json.load(f)

    # Registered IDs that pass the paper_filter
    registered_ids = {
        k for k, v in paper_info.items() if _PAPER_FILTER(v)
    }

    ocr_files = sorted(
        f for f in os.listdir(_OCR_DIR)
        if f.endswith(".txt") and f not in {".DS_Store", ".gitkeep"}
    )
    # Apply filter (same logic as load_papers in run_extraction.py)
    ocr_files = [f for f in ocr_files if f.removesuffix(".txt") in registered_ids]

    code_to_docid: dict[str, int] = {}
    code_to_meta: dict[str, dict] = {}
    for i, fname in enumerate(ocr_files):
        code = fname.removesuffix(".txt")
        code_to_docid[code] = i
        meta = {k: _clean(v) for k, v in paper_info.get(code, {}).items()}
        meta["paper_code"] = code
        code_to_meta[code] = meta

    return code_to_docid, code_to_meta


# ---------------------------------------------------------------------------
# Ground truth → record conversion
# ---------------------------------------------------------------------------


def build_gt_records(
    df: pd.DataFrame,
    code_to_docid: dict[str, int],
    code_to_meta: dict[str, dict],
) -> list[dict]:
    """Convert GT DataFrame rows to final.json-compatible dicts.

    Rows whose reference_id has no OCR file (or was filtered out) are skipped.
    Internal bookkeeping fields are prefixed with '_' and stripped before output.
    """
    records: list[dict] = []
    for i, row in df.iterrows():
        ref_id = str(row["reference_id"])
        if ref_id not in code_to_docid:
            continue

        meta = code_to_meta[ref_id]
        record: dict = dict(meta)  # metadata + paper_code

        record["document_id"] = code_to_docid[ref_id]

        # Entity schema fields (ObservationSchema: name, abbreviations, site_type, latitude, longitude)
        record["name"] = _clean(row.get("site_name"))
        record["abbreviations"] = None   # not in GT; always None for synthetic data
        record["site_type"] = _clean(row.get("habitat"))
        record["latitude"] = _clean(row.get("latitude"))
        record["longitude"] = _clean(row.get("longitude"))
        record["entity_id"] = f"gt_{i}"

        # Measurement event fields
        # (MeasurementEventSchema: date, nfix_method, substrate_type, sample_depth, additional_details)
        record["date"] = _format_date(row)
        record["nfix_method"] = None    # not in GT
        record["substrate_type"] = _clean(row.get("substrate"))
        record["sample_depth"] = None   # not in GT
        record["additional_details"] = _clean(row.get("substrate_details"))

        # Measurement fields — infer specific attribute type from units
        units_raw = _clean(row.get("units"))
        inferred_attr = _classify_attribute(units_raw)
        record["attribute"] = inferred_attr
        record["attribute_terms"] = []
        record["value"] = str(row["value"])
        record["units"] = units_raw

        # Provenance (unknown for synthetic records; judge falls back to full doc)
        record["page_number"] = None
        record["table_number"] = None
        record["row_index"] = None
        record["column_index"] = None
        record["source"] = "text"
        record["context"] = None

        # Provenance tracking for train/test exclusion
        record["gt_row_index"] = i
        record["donor_gt_row_index"] = None  # set by create_invalid_record for invalid entries

        # Internal bookkeeping (stripped before output)
        record["_orig_idx"] = i
        record["_paper_code"] = ref_id

        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Valid-set sampling
# ---------------------------------------------------------------------------


def sample_valid_set(records: list[dict], rng: random.Random) -> list[dict]:
    """Greedy paper sampling: add papers one at a time until ≥50% of rows selected."""
    by_paper: dict[str, list[dict]] = {}
    for r in records:
        by_paper.setdefault(r["_paper_code"], []).append(r)

    papers = list(by_paper.keys())
    rng.shuffle(papers)

    threshold = len(records) / 2
    selected: list[dict] = []
    for paper in papers:
        selected.extend(by_paper[paper])
        if len(selected) >= threshold:
            break

    return selected


# ---------------------------------------------------------------------------
# Invalid record construction
# ---------------------------------------------------------------------------


def _get_candidates(
    record: dict,
    pool: list[dict],
    pred,
) -> list[dict]:
    """Return records from pool (excluding record itself) that satisfy pred."""
    orig_idx = record["_orig_idx"]
    return [r for r in pool if r["_orig_idx"] != orig_idx and pred(r)]


def create_invalid_record(
    record: dict,
    paper_pool: list[dict],
    global_pool: list[dict],
    rng: random.Random,
) -> dict | None:
    """Create an invalid counterpart of record.

    Strategies: change_value, change_entity, change_units.
    Tries same-paper candidates first; falls back to global pool if needed.
    Returns None only if no candidates exist anywhere.
    """
    ek = _entity_key(record)
    attr = record["attribute"]
    val = record["value"]
    units = record.get("units")

    def val_pred(r):
        return r["value"] != val and (_entity_key(r) != ek or r["attribute"] != attr)

    def ent_pred(r):
        return _entity_key(r) != ek

    def resolve(pred) -> list[dict]:
        cands = _get_candidates(record, paper_pool, pred)
        if not cands:
            cands = _get_candidates(record, global_pool, pred)
        return cands

    val_cands = resolve(val_pred)
    ent_cands = resolve(ent_pred)
    unit_cands = _unit_candidates(attr, units) if units else []

    options: list[str] = []
    if val_cands:
        options.append("change_value")
    if ent_cands:
        options.append("change_entity")
    if unit_cands:
        options.append("change_units")

    if not options:
        return None

    mod_type = rng.choice(options)
    invalid = dict(record)
    donor_idx: int | None = None

    if mod_type == "change_value":
        src = rng.choice(val_cands)
        invalid["value"] = src["value"]
        donor_idx = src["_orig_idx"]
    elif mod_type == "change_entity":
        src = rng.choice(ent_cands)
        for f in _JUDGE_ENTITY_FIELDS:
            invalid[f] = src.get(f)
        donor_idx = src["_orig_idx"]
    elif mod_type == "change_units":
        invalid["units"] = rng.choice(unit_cands)
        # no donor row — unit was chosen from canonical list, not another GT record

    invalid["donor_gt_row_index"] = donor_idx
    invalid["label"] = "invalid"
    invalid["modification_type"] = mod_type
    return invalid


# ---------------------------------------------------------------------------
# Noise and OCR-table invalid record construction
# ---------------------------------------------------------------------------


def create_noise_record(record: dict, rng: random.Random) -> dict:
    """Create an invalid record by perturbing with Gaussian noise (value) or a
    fabricated entity name, chosen uniformly at random when both are available."""
    invalid = dict(record)

    try:
        val = float(record["value"])
        can_noise_val = True
    except (TypeError, ValueError):
        can_noise_val = False

    options = ["noise_value", "noise_entity"] if can_noise_val else ["noise_entity"]
    mod_type = rng.choice(options)

    if mod_type == "noise_value":
        scale = max(abs(val) * 0.3, 0.1)
        for _ in range(100):
            new_val = val + rng.gauss(0, scale)
            if abs(new_val - val) > 1e-10:
                break
        invalid["value"] = _format_noisy_value(record["value"], new_val)
    else:
        invalid["name"] = rng.choice(_MADE_UP_NAMES)
        invalid["abbreviations"] = None

    invalid["donor_gt_row_index"] = None
    invalid["label"] = "invalid"
    invalid["modification_type"] = mod_type
    return invalid


def create_table_record(
    record: dict,
    ocr_values: list[str],
    rng: random.Random,
) -> dict | None:
    """Create an invalid record by replacing value with one from an OCR table cell.

    Returns None if no suitable numeric table value exists (caller should fall back).
    """
    candidates = [v for v in ocr_values if v != record["value"]]
    if not candidates:
        return None

    invalid = dict(record)
    invalid["value"] = rng.choice(candidates)
    invalid["donor_gt_row_index"] = None
    invalid["label"] = "invalid"
    invalid["modification_type"] = "table_value"
    return invalid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    print(f"Seed: {args.seed}")

    # Build lookup maps
    code_to_docid, code_to_meta = build_document_id_map()
    print(f"Document map: {len(code_to_docid)} papers pass filter and have OCR files")

    # Load and convert GT
    df = pd.read_csv(_GT_FILE)
    print(f"Loaded {len(df):,} GT rows from {_GT_FILE.name}")

    all_records = build_gt_records(df, code_to_docid, code_to_meta)
    print(f"Converted {len(all_records):,} records (with matching OCR files)")

    # Attribute distribution after inference
    attr_counts: dict[str, int] = {}
    for r in all_records:
        attr_counts[r["attribute"]] = attr_counts.get(r["attribute"], 0) + 1
    print("Inferred attribute distribution:")
    for k, v in sorted(attr_counts.items()):
        print(f"  {k}: {v}")

    # Sample valid set (X_V)
    xv = sample_valid_set(all_records, rng)
    print(f"\nSampled {len(xv):,} valid records ({len(xv) / len(all_records) * 100:.1f}% of total)")

    # Build same-paper index over the full record pool
    by_paper: dict[str, list[dict]] = {}
    for r in all_records:
        by_paper.setdefault(r["_paper_code"], []).append(r)

    # Create invalid counterparts (X_I)
    xi: list[dict] = []
    skipped = 0
    for record in xv:
        paper_recs = by_paper[record["_paper_code"]]
        inv = create_invalid_record(record, paper_recs, all_records, rng)
        if inv is None:
            skipped += 1
            continue
        xi.append(inv)

    if skipped:
        print(f"Warning: {skipped} valid record(s) had no invalid candidate — dropped for balance.")
        xv = xv[: len(xi)]

    # Label valid records
    for r in xv:
        r["label"] = "valid"
        r["modification_type"] = None

    # --- Additional diversity subsets (each ~half the valid set) ---
    xv_shuffled = list(xv)
    rng.shuffle(xv_shuffled)
    mid = len(xv_shuffled) // 2
    xv_noise = xv_shuffled[:mid]
    xv_table = xv_shuffled[mid:]

    # Subset 2: noise invalids
    xi_noise: list[dict] = [create_noise_record(r, rng) for r in xv_noise]
    print(f"Created {len(xi_noise):,} noise invalid records")

    # Subset 3: OCR-table invalids (text fallback if no table numbers; noise last resort)
    ocr_table_cache: dict[str, list[str]] = {}
    ocr_text_cache: dict[str, list[str]] = {}
    for code in {r["_paper_code"] for r in xv_table}:
        path = _OCR_DIR / f"{code}.txt"
        ocr_table_cache[code] = _extract_table_values(path)
        ocr_text_cache[code] = _extract_text_values(path)

    xi_table: list[dict] = []
    n_noise_fallback = 0
    for r in xv_table:
        code = r["_paper_code"]
        vals = ocr_table_cache.get(code, []) or ocr_text_cache.get(code, [])
        rec = create_table_record(r, vals, rng)
        if rec is None:
            rec = create_noise_record(r, rng)
            n_noise_fallback += 1
        xi_table.append(rec)
    if n_noise_fallback:
        print(f"  ({n_noise_fallback} table records fell back to noise — no numeric values in OCR)")
    print(f"Created {len(xi_table):,} table/fallback invalid records")

    # Combine and assign sequential measurement_ids; strip internal fields
    combined = xv + xi + xi_noise + xi_table
    output: list[dict] = []
    for mid, r in enumerate(combined):
        rec = {k: v for k, v in r.items() if not k.startswith("_")}
        rec["measurement_id"] = mid
        output.append(rec)

    # Summary
    mod_counts: dict[str | None, int] = {}
    for r in output:
        mt = r.get("modification_type")
        mod_counts[mt] = mod_counts.get(mt, 0) + 1
    print("\nLabel / modification-type distribution:")
    for k, v in sorted(mod_counts.items(), key=lambda x: str(x[0])):
        print(f"  {str(k):20s} {v:5d}")

    # Save
    _OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(output):,} records → {_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
