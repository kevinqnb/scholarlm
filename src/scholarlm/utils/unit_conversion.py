"""Unit conversion for extracted measurement values.

Applies per-attribute multiplicative factors to convert extracted values
(which may be in various reported units) to a single standard unit per
attribute, matching the units used in the ground truth dataset.

Typical usage
-------------
    from scholarlm.utils.unit_conversion import apply_unit_conversion

    unit_conversion_table = {
        "max_depth":  {"m": 1.0, "cm": 0.01, "feet": 0.3048, "km": 1000.0},
        "surface_area": {"m^2": 1.0, "ha": 1e4, "km^2": 1e6},
        "ph": {},  # dimensionless — no conversion
    }

    ext_df = apply_unit_conversion(ext_df, unit_conversion_table)
    # ext_df now has a "converted_value" column with values in standard units
"""
from __future__ import annotations

import pandas as pd


def apply_unit_conversion(
    df: pd.DataFrame,
    unit_conversion_table: dict[str, dict[str, float]],
    value_col: str = "value",
    unit_col: str = "units",
    attribute_col: str = "attribute",
    out_col: str = "converted_value",
) -> pd.DataFrame:
    """Apply per-attribute unit conversions and return a new column of standardised values.

    For each row, looks up ``df[attribute_col]`` in ``unit_conversion_table`` to
    obtain a ``{unit: multiplier}`` mapping, then multiplies ``df[value_col]`` by the
    factor for that row's unit.  Rows whose attribute or unit is absent from the table
    are assigned a factor of 1.0 (i.e. passed through unchanged).  Non-numeric values
    become ``NaN`` after conversion.

    Args:
        df: Extraction DataFrame.  Not modified in place.
        unit_conversion_table: ``{attribute: {unit: factor}}``.  An empty inner dict
            (e.g. for a dimensionless attribute like pH) means no conversion is applied.
        value_col: Column holding the raw extracted value (string or numeric).
        unit_col: Column holding the unit string.
        attribute_col: Column holding the attribute name.
        out_col: Name of the output column added to the returned DataFrame.

    Returns:
        Copy of ``df`` with ``out_col`` appended.
    """
    df = df.copy()
    numeric_values = pd.to_numeric(df[value_col], errors="coerce")

    factors = pd.Series(1.0, index=df.index)
    for attribute, unit_map in unit_conversion_table.items():
        attr_mask = df[attribute_col] == attribute
        for unit, factor in unit_map.items():
            unit_mask = df[unit_col] == unit
            factors.loc[attr_mask & unit_mask] = factor

    df[out_col] = numeric_values * factors
    return df
