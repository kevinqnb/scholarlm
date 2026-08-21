#!/usr/bin/env python3
"""
Generate a LaTeX table from the per-cell summary statistics produced by
analysis/meta_updated.py.

CSV lives at: results/meta_{dataset}_{extraction_model}_{extraction_date}.csv
Columns: dataset, ecosystem, attribute, setting, unit, n, n_eff, sum_w, mean, std,
         q1, median, q3

One table per dataset x ecosystem. Rows: setting (ground truth / extracted /
judge-filtered / NTP- and probe-confidence weighted). Columns: attribute, showing
"median (Q1, Q3)" -- weighted-Hazen quantiles (analysis/meta_updated.py's
weighted_hazen_quantile), identical to the classic unweighted Hazen (1914)
plotting-position quantile when every row's weight is 1. The NTP-/probe-weighted
rows use ALL rows for that (ecosystem, attribute) cell, weighted continuously by
that method's confidence -- never a hard >= threshold filter (see
analysis/meta_updated.py's module docstring). Cells with n < MIN_N are rendered
as "--".

Usage:
  python results/generate_meta_table.py                                # walks results/, writes .tex files
  python results/generate_meta_table.py results/meta_pond_gemma-3-27b_2026_05_05.csv  # single csv, stdout
  python results/generate_meta_table.py results/meta_pond_gemma-3-27b_2026_05_05.csv --attributes tn tp chla ph
"""
import argparse
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")

MIN_N = 5  # cells with fewer contributing rows than this are rendered as "--"

ECOSYSTEMS = ["pond", "lake", "wetland"]
ATTRIBUTES = ["surface_area", "max_depth", "vegetation_cover", "ph", "tn", "tp", "chla"]
SETTINGS = [
    "ground_truth", "judge_filtered", "extracted",
    "ntp_weighted", "probe_weighted",
]

SETTING_HEADERS = {
    "ground_truth":   "Ground truth",
    "judge_filtered": "Judge-filtered",
    "extracted":      "Unfiltered",
    "ntp_weighted":   "NTP-weighted",
    "probe_weighted": "Probe-weighted",
}

ATTRIBUTE_LABELS = {
    "surface_area": "Surface area", "max_depth": "Max depth",
    "vegetation_cover": "Veg. cover", "ph": "pH",
    "tn": "TN", "tp": "TP", "chla": "Chl-a",
}


def fmt(median, q1, q3, n) -> str:
    if pd.isna(median) or n < MIN_N:
        return "--"
    if pd.isna(q1) or pd.isna(q3):
        return rf"{median:.2g}"
    return rf"{median:.2g} ({q1:.2g}, {q3:.2g})"


def build_lookup(df: pd.DataFrame) -> dict:
    """(ecosystem, attribute, setting) -> (median, q1, q3, n)."""
    lookup = {}
    for _, row in df.iterrows():
        lookup[(row["ecosystem"], row["attribute"], row["setting"])] = (
            row["median"], row["q1"], row["q3"], row["n"],
        )
    return lookup


def generate_latex(
    lookup: dict, ecosystem: str, settings: list, attributes: list,
    caption: str | None, tex_label: str | None,
) -> str:
    n_attrs = len(attributes)
    col_spec = "l" + "c" * n_attrs

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header = " & ".join([ATTRIBUTE_LABELS[a] for a in attributes])
    lines.append(rf"Setting & {header} \\")
    lines.append(r"\midrule")

    for setting in settings:
        cells = []
        for attribute in attributes:
            median, q1, q3, n = lookup.get(
                (ecosystem, attribute, setting), (float("nan"), float("nan"), float("nan"), 0)
            )
            cells.append(fmt(median, q1, q3, n))
        row_str = " & ".join(cells)
        lines.append(rf"{SETTING_HEADERS[setting]} & {row_str} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    if caption is not None:
        lines.append(rf"\caption{{{caption}}}")
    if tex_label is not None:
        lines.append(rf"\label{{{tex_label}}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def make_caption(
    dataset: str, extraction_model: str, extraction_date: str, ecosystem: str, subset: bool,
) -> str:
    scope = " (selected attributes)" if subset else ""
    return (
        rf"\textbf{{{ecosystem.capitalize()} attribute statistics{scope} ({dataset}, \texttt{{{extraction_model}}}).}} "
        rf"Median (Q1, Q3) for each attribute, broken down by setting. All quantiles use "
        rf"weighted-Hazen plotting positions, identical to the classic unweighted Hazen (1914) "
        rf"convention when every row's weight is 1. The NTP- and probe-weighted rows include "
        rf"every extracted row for that cell, weighted continuously by that method's confidence "
        rf"-- never a hard threshold filter. Cells with fewer than {MIN_N} contributing rows are "
        rf"omitted (--)."
    )


def parse_csv_stem(stem: str) -> tuple | None:
    """Parse (dataset, extraction_model, extraction_date) from 'meta_{dataset}_{model}_{date}'."""
    if not stem.startswith("meta_"):
        return None
    parts = stem[len("meta_"):].split("_")
    if len(parts) < 3:
        return None
    # date is always the last 3 underscore-joined tokens (YYYY_mm_dd)
    date = "_".join(parts[-3:])
    remainder = parts[:-3]
    if not remainder:
        return None
    dataset = remainder[0]
    model = "_".join(remainder[1:]) if len(remainder) > 1 else ""
    return dataset, model, date


def process_csv(csv_path: Path, attributes: list | None = None, print_to_stdout: bool = False) -> None:
    parsed = parse_csv_stem(csv_path.stem)
    if parsed is None:
        print(f"Skipping {csv_path}: cannot parse dataset/model/date from filename.")
        return
    dataset, extraction_model, extraction_date = parsed

    df = pd.read_csv(csv_path)
    required_cols = {"ecosystem", "attribute", "setting", "median", "q1", "q3", "n"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Skipping {csv_path}: missing columns {sorted(missing)} "
              f"(stale CSV -- rerun analysis/meta_updated.py to regenerate).")
        return

    subset = attributes is not None
    attrs = attributes if subset else ATTRIBUTES
    settings = [s for s in SETTINGS if s in df["setting"].unique()]
    lookup = build_lookup(df)

    tables = []
    for ecosystem in ECOSYSTEMS:
        caption = make_caption(dataset, extraction_model, extraction_date, ecosystem, subset)
        label = f"tab:meta-{dataset}-{extraction_model}-{ecosystem}" + ("-subset" if subset else "")
        tables.append(generate_latex(lookup, ecosystem, settings, attrs, caption=caption, tex_label=label))
    latex = "\n\n".join(tables)

    if print_to_stdout:
        print(f"% -- {dataset} / {extraction_model} / {extraction_date} --")
        print(latex)
        print()
    else:
        out_dir = RESULTS_DIR / "meta" / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_subset" if subset else ""
        out_path = out_dir / f"meta_table_{extraction_model}_{extraction_date}{suffix}.tex"
        out_path.write_text(latex + "\n")
        print(f"Wrote {out_path}")


def main() -> None:
    for csv_path in sorted(RESULTS_DIR.glob("meta_*.csv")):
        process_csv(csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", nargs="?", type=Path, default=None,
                         help="Single CSV to process (prints to stdout). Omit to walk results/ and write .tex files.")
    parser.add_argument("--attributes", nargs="+", default=None, choices=ATTRIBUTES,
                         help="Restrict the table to this attribute subset (default: all 7).")
    args = parser.parse_args()

    if args.csv_path is not None:
        process_csv(args.csv_path, attributes=args.attributes, print_to_stdout=True)
    else:
        main()
