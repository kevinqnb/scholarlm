#!/usr/bin/env python3
"""
Generate a LaTeX table from the per-cell summary statistics produced by
analysis/meta.py.

CSV lives at: results/meta_{dataset}_{extraction_model}_{extraction_date}.csv
Columns: dataset, ecosystem, attribute, setting, unit, n, n_eff, sum_w, mean,
         std, q1, median, q3, whisker_lo, whisker_hi, n_outliers, w_outliers,
         quantiles_clamped

One table per dataset. Rows: ecosystem x attribute. Columns: one per setting,
showing "mean $\\pm$ std". Cells with n < MIN_N are rendered as "--".

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
SETTINGS = ["ground_truth", "extracted", "judge_filtered", "ntp_weighted", "probe_weighted",
            "ntp_threshold", "probe_threshold"]

SETTING_HEADERS = {
    "ground_truth":    "GT",
    "extracted":       "Extracted",
    "judge_filtered":  "Judge-filt.",
    "ntp_weighted":    "NTP-wt.",
    "probe_weighted":  "Probe-wt.",
    "ntp_threshold":   "NTP $\\geq$.75",
    "probe_threshold": "Probe $\\geq$.75",
}

ATTRIBUTE_LABELS = {
    "surface_area": "Surface area", "max_depth": "Max depth",
    "vegetation_cover": "Veg. cover", "ph": "pH",
    "tn": "TN", "tp": "TP", "chla": "Chl-a",
}


def fmt(mean, std, n) -> str:
    if pd.isna(mean) or n < MIN_N:
        return "--"
    if pd.isna(std):
        return rf"{mean:.2g}"
    return rf"{mean:.2g} $\pm$ {std:.2g}"


def build_lookup(df: pd.DataFrame) -> dict:
    """(ecosystem, attribute, setting) -> (mean, std, n)."""
    lookup = {}
    for _, row in df.iterrows():
        lookup[(row["ecosystem"], row["attribute"], row["setting"])] = (
            row["mean"], row["std"], row["n"],
        )
    return lookup


def generate_latex(
    lookup: dict, settings: list, attributes: list, caption: str | None, tex_label: str | None,
) -> str:
    n_settings = len(settings)
    col_spec = "ll" + "c" * n_settings

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header = " & ".join([SETTING_HEADERS[s] for s in settings])
    lines.append(rf"Ecosystem & Attribute & {header} \\")
    lines.append(r"\midrule")

    for eco_idx, ecosystem in enumerate(ECOSYSTEMS):
        lines.append(rf"\multirow{{{len(attributes)}}}{{*}}{{{ecosystem.capitalize()}}}")
        for attribute in attributes:
            cells = []
            for setting in settings:
                mean, std, n = lookup.get((ecosystem, attribute, setting), (float("nan"), float("nan"), 0))
                cells.append(fmt(mean, std, n))
            row_str = " & ".join(cells)
            lines.append(rf"& {ATTRIBUTE_LABELS[attribute]} & {row_str} \\")
        if eco_idx < len(ECOSYSTEMS) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    if caption is not None:
        lines.append(rf"\caption{{{caption}}}")
    if tex_label is not None:
        lines.append(rf"\label{{{tex_label}}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def make_caption(dataset: str, extraction_model: str, extraction_date: str, subset: bool) -> str:
    scope = " (selected attributes)" if subset else ""
    return (
        rf"\textbf{{Per-ecosystem attribute statistics{scope} ({dataset}, \texttt{{{extraction_model}}}).}} "
        rf"Mean $\pm$ standard deviation for each attribute, broken down by ecosystem class and "
        rf"extraction setting. Ground truth (GT), judge-filtered, and NTP/probe threshold "
        rf"($\geq$0.75, hard-gated, unweighted) cells use unweighted statistics; NTP- and "
        rf"probe-weighted cells use reliability-weighted statistics over the full extracted "
        rf"dataset. All settings share the same weighted-quantile estimator (Hazen plotting "
        rf"positions), so quartiles/whiskers are directly comparable across columns. Cells with "
        rf"fewer than {MIN_N} contributing rows are omitted (--)."
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
    required_cols = {"ecosystem", "attribute", "setting", "mean", "std", "n"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Skipping {csv_path}: missing columns {sorted(missing)} "
              f"(stale CSV -- rerun analysis/meta.py to regenerate).")
        return

    subset = attributes is not None
    attrs = attributes if subset else ATTRIBUTES
    settings = [s for s in SETTINGS if s in df["setting"].unique()]
    lookup = build_lookup(df)
    caption = make_caption(dataset, extraction_model, extraction_date, subset)
    label = f"tab:meta-{dataset}-{extraction_model}" + ("-subset" if subset else "")
    latex = generate_latex(lookup, settings, attrs, caption=caption, tex_label=label)

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
