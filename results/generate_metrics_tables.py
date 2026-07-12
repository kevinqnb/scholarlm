#!/usr/bin/env python3
"""
Generate LaTeX calibration-error tables from the combined metrics CSV produced
by analysis/calibration.py.

CSV lives at: results/metrics_{extraction_model}_{probe_type}.csv
Columns: Dataset type, Judge model, Train dataset, Test dataset, Type, ...,
         ECE, ECE_lo, ECE_hi, ECE_em, ECE_em_lo, ECE_em_hi,
         RMSCE_db, RMSCE_db_lo, RMSCE_db_hi

Generates one table per (judge_model, extraction_model, probe_type,
dataset_pair, metric_variant) combo. Rows: Within / Cross-domain x
{<ds A> NTP, <ds A> Probe, <ds B> NTP, <ds B> Probe}. Columns: Synthetic |
Extracted. Cells show "value $\\pm$ half-width" from the bootstrap CI computed
in analysis/calibration.py.

Three metric variants are generated per dataset pair:
  - ece      — L1 ECE, equal-width bins (plug-in)
  - ece_em   — L1 ECE, adaptive equal-mass bins (plug-in)
  - rmsce_db — debiased L2 RMS calibration error, equal-mass bins

Usage:
  python results/generate_metrics_tables.py                                        # walks results/
  python results/generate_metrics_tables.py results/metrics_gemma-3-27b_head.csv  # single csv, stdout
"""

import sys
import itertools
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")

# Fixed dataset order — shared with analysis/calibration.py's DATASET_PAIRS so
# that pair subfolder names (e.g. "pond_nfix") line up with the figures.
DATASETS = ["pond", "nfix", "supermat"]
DATASET_PAIRS = list(itertools.combinations(DATASETS, 2))

_DS_LABELS = {"pond": "PLW", "nfix": "NF", "supermat": "SM"}

# (metric_key, value_col, lo_col, hi_col, short_name, description)
_METRIC_VARIANTS = [
    ("ece", "ECE", "ECE_lo", "ECE_hi",
     "ECE", "Expected Calibration Error (equal-width bins, plug-in)"),
    ("ece_em", "ECE_em", "ECE_em_lo", "ECE_em_hi",
     "Adaptive ECE", "Expected Calibration Error (adaptive equal-mass bins, plug-in)"),
    ("rmsce_db", "RMSCE_db", "RMSCE_db_lo", "RMSCE_db_hi",
     "Debiased RMSCE", "Debiased RMS Calibration Error (Kumar, Liang \\& Ma 2019, equal-mass bins)"),
]

_GROUPS = ["Within", "Cross"]
_GROUP_SIZE = 4


def fmt(val, lo, hi) -> str:
    if val is None or pd.isna(val):
        return "--"
    if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return f"{val:.2f}"
    half_width = (hi - lo) / 2.0
    return rf"{val:.2f} $\pm$ {half_width:.2f}"


def build_lookup(df: pd.DataFrame, train_ds: str, value_col: str, lo_col: str, hi_col: str) -> dict:
    """Return lookup: (test_setting, type) -> (value, ci_low, ci_high) for a given training dataset.

    test_setting is constructed as "Syn. {test_ds}" or "Real {test_ds}".
    """
    sub = df[df["Train dataset"] == train_ds]
    lookup = {}
    for _, row in sub.iterrows():
        prefix = "Syn." if row["Dataset type"] == "syn" else "Real"
        test_setting = f"{prefix} {row['Test dataset']}"
        lookup[(test_setting, row["Type"])] = (row[value_col], row[lo_col], row[hi_col])
    return lookup


def row_defs(ds_a: str, ds_b: str) -> list:
    label_a, label_b = _DS_LABELS[ds_a], _DS_LABELS[ds_b]
    return [
        ("Within", ds_a, ds_a, "NTP",   f"{label_a} NTP"),
        ("Within", ds_a, ds_a, "Probe", f"{label_a} Probe"),
        ("Within", ds_b, ds_b, "NTP",   f"{label_b} NTP"),
        ("Within", ds_b, ds_b, "Probe", f"{label_b} Probe"),
        ("Cross",  ds_a, ds_b, "NTP",   f"{label_a} NTP"),
        ("Cross",  ds_a, ds_b, "Probe", f"{label_a} Probe"),
        ("Cross",  ds_b, ds_a, "NTP",   f"{label_b} NTP"),
        ("Cross",  ds_b, ds_a, "Probe", f"{label_b} Probe"),
    ]


def generate_latex(
    ds_a: str,
    ds_b: str,
    lookup_a: dict,
    lookup_b: dict,
    caption: str | None = None,
    tex_label: str | None = None,
) -> str:
    train_lookup = {ds_a: lookup_a, ds_b: lookup_b}
    row_definitions = row_defs(ds_a, ds_b)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{ll cc}")
    lines.append(r"\toprule")
    lines.append(r"& & Synthetic & Extracted \\")
    lines.append(r"\midrule")

    for gi, group in enumerate(_GROUPS):
        group_rows = [
            (train_ds, test_ds, kind, display)
            for grp, train_ds, test_ds, kind, display in row_definitions
            if grp == group
        ]
        lines.append(rf"\multirow{{{_GROUP_SIZE}}}{{*}}{{\textit{{{group}}}}}")
        for train_ds, test_ds, kind, display in group_rows:
            lkp = train_lookup[train_ds]
            syn_val, syn_lo, syn_hi = lkp.get((f"Syn. {test_ds}", kind), (None, None, None))
            real_val, real_lo, real_hi = lkp.get((f"Real {test_ds}", kind), (None, None, None))
            syn_str = fmt(syn_val, syn_lo, syn_hi)
            real_str = fmt(real_val, real_lo, real_hi)
            lines.append(rf"& {display} & {syn_str} & {real_str} \\")
        if gi < len(_GROUPS) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    if caption is not None:
        lines.append(rf"\caption{{{caption}}}")
    if tex_label is not None:
        lines.append(rf"\label{{{tex_label}}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def make_caption(judge_model: str, extraction_model: str, ds_a: str, ds_b: str, description: str) -> str:
    return (
        rf"\textbf{{{description} (\texttt{{{judge_model}}}).}} "
        rf"Probe and NTP validation probabilities evaluated within and cross-domain "
        rf"between {_DS_LABELS[ds_a]} and {_DS_LABELS[ds_b]} for synthetic and real, "
        rf"extraction test settings. Intervals are 95\% bootstrap confidence intervals. "
        rf"Extracted measurements collected from \texttt{{{extraction_model}}}."
    )


def parse_csv_stem(stem: str) -> tuple[str, str] | None:
    """Parse extraction_model and probe_type from 'metrics_{extraction_model}_{probe_type}'."""
    if not stem.startswith("metrics_"):
        return None
    remainder = stem[len("metrics_"):]
    last_us = remainder.rfind("_")
    if last_us == -1:
        return None
    return remainder[:last_us], remainder[last_us + 1:]


def process_csv(csv_path: Path, print_to_stdout: bool = False) -> None:
    parsed = parse_csv_stem(csv_path.stem)
    if parsed is None:
        print(f"Skipping {csv_path}: cannot parse extraction_model/probe_type from filename.")
        return
    extraction_model, probe_type = parsed

    df = pd.read_csv(csv_path)

    required_cols = {col for _, col, lo_col, hi_col, _, _ in _METRIC_VARIANTS for col in (col, lo_col, hi_col)}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Skipping {csv_path}: missing columns {sorted(missing)} "
              f"(stale CSV — rerun analysis/calibration.py to regenerate).")
        return

    for judge_model in sorted(df["Judge model"].unique()):
        judge_df = df[df["Judge model"] == judge_model]
        available = set(judge_df["Train dataset"].unique())

        for ds_a, ds_b in DATASET_PAIRS:
            if ds_a not in available or ds_b not in available:
                continue

            for metric_key, value_col, lo_col, hi_col, _short_name, description in _METRIC_VARIANTS:
                lookup_a = build_lookup(judge_df, ds_a, value_col, lo_col, hi_col)
                lookup_b = build_lookup(judge_df, ds_b, value_col, lo_col, hi_col)

                caption = make_caption(judge_model, extraction_model, ds_a, ds_b, description)
                pair_name = f"{ds_a}_{ds_b}"
                label = f"tab:{judge_model}-{extraction_model}-{pair_name}-{metric_key}"
                latex = generate_latex(
                    ds_a, ds_b, lookup_a, lookup_b,
                    caption=caption, tex_label=label,
                )

                if print_to_stdout:
                    print(f"% ── {judge_model} / {extraction_model} / {probe_type} / {pair_name} / {metric_key} ──")
                    print(latex)
                    print()
                else:
                    out_dir = RESULTS_DIR / judge_model / extraction_model / probe_type / pair_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"metrics_table_{metric_key}.tex"
                    out_path.write_text(latex + "\n")
                    print(f"Wrote {out_path}")


def main() -> None:
    for csv_path in sorted(RESULTS_DIR.glob("metrics_*.csv")):
        process_csv(csv_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_csv(Path(sys.argv[1]), print_to_stdout=True)
    else:
        main()
