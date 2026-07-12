#!/usr/bin/env python3
"""
Generate a baseline-comparison LaTeX table from baselines_{dataset}.csv files.

Rows are MeasurementLM backbones and external baselines (NuExtract,
ChatExtract, GLiNER); columns are datasets. Each cell shows
"F1 (recovery, validity)" rounded to 2 decimal places, where F1 is the
harmonic mean of recovery and validity. Missing data (NaN) is rendered as
"--". Bold marks the best F1 per column.

When run with no arguments, reads from results/baselines/ and writes
baselines_table.tex there. Pass explicit CSV paths as arguments (one per
dataset, in DATASETS order) to print the table to stdout instead.
"""

import sys
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")

DATASETS = [
    ("pond", r"\pond"),
    ("nfix", r"\nfix"),
    ("supermat", r"\supermat"),
]

ROWS = [
    (r"\texttt{llama-3.1-8b}", "llama-3.1-8b"),
    (r"\texttt{gemma-3-27b}", "gemma-3-27b"),
    (r"\texttt{gpt-oss-120b}", "gpt-oss-120b"),
    ("NuExtract", "nuextract-2.0-8b"),
    ("ChatExtract", "chatextract-gemma-3-27b"),
    ("GLiNER", "gliner-large-v1"),
]


def load_baselines(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _collect_values(by_dataset):
    """Return (f1, recovery, validity) each shaped (n_rows, n_cols)."""
    n_cols = len(DATASETS)
    f1   = [[float("nan")] * n_cols for _ in ROWS]
    recov = [[float("nan")] * n_cols for _ in ROWS]
    valid = [[float("nan")] * n_cols for _ in ROWS]

    for row_idx, (_, model_key) in enumerate(ROWS):
        for col_idx, (dataset, _) in enumerate(DATASETS):
            by_model = by_dataset.get(dataset, {})
            if model_key in by_model:
                r = by_model[model_key]
                f1[row_idx][col_idx] = r.get("f1", float("nan"))
                recov[row_idx][col_idx] = r.get("recovery", float("nan"))
                valid[row_idx][col_idx] = r.get("validity", float("nan"))

    return f1, recov, valid


def _generate_table(by_dataset, caption: str, label: str) -> str:
    f1, recov, valid = _collect_values(by_dataset)
    n_cols = len(DATASETS)

    best = [
        max((f1[r][c] for r in range(len(ROWS)) if not pd.isna(f1[r][c])), default=None)
        for c in range(n_cols)
    ]

    def fmt_cell(f1_v, recov_v, valid_v, col_idx) -> str:
        if pd.isna(f1_v) or pd.isna(recov_v) or pd.isna(valid_v):
            return "--"
        cell = f"{f1_v:.2f} ({recov_v:.2f}, {valid_v:.2f})"
        if best[col_idx] is not None and f1_v == best[col_idx]:
            cell = r"\textbf{" + cell + "}"
        return cell

    dataset_headers = " & ".join(label_ for _, label_ in DATASETS)
    col_spec = "c" * n_cols
    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"  \small")
    lines.append(r"  \setlength{\tabcolsep}{4pt}")
    lines.append(r"  \centering")
    lines.append(rf"  \begin{{tabular}}{{l {col_spec}}}")
    lines.append(r"    \toprule")
    lines.append(f"    & {dataset_headers} \\\\")
    lines.append(r"    \midrule")

    for row_idx, (row_label, _) in enumerate(ROWS):
        cells = [fmt_cell(f1[row_idx][c], recov[row_idx][c], valid[row_idx][c], c) for c in range(n_cols)]
        lines.append(f"    {row_label} & {' & '.join(cells)} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(f"  {caption}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def generate_table(csv_paths: dict) -> str:
    """csv_paths: dict mapping dataset name -> CSV path."""
    by_dataset = {}
    for dataset, path in csv_paths.items():
        df = load_baselines(path)
        by_dataset[dataset] = {row["model"]: row for _, row in df.iterrows()}

    return _generate_table(
        by_dataset,
        caption=(
            r"\caption{\textbf{Baseline Comparison.} F1 (harmonic mean of recovery "
            r"and validity rate) with recovery and validity shown in parentheses, "
            r"for each MeasurementLM backbone and external baseline (NuExtract, "
            r"ChatExtract, GLiNER) across the \pond, \nfix, and \supermat datasets. "
            r"Bold marks the best F1 per column.}"
        ),
        label="tab:baselines",
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_paths = {dataset: sys.argv[i + 1] for i, (dataset, _) in enumerate(DATASETS) if i + 1 < len(sys.argv)}
        print(generate_table(csv_paths))
    else:
        baselines_dir = RESULTS_DIR / "baselines"
        csv_paths = {dataset: str(baselines_dir / f"baselines_{dataset}.csv") for dataset, _ in DATASETS}
        table_tex = generate_table(csv_paths)
        table_path = baselines_dir / "baselines_table.tex"
        table_path.write_text(table_tex + "\n")
        print(f"Wrote {table_path}")
