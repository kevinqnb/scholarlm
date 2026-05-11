#!/usr/bin/env python3
"""
Generate a unified ablation LaTeX table from ablation_pond.csv and ablation_nfix.csv.

Each CSV has columns:
  dataset, model,
  baseline_recovery, baseline_hallucination,
  ablation_1_recovery, ablation_1_hallucination,
  ...
  ablation_6_recovery, ablation_6_hallucination

The unified table has two dataset column groups (PLW, NF), each with 3 model sub-columns.
Each cell shows "recovery (hallucination)" rounded to 2 decimal places.
Missing data (NaN) is rendered as "--".

When run with no arguments, reads from results/ablation/ and writes ablation_table.tex there.
Pass explicit CSV paths as arguments to print LaTeX to stdout instead.
"""

import sys
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")


def load_ablation(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def generate_latex(pond_csv: str, nfix_csv: str) -> str:
    df_pond = load_ablation(pond_csv)
    df_nfix = load_ablation(nfix_csv)

    models = ["llama-3.1-8b", "gemma-3-27b", "gpt-oss-120b"]
    model_display = {
        "llama-3.1-8b": r"\texttt{llama-3.1-8b}",
        "gemma-3-27b": r"\texttt{gemma-3-27b}",
        "gpt-oss-120b": r"\texttt{gpt-oss-120b}",
    }

    # Row labels and corresponding CSV column prefixes
    rows = [
        ("Base", "baseline"),
        ("$A_1$", "ablation_1"),
        ("$A_2$", "ablation_2"),
        ("$A_3$", "ablation_3"),
        ("$A_4$", "ablation_4"),
        ("$A_5$", "ablation_5"),
        ("$A_6$", "ablation_6"),
    ]

    # Index each dataframe by model for easy lookup
    pond_by_model = {row["model"]: row for _, row in df_pond.iterrows()}
    nfix_by_model = {row["model"]: row for _, row in df_nfix.iterrows()}

    # First pass: collect all raw values (rows × cols) to find per-column bests
    n_cols = len(models) * 2
    rec_vals = [[float("nan")] * n_cols for _ in rows]
    hal_vals = [[float("nan")] * n_cols for _ in rows]

    for row_idx, (_, col_prefix) in enumerate(rows):
        col_idx = 0
        for df_by_model in [pond_by_model, nfix_by_model]:
            for model in models:
                if model in df_by_model:
                    row_data = df_by_model[model]
                    rec_vals[row_idx][col_idx] = row_data.get(f"{col_prefix}_recovery", float("nan"))
                    hal_vals[row_idx][col_idx] = row_data.get(f"{col_prefix}_hallucination", float("nan"))
                col_idx += 1

    best_rec = [
        max((rec_vals[r][c] for r in range(len(rows)) if not pd.isna(rec_vals[r][c])), default=None)
        for c in range(n_cols)
    ]
    best_hal = [
        min((hal_vals[r][c] for r in range(len(rows)) if not pd.isna(hal_vals[r][c])), default=None)
        for c in range(n_cols)
    ]

    def fmt_cell(rec, hal, col_idx) -> str:
        if pd.isna(rec) or pd.isna(hal):
            return "--"
        rec_str = f"{rec:.2f}"
        hal_str = f"{hal:.2f}"
        if best_rec[col_idx] is not None and rec == best_rec[col_idx]:
            rec_str = r"\textbf{" + rec_str + "}"
        if best_hal[col_idx] is not None and hal == best_hal[col_idx]:
            hal_str = r"\textbf{" + hal_str + "}"
        return f"{rec_str} ({hal_str})"

    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"  \small")
    lines.append(r"  \setlength{\tabcolsep}{4pt}")
    lines.append(r"  \centering")
    lines.append(r"  \begin{tabular}{l ccc ccc}")
    lines.append(r"    \toprule")

    # Dataset header row
    lines.append(
        r"    & \multicolumn{3}{c}{\pond} & \multicolumn{3}{c}{\nfix} \\"
    )
    lines.append(r"    \cmidrule(lr){2-4} \cmidrule(lr){5-7}")

    # Model header row
    model_headers = " & ".join(model_display[m] for m in models)
    lines.append(f"    & {model_headers} & {model_headers} \\\\")
    lines.append(r"    \midrule")

    # Data rows
    for row_idx, (row_label, _) in enumerate(rows):
        cells = [fmt_cell(rec_vals[row_idx][c], hal_vals[row_idx][c], c) for c in range(n_cols)]
        lines.append(f"    {row_label} & {' & '.join(cells)} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(
        r"  \caption{\textbf{Ablation Performance.} Recovery rate and "
        r"(error rate) evaluated for the complete, base pipeline and "
        r"ablations 1--6; performed for the \pond and \nfix datasets "
        r"and for each extraction LLM.}"
    )
    lines.append(r"  \label{tab:ablation}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pond_csv = sys.argv[1]
        nfix_csv = sys.argv[2] if len(sys.argv) > 2 else "ablation_nfix.csv"
        print(generate_latex(pond_csv, nfix_csv))
    else:
        ablation_dir = RESULTS_DIR / "ablation"
        pond_csv = ablation_dir / "ablation_pond.csv"
        nfix_csv = ablation_dir / "ablation_nfix.csv"
        out_path = ablation_dir / "ablation_table.tex"
        out_path.write_text(generate_latex(str(pond_csv), str(nfix_csv)) + "\n")
        print(f"Wrote {out_path}")