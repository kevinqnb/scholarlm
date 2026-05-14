#!/usr/bin/env python3
"""
Generate ablation LaTeX tables from ablation_pond.csv and ablation_nfix.csv.

Produces two tables: one for recovery rate and one for validity rate.
Each cell shows "value ± margin" with 95% Wilson CIs, rounded to 2 decimal places.
Missing data (NaN) is rendered as "--".

When run with no arguments, reads from results/ablation/ and writes
ablation_table_recovery.tex and ablation_table_hallucination.tex there.
Pass explicit CSV paths as arguments to print both tables to stdout instead.
"""

import sys
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")

MODELS = ["llama-3.1-8b", "gemma-3-27b", "gpt-oss-120b"]
MODEL_DISPLAY = {
    "llama-3.1-8b": r"\texttt{llama-3.1-8b}",
    "gemma-3-27b": r"\texttt{gemma-3-27b}",
    "gpt-oss-120b": r"\texttt{gpt-oss-120b}",
}
ROWS = [
    ("Base", "baseline"),
    ("$A_1$", "ablation_1"),
    ("$A_2$", "ablation_2"),
    ("$A_3$", "ablation_3"),
    ("$A_4$", "ablation_4"),
    ("$A_5$", "ablation_5"),
    ("$A_6$", "ablation_6"),
]


def load_ablation(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _collect_values(pond_by_model, nfix_by_model, metric: str):
    """Return (vals, lo_vals, hi_vals) each shaped (n_rows, n_cols)."""
    n_cols = len(MODELS) * 2
    vals   = [[float("nan")] * n_cols for _ in ROWS]
    lo_vals = [[float("nan")] * n_cols for _ in ROWS]
    hi_vals = [[float("nan")] * n_cols for _ in ROWS]

    for row_idx, (_, col_prefix) in enumerate(ROWS):
        col_idx = 0
        for df_by_model in [pond_by_model, nfix_by_model]:
            for model in MODELS:
                if model in df_by_model:
                    r = df_by_model[model]
                    vals[row_idx][col_idx]    = r.get(f"{col_prefix}_{metric}",        float("nan"))
                    lo_vals[row_idx][col_idx] = r.get(f"{col_prefix}_{metric}_ci_lo",  float("nan"))
                    hi_vals[row_idx][col_idx] = r.get(f"{col_prefix}_{metric}_ci_hi",  float("nan"))
                col_idx += 1

    return vals, lo_vals, hi_vals


def _generate_table(pond_by_model, nfix_by_model, metric: str, caption: str, label: str) -> str:
    higher_is_better = metric in ("recovery", "validity")
    vals, lo_vals, hi_vals = _collect_values(pond_by_model, nfix_by_model, metric)
    n_cols = len(MODELS) * 2

    if higher_is_better:
        best = [
            max((vals[r][c] for r in range(len(ROWS)) if not pd.isna(vals[r][c])), default=None)
            for c in range(n_cols)
        ]
    else:
        best = [
            min((vals[r][c] for r in range(len(ROWS)) if not pd.isna(vals[r][c])), default=None)
            for c in range(n_cols)
        ]

    def fmt_cell(v, lo, hi, col_idx) -> str:
        if pd.isna(v) or pd.isna(lo) or pd.isna(hi):
            return "--"
        v_str = f"{v:.2f}"
        if best[col_idx] is not None and v == best[col_idx]:
            v_str = r"\textbf{" + v_str + "}"
        margin = (hi - lo) / 2
        return rf"{v_str} $\pm$ {margin:.2f}"

    model_headers = " & ".join(MODEL_DISPLAY[m] for m in MODELS)
    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"  \small")
    lines.append(r"  \setlength{\tabcolsep}{4pt}")
    lines.append(r"  \centering")
    lines.append(r"  \begin{tabular}{l ccc ccc}")
    lines.append(r"    \toprule")
    lines.append(r"    & \multicolumn{3}{c}{\pond} & \multicolumn{3}{c}{\nfix} \\")
    lines.append(r"    \cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    lines.append(f"    & {model_headers} & {model_headers} \\\\")
    lines.append(r"    \midrule")

    for row_idx, (row_label, _) in enumerate(ROWS):
        cells = [fmt_cell(vals[row_idx][c], lo_vals[row_idx][c], hi_vals[row_idx][c], c) for c in range(n_cols)]
        lines.append(f"    {row_label} & {' & '.join(cells)} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(f"  {caption}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def generate_tables(pond_csv: str, nfix_csv: str) -> tuple[str, str]:
    df_pond = load_ablation(pond_csv)
    df_nfix = load_ablation(nfix_csv)

    pond_by_model = {row["model"]: row for _, row in df_pond.iterrows()}
    nfix_by_model = {row["model"]: row for _, row in df_nfix.iterrows()}

    recovery_table = _generate_table(
        pond_by_model, nfix_by_model,
        metric="recovery",
        caption=(
            r"\caption{\textbf{Ablation Recovery Rate.} Recovery rate with 95\% Wilson CIs "
            r"for the base pipeline and ablations $A_1$--$A_6$ across the \pond and \nfix "
            r"datasets and each extraction LLM. Bold marks the best value per column.}"
        ),
        label="tab:ablation_recovery",
    )

    hallucination_table = _generate_table(
        pond_by_model, nfix_by_model,
        metric="validity",
        caption=(
            r"\caption{\textbf{Ablation Validity Rate.} Validity rate with 95\% Wilson CIs "
            r"for the base pipeline and ablations $A_1$--$A_6$ across the \pond and \nfix "
            r"datasets and each extraction LLM. Bold marks the best value per column.}"
        ),
        label="tab:ablation_validity",
    )

    return recovery_table, hallucination_table


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pond_csv = sys.argv[1]
        nfix_csv = sys.argv[2] if len(sys.argv) > 2 else "ablation_nfix.csv"
        rec_tex, hal_tex = generate_tables(pond_csv, nfix_csv)
        print(rec_tex)
        print()
        print(hal_tex)
    else:
        ablation_dir = RESULTS_DIR / "ablation"
        pond_csv = ablation_dir / "ablation_pond.csv"
        nfix_csv = ablation_dir / "ablation_nfix.csv"
        rec_tex, hal_tex = generate_tables(str(pond_csv), str(nfix_csv))
        rec_path = ablation_dir / "ablation_table_recovery.tex"
        hal_path = ablation_dir / "ablation_table_hallucination.tex"
        rec_path.write_text(rec_tex + "\n")
        hal_path.write_text(hal_tex + "\n")
        print(f"Wrote {rec_path}")
        print(f"Wrote {hal_path}")
