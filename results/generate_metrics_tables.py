#!/usr/bin/env python3
"""
Generate the LaTeX metrics table from metrics_pond.csv and metrics_nfix.csv.

Each CSV has columns: Test setting, Type, Accuracy, Precision, Recall, F1, AUROC, ECE
- "Type" is either "Probe" or "NTP"
- "Test setting" is one of: "Syn. pond", "Syn. nfix", "Real pond", "Real nfix"
  (where "Real" maps to "Ext." in the table)

The table has two column groups:
  - pond-Trained (from metrics_pond.csv): F1, AUC, ECE
  - nfix-Trained (from metrics_nfix.csv): F1, AUC, ECE

Rows are grouped by Type (Probe, NTP), with 4 test settings each.

When run with no arguments, walks results/{judge_model}/{extraction_model}/{probe_type}/
and writes metrics_table.tex in each subdirectory that contains both CSVs.
Pass explicit CSV paths as arguments to print LaTeX to stdout instead.
"""

import sys
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")


def load_metrics(pond_csv: str, nfix_csv: str) -> dict:
    """Load both CSVs and return a nested lookup dict.

    Returns:
        dict keyed by (type, test_setting) -> {"pond": {F1, AUROC, ECE}, "nfix": {F1, AUROC, ECE}}
    """
    df_pond = pd.read_csv(pond_csv, index_col=0)
    df_nfix = pd.read_csv(nfix_csv, index_col=0)

    metrics = {}
    for df, train_key in [(df_pond, "pond"), (df_nfix, "nfix")]:
        for _, row in df.iterrows():
            key = (row["Type"], row["Test setting"])
            if key not in metrics:
                metrics[key] = {}
            metrics[key][train_key] = {
                "F1": row["F1"],
                "AUROC": row["AUROC"],
                "ECE": row["ECE"],
            }
    return metrics


def fmt(val: float) -> str:
    """Format a metric value to 2 decimal places."""
    return f"{val:.2f}"


def generate_latex(
    pond_csv: str,
    nfix_csv: str,
    caption: str | None = None,
    label: str = "tab:llama-metrics",
) -> str:
    metrics = load_metrics(pond_csv, nfix_csv)

    # Row order: the test settings within each Type group
    test_settings = ["Syn. pond", "Real pond", "Syn. nfix", "Real nfix"]

    # Display names for test settings (Real -> Ext.)
    display_names = {
        "Syn. pond": r"Syn. \pond",
        "Real pond": r"Ext. \pond",
        "Syn. nfix": r"Syn. \nfix",
        "Real nfix": r"Ext. \nfix",
    }

    # Type groups in display order
    type_groups = ["Probe", "NTP"]

    # Build lines
    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{ll ccc ccc}")
    lines.append(r"\toprule")
    lines.append(
        r"& & \multicolumn{3}{c}{\pond-Trained} & \multicolumn{3}{c}{\nfix-Trained} \\"
    )
    lines.append(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}")
    lines.append(r"& & F1 & AUC & ECE & F1 & AUC & ECE \\")
    lines.append(r"\midrule")

    for i, type_name in enumerate(type_groups):
        n_rows = len(test_settings)
        lines.append(rf"\multirow{{{n_rows}}}{{*}}{{{type_name}}}")

        for setting in test_settings:
            key = (type_name, setting)
            m = metrics[key]
            pond = m["pond"]
            nfix = m["nfix"]

            row = (
                f"& {display_names[setting]} "
                f"& {fmt(pond['F1'])} & {fmt(pond['AUROC'])} & {fmt(pond['ECE'])} "
                f"& {fmt(nfix['F1'])} & {fmt(nfix['AUROC'])} & {fmt(nfix['ECE'])} \\\\"
            )
            lines.append(row)

        # Add midrule between groups (not after the last one)
        if i < len(type_groups) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    if caption is None:
        caption = (
            r"\textbf{Performance Metrics (\texttt{llama-3.1-8b}).} "
            r"for probe and NTP predicted validation probabilities, "
            r"trained on the synthetic \pond (left-three columns) and synthetic \nfix "
            r"(right-three columns) datasets. Evaluated for test synthetic and "
            r"extracted \pond and \nfix  datasets. Data extracted with \texttt{gemma-3-27b}."
        )

    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def make_caption(judge_model: str, extraction_model: str) -> str:
    return (
        rf"\textbf{{Performance Metrics (\texttt{{{judge_model}}}).}} "
        r"for probe and NTP predicted validation probabilities, "
        r"trained on the synthetic \pond (left-three columns) and synthetic \nfix "
        r"(right-three columns) datasets. Evaluated for test synthetic and "
        rf"extracted \pond and \nfix datasets. Data extracted with \texttt{{{extraction_model}}}."
    )


def main() -> None:
    for pond_csv in sorted(RESULTS_DIR.glob("*/*/*/metrics_pond.csv")):
        nfix_csv = pond_csv.parent / "metrics_nfix.csv"
        if not nfix_csv.exists():
            print(f"Skipping {pond_csv.parent} — missing metrics_nfix.csv", file=sys.stderr)
            continue

        subdir = pond_csv.parent
        probe_type = subdir.name
        extraction_model = subdir.parent.name
        judge_model = subdir.parent.parent.name

        caption = make_caption(judge_model, extraction_model)
        label = f"tab:{judge_model}-{extraction_model}-{probe_type}"
        latex = generate_latex(str(pond_csv), str(nfix_csv), caption=caption, label=label)

        out_path = subdir / "metrics_table.tex"
        out_path.write_text(latex + "\n")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pond_csv = sys.argv[1]
        nfix_csv = sys.argv[2] if len(sys.argv) > 2 else "metrics_nfix.csv"
        print(generate_latex(pond_csv, nfix_csv))
    else:
        main()