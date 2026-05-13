#!/usr/bin/env python3
"""
Generate LaTeX metrics tables from a single metrics CSV.

The CSV is produced by analysis/synthetic_probe_test.py (compute_metrics),
with columns: Dataset type, Judge model, Train dataset, Test dataset, Type,
Accuracy, Precision, Recall, F1, AUROC, ECE

Generates two tables per judge model (one for dtype=syn, one for dtype=real).
Each table has hierarchical rows: In-domain / Cross-domain (outer),
PLW NTP, PLW Probe, NF NTP, NF Probe (inner), and columns Prec., Rec., F1, AUC, ECE.

Usage:
  python results/generate_metrics_tables.py                           # walks results/ for metrics_*.csv
  python results/generate_metrics_tables.py results/metrics_X_Y.csv  # prints all tables to stdout
"""

import sys
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")

# (domain_group, train_ds, test_ds, type, display_label)
# PLW = pond, NF = nfix; label refers to the test dataset
_ROW_DEFS = [
    ("In-domain",    "pond", "pond", "NTP",   r"PLW NTP"),
    ("In-domain",    "pond", "pond", "Probe",  r"PLW Probe"),
    ("In-domain",    "nfix", "nfix", "NTP",    r"NF NTP"),
    ("In-domain",    "nfix", "nfix", "Probe",  r"NF Probe"),
    ("Cross-domain", "nfix", "pond", "NTP",    r"PLW NTP"),
    ("Cross-domain", "nfix", "pond", "Probe",  r"PLW Probe"),
    ("Cross-domain", "pond", "nfix", "NTP",    r"NF NTP"),
    ("Cross-domain", "pond", "nfix", "Probe",  r"NF Probe"),
]

_GROUPS = ["In-domain", "Cross-domain"]
_GROUP_SIZE = 4


def fmt(val: float) -> str:
    return f"{val:.2f}"


def generate_latex(
    df: pd.DataFrame,
    judge_model: str,
    dtype: str,
    caption: str | None = None,
    tex_label: str | None = None,
) -> str:
    sub = df[(df["Judge model"] == judge_model) & (df["Dataset type"] == dtype)]

    lookup = {
        (row["Train dataset"], row["Test dataset"], row["Type"]): row
        for _, row in sub.iterrows()
    }

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{ll ccccc}")
    lines.append(r"\toprule")
    lines.append(r"& & Prec. & Rec. & F1 & AUC & ECE \\")
    lines.append(r"\midrule")

    for gi, group in enumerate(_GROUPS):
        group_rows = [
            (train_ds, test_ds, kind, display)
            for grp, train_ds, test_ds, kind, display in _ROW_DEFS
            if grp == group
        ]
        lines.append(rf"\multirow{{{_GROUP_SIZE}}}{{*}}{{\textit{{{group}}}}}")
        for train_ds, test_ds, kind, display in group_rows:
            row = lookup.get((train_ds, test_ds, kind))
            if row is None:
                vals = ["--"] * 5
            else:
                vals = [
                    fmt(row["Precision"]), fmt(row["Recall"]), fmt(row["F1"]),
                    fmt(row["AUROC"]), fmt(row["ECE"]),
                ]
            lines.append(f"& {display} & " + " & ".join(vals) + r" \\")
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


def make_caption(judge_model: str, extraction_model: str, dtype: str) -> str:
    dtype_str = "Synthetic" if dtype == "syn" else "Extracted"
    return (
        rf"\textbf{{Performance Metrics (\texttt{{{judge_model}}}, {dtype_str}).}} "
        r"Probe and NTP validation probabilities evaluated in-domain and cross-domain. "
        rf"Measurements extracted with \texttt{{{extraction_model}}}."
    )


def process_csv(csv_path: Path, print_to_stdout: bool = False) -> None:
    stem = csv_path.stem  # e.g. "metrics_gemma-3-27b_head"
    rest = stem[len("metrics_"):]  # strip leading "metrics_"
    extraction_model, probe_type = rest.rsplit("_", 1)

    df = pd.read_csv(csv_path)
    judge_models = sorted(df["Judge model"].unique())

    for judge_model in judge_models:
        for dtype in ["syn", "real"]:
            caption = make_caption(judge_model, extraction_model, dtype)
            label = f"tab:{judge_model}-{extraction_model}-{probe_type}-{dtype}"
            latex = generate_latex(df, judge_model, dtype, caption=caption, tex_label=label)

            if print_to_stdout:
                print(f"% ── {judge_model} / {dtype} ──────────────────────────")
                print(latex)
                print()
            else:
                out_dir = RESULTS_DIR / judge_model / extraction_model / probe_type
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"metrics_table_{dtype}.tex"
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
