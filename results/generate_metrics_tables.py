#!/usr/bin/env python3
"""
Generate LaTeX ECE tables from per-training-dataset metrics CSVs.

CSVs live at: results/{judge_model}/{extraction_model}/{probe_type}/metrics_{train_dataset}.csv
Columns: Test setting, Type, ..., ECE

Generates one table per directory (one per judge/extraction/probe_type combo).
Rows: Within / Cross-domain x {PLW NTP, PLW Probe, NF NTP, NF Probe}.
Columns: Synthetic | Extracted.

Usage:
  python results/generate_metrics_tables.py                                      # walks results/
  python results/generate_metrics_tables.py results/mistral-7b/gemma-3-27b/head  # single dir, stdout
"""

import sys
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")

# (group, train_dataset, test_dataset, type, display_label)
# PLW = pond, NF = nfix; label indicates the *training* dataset
_ROW_DEFS = [
    ("Within", "pond", "pond", "NTP",   r"PLW NTP"),
    ("Within", "pond", "pond", "Probe", r"PLW Probe"),
    ("Within", "nfix", "nfix", "NTP",   r"NF NTP"),
    ("Within", "nfix", "nfix", "Probe", r"NF Probe"),
    ("Cross",  "pond", "nfix", "NTP",   r"PLW NTP"),
    ("Cross",  "pond", "nfix", "Probe", r"PLW Probe"),
    ("Cross",  "nfix", "pond", "NTP",   r"NF NTP"),
    ("Cross",  "nfix", "pond", "Probe", r"NF Probe"),
]

_GROUPS = ["Within", "Cross"]
_GROUP_SIZE = 4

# Maps (test_dataset, dtype) -> exact "Test setting" string in the CSV
_TEST_SETTING = {
    ("pond", "syn"):  "Syn. pond",
    ("nfix", "syn"):  "Syn. nfix",
    ("pond", "real"): "Real pond",
    ("nfix", "real"): "Real nfix",
}


def fmt(val) -> str:
    if val is None or pd.isna(val):
        return "--"
    return f"{val:.2f}"


def load_lookup(csv_path: Path) -> dict:
    """Return a lookup: (test_setting, type) -> ECE."""
    df = pd.read_csv(csv_path)
    return {
        (row["Test setting"], row["Type"]): row["ECE"]
        for _, row in df.iterrows()
    }


def generate_latex(
    pond_lookup: dict,
    nfix_lookup: dict,
    judge_model: str,
    extraction_model: str,
    caption: str | None = None,
    tex_label: str | None = None,
) -> str:
    train_lookup = {"pond": pond_lookup, "nfix": nfix_lookup}

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
            for grp, train_ds, test_ds, kind, display in _ROW_DEFS
            if grp == group
        ]
        lines.append(rf"\multirow{{{_GROUP_SIZE}}}{{*}}{{\textit{{{group}}}}}")
        for train_ds, test_ds, kind, display in group_rows:
            lkp = train_lookup[train_ds]
            syn_key  = (_TEST_SETTING[(test_ds, "syn")],  kind)
            real_key = (_TEST_SETTING[(test_ds, "real")], kind)
            syn_str  = fmt(lkp.get(syn_key))
            real_str = fmt(lkp.get(real_key))
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


def make_caption(judge_model: str, extraction_model: str) -> str:
    return (
        rf"\textbf{{Expected Calibration Error Performance (\texttt{{{judge_model}}}).}} "
        r"Probe and NTP validation probabilities evaluated within and cross-domain "
        r"for synthetic and real, extraction test settings. "
        rf"Extracted measurements collected from \texttt{{{extraction_model}}}."
    )


def process_dir(result_dir: Path, print_to_stdout: bool = False) -> None:
    pond_csv = result_dir / "metrics_pond.csv"
    nfix_csv = result_dir / "metrics_nfix.csv"
    if not pond_csv.exists() or not nfix_csv.exists():
        return

    parts = result_dir.relative_to(RESULTS_DIR).parts
    if len(parts) != 3:
        return
    judge_model, extraction_model, probe_type = parts

    pond_lookup = load_lookup(pond_csv)
    nfix_lookup = load_lookup(nfix_csv)

    caption = make_caption(judge_model, extraction_model)
    label = f"tab:{judge_model}-{extraction_model}"
    latex = generate_latex(
        pond_lookup, nfix_lookup, judge_model, extraction_model,
        caption=caption, tex_label=label,
    )

    if print_to_stdout:
        print(f"% ── {judge_model} / {extraction_model} / {probe_type} ──────────────────")
        print(latex)
        print()
    else:
        out_path = result_dir / "metrics_table.tex"
        out_path.write_text(latex + "\n")
        print(f"Wrote {out_path}")


def main() -> None:
    for pond_csv in sorted(RESULTS_DIR.glob("*/*/*/metrics_pond.csv")):
        process_dir(pond_csv.parent)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_dir(Path(sys.argv[1]), print_to_stdout=True)
    else:
        main()
