#!/usr/bin/env python3
"""
Generate LaTeX ECE tables from the combined metrics CSV produced by
analysis/synthetic_probe_test.py.

CSV lives at: results/metrics_{extraction_model}_{probe_type}.csv
Columns: Dataset type, Judge model, Train dataset, Test dataset, Type, ..., ECE

Generates one table per (judge_model, extraction_model, probe_type) combo.
Rows: Within / Cross-domain x {PLW NTP, PLW Probe, NF NTP, NF Probe}.
Columns: Synthetic | Extracted.

Usage:
  python results/generate_metrics_tables.py                                        # walks results/
  python results/generate_metrics_tables.py results/metrics_gemma-3-27b_head.csv  # single csv, stdout
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


def fmt(val) -> str:
    if val is None or pd.isna(val):
        return "--"
    return f"{val:.2f}"


def build_lookup(df: pd.DataFrame, train_ds: str) -> dict:
    """Return lookup: (test_setting, type) -> ECE for a given training dataset.

    test_setting is constructed as "Syn. {test_ds}" or "Real {test_ds}".
    """
    sub = df[df["Train dataset"] == train_ds]
    lookup = {}
    for _, row in sub.iterrows():
        prefix = "Syn." if row["Dataset type"] == "syn" else "Real"
        test_setting = f"{prefix} {row['Test dataset']}"
        lookup[(test_setting, row["Type"])] = row["ECE"]
    return lookup


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
            syn_str  = fmt(lkp.get((f"Syn. {test_ds}", kind)))
            real_str = fmt(lkp.get((f"Real {test_ds}", kind)))
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

    for judge_model in sorted(df["Judge model"].unique()):
        judge_df = df[df["Judge model"] == judge_model]
        pond_lookup = build_lookup(judge_df, "pond")
        nfix_lookup = build_lookup(judge_df, "nfix")

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
            out_dir = RESULTS_DIR / judge_model / extraction_model / probe_type
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "metrics_table.tex"
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
