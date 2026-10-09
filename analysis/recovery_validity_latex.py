"""Render pre-computed ``analysis/recovery_validity.py`` result CSVs as one LaTeX
table: a row per method, a column per dataset (row groups split by one rule), each cell a (recovery, validity)
pair with its bootstrap interval.

This only *formats* numbers that ``recovery_validity.py`` already wrote -- it
computes no metric and never re-reads a match cache, so it cannot change a
reported number. What it does guard is the mapping from table cell to CSV row:

  - every cell names its experiment id explicitly in the table spec (no id is
    inferred from a method name, and a cell that has no run is an explicit
    ``null``, rendered ``--``, never a silent blank);
  - each dataset column points at the *analysis config* that produced its CSV;
    the CSV path is read off that config's ``recovery_validity.output``, and the
    CSV must (a) carry that config's id in ``analysis_config_id`` on every row,
    and (b) contain exactly that config's ``experiment_ids`` -- a CSV left over
    from before the config was edited is a hard error, not a table;
  - the row's dataset, the validity columns (present, not NaN), and
    ``ci_lo <= point <= ci_hi`` are asserted for every cell used;
  - every id in the spec must be unique, so one run cannot fill two cells.

Interval format (``ci_format``, required): the bootstrap intervals are
*percentile* intervals and need not be symmetric about the point estimate.
``pm`` prints ``point $\\pm$ (hi - lo) / 2`` (the half-width), which is exact only
for a symmetric interval; ``interval`` prints ``point [lo, hi]`` and is always
exact. Values are percentages. The best point estimate per column and metric is bolded
(the number only, not its interval).

Usage
-----
    python analysis/recovery_validity_latex.py --config analysis/analysis-configs/recovery-validity/<id>.yaml

Table spec (``params.recovery_validity_latex`` in the analysis config; every key
required, no defaults)::

    datasets:            # ordered: column order. key = column header
      PLW: {dataset: pond, analysis_config: <recovery_validity analysis config id>}
    decimals: 1
    ci_format: pm | interval
    output: analysis/results/recovery-validity/<name>.tex
    caption: "..."
    label: tab:...
    blocks:              # ordered row groups, separated by a single \\midrule
      - rows:
          - label: "Direct Gemma-3-27B"
            ids: {PLW: <experiment id>, NF: <id or null>, SM: ...}   # every column key
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from analysis.common.config import (  # noqa: E402
    _load_envelope, analysis_config_path, get_section,
)

SECTION = "recovery_validity_latex"
SECTION_KEYS = ("datasets", "decimals", "ci_format", "output", "caption", "label", "blocks")
CI_FORMATS = ("pm", "interval")
MISSING_CELL = "--"

# recovery_max_weight_matching is not rendered; requiring it rejects a CSV written
# before 2026-10-07, whose `recovery` column was the max-weight matching count rather
# than today's any-edge count (see analysis/recovery_validity.py's module docstring).
_NEEDED_COLUMNS = (
    "experiment_id", "dataset", "analysis_config_id",
    "recovery", "recovery_ci_lo", "recovery_ci_hi", "recovery_max_weight_matching",
    "validity", "validity_ci_lo", "validity_ci_hi",
)


# ---------------------------------------------------------------------------
# Loading + validation
# ---------------------------------------------------------------------------


def load_spec(path: Path) -> dict:
    """Load and validate the table-spec analysis config; returns the
    ``params.recovery_validity_latex`` section (with the envelope checked).

    Raises:
        ValueError/KeyError: malformed envelope or section, bad types, a block
            row whose ``ids`` keys are not exactly the dataset column keys, or
            an experiment id used in more than one cell of the same column.
    """
    cfg = _load_envelope(path, "recovery-validity")
    if set(cfg["params"]) != {SECTION}:
        raise ValueError(f"{path}: params keys {sorted(cfg['params'])} must be exactly [{SECTION!r}]")
    spec = get_section(cfg, SECTION, required_keys=SECTION_KEYS)

    if spec["ci_format"] not in CI_FORMATS:
        raise ValueError(f"{path}: ci_format must be one of {CI_FORMATS}, got {spec['ci_format']!r}")
    decimals = spec["decimals"]
    if isinstance(decimals, bool) or not isinstance(decimals, int) or decimals < 0:
        raise ValueError(f"{path}: decimals must be a non-negative int, got {decimals!r}")
    for key in ("output", "caption", "label"):
        if not isinstance(spec[key], str) or not spec[key]:
            raise ValueError(f"{path}: {key} must be a non-empty string, got {spec[key]!r}")

    datasets = spec["datasets"]
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError(f"{path}: datasets must be a non-empty mapping")
    for col, block in datasets.items():
        if not isinstance(block, dict) or set(block) != {"dataset", "analysis_config"}:
            raise ValueError(f"{path}: datasets.{col} keys must be exactly ['analysis_config', 'dataset']")

    blocks = spec["blocks"]
    if not isinstance(blocks, list) or not blocks:
        raise ValueError(f"{path}: blocks must be a non-empty list")
    used: dict[str, list[str]] = {col: [] for col in datasets}
    for block in blocks:
        if not isinstance(block, dict) or set(block) != {"rows"}:
            raise ValueError(f"{path}: each block needs exactly the key 'rows', got {block!r}")
        if not isinstance(block["rows"], list) or not block["rows"]:
            raise ValueError(f"{path}: a block has no rows")
        for row in block["rows"]:
            if not isinstance(row, dict) or set(row) != {"label", "ids"}:
                raise ValueError(f"{path}: each row needs exactly keys ['ids', 'label'], got {row!r}")
            if set(row["ids"]) != set(datasets):
                raise ValueError(
                    f"{path}: row {row['label']!r} ids keys {sorted(row['ids'])} must be exactly "
                    f"the dataset columns {sorted(datasets)} (use null for a missing run)"
                )
            for col, exp_id in row["ids"].items():
                if exp_id is not None:
                    if not isinstance(exp_id, str) or not exp_id:
                        raise ValueError(f"{path}: row {row['label']!r} ids.{col} must be a string or null")
                    used[col].append(exp_id)
    for col, ids in used.items():
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"{path}: column {col!r} uses experiment id(s) in more than one cell: {dupes}")
    return spec


def load_results(col: str, dataset: str, analysis_config_id: str) -> pd.DataFrame:
    """The recovery_validity CSV written by ``analysis_config_id``, indexed by
    experiment_id, after checking it really is that config's output.

    Raises:
        FileNotFoundError: the referenced analysis config or its CSV is missing.
        ValueError: the config is not a recovery_validity config for ``dataset``,
            the CSV lacks needed columns / has duplicate ids, carries a
            different ``analysis_config_id`` or ``dataset``, or its id set is
            not exactly the config's ``experiment_ids`` (stale CSV).
    """
    config_path = analysis_config_path("recovery-validity", analysis_config_id)
    cfg = _load_envelope(config_path, "recovery-validity")
    params = cfg["params"]
    if "recovery_validity" not in params or "experiment_ids" not in params:
        raise ValueError(f"{config_path}: not a recovery_validity analysis config")
    output = Path(params["recovery_validity"]["output"])
    csv_path = output if output.is_absolute() else _REPO_ROOT / output
    if not csv_path.exists():
        raise FileNotFoundError(f"{col}: {csv_path} (output of {analysis_config_id}) does not exist -- run it first")

    df = pd.read_csv(csv_path)
    missing_cols = [c for c in _NEEDED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{csv_path}: missing column(s) {missing_cols}")
    if df["experiment_id"].duplicated().any():
        raise ValueError(f"{csv_path}: duplicate experiment_id rows")
    if set(df["analysis_config_id"]) != {analysis_config_id}:
        raise ValueError(
            f"{csv_path}: analysis_config_id values {sorted(set(df['analysis_config_id']), key=str)} "
            f"!= {analysis_config_id!r}"
        )
    if set(df["dataset"]) != {dataset}:
        raise ValueError(f"{csv_path}: dataset values {sorted(set(df['dataset']))} != {dataset!r}")
    extra = sorted(set(df["experiment_id"]) - set(params["experiment_ids"]))
    absent = sorted(set(params["experiment_ids"]) - set(df["experiment_id"]))
    if extra or absent:
        raise ValueError(
            f"{csv_path} is stale relative to {config_path}: CSV has {len(extra)} id(s) the config "
            f"no longer lists ({extra}), and is missing {len(absent)} id(s) the config lists "
            f"({absent}). Rerun `python analysis/recovery_validity.py --config {config_path}`."
        )
    return df.set_index("experiment_id")


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_estimate(
    point: float, lo: float, hi: float, *, decimals: int, ci_format: str, bold: bool = False,
) -> str:
    """One metric as a percentage with its interval, e.g. ``38.8 $\\pm$ 5.4``.
    ``bold`` wraps only the point estimate in ``\\textbf``, never the interval.

    Raises:
        ValueError: any value is NaN or outside [0, 1], or lo <= point <= hi fails.
    """
    for name, v in (("point", point), ("ci_lo", lo), ("ci_hi", hi)):
        if not (isinstance(v, (int, float)) and math.isfinite(v) and 0.0 <= v <= 1.0):
            raise ValueError(f"{name}={v!r} is not a finite rate in [0, 1]")
    if not lo <= point <= hi:
        raise ValueError(f"point {point} outside its interval [{lo}, {hi}]")
    p = f"{100 * point:.{decimals}f}"
    if bold:
        p = f"\\textbf{{{p}}}"
    if ci_format == "pm":
        return f"{p} $\\pm$ {100 * (hi - lo) / 2:.{decimals}f}"
    if ci_format == "interval":
        return f"{p} [{100 * lo:.{decimals}f}, {100 * hi:.{decimals}f}]"
    raise ValueError(f"ci_format must be one of {CI_FORMATS}, got {ci_format!r}")


def format_cell(
    row: pd.Series, *, decimals: int, ci_format: str, bold_recovery: bool = False, bold_validity: bool = False,
) -> str:
    """``(recovery, validity)`` for one result row."""
    try:
        recovery = format_estimate(
            row["recovery"], row["recovery_ci_lo"], row["recovery_ci_hi"], decimals=decimals, ci_format=ci_format, bold=bold_recovery)
        validity = format_estimate(
            row["validity"], row["validity_ci_lo"], row["validity_ci_hi"], decimals=decimals, ci_format=ci_format,
            bold=bold_validity)
    except ValueError as e:
        raise ValueError(f"{row.name}: {e}") from e
    return f"({recovery}, {validity})"


def build_table(spec: dict, results: dict[str, pd.DataFrame]) -> str:
    """The full ``table`` environment as a string. ``results`` maps each
    dataset column key to its (experiment_id-indexed) result frame."""
    cols = list(spec["datasets"])
    # Best (max raw point estimate) recovery / validity among the rows actually
    # shown in each column; exact ties are all bolded.
    best: dict[str, dict[str, float]] = {}
    for col in cols:
        shown = [r["ids"][col] for b in spec["blocks"] for r in b["rows"] if r["ids"][col] is not None]
        missing = [i for i in shown if i not in results[col].index]
        if missing:
            raise KeyError(f"column {col}: {missing} not in that column's results CSV")
        best[col] = {m: results[col].loc[shown, m].max() for m in ("recovery", "validity")}
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{spec['caption']}}}",
        f"\\label{{{spec['label']}}}",
        f"\\begin{{tabular}}{{l{'c' * len(cols)}}}",
        "\\toprule",
        "Method & " + " & ".join(cols) + " \\\\",
    ]
    for block in spec["blocks"]:
        lines.append("\\midrule")
        for row in block["rows"]:
            cells = []
            for col in cols:
                exp_id = row["ids"][col]
                if exp_id is None:
                    cells.append(MISSING_CELL)
                    continue
                if exp_id not in results[col].index:
                    raise KeyError(f"row {row['label']!r}, column {col}: {exp_id!r} not in that column's results CSV")
                res = results[col].loc[exp_id]
                cells.append(format_cell(
                    res, decimals=spec["decimals"], ci_format=spec["ci_format"],
                    bold_recovery=res["recovery"] == best[col]["recovery"],
                    bold_validity=res["validity"] == best[col]["validity"]))
            lines.append(f"{row['label']} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="analysis-configs/recovery-validity/<id>.yaml table spec")
    args = parser.parse_args(argv)

    spec = load_spec(args.config)
    results = {
        col: load_results(col, block["dataset"], block["analysis_config"])
        for col, block in spec["datasets"].items()
    }
    text = build_table(spec, results)
    output = Path(spec["output"])
    output = output if output.is_absolute() else _REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    print(text)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
