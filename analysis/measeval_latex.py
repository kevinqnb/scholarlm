"""Format the measeval_evaluation.py result CSV as a LaTeX table (methods x annotation types).

Each cell is the official MeasEval scorer's F-measure, as a percentage, for one
annotation type (Quantity, Unit, MeasuredEntity); the best value per column is
bolded. Formatting only: no metric is computed. The cell-to-row mapping is guarded:
every row names one experiment id, the CSV must belong to the named
measeval_evaluation config, hold exactly that config's ids, be a non-dev run, and
have one row per (id, annotation type).

Usage
-----
    python analysis/measeval_latex.py --config analysis/analysis-configs/measeval/<id>.yaml

Table spec (``params.measeval_latex``, every key required)::

    analysis_config: <measeval_evaluation analysis config id>
    decimals: 1
    output: analysis/results/measeval/<name>.tex
    caption: "..."
    label: tab:...
    blocks:              # ordered row groups, separated by a single \\midrule
      - rows:
          - label: "Pipeline (\\texttt{gpt-oss-120b})"
            id: <experiment id>
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

SECTION = "measeval_latex"
SECTION_KEYS = ("analysis_config", "decimals", "output", "caption", "label", "blocks")

# Column order and headers: (annot_type in the CSV, header).
COLUMNS = (("Quantity", "Quantity"), ("Unit", "Unit"), ("MeasuredEntity", "Entity"))

_NEEDED_COLUMNS = ("experiment_id", "dataset", "dev", "annot_type", "analysis_config_id", "f_measure")


# ---------------------------------------------------------------------------
# Loading + validation
# ---------------------------------------------------------------------------


def load_spec(path: Path) -> dict:
    """Load and validate a table-spec config.

    Args:
        path: Config path.

    Returns:
        The ``params.measeval_latex`` section.

    Raises:
        ValueError, KeyError: Bad envelope, keys or types, or an id used in more than one row.
    """
    cfg = _load_envelope(path, "measeval")
    if set(cfg["params"]) != {SECTION}:
        raise ValueError(f"{path}: params keys {sorted(cfg['params'])} must be exactly [{SECTION!r}]")
    spec = get_section(cfg, SECTION, required_keys=SECTION_KEYS)

    decimals = spec["decimals"]
    if isinstance(decimals, bool) or not isinstance(decimals, int) or decimals < 0:
        raise ValueError(f"{path}: decimals must be a non-negative int, got {decimals!r}")
    for key in ("analysis_config", "output", "caption", "label"):
        if not isinstance(spec[key], str) or not spec[key]:
            raise ValueError(f"{path}: {key} must be a non-empty string, got {spec[key]!r}")

    blocks = spec["blocks"]
    if not isinstance(blocks, list) or not blocks:
        raise ValueError(f"{path}: blocks must be a non-empty list")
    used: list[str] = []
    for block in blocks:
        if not isinstance(block, dict) or set(block) != {"rows"}:
            raise ValueError(f"{path}: each block needs exactly the key 'rows', got {block!r}")
        if not isinstance(block["rows"], list) or not block["rows"]:
            raise ValueError(f"{path}: a block has no rows")
        for row in block["rows"]:
            if not isinstance(row, dict) or set(row) != {"label", "id"}:
                raise ValueError(f"{path}: each row needs exactly keys ['id', 'label'], got {row!r}")
            if not isinstance(row["id"], str) or not row["id"]:
                raise ValueError(f"{path}: row {row['label']!r} id must be a non-empty string")
            used.append(row["id"])
    dupes = sorted({i for i in used if used.count(i) > 1})
    if dupes:
        raise ValueError(f"{path}: experiment id(s) used in more than one row: {dupes}")
    return spec


def load_results(analysis_config_id: str) -> pd.DataFrame:
    """Load the CSV a measeval_evaluation config wrote, after checking it is current.

    Args:
        analysis_config_id: Config that wrote the CSV.

    Returns:
        F-measure per experiment id (index) and annotation type (columns).

    Raises:
        FileNotFoundError: The referenced config or its CSV is missing.
        ValueError: The config is not a measeval_evaluation config, the CSV lacks
            needed columns, carries another ``analysis_config_id`` or dataset, is a
            dev run, has a missing or duplicate (id, annot_type) row, or its id set is
            not exactly the config's ``experiment_ids`` (stale CSV).
    """
    config_path = analysis_config_path("measeval", analysis_config_id)
    cfg = _load_envelope(config_path, "measeval")
    params = cfg["params"]
    if "measeval_evaluation" not in params or "experiment_ids" not in params:
        raise ValueError(f"{config_path}: not a measeval_evaluation analysis config")
    output = Path(params["measeval_evaluation"]["output"])
    csv_path = output if output.is_absolute() else _REPO_ROOT / output
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} (output of {analysis_config_id}) does not exist -- run it first")

    df = pd.read_csv(csv_path)
    missing_cols = [c for c in _NEEDED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{csv_path}: missing column(s) {missing_cols}")
    if set(df["analysis_config_id"]) != {analysis_config_id}:
        raise ValueError(
            f"{csv_path}: analysis_config_id values {sorted(set(df['analysis_config_id']), key=str)} "
            f"!= {analysis_config_id!r}"
        )
    if set(df["dataset"]) != {"measeval"}:
        raise ValueError(f"{csv_path}: dataset values {sorted(set(df['dataset']))} != ['measeval']")
    if df["dev"].astype(bool).any():
        raise ValueError(f"{csv_path}: contains dev-mode rows, which are not comparable scores")
    extra = sorted(set(df["experiment_id"]) - set(params["experiment_ids"]))
    absent = sorted(set(params["experiment_ids"]) - set(df["experiment_id"]))
    if extra or absent:
        raise ValueError(
            f"{csv_path} is stale relative to {config_path}: CSV has {len(extra)} id(s) the config "
            f"no longer lists ({extra}), and is missing {len(absent)} id(s) the config lists "
            f"({absent}). Rerun `python analysis/measeval_evaluation.py --config {config_path}`."
        )
    types = [t for t, _ in COLUMNS]
    if df.duplicated(["experiment_id", "annot_type"]).any():
        raise ValueError(f"{csv_path}: duplicate (experiment_id, annot_type) rows")
    table = df[df["annot_type"].isin(types)].pivot(index="experiment_id", columns="annot_type", values="f_measure")
    table = table.reindex(columns=types)
    holes = table.isna()
    if holes.any().any():
        # A missing row, or a scorer that printed no F (no data / no match for that type).
        bad = [(i, t) for i in table.index for t in types if holes.loc[i, t]]
        raise ValueError(f"{csv_path}: no F-measure for (experiment_id, annot_type) {bad}")
    return table


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_f(value: float, *, decimals: int, bold: bool = False) -> str:
    """Format one F-measure as a percentage, e.g. ``71.4``.

    Args:
        value: F-measure in [0, 1].
        decimals: Decimal places.
        bold: Wrap in ``\\textbf``.

    Returns:
        LaTeX string.

    Raises:
        ValueError: ``value`` is not a finite number in [0, 1].
    """
    if isinstance(value, bool) or not (isinstance(value, (int, float)) and math.isfinite(value) and 0.0 <= value <= 1.0):
        raise ValueError(f"F-measure {value!r} is not a finite rate in [0, 1]")
    text = f"{100 * value:.{decimals}f}"
    return f"\\textbf{{{text}}}" if bold else text


def build_table(spec: dict, results: pd.DataFrame) -> str:
    """Render the full LaTeX ``table`` environment.

    Args:
        spec: Output of ``load_spec``.
        results: Output of ``load_results``.

    Returns:
        LaTeX source.

    Raises:
        KeyError: A spec id is missing from the results.
    """
    shown = [r["id"] for b in spec["blocks"] for r in b["rows"]]
    missing = [i for i in shown if i not in results.index]
    if missing:
        raise KeyError(f"{missing} not in the results CSV")
    # Best value per column among the rows shown; exact ties are all bolded.
    best = {t: results.loc[shown, t].max() for t, _ in COLUMNS}
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{spec['caption']}}}",
        f"\\label{{{spec['label']}}}",
        f"\\begin{{tabular}}{{l{'c' * len(COLUMNS)}}}",
        "\\toprule",
        "Method & " + " & ".join(h for _, h in COLUMNS) + " \\\\",
    ]
    for block in spec["blocks"]:
        lines.append("\\midrule")
        for row in block["rows"]:
            res = results.loc[row["id"]]
            try:
                cells = [format_f(res[t], decimals=spec["decimals"], bold=res[t] == best[t]) for t, _ in COLUMNS]
            except ValueError as e:
                raise ValueError(f"{row['id']}: {e}") from e
            lines.append(f"{row['label']} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI: build the table from ``--config``, write it, and print it.

    Args:
        argv: Arguments (None = sys.argv).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="analysis-configs/measeval/<id>.yaml table spec")
    args = parser.parse_args(argv)

    spec = load_spec(args.config)
    text = build_table(spec, load_results(spec["analysis_config"]))
    output = Path(spec["output"])
    output = output if output.is_absolute() else _REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    print(text)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
