"""Format recovery_validity.py result CSVs as a LaTeX table (methods x datasets).

Each cell is (recovery, validity) as percentages with bootstrap intervals; the best
point per column and metric is bolded. Formatting only: no metric is computed. The
cell-to-row mapping is guarded: ids are explicit (null renders "--"), each column's
CSV must belong to its named config and hold exactly that config's ids, and each
value must lie in its interval.

``ci_format``: ``pm`` prints the half-width (exact only for symmetric intervals);
``interval`` prints [lo, hi].

Usage
-----
    python analysis/recovery_validity_latex.py --config analysis/analysis-configs/recovery-validity/<id>.yaml

Table spec (``params.recovery_validity_latex``, every key required)::

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
OPTIONAL_KEYS = ("figure",)
CI_FORMATS = ("pm", "interval")
MISSING_CELL = "--"

# recovery_max_weight_matching isn't rendered; requiring it rejects older CSVs whose
# `recovery` column meant the matching count.
_NEEDED_COLUMNS = (
    "experiment_id", "dataset", "analysis_config_id",
    "recovery", "recovery_ci_lo", "recovery_ci_hi", "recovery_max_weight_matching",
    "validity", "validity_ci_lo", "validity_ci_hi",
)


# ---------------------------------------------------------------------------
# Loading + validation
# ---------------------------------------------------------------------------


def load_spec(path: Path) -> dict:
    """Load and validate a table-spec config.

    Args:
        path: Config path.

    Returns:
        The ``params.recovery_validity_latex`` section.

    Raises:
        ValueError, KeyError: Bad envelope, keys or types, a row whose ids don't
            cover exactly the dataset columns, or an id reused within a column.
    """
    cfg = _load_envelope(path, "recovery-validity")
    if set(cfg["params"]) != {SECTION}:
        raise ValueError(f"{path}: params keys {sorted(cfg['params'])} must be exactly [{SECTION!r}]")
    spec = get_section(cfg, SECTION, required_keys=SECTION_KEYS, optional_keys=OPTIONAL_KEYS)
    has_figure = "figure" in spec
    if has_figure and (not isinstance(spec["figure"], dict) or set(spec["figure"]) != {"output_dir", "group_font", "label_font"}
                       or not isinstance(spec["figure"]["output_dir"], str) or not spec["figure"]["output_dir"]
                       or spec["figure"]["group_font"] not in ("mono", "italic")
                       or spec["figure"]["label_font"] not in ("mono", "serif")):
        raise ValueError(
            f"{path}: figure must be a mapping with exactly output_dir (non-empty string) and "
            f"group_font ('mono' for model names, 'italic' otherwise) and label_font ('mono' for model/method names, 'serif')")

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
        ac = block["analysis_config"]
        ac_list = [ac] if isinstance(ac, str) else ac
        if not isinstance(ac_list, list) or not ac_list or not all(isinstance(a, str) and a for a in ac_list) \
                or len(set(ac_list)) != len(ac_list):
            raise ValueError(
                f"{path}: datasets.{col}.analysis_config must be a config id or a non-empty list of distinct ids, got {ac!r}")

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
            expected_keys = {"label", "ids", "short_label", "groups"} if has_figure else {"label", "ids"}
            if not isinstance(row, dict) or set(row) != expected_keys:
                raise ValueError(f"{path}: each row needs exactly keys {sorted(expected_keys)}, got {row!r}")
            if has_figure and not (
                isinstance(row["short_label"], str) and row["short_label"]
                and isinstance(row["groups"], list) and row["groups"] and len(set(row["groups"])) == len(row["groups"])
                and all(isinstance(g, str) and g for g in row["groups"])
            ):
                raise ValueError(
                    f"{path}: row {row['label']!r} needs a non-empty short_label and a non-empty list of distinct groups")
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
    if has_figure:
        groups_of: dict[str, list[str]] = {}
        for b in blocks:
            for r in b["rows"]:
                for g in r["groups"]:
                    groups_of.setdefault(g, []).append(r["short_label"])
        if any(len(v) != len(set(v)) for v in groups_of.values()):
            raise ValueError(f"{path}: short_label repeated within a group: {groups_of}")
    for col, ids in used.items():
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"{path}: column {col!r} uses experiment id(s) in more than one cell: {dupes}")
    return spec


def _load_one_results(col: str, dataset: str, analysis_config_id: str) -> pd.DataFrame:
    """Load the results CSV a recovery-validity config wrote, after checking it is current.

    Args:
        col: Table column key (for error messages).
        dataset: Expected dataset.
        analysis_config_id: Config that wrote the CSV.

    Returns:
        Results indexed by experiment_id.

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


def load_results(col: str, dataset: str, analysis_config) -> pd.DataFrame:
    """Load and merge the results of one or several recovery-validity configs for a column.

    Args:
        col: Table column key (for error messages).
        dataset: Expected dataset.
        analysis_config: One config id or a list of ids; each is validated by
            ``_load_one_results``.

    Returns:
        Results indexed by experiment_id.

    Raises:
        ValueError: An experiment id appears in more than one config, or the configs
            scored against different ground-truth files (numbers not comparable),
            plus anything ``_load_one_results`` raises.
    """
    ids = [analysis_config] if isinstance(analysis_config, str) else list(analysis_config)
    frames = [_load_one_results(col, dataset, a) for a in ids]
    merged = pd.concat(frames)
    if merged.index.duplicated().any():
        raise ValueError(f"{col}: experiment id(s) in more than one of {ids}: {sorted(set(merged.index[merged.index.duplicated()]))}")
    gts = set(merged["ground_truth_file"])
    if len(gts) != 1:
        raise ValueError(f"{col}: configs {ids} scored against different ground-truth files: {sorted(gts)}")
    return merged


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_estimate(
    point: float, lo: float, hi: float, *, decimals: int, ci_format: str, bold: bool = False,
) -> str:
    """Format one rate as a percentage with its interval, e.g. ``38.8 $\\pm$ 5.4``.

    Args:
        point: Point estimate in [0, 1].
        lo: Interval lower bound.
        hi: Interval upper bound.
        decimals: Decimal places.
        ci_format: ``"pm"`` or ``"interval"``.
        bold: Bold the point estimate only.

    Returns:
        LaTeX string.

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
    """Format one table cell as ``(recovery, validity)``.

    Args:
        row: One results row.
        decimals: Decimal places.
        ci_format: ``"pm"`` or ``"interval"``.
        bold_recovery: Bold the recovery point.
        bold_validity: Bold the validity point.

    Returns:
        LaTeX string.

    Raises:
        ValueError: A value is invalid (message prefixed with the experiment id).
    """
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
    """Render the full LaTeX ``table`` environment.

    Args:
        spec: Output of ``load_spec``.
        results: Column key -> results frame from ``load_results``.

    Returns:
        LaTeX source.

    Raises:
        KeyError: A spec id is missing from its column's CSV.
    """
    cols = list(spec["datasets"])
    # Best point per column among the rows shown; exact ties are all bolded.
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


def build_figures(spec: dict, results: dict[str, pd.DataFrame]) -> list[Path]:
    """Write one recovery/validity interval figure per dataset column.

    Rows whose id is null for a column are omitted from that column's figure (as the
    table shows ``--``). A row listing several groups appears in each of them. Colours are keyed on ``short_label`` in config order, so a
    method keeps its colour across groups and datasets.

    Args:
        spec: Output of ``load_spec`` (must carry ``figure``).
        results: Column key -> results frame from ``load_results``.

    Returns:
        Written figure paths.

    Raises:
        KeyError: A spec id is missing from its column's CSV.
    """
    from analysis.common.recovery_validity_plot import plot_recovery_validity

    out_dir = Path(spec["figure"]["output_dir"])
    out_dir = out_dir if out_dir.is_absolute() else _REPO_ROOT / out_dir
    rows = [r for b in spec["blocks"] for r in b["rows"]]
    # A method missing from some column (null id) takes the last colours, so every figure uses an
    # unbroken prefix of tab10 (no skipped hue) while each method keeps its colour across datasets.
    missing_somewhere = {r["short_label"] for r in rows if any(i is None for i in r["ids"].values())}
    ordered = list(dict.fromkeys(r["short_label"] for r in rows))
    color_keys = [k for k in ordered if k not in missing_somewhere] + [k for k in ordered if k in missing_somewhere]
    written = []
    for col in spec["datasets"]:
        entries = []
        for row in rows:
            exp_id = row["ids"][col]
            if exp_id is None:
                continue
            if exp_id not in results[col].index:
                raise KeyError(f"row {row['label']!r}, column {col}: {exp_id!r} not in that column's results CSV")
            res = results[col].loc[exp_id]
            for group in row["groups"]:
                entries.append({
                    "label": row["short_label"], "group": group, "color_key": row["short_label"],
                    "recovery": (res["recovery"], res["recovery_ci_lo"], res["recovery_ci_hi"]),
                    "validity": (res["validity"], res["validity_ci_lo"], res["validity_ci_hi"]),
                })
        path = out_dir / f"{col}.pdf"
        plot_recovery_validity(entries, color_keys, path, group_font=spec["figure"]["group_font"],
                               label_font=spec["figure"]["label_font"])
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> None:
    """CLI: build the table from ``--config``, write it, and print it.

    Args:
        argv: Arguments (None = sys.argv).
    """
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
    if "figure" in spec:
        for path in build_figures(spec, results):
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
