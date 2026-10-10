"""Format v4 calibration metrics CSVs as LaTeX tables (formatting only).

Reads metrics_{probe,ntp}.csv written by calibration.py (``labels: llm_matching``) or
calibration_validated.py (``labels: human_validated``) for a v4 config. Captions state the
config's recalibration method and fit settings, which the CSVs are checked against.
For each of syn and real it writes three tables into ``output_dir``:
  - calibration_{setting}_smece.tex: smooth ECE, training-dataset rows x test-dataset columns.
  - calibration_{setting}_classification.tex: N, label rate, Acc/Prec/Rec/F1/AUROC.
  - calibration_{setting}_ece_variants.tex: ECE, adaptive ECE, debiased RMSCE.

Validity is asserted equal to Precision and not printed. CSV rows must exactly
cover the config's datasets, and their recalibration columns must match the config. Non-finite values are errors, except Precision/F1 with
no predicted positives, which print as "--". Intervals are printed as stored
([lo, hi]); for |gap|-type errors they may lie above the point.

Usage
-----
    python analysis/calibration_latex.py --config analysis/analysis-configs/<calibration | calibration-validated>/<id>.yaml

Table spec (``params.calibration_latex``, every key required)::

    calibration_config: <id of a v4 calibration or calibration-validated analysis config>
    labels: llm_matching | human_validated   # calibration config | calibration-validated config
    datasets: {PLW: pond, NF: nfix, SM: supermat}   # ordered; key = label
    decimals: 3
    output_dir: analysis/results/<calibration | calibration-validated>/<name>
    label_prefix: tab:calibration
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
    _load_envelope, analysis_config_path, analysis_results_dir, get_section, load_calibration_v4_config,
    load_calibration_validated_config,
)

SECTION = "calibration_latex"
SECTION_KEYS = ("calibration_config", "labels", "datasets", "decimals", "output_dir", "label_prefix")
# Real-cell label source -> loader for the referenced calibration config.
LABEL_LOADERS = {"llm_matching": load_calibration_v4_config, "human_validated": load_calibration_validated_config}
# Real-cell label source -> analysis type (of the calibration config, its results, and this config).
LABEL_TYPES = {"llm_matching": "calibration", "human_validated": "calibration-validated"}
# Real-cell label source -> script that writes the metrics CSVs (for error messages).
LABEL_SCRIPTS = {"llm_matching": "calibration.py", "human_validated": "calibration_validated.py"}
MISSING_CELL = "--"
SETTINGS = {"syn": "synthetic", "real": "real"}
METHODS = (("NTP", "ntp"), ("Probe", "probe"))  # (CSV Type value, metrics_<file>.csv suffix)

# (CSV column, header) -- point estimates in the classification table.
CLASSIFICATION = (("Accuracy", "Acc."), ("Precision", "Prec."), ("Recall", "Rec."),
                  ("F1", "F1"), ("AUROC", "AUROC"))
# (CSV value column, lo column, hi column, header) -- intervals in the variants table.
# Resampling inflates |gap|-type errors, so an interval can sit above its point; they
# are printed as exact [lo, hi] and lo <= point is not required.
ECE_VARIANTS = (("ECE", "ECE_lo", "ECE_hi", "ECE"),
                ("ECE_em", "ECE_em_lo", "ECE_em_hi", "Adaptive ECE"),
                ("RMSCE_db", "RMSCE_db_lo", "RMSCE_db_hi", "Debiased RMSCE"))
SMECE = ("SmECE", "SmECE_lo", "SmECE_hi")

_KEY = ["Dataset type", "Train dataset", "Test dataset", "Type"]
_NEEDED_COLUMNS = (
    ["Dataset type", "Judge model", "Train dataset", "Test dataset", "Type", "N", "Label rate",
     "Recalibration", "Fit source", "Fit n", "Fit seed", "pi_te", "Doc resamples", "Validity"]
    + [c for c, _ in CLASSIFICATION]
    + [c for v in ECE_VARIANTS for c in v[:3]]
    + list(SMECE)
)


# ---------------------------------------------------------------------------
# Loading + validation
# ---------------------------------------------------------------------------


def load_spec(path: Path) -> dict:
    """Load and validate a calibration_latex table spec.

    Args:
        path: Config path (in calibration/ or calibration-validated/, matching ``labels``).

    Returns:
        The ``params.calibration_latex`` section.

    Raises:
        ValueError, KeyError: Bad location, envelope, keys or values, or repeated datasets.
    """
    if Path(path).parent.name not in LABEL_TYPES.values():
        raise ValueError(f"{path}: a calibration_latex config must live in one of "
                         f"{sorted(LABEL_TYPES.values())}/, found in {Path(path).parent.name}/")
    cfg = _load_envelope(path, Path(path).parent.name)
    if set(cfg["params"]) != {SECTION}:
        raise ValueError(f"{path}: params keys {sorted(cfg['params'])} must be exactly [{SECTION!r}]")
    spec = get_section(cfg, SECTION, required_keys=SECTION_KEYS)
    if spec["labels"] not in LABEL_LOADERS:
        raise ValueError(f"{path}: labels must be one of {tuple(LABEL_LOADERS)}, got {spec['labels']!r}")
    if LABEL_TYPES[spec["labels"]] != Path(path).parent.name:
        raise ValueError(f"{path}: labels {spec['labels']!r} formats {LABEL_TYPES[spec['labels']]} results, "
                         f"but the config lives in {Path(path).parent.name}/")
    d = spec["decimals"]
    if isinstance(d, bool) or not isinstance(d, int) or d < 0:
        raise ValueError(f"{path}: decimals must be a non-negative int, got {d!r}")
    for key in ("calibration_config", "output_dir", "label_prefix"):
        if not isinstance(spec[key], str) or not spec[key]:
            raise ValueError(f"{path}: {key} must be a non-empty string, got {spec[key]!r}")
    ds = spec["datasets"]
    if not isinstance(ds, dict) or not ds or not all(isinstance(k, str) and isinstance(v, str) for k, v in ds.items()):
        raise ValueError(f"{path}: datasets must be a non-empty mapping of label -> dataset name")
    if len(set(ds.values())) != len(ds):
        raise ValueError(f"{path}: datasets maps two labels to one dataset: {ds}")
    return spec


def load_calibration_config(spec: dict) -> dict:
    """Load the referenced calibration config with the loader for ``spec["labels"]``.

    Args:
        spec: Output of ``load_spec``.

    Returns:
        The calibration config dict.
    """
    return LABEL_LOADERS[spec["labels"]](analysis_config_path(LABEL_TYPES[spec["labels"]], spec["calibration_config"]))


def _check_recalibration_columns(df: pd.DataFrame, cal_params: dict, csv_path: Path) -> None:
    """Check a metrics CSV's recalibration columns against the config that should have written it.

    Syn rows are never recalibrated, so their columns are empty. Real rows carry the
    config's method, fit source and fit seed; ``Fit n`` is ``fit_n`` for sample, 0 for
    manual and the evaluated row count for oracle; ``pi_te`` is the config's
    ``pi_te_estimate`` for manual (otherwise it is read from the fit rows, so only finite).

    Args:
        df: One metrics CSV.
        cal_params: ``params`` of the calibration config.
        csv_path: Path (for messages).

    Raises:
        ValueError: Any column disagrees with the config.
    """
    cols = ["Recalibration", "Fit source", "Fit n", "Fit seed", "pi_te"]
    syn, real = df[df["Dataset type"] == "syn"], df[df["Dataset type"] == "real"]
    if syn[cols].notna().any().any():
        raise ValueError(f"{csv_path}: syn rows have recalibration columns {syn[cols].notna().any().to_dict()}")
    source = cal_params["fit_source"]
    for col, want in (("Recalibration", cal_params["recalibration"]), ("Fit source", source)):
        if not (real[col] == want).all():
            raise ValueError(f"{csv_path}: real rows' {col} {sorted(set(real[col]))} != config {want!r}")
    if source == "sample":
        if not (real["Fit n"] == cal_params["fit_n"]).all() or not (real["Fit seed"] == cal_params["fit_seed"]).all():
            raise ValueError(f"{csv_path}: real rows' Fit n / Fit seed {sorted(set(real['Fit n']))} / "
                             f"{sorted(set(real['Fit seed']))} != config {cal_params['fit_n']} / {cal_params['fit_seed']}")
    elif source == "oracle":
        if not (real["Fit n"] == real["N"]).all() or real["Fit seed"].notna().any():
            raise ValueError(f"{csv_path}: oracle rows must fit on all N evaluated rows and carry no Fit seed")
    else:
        if not (real["Fit n"] == 0).all() or real["Fit seed"].notna().any():
            raise ValueError(f"{csv_path}: manual rows must have Fit n 0 and no Fit seed")
        for ds, block in cal_params["datasets"].items():
            got = real.loc[real["Test dataset"] == ds, "pi_te"]
            if not (got == block["pi_te_estimate"]).all():
                raise ValueError(f"{csv_path}: {ds} pi_te {sorted(set(got))} != config {block['pi_te_estimate']}")
    if not (real["pi_te"].between(0, 1, inclusive="neither")).all():
        raise ValueError(f"{csv_path}: real rows' pi_te outside (0, 1)")
    if not (df["Doc resamples"] == cal_params["n_boot"]).all():
        raise ValueError(f"{csv_path}: Doc resamples {sorted(set(df['Doc resamples']))} != config n_boot {cal_params['n_boot']}")


def load_metrics(spec: dict) -> tuple[dict[str, pd.DataFrame], str]:
    """Load and validate both metrics CSVs against the referenced calibration config.

    Args:
        spec: Output of ``load_spec``.

    Returns:
        ``({"NTP": df, "Probe": df}, judge_model)``.

    Raises:
        FileNotFoundError: A metrics CSV is missing.
        ValueError: Wrong columns, rows, datasets or judges, or recalibration columns
            that disagree with the config.
    """
    cal_id = spec["calibration_config"]
    cal_cfg = load_calibration_config(spec)
    config_datasets = set(cal_cfg["params"]["datasets"])
    if set(spec["datasets"].values()) != config_datasets:
        raise ValueError(f"spec datasets {sorted(spec['datasets'].values())} != {cal_id} datasets {sorted(config_datasets)}")
    results_dir = analysis_results_dir(LABEL_TYPES[spec["labels"]]) / cal_id

    frames, judges = {}, set()
    for kind, suffix in METHODS:
        csv_path = results_dir / f"metrics_{suffix}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} does not exist -- run analysis/{LABEL_SCRIPTS[spec['labels']]} on {cal_id} first")
        df = pd.read_csv(csv_path)
        missing = [c for c in _NEEDED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{csv_path}: missing column(s) {missing}")
        if set(df["Type"]) != {kind}:
            raise ValueError(f"{csv_path}: Type values {sorted(set(df['Type']))} != {kind!r}")
        if df.duplicated(_KEY).any():
            raise ValueError(f"{csv_path}: duplicate (setting, train, test, type) rows")
        expected = {(s, tr, te) for s in SETTINGS for tr in config_datasets for te in config_datasets}
        got = set(zip(df["Dataset type"], df["Train dataset"], df["Test dataset"]))
        if got != expected or len(df) != len(expected):
            raise ValueError(f"{csv_path}: rows {len(df)} do not cover exactly {{syn,real}} x train x test of {sorted(config_datasets)}")
        _check_recalibration_columns(df, cal_cfg["params"], csv_path)
        judges |= set(df["Judge model"])
        frames[kind] = df
    if len(judges) != 1:
        raise ValueError(f"metrics CSVs for {cal_id} mix judge models: {sorted(judges)}")
    return frames, judges.pop()


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _check_finite(name: str, v: float) -> None:
    """Fail unless ``v`` is a finite number.

    Args:
        name: Value name (for the message).
        v: Value.

    Raises:
        ValueError: ``v`` is non-numeric or non-finite.
    """
    if not (isinstance(v, (int, float)) and math.isfinite(v)):
        raise ValueError(f"{name}={v!r} is not finite")


def format_estimate(point: float, lo: float, hi: float, *, decimals: int, bold: bool = False) -> str:
    """Format a value with its interval, e.g. ``0.051 [0.042, 0.060]`` (bounds not clipped).

    Args:
        point: Point estimate in [0, 1].
        lo: Interval lower bound.
        hi: Interval upper bound.
        decimals: Decimal places.
        bold: Bold the point only.

    Returns:
        LaTeX string.

    Raises:
        ValueError: a value is non-finite, the point is outside [0, 1], or lo > hi.
    """
    for name, v in (("point", point), ("ci_lo", lo), ("ci_hi", hi)):
        _check_finite(name, v)
    if not 0.0 <= point <= 1.0:
        raise ValueError(f"point {point} outside [0, 1]")
    if lo > hi:
        raise ValueError(f"interval [{lo}, {hi}] has lo > hi")
    p = f"{point:.{decimals}f}"
    if bold:
        p = f"\\textbf{{{p}}}"
    return f"{p} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def format_point(v: float, *, decimals: int) -> str:
    """Format a finite value in [0, 1] with fixed decimals.

    Args:
        v: Value.
        decimals: Decimal places.

    Returns:
        Formatted string.

    Raises:
        ValueError: Non-finite or outside [0, 1].
    """
    _check_finite("value", v)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"value {v} outside [0, 1]")
    return f"{v:.{decimals}f}"


def _cell(row: pd.Series, cols: tuple, *, spec: dict, bold: bool = False) -> str:
    """Format one interval cell, prefixing errors with the row's identity.

    Args:
        row: Metrics row.
        cols: (value, lo, hi) column names.
        spec: Table spec (for ``decimals``).
        bold: Bold the point.

    Returns:
        LaTeX string.
    """
    try:
        return format_estimate(row[cols[0]], row[cols[1]], row[cols[2]], decimals=spec["decimals"], bold=bold)
    except ValueError as e:
        raise ValueError(f"{row['Dataset type']}/{row['Type']} train={row['Train dataset']} "
                         f"test={row['Test dataset']} {cols[0]}: {e}") from e


def _lookup(frames: dict[str, pd.DataFrame], setting: str) -> dict[tuple[str, str, str], pd.Series]:
    """Index one setting's rows by (Type, train dataset, test dataset).

    Args:
        frames: Output of ``load_metrics``.
        setting: ``"syn"`` or ``"real"``.

    Returns:
        ``{(Type, train, test): row}``.
    """
    out = {}
    for kind, df in frames.items():
        for _, r in df[df["Dataset type"] == setting].iterrows():
            out[(kind, r["Train dataset"], r["Test dataset"])] = r
    return out


def _wrap(spec: dict, setting: str, name: str, body: list[str], caption: str, colspec: str) -> str:
    """Wrap table body lines in a LaTeX ``table`` / ``tabular`` environment.

    Args:
        spec: Table spec (for ``label_prefix``).
        setting: ``"syn"`` or ``"real"`` (part of the label).
        name: Table name (part of the label).
        body: Rows, including the header.
        caption: Caption text.
        colspec: tabular column spec.

    Returns:
        LaTeX source.
    """
    return "\n".join([
        "\\begin{table}[t]", "\\centering", "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{spec['label_prefix']}-{setting}-{name}}}",
        f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", *body, "\\bottomrule", "\\end{tabular}", "\\end{table}", "",
    ])


def _n_constant_down_columns(frames: dict, setting: str) -> bool:
    """Whether N is constant down each test-dataset column.

    Args:
        frames: Output of ``load_metrics``.
        setting: ``"syn"`` or ``"real"``.

    Returns:
        True if every (method, train) row has the same N for each test dataset.
    """
    df = pd.concat(frames.values())
    return bool(df[df["Dataset type"] == setting].groupby("Test dataset")["N"].nunique().eq(1).all())


def _auto_label_text(cal: dict) -> str:
    """Name the automatic (non-human) validity label the config used.

    Args:
        cal: ``params`` of the calibration config.

    Returns:
        ``"judge-or-match"`` if every dataset uses matching labels, else ``"judge"``.

    Raises:
        ValueError: Datasets disagree, so one caption cannot describe them.
    """
    flags = {block["use_matching_labels"] for block in cal["datasets"].values()}
    if len(flags) != 1:
        raise ValueError(f"datasets mix use_matching_labels {flags}; the caption cannot name one label source")
    return "judge-or-match" if flags.pop() else "judge"


def _recalibration_text(cal: dict) -> str:
    """Caption sentence describing the real-cell recalibration the config used.

    Args:
        cal: ``params`` of the calibration config.

    Returns:
        One sentence (no trailing period).
    """
    n, seed, source = cal["fit_n"], cal["fit_seed"], cal["fit_source"]
    auto = _auto_label_text(cal)
    method = cal["recalibration"]
    if method == "intercept_fit":
        how = "a single intercept shift (slope 1) fitted by maximum likelihood"
    elif method == "platt_fit":
        how = "Platt scaling (slope and intercept) fitted by maximum likelihood"
    else:
        how = "a prior-shift correction (slope 1; intercept moved by the log-odds change from the training prevalence)"
    if source == "sample":
        return (f"Recalibrated by {how} on {n} {auto}-labelled rows drawn once (seed {seed}) "
                "from the probe's training documents")
    if source == "manual":
        return f"Recalibrated by {how}, to an assumed valid rate per test dataset"
    return (f"Recalibrated by {how} on the evaluated rows themselves (oracle fit, a diagnostic that overstates "
            "calibration)")


def _caption_tail(spec: dict, frames: dict, setting: str, judge: str, cal: dict, *, with_ci: bool = True) -> str:
    """Shared caption text: test setting, recalibration, judge, row meaning, intervals.

    Args:
        spec: Table spec.
        frames: Output of ``load_metrics``.
        setting: ``"syn"`` or ``"real"``.
        judge: Judge model name.
        cal: ``params`` of the calibration config.
        with_ci: Include the interval description.

    Returns:
        Caption text.
    """
    labels = ", ".join(f"{k}: {v}" for k, v in spec["datasets"].items())
    ci = (f"bracketed 95\\% percentile interval over {cal['n_boot']} document-level bootstrap resamples of the "
          "evaluation set; point: the un-resampled value")
    if setting == "syn":
        s = "Synthetic test sets; no recalibration."
    else:
        s = _recalibration_text(cal) + "."
        if spec["labels"] == "human_validated":
            s += " Scored against human validity labels on the validated rows"
        else:
            s += f" Scored against {_auto_label_text(cal)} validity labels"
        # Checked from the data: exclusion is per cell, so N may vary.
        n_note = ("$N$ is the same down each column" if _n_constant_down_columns(frames, setting)
                  else "$N$ differs down a column")
        s += f", all outside the probe's training documents; {n_note}."
    return f"{s} Judge model \\texttt{{{judge}}}. Rows: dataset the probe was trained on; {labels}." + (f" Intervals: {ci}." if with_ci else "")


def build_smece_table(spec: dict, frames: dict, setting: str, judge: str, cal: dict) -> str:
    """Main table: smooth ECE, NTP then Probe blocks, train rows x test columns, best bolded.

    Args:
        spec: Table spec.
        frames: Output of ``load_metrics``.
        setting: ``"syn"`` or ``"real"``.
        judge: Judge model name.
        cal: ``params`` of the calibration config.

    Returns:
        LaTeX source.
    """
    lk = _lookup(frames, setting)
    labels = spec["datasets"]
    # Lowest smECE per test column across every row shown; exact ties all bold.
    best = {te: min(lk[(k, tr, te)][SMECE[0]] for k, _ in METHODS for tr in labels.values()) for te in labels.values()}
    body = [
        f"& \\multicolumn{{{len(labels)}}}{{c}}{{Test dataset}} \\\\",
        f"\\cmidrule(l){{2-{len(labels) + 1}}}",
        "Method (train) & " + " & ".join(labels) + " \\\\",
    ]
    for i, (kind, _) in enumerate(METHODS):
        body.append("\\midrule")
        for tl, tr in labels.items():
            cells = [_cell(lk[(kind, tr, te)], SMECE, spec=spec, bold=lk[(kind, tr, te)][SMECE[0]] == best[te])
                     for te in labels.values()]
            body.append(f"{kind} ({tl}) & " + " & ".join(cells) + " \\\\")
    caption = f"Smooth ECE (lower is better; best per column in bold). " + _caption_tail(spec, frames, setting, judge, cal)
    return _wrap(spec, setting, "smece", body, caption, "l" + "c" * len(labels))


def _long_rows(spec: dict, frames: dict, setting: str, cell_fn) -> list[str]:
    """Body rows for the long tables: method x train x test, NTP block then Probe block.

    Args:
        spec: Table spec.
        frames: Output of ``load_metrics``.
        setting: ``"syn"`` or ``"real"``.
        cell_fn: Row -> LaTeX cells after the three stub columns.

    Returns:
        Body lines.
    """
    lk = _lookup(frames, setting)
    labels = spec["datasets"]
    body = []
    for kind, _ in METHODS:
        body.append("\\midrule")
        for tl, tr in labels.items():
            for tel, te in labels.items():
                body.append(f"{kind} & {tl} & {tel} & " + cell_fn(lk[(kind, tr, te)]) + " \\\\")
    return body


def build_classification_table(spec: dict, frames: dict, setting: str, judge: str, cal: dict) -> str:
    """Classification table: N, label rate and threshold-0.5 metrics per cell.

    Args:
        spec: Table spec.
        frames: Output of ``load_metrics``.
        setting: ``"syn"`` or ``"real"``.
        judge: Judge model name.
        cal: ``params`` of the calibration config.

    Returns:
        LaTeX source.

    Raises:
        ValueError: Validity != Precision, or a value is invalid.
    """
    d = spec["decimals"]

    undefined: list[str] = []

    def cells(r: pd.Series) -> str:
        """LaTeX cells for one row; Precision/F1 print "--" when nothing is predicted positive."""
        where = f"{setting}/{r['Type']} train={r['Train dataset']} test={r['Test dataset']}"
        # No predicted positives (Precision NaN, Recall 0, Validity 0.0): print "--".
        no_pred_pos = math.isnan(r["Precision"]) and r["Recall"] == 0.0 and r["Validity"] == 0.0
        if no_pred_pos:
            undefined.append(where)
        elif r["Validity"] != r["Precision"]:  # Validity is Precision by definition
            raise ValueError(f"Validity {r['Validity']} != Precision {r['Precision']} in {where}")
        try:
            metrics = []
            for c, _ in CLASSIFICATION:
                if no_pred_pos and c in ("Precision", "F1"):
                    if not math.isnan(r[c]):
                        raise ValueError(f"{c}={r[c]} but Precision is NaN")
                    metrics.append(MISSING_CELL)
                else:
                    metrics.append(format_point(r[c], decimals=d))
            n, rate = int(r["N"]), format_point(r["Label rate"], decimals=d)
        except ValueError as e:
            raise ValueError(f"{where}: {e}") from e
        return " & ".join([f"{n:,}".replace(",", "{,}"), rate, *metrics])

    body = ["Method & Train & Test & $N$ & Pos. rate & " + " & ".join(h for _, h in CLASSIFICATION) + " \\\\",
            *_long_rows(spec, frames, setting, cells)]
    note = ""
    if undefined:
        print(f"[calibration_latex] {setting}: Precision/F1 undefined (no predicted positives) in: {undefined}")
        note = (f" -- : undefined (no row predicted valid at threshold 0.5; {len(undefined)} cell(s)"
                " in this table).")
    caption = ("Classification metrics at threshold 0.5 (no intervals stored). Pos. rate: fraction of "
               "test rows labelled valid." + note + " " + _caption_tail(spec, frames, setting, judge, cal, with_ci=False))
    return _wrap(spec, setting, "classification", body, caption, "lll" + "r" + "c" * (1 + len(CLASSIFICATION)))


def build_variants_table(spec: dict, frames: dict, setting: str, judge: str, cal: dict) -> str:
    """ECE-variants table: ECE, adaptive ECE and debiased RMSCE with intervals.

    Args:
        spec: Table spec.
        frames: Output of ``load_metrics``.
        setting: ``"syn"`` or ``"real"``.
        judge: Judge model name.
        cal: ``params`` of the calibration config.

    Returns:
        LaTeX source.
    """
    def cells(r: pd.Series) -> str:
        """Interval cells for each ECE variant in one row."""
        return " & ".join(_cell(r, v[:3], spec=spec) for v in ECE_VARIANTS)

    body = ["Method & Train & Test & " + " & ".join(v[3] for v in ECE_VARIANTS) + " \\\\",
            *_long_rows(spec, frames, setting, cells)]
    caption = ("Alternative calibration errors: equal-width ECE, equal-mass (adaptive) ECE, and debiased "
               "RMSCE (Kumar et al., 2019). " + _caption_tail(spec, frames, setting, judge, cal))
    return _wrap(spec, setting, "ece-variants", body, caption, "lll" + "c" * len(ECE_VARIANTS))


def build_tables(spec: dict, frames: dict, judge: str, cal: dict) -> dict[str, str]:
    """Build all three tables for both settings.

    Args:
        spec: Table spec.
        frames: Output of ``load_metrics``.
        judge: Judge model name.
        cal: ``params`` of the calibration config.

    Returns:
        ``{filename: LaTeX source}``.
    """
    out = {}
    for setting in SETTINGS:
        for name, fn in (("smece", build_smece_table), ("classification", build_classification_table),
                         ("ece_variants", build_variants_table)):
            out[f"calibration_{setting}_{name}.tex"] = fn(spec, frames, setting, judge, cal)
    return out


def main(argv: list[str] | None = None) -> None:
    """CLI: build every table from ``--config`` and write it to ``output_dir``.

    Args:
        argv: Arguments (None = sys.argv).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="analysis-configs/<calibration | calibration-validated>/<id>.yaml table spec")
    args = parser.parse_args(argv)

    spec = load_spec(args.config)
    frames, judge = load_metrics(spec)
    tables = build_tables(spec, frames, judge, load_calibration_config(spec)["params"])
    out_dir = Path(spec["output_dir"])
    out_dir = out_dir if out_dir.is_absolute() else _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, text in tables.items():
        (out_dir / fname).write_text(text)
        print(f"% ---- {fname}\n{text}")
    print(f"Wrote {len(tables)} tables to {out_dir}")


if __name__ == "__main__":
    main()
