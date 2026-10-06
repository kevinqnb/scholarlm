"""Render the ``metrics_{probe,ntp}.csv`` files written by
``analysis/calibration_updated_v3.py`` as LaTeX tables, one set per test setting
(synthetic and real are never mixed in a table).

This only *formats* numbers that calibration_updated_v3.py already wrote; it
computes no metric. Per setting it writes three files into ``output_dir``:

  calibration_{setting}_smece.tex       main table: smooth ECE (+ interval).
        Rows: an NTP block then a Probe block (one midrule between), each with
        one row per *training* dataset (the probe / NTP calibrator was trained
        on that dataset's synthetic data). Columns: the *test* dataset.
  calibration_{setting}_classification.tex   N, label rate, Acc, Prec, Rec, F1, AUROC
        (point estimates; the CSV stores no intervals for these). 18 rows:
        {NTP, Probe} x train x test.
  calibration_{setting}_ece_variants.tex     ECE, adaptive (equal-mass) ECE and
        debiased RMSCE, each with its interval. Same 18 rows.

Not shown: ``Validity``. It is Precision by definition (TP / (TP + FP)), so it is
asserted equal to Precision and omitted rather than printed twice.

Guards (the CSVs carry no config id, so provenance cannot be fully proven here):
  - each CSV has exactly {syn, real} x train x test rows, no duplicate keys, its
    ``Type`` column matches the file, and both files share one ``Judge model``;
  - the dataset sets equal the calibration config's ``datasets``; ``Platt N`` is
    NaN on syn rows and equals the config's ``platt_n`` on real rows;
  - every cell used is finite (NaN is an error, never rendered) -- except Precision
    and F1 when no row is predicted valid (Recall == 0, stored Validity == 0), which
    print as ``--`` with a caption note and a console line naming the cells; the point
    estimate lies in [0, 1] (the smooth-ECE interval is never clipped: its
    lower bound can be negative);
  - ``lo <= point <= hi`` is asserted for every interval except the plug-in ECE
    and adaptive-ECE columns: their percentile-bootstrap intervals can sit above
    the point estimate when the true ECE is near zero (expected; e.g. syn NTP
    nfix->nfix), so those print as ``point [lo, hi]`` exactly as stored.

Usage
-----
    python analysis/calibration_latex.py --config analysis/analysis-configs/<id>.yaml

Table spec (``params.calibration_latex``; every key required, no defaults)::

    calibration_config: <id of a calibration_updated_v3 analysis config>
    datasets: {PLW: pond, NF: nfix, SM: supermat}   # ordered; key = label
    decimals: 3
    ci_format: pm | interval
    output_dir: analysis/results/calibration/<name>
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

from analysis.analysis_config import (  # noqa: E402
    ANALYSIS_CONFIGS_ROOT, _load_envelope, get_section, load_calibration_v3_config,
)

SECTION = "calibration_latex"
SECTION_KEYS = ("calibration_config", "datasets", "decimals", "ci_format", "output_dir", "label_prefix")
CI_FORMATS = ("pm", "interval")
MISSING_CELL = "--"
SETTINGS = {"syn": "synthetic", "real": "real"}
METHODS = (("NTP", "ntp"), ("Probe", "probe"))  # (CSV Type value, metrics_<file>.csv suffix)

# (CSV column, header) -- point estimates in the classification table.
CLASSIFICATION = (("Accuracy", "Acc."), ("Precision", "Prec."), ("Recall", "Rec."),
                  ("F1", "F1"), ("AUROC", "AUROC"))
# (CSV value column, lo column, hi column, header) -- intervals in the variants table.
# Last field: plug-in ECE with a percentile bootstrap interval. That interval can
# lie entirely above the point estimate when the true ECE is near zero (bootstrap
# replicates carry extra noise-floor bias), so these columns skip the lo <= point
# check and always print the exact [lo, hi]. Debiased RMSCE keeps the strict check.
ECE_VARIANTS = (("ECE", "ECE_lo", "ECE_hi", "ECE", True),
                ("ECE_em", "ECE_em_lo", "ECE_em_hi", "Adaptive ECE", True),
                ("RMSCE_db", "RMSCE_db_lo", "RMSCE_db_hi", "Debiased RMSCE", False))
SMECE = ("SmECE", "SmECE_lo", "SmECE_hi")

_KEY = ["Dataset type", "Train dataset", "Test dataset", "Type"]
_NEEDED_COLUMNS = (
    ["Dataset type", "Judge model", "Train dataset", "Test dataset", "Type", "N", "Label rate",
     "Platt N", "Validity"]
    + [c for c, _ in CLASSIFICATION]
    + [c for v in ECE_VARIANTS for c in v[:3]]
    + list(SMECE)
)


# ---------------------------------------------------------------------------
# Loading + validation
# ---------------------------------------------------------------------------


def load_spec(path: Path) -> dict:
    """The ``params.calibration_latex`` section, validated.

    Raises:
        ValueError/KeyError: malformed envelope or section, bad value types, or
            dataset labels/names that are empty or repeated.
    """
    cfg = _load_envelope(path)
    if set(cfg["params"]) != {SECTION}:
        raise ValueError(f"{path}: params keys {sorted(cfg['params'])} must be exactly [{SECTION!r}]")
    spec = get_section(cfg, SECTION, required_keys=SECTION_KEYS)
    if spec["ci_format"] not in CI_FORMATS:
        raise ValueError(f"{path}: ci_format must be one of {CI_FORMATS}, got {spec['ci_format']!r}")
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


def load_metrics(spec: dict) -> tuple[dict[str, pd.DataFrame], str]:
    """``{Type: frame}`` for NTP and Probe, plus the single judge model, after
    validating the CSVs against the referenced calibration config.

    Raises:
        FileNotFoundError: a metrics CSV is missing.
        ValueError: any guard in the module docstring fails.
    """
    cal_id = spec["calibration_config"]
    cal_cfg = load_calibration_v3_config(ANALYSIS_CONFIGS_ROOT / f"{cal_id}.yaml")
    config_datasets = set(cal_cfg["params"]["datasets"])
    if set(spec["datasets"].values()) != config_datasets:
        raise ValueError(f"spec datasets {sorted(spec['datasets'].values())} != {cal_id} datasets {sorted(config_datasets)}")
    platt_n = cal_cfg["params"]["platt_n"]
    results_dir = _REPO_ROOT / "analysis" / "results" / "calibration" / cal_id

    frames, judges = {}, set()
    for kind, suffix in METHODS:
        csv_path = results_dir / f"metrics_{suffix}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} does not exist -- run calibration_updated_v3.py on {cal_id} first")
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
        syn, real = df[df["Dataset type"] == "syn"], df[df["Dataset type"] == "real"]
        if syn["Platt N"].notna().any():
            raise ValueError(f"{csv_path}: syn rows have a Platt N")
        if not (real["Platt N"] == platt_n).all():
            raise ValueError(f"{csv_path}: real rows' Platt N {sorted(set(real['Platt N']))} != config platt_n {platt_n}")
        judges |= set(df["Judge model"])
        frames[kind] = df
    if len(judges) != 1:
        raise ValueError(f"metrics CSVs for {cal_id} mix judge models: {sorted(judges)}")
    return frames, judges.pop()


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _check_finite(name: str, v: float) -> None:
    if not (isinstance(v, (int, float)) and math.isfinite(v)):
        raise ValueError(f"{name}={v!r} is not finite")


def format_estimate(
    point: float, lo: float, hi: float, *, decimals: int, ci_format: str, bold: bool = False,
    plugin_percentile: bool = False,
) -> str:
    """``point`` with its interval, e.g. ``0.051 $\\pm$ 0.009``; ``bold`` wraps
    only the point estimate. Bounds are not range-checked (smooth-ECE's lower
    bound can be negative) and never clipped.

    Raises:
        ValueError: a value is non-finite, the point is outside [0, 1], or
            lo <= point <= hi fails.
    """
    for name, v in (("point", point), ("ci_lo", lo), ("ci_hi", hi)):
        _check_finite(name, v)
    if not 0.0 <= point <= 1.0:
        raise ValueError(f"point {point} outside [0, 1]")
    if not lo <= point <= hi and not plugin_percentile:
        raise ValueError(f"point {point} outside its interval [{lo}, {hi}]")
    if plugin_percentile:
        ci_format = "interval"  # percentile interval need not be centred on (or contain) the point
    p = f"{point:.{decimals}f}"
    if bold:
        p = f"\\textbf{{{p}}}"
    if ci_format == "pm":
        return f"{p} $\\pm$ {(hi - lo) / 2:.{decimals}f}"
    if ci_format == "interval":
        return f"{p} [{lo:.{decimals}f}, {hi:.{decimals}f}]"
    raise ValueError(f"ci_format must be one of {CI_FORMATS}, got {ci_format!r}")


def format_point(v: float, *, decimals: int) -> str:
    _check_finite("value", v)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"value {v} outside [0, 1]")
    return f"{v:.{decimals}f}"


def _cell(row: pd.Series, cols: tuple, *, spec: dict, bold: bool = False, plugin_percentile: bool = False) -> str:
    try:
        return format_estimate(row[cols[0]], row[cols[1]], row[cols[2]], decimals=spec["decimals"],
                               ci_format=spec["ci_format"], bold=bold, plugin_percentile=plugin_percentile)
    except ValueError as e:
        raise ValueError(f"{row['Dataset type']}/{row['Type']} train={row['Train dataset']} "
                         f"test={row['Test dataset']} {cols[0]}: {e}") from e


def _lookup(frames: dict[str, pd.DataFrame], setting: str) -> dict[tuple[str, str, str], pd.Series]:
    """(Type, train, test) -> row, for one setting."""
    out = {}
    for kind, df in frames.items():
        for _, r in df[df["Dataset type"] == setting].iterrows():
            out[(kind, r["Train dataset"], r["Test dataset"])] = r
    return out


def _wrap(spec: dict, setting: str, name: str, body: list[str], caption: str, colspec: str) -> str:
    return "\n".join([
        "\\begin{table}[t]", "\\centering", "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{spec['label_prefix']}-{setting}-{name}}}",
        f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", *body, "\\bottomrule", "\\end{tabular}", "\\end{table}", "",
    ])


def _caption_tail(spec: dict, setting: str, judge: str, platt_n: int, *, with_ci: bool = True) -> str:
    labels = ", ".join(f"{k}: {v}" for k, v in spec["datasets"].items())
    ci = ("$\\pm$ half-width of the" if spec["ci_format"] == "pm" else "bracketed") + " 95\\% bootstrap interval"
    if setting == "syn":
        s = "Synthetic test sets; no recalibration."
    else:
        s = (f"Real extractions, Platt-scaled on {platt_n} labelled rows per test dataset. "
             "Off-diagonal cells exclude the training probe's documents, so the test rows (N) "
             "differ down a column.")
    return f"{s} Judge model \\texttt{{{judge}}}. Rows: dataset the probe / NTP calibrator was trained on; {labels}." + (f" Intervals: {ci}." if with_ci else "")


def build_smece_table(spec: dict, frames: dict, setting: str, judge: str, platt_n: int) -> str:
    """Main table: smooth ECE, NTP block then Probe block, train rows x test columns."""
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
    caption = f"Smooth ECE (lower is better; best per column in bold). " + _caption_tail(spec, setting, judge, platt_n)
    return _wrap(spec, setting, "smece", body, caption, "l" + "c" * len(labels))


def _long_rows(spec: dict, frames: dict, setting: str, cell_fn) -> list[str]:
    """18-row body: NTP block, midrule, Probe block; stub columns method / train / test."""
    lk = _lookup(frames, setting)
    labels = spec["datasets"]
    body = []
    for kind, _ in METHODS:
        body.append("\\midrule")
        for tl, tr in labels.items():
            for tel, te in labels.items():
                body.append(f"{kind} & {tl} & {tel} & " + cell_fn(lk[(kind, tr, te)]) + " \\\\")
    return body


def build_classification_table(spec: dict, frames: dict, setting: str, judge: str, platt_n: int) -> str:
    d = spec["decimals"]

    undefined: list[str] = []

    def cells(r: pd.Series) -> str:
        where = f"{setting}/{r['Type']} train={r['Train dataset']} test={r['Test dataset']}"
        # No predicted positives: Precision is 0/0 and F1 follows. Rendered as an explicit
        # marker only when the row is consistent with that (Recall == 0 and stored
        # Validity == 0.0, which validity_rate_from_labels returns for that case).
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
               "test rows labelled valid." + note + " " + _caption_tail(spec, setting, judge, platt_n, with_ci=False))
    return _wrap(spec, setting, "classification", body, caption, "lll" + "r" + "c" * (1 + len(CLASSIFICATION)))


def build_variants_table(spec: dict, frames: dict, setting: str, judge: str, platt_n: int) -> str:
    def cells(r: pd.Series) -> str:
        return " & ".join(_cell(r, v[:3], spec=spec, plugin_percentile=v[4]) for v in ECE_VARIANTS)

    body = ["Method & Train & Test & " + " & ".join(v[3] for v in ECE_VARIANTS) + " \\\\",
            *_long_rows(spec, frames, setting, cells)]
    caption = ("Alternative calibration errors: equal-width ECE, equal-mass (adaptive) ECE, and debiased "
               "RMSCE (Kumar et al., 2019). " + _caption_tail(spec, setting, judge, platt_n))
    return _wrap(spec, setting, "ece-variants", body, caption, "lll" + "c" * len(ECE_VARIANTS))


def build_tables(spec: dict, frames: dict, judge: str, platt_n: int) -> dict[str, str]:
    """``{filename: tex}`` for every table, both settings."""
    out = {}
    for setting in SETTINGS:
        for name, fn in (("smece", build_smece_table), ("classification", build_classification_table),
                         ("ece_variants", build_variants_table)):
            out[f"calibration_{setting}_{name}.tex"] = fn(spec, frames, setting, judge, platt_n)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="analysis-configs/<id>.yaml table spec")
    args = parser.parse_args(argv)

    spec = load_spec(args.config)
    frames, judge = load_metrics(spec)
    platt_n = load_calibration_v3_config(ANALYSIS_CONFIGS_ROOT / f"{spec['calibration_config']}.yaml")["params"]["platt_n"]
    tables = build_tables(spec, frames, judge, platt_n)
    out_dir = Path(spec["output_dir"])
    out_dir = out_dir if out_dir.is_absolute() else _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, text in tables.items():
        (out_dir / fname).write_text(text)
        print(f"% ---- {fname}\n{text}")
    print(f"Wrote {len(tables)} tables to {out_dir}")


if __name__ == "__main__":
    main()
