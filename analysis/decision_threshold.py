"""Recovery/validity curves over a decision threshold on probe / NTP probabilities.

For each configured (train, test) calibration cell and each method (probe, NTP),
extractions whose predicted probability is ``>= t`` are kept and scored at every
threshold t: recovery is the fraction of ground-truth rows with a dataset-threshold
match edge to a KEPT extraction (any-edge, the same headline definition as
analysis/recovery_validity.py), validity the fraction of kept extractions that are
judged valid OR matched (recovery_validity.py's validity labels, unchanged).

Probabilities are the recalibrated ``probe_probs`` / ``ntp_probs`` of a v4 calibration
run's ``real`` cells (analysis/calibration_updated_v4.py's predictions.pkl). Those are
scored per final.json row and exist only for the test documents (the fit pool and the
probe's synthetic-training documents are excluded), so:

  - scoring happens on recovery_validity.py's own row space -- postprocessed.json
    extractions, ground truth from the calibration config's ground_truth_file, edges
    from the match cache at the dataset's fuzzy_threshold -- loaded through its own
    guarded loader (``_load_checked_inputs``), with split postprocessed rows
    inheriting their parent's probability by measurement_id, exactly as they inherit
    its judgement;
  - ground truth AND extractions are restricted to documents outside the cell's
    ``excluded_documents``. The t = 0 point (everything kept) is asserted equal to
    recovery_validity.py's verified full-frame recovered/validity masks restricted to
    those documents -- it does NOT equal the published all-document numbers.

A permutation control (probabilities shuffled across scored rows, ``n_permutations``
draws seeded by the envelope seed) is written and plotted alongside: its validity
should stay near the base rate at every threshold.

Usage
-----
    python analysis/decision_threshold.py analysis/analysis-configs/<id>.yaml
    bash analysis/submit.sh decision_threshold <id> --walltime HH:MM:SS --omp N

Writes analysis/results/decision_threshold/<config id>/curves.csv and
figures/<train>-<test>-<method>.pdf (+ decision_threshold_colorbar.pdf,
decision_threshold_legend.pdf).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))
sys.path.insert(0, str(_REPO_ROOT))

from analysis.common.config import (  # noqa: E402
    ANALYSIS_CONFIGS_ROOT, _is_int, _load_envelope, load_calibration_v4_config,
)
from analysis.common.metrics import recovery_rate_from_labels, validity_rate_from_labels  # noqa: E402

PARAM_KEYS = ("calibration_config_id", "cells", "thresholds", "highlight_threshold",
              "n_permutations", "invalid_datasets")
# (method, score column in prediction_store.check_real_cell's rows)
METHODS = (("probe", "probe_prob"), ("ntp", "ntp_prob"))
RESULTS_ROOT = _REPO_ROOT / "analysis" / "results" / "decision_threshold"
_THRESHOLD_NORM = (0.0, 1.0)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def load_decision_threshold_config(path: Path) -> dict:
    """Load and validate an analysis-configs/<id>.yaml for this script.

    ``params`` holds exactly PARAM_KEYS:
      - calibration_config_id: a v4 calibration config (its predictions.pkl is read);
      - cells: non-empty list of unique {train, test} mappings, both datasets of that
        calibration config;
      - thresholds: explicit, strictly increasing numbers in [0, 1], containing 0.0
        (the known-answer point) -- explicit, not generated, since the filter is
        inclusive and float-noise grid values would move rows across it;
      - highlight_threshold: one of thresholds (drawn as a diamond);
      - n_permutations: positive int, permutation-control draws;
      - invalid_datasets: mapping dataset -> reason (may be empty); every output row
        of a cell testing on that dataset carries the reason.

    Raises:
        ValueError: any of the above fails.
        FileNotFoundError: the calibration config does not exist.
    """
    cfg = _load_envelope(path)
    params = cfg["params"]
    if set(params) != set(PARAM_KEYS):
        raise ValueError(f"{path}: params keys {sorted(params)} must be exactly {sorted(PARAM_KEYS)}")

    cal_path = ANALYSIS_CONFIGS_ROOT / f"{params['calibration_config_id']}.yaml"
    if not cal_path.exists():
        raise FileNotFoundError(f"{path}: calibration config {cal_path} does not exist")
    datasets = set(load_calibration_v4_config(cal_path)["params"]["datasets"])

    cells = params["cells"]
    if not isinstance(cells, list) or not cells or not all(
        isinstance(c, dict) and set(c) == {"train", "test"} and c["train"] in datasets and c["test"] in datasets
        for c in cells
    ):
        raise ValueError(f"{path}: params.cells must be a non-empty list of {{train, test}} mappings "
                         f"over the calibration config's datasets {sorted(datasets)}, got {cells!r}")
    pairs = [(c["train"], c["test"]) for c in cells]
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"{path}: params.cells has duplicate cells: {pairs}")

    ts = params["thresholds"]
    if not isinstance(ts, list) or not ts or not all(_is_number(t) for t in ts):
        raise ValueError(f"{path}: params.thresholds must be a non-empty list of numbers, got {ts!r}")
    if any(not 0.0 <= t <= 1.0 for t in ts):
        raise ValueError(f"{path}: params.thresholds must lie in [0, 1], got {ts}")
    if any(b <= a for a, b in zip(ts, ts[1:])):
        raise ValueError(f"{path}: params.thresholds must be strictly increasing, got {ts}")
    if 0.0 not in ts:
        raise ValueError(f"{path}: params.thresholds must contain 0.0 (the unfiltered known-answer point), got {ts}")
    if params["highlight_threshold"] not in ts:
        raise ValueError(f"{path}: params.highlight_threshold {params['highlight_threshold']!r} is not in thresholds")

    if not _is_int(params["n_permutations"]) or params["n_permutations"] <= 0:
        raise ValueError(f"{path}: params.n_permutations must be a positive int, got {params['n_permutations']!r}")

    invalid = params["invalid_datasets"]
    if not isinstance(invalid, dict) or not all(
        k in datasets and isinstance(v, str) and v for k, v in invalid.items()
    ):
        raise ValueError(f"{path}: params.invalid_datasets must map calibration datasets to non-empty "
                         f"reasons, got {invalid!r}")
    return cfg


# ---------------------------------------------------------------------------
# Row space: restriction to scored documents, per-row probabilities
# ---------------------------------------------------------------------------


def restrict_to_scored_documents(gt_df: pd.DataFrame, ext_df: pd.DataFrame, edges, excluded_documents: set):
    """Masks of the ground-truth / extraction rows outside ``excluded_documents``, and
    ``edges`` re-indexed into those kept rows. Document ids are compared as strings,
    as recovery_validity._verify_matching does.

    Raises:
        AssertionError: an edge touches an excluded row (edges never cross documents,
            so either both ends are kept or both are excluded).
    """
    excluded = {str(d) for d in excluded_documents}
    gt_keep = ~gt_df["document_id"].astype(str).isin(excluded).to_numpy()
    ext_keep = ~ext_df["document_id"].astype(str).isin(excluded).to_numpy()
    gt_pos = np.cumsum(gt_keep) - 1
    ext_pos = np.cumsum(ext_keep) - 1
    sub_edges = []
    for g, e in edges:
        if gt_keep[g] != ext_keep[e]:
            raise AssertionError(f"edge ({g}, {e}) joins a kept row to an excluded-document row")
        if gt_keep[g]:
            sub_edges.append((int(gt_pos[g]), int(ext_pos[e])))
    return gt_keep, ext_keep, sub_edges


def row_probabilities(ext_df: pd.DataFrame, scored: pd.DataFrame, column: str) -> np.ndarray:
    """``scored[column]`` for every ``ext_df`` row, joined on measurement_id (split
    postprocessed rows share their parent's id and so its probability).

    Raises:
        ValueError: scored measurement_ids are not unique, the two sides' id sets
            differ, or they disagree on document_id at a shared id.
    """
    if not scored["measurement_id"].is_unique:
        raise ValueError("scored measurement_ids are not unique")
    ext_ids, scored_ids = set(ext_df["measurement_id"]), set(scored["measurement_id"])
    if ext_ids != scored_ids:
        raise ValueError(f"extraction and scored measurement_id sets differ: {len(ext_ids - scored_ids)} "
                         f"extraction id(s) unscored, {len(scored_ids - ext_ids)} scored id(s) not extracted")
    joined = ext_df[["measurement_id", "document_id"]].merge(
        scored[["measurement_id", "document_id", column]], on="measurement_id", how="left",
        suffixes=("", "__scored"), validate="many_to_one")
    if (joined["document_id"].astype(str) != joined["document_id__scored"].astype(str)).any():
        raise ValueError("extraction and scored rows disagree on document_id at a shared measurement_id")
    probs = joined[column].to_numpy(dtype=float)
    assert len(probs) == len(ext_df) and np.isfinite(probs).all()
    return probs


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------


def decision_threshold_curve(n_gt: int, edges, validity_labels: np.ndarray, probs: np.ndarray,
                             thresholds) -> pd.DataFrame:
    """Recovery and validity of the rows with ``probs >= t``, for each t.

    ``edges`` are (gt_idx, ex_idx) already at the dataset fuzzy threshold, indexed into
    ``validity_labels``/``probs``. Each point is computed by analysis.metrics and
    cross-checked against an independent vectorised computation. With nothing kept,
    recovery is 0 and validity is NaN (undefined, not 0).

    Raises:
        AssertionError: the two computations disagree, or recovery / n_kept increase
            with t.
    """
    validity_labels = np.asarray(validity_labels, dtype=bool)
    probs = np.asarray(probs, dtype=float)
    assert validity_labels.shape == probs.shape, (validity_labels.shape, probs.shape)
    edge_arr = np.asarray(edges, dtype=np.int64).reshape(-1, 2)

    rows = []
    for t in thresholds:
        keep = probs >= t
        n_kept = int(keep.sum())
        recovery = recovery_rate_from_labels(n_gt, edges, keep)
        recovered = np.zeros(n_gt, dtype=bool)
        recovered[edge_arr[keep[edge_arr[:, 1]], 0]] = True
        assert recovery == recovered.mean(), (t, recovery, recovered.mean())
        if n_kept:
            validity = validity_rate_from_labels(validity_labels, keep)
            assert validity == validity_labels[keep].mean(), (t, validity)
        else:
            validity = np.nan
        rows.append({"threshold": float(t), "n_kept": n_kept, "recovery": recovery, "validity": validity})
    df = pd.DataFrame(rows)
    for col in ("recovery", "n_kept"):
        if (np.diff(df[col].to_numpy()) > 0).any():
            raise AssertionError(f"{col} increases with the decision threshold: {df[col].tolist()}")
    return df


def permutation_control(n_gt: int, edges, validity_labels: np.ndarray, ext_df: pd.DataFrame,
                        scored: pd.DataFrame, column: str, thresholds, *, n_permutations: int,
                        seed: int) -> pd.DataFrame:
    """Mean curve over ``n_permutations`` shuffles of ``scored[column]`` across scored
    rows (split children still share their parent's shuffled value). Validity is
    averaged over the draws that kept at least one row; NaN if none did. The kept-row
    count is a mean (``n_kept_mean``), not a count."""
    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(n_permutations):
        shuffled = scored.assign(**{column: rng.permutation(scored[column].to_numpy())})
        probs = row_probabilities(ext_df, shuffled, column)
        curves.append(decision_threshold_curve(n_gt, edges, validity_labels, probs, thresholds))
    validity = np.stack([c["validity"].to_numpy() for c in curves])
    has_rows = ~np.isnan(validity)
    n_with_rows = has_rows.sum(axis=0)
    mean_validity = np.full(len(thresholds), np.nan)
    ok = n_with_rows > 0
    mean_validity[ok] = np.where(has_rows, validity, 0.0).sum(axis=0)[ok] / n_with_rows[ok]
    return pd.DataFrame({
        "threshold": [float(t) for t in thresholds],
        "n_kept_mean": np.mean([c["n_kept"].to_numpy() for c in curves], axis=0),
        "recovery": np.mean([c["recovery"].to_numpy() for c in curves], axis=0),
        "validity": mean_validity,
    })


# ---------------------------------------------------------------------------
# Plots (styled as recovery_validity.plot_fuzzy_threshold_curve)
# ---------------------------------------------------------------------------


def plot_decision_threshold_curve(curve: pd.DataFrame, permuted: pd.DataFrame, highlight: float,
                                  out_path: Path) -> None:
    """Recovery (x) vs validity (y): grey line through the observed sweep, points
    coloured by threshold on coolwarm (fixed 0..1 norm), the highlight threshold a
    black-edged diamond, the permutation control a dotted grey line. Thresholds that
    keep no rows have no validity and are not drawn.

    Raises:
        ValueError: the highlight threshold is absent or keeps no rows.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    curve = curve[curve["n_kept"] > 0]
    permuted = permuted[permuted["validity"].notna()]
    is_hl = (curve["threshold"] == highlight).to_numpy()
    if is_hl.sum() != 1:
        raise ValueError(f"highlight threshold {highlight} is missing or keeps no rows")

    norm = mcolors.Normalize(*_THRESHOLD_NORM)
    fig, ax = plt.subplots(figsize=(4.0, 3.8))
    ax.plot(permuted["recovery"], permuted["validity"], ":", color="grey", lw=2.0, zorder=1)
    ax.plot(curve["recovery"], curve["validity"], "-", color="grey", lw=3.0, zorder=2)
    ax.scatter(curve["recovery"][~is_hl], curve["validity"][~is_hl], c=curve["threshold"][~is_hl],
               cmap=plt.cm.coolwarm, norm=norm, marker="o", s=45, zorder=3)
    # Dark edge: 0.5 sits at coolwarm's near-white midpoint.
    ax.scatter(curve["recovery"][is_hl], curve["validity"][is_hl], c=curve["threshold"][is_hl],
               cmap=plt.cm.coolwarm, norm=norm, marker="D", s=60, zorder=4, edgecolors="k", linewidths=0.9)
    ax.set_xlabel("Recovery")
    ax.set_ylabel("Validity")
    ax.grid(alpha=0.25, linestyle="-", linewidth=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def save_decision_threshold_colorbar(out_path: Path) -> None:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=mcolors.Normalize(*_THRESHOLD_NORM))
    sm.set_array([])
    fig, ax = plt.subplots(figsize=(0.35, 3.2))
    plt.colorbar(sm, cax=ax, label="Decision Threshold")
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def save_decision_threshold_legend(out_path: Path) -> None:
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    handles = [mlines.Line2D([], [], color="grey", lw=2, linestyle="-", label="Observed"),
               mlines.Line2D([], [], color="grey", lw=2, linestyle=":", label="Permuted")]
    fig, ax = plt.subplots(figsize=(3.0, 0.35))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=2, fontsize=13, frameon=False, handlelength=2.0)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Inputs and orchestration
# ---------------------------------------------------------------------------


def resolve_inputs(cfg: dict) -> dict:
    """The calibration config, its resolved run directories (calibration_ids), and the
    path of its predictions.pkl -- everything a run reads, checked to exist."""
    from analysis.common import calibration_ids as cids

    cal_id = cfg["params"]["calibration_config_id"]
    cal_cfg = load_calibration_v4_config(ANALYSIS_CONFIGS_ROOT / f"{cal_id}.yaml")
    predictions_path = _REPO_ROOT / "analysis" / "results" / "calibration" / cal_id / "predictions.pkl"
    if not predictions_path.exists():
        raise FileNotFoundError(f"{predictions_path} does not exist -- run calibration_v4 on {cal_id} first")
    return {"calibration_config": cal_cfg, "calibration_inputs": cids.resolve_calibration_inputs(cal_cfg),
            "predictions_path": predictions_path}


def score_cell(cfg: dict, inputs: dict, predictions: dict, train: str, test: str) -> pd.DataFrame:
    """Observed and permuted curves for both methods of one (train, test) cell."""
    from analysis import recovery_validity as rv
    from analysis.common.provenance import repo_relative, sha256_file
    from analysis.common.prediction_store import check_real_cell

    params = cfg["params"]
    cal_cfg = inputs["calibration_config"]
    ds_block = cal_cfg["params"]["datasets"][test]
    ds_inputs = inputs["calibration_inputs"]["datasets"][test]
    extraction_id = ds_block["extraction_id"]

    data = rv._load_checked_inputs(extraction_id, ds_inputs["ground_truth_path"])
    assert data["dataset"] == test, (data["dataset"], test)
    gt_df, ext_df, mcfg = data["ground_truth_df"], data["extraction_df"], data["cfg"]
    n_gt, n_ext = len(gt_df), len(ext_df)
    edges, _w = rv.filter_edges_by_threshold(data["raw_edges"], data["raw_weights"], mcfg["fuzzy_threshold"])
    recovered = rv.gt_recovered_mask(n_gt, edges)
    rv._verify_recovery(gt_df, ext_df, mcfg, mcfg["fuzzy_threshold"], data["cache_path"], recovered)
    judge_combine_id, _judge_ids, judged = rv._resolve_judged_labels(
        test, extraction_id, ds_block["judge_combine_id"], ext_df)
    validity_labels = rv._verify_validity(gt_df, ext_df, mcfg, mcfg["fuzzy_threshold"], data["cache_path"],
                                          rv.ext_matched_mask(n_ext, edges), judged)

    judge_models = list(predictions["real"])
    assert judge_models == [inputs["calibration_inputs"]["judge_model"]], judge_models
    cell = predictions["real"][judge_models[0]][train][test]
    final_path = ds_inputs["extraction_dir"] / "final.json"
    combined_path = ds_inputs["judge_combine_dir"] / "combined.json"
    if cell["combined_sha256"] != sha256_file(combined_path):
        raise ValueError(f"{train}->{test}: combined.json changed since predictions.pkl was built")
    with open(final_path) as f:
        final_df = pd.DataFrame(json.load(f))
    excluded = set(cell["excluded_documents"])
    scored = check_real_cell(cell, final_df, excluded, sha256_file(final_path), params["calibration_config_id"])

    gt_keep, ext_keep, sub_edges = restrict_to_scored_documents(gt_df, ext_df, edges, excluded)
    ext_sub = ext_df[ext_keep].reset_index(drop=True)
    labels_sub = validity_labels[ext_keep]
    n_gt_sub = int(gt_keep.sum())
    print(f"  {train}->{test}: {len(excluded)} excluded documents | GT {n_gt_sub}/{n_gt} | "
          f"extractions {len(ext_sub)}/{n_ext} | edges {len(sub_edges)}/{len(edges)}")

    meta = {
        "analysis_config_id": cfg["id"], "calibration_config_id": params["calibration_config_id"],
        "train_dataset": train, "test_dataset": test, "extraction_id": extraction_id,
        "judge_combine_id": judge_combine_id, "ground_truth_file": repo_relative(data["ground_truth_path"]),
        "extraction_file": repo_relative(data["extraction_file_path"]),
        "fuzzy_threshold": mcfg["fuzzy_threshold"], "n_excluded_documents": len(excluded),
        "n_gt": n_gt_sub, "n_ext": len(ext_sub),
        "invalid_reason": params["invalid_datasets"].get(test),
    }
    frames = []
    for method, column in METHODS:
        probs = row_probabilities(ext_sub, scored, column)
        curve = decision_threshold_curve(n_gt_sub, sub_edges, labels_sub, probs, params["thresholds"])
        # Known answer: keeping everything reproduces recovery_validity's verified masks on these documents.
        at0 = curve[curve["threshold"] == 0.0].iloc[0]
        assert at0["n_kept"] == len(ext_sub), (method, at0["n_kept"], len(ext_sub))
        assert at0["recovery"] == recovered[gt_keep].mean(), (method, at0["recovery"], recovered[gt_keep].mean())
        assert at0["validity"] == labels_sub.mean(), (method, at0["validity"], labels_sub.mean())
        permuted = permutation_control(n_gt_sub, sub_edges, labels_sub, ext_sub, scored, column,
                                       params["thresholds"], n_permutations=params["n_permutations"],
                                       seed=cfg["seed"])
        for name, df in (("observed", curve), ("permuted_mean", permuted)):
            frames.append(df.assign(method=method, curve=name,
                                    is_highlight=df["threshold"] == params["highlight_threshold"], **meta))
    out = pd.concat(frames, ignore_index=True)
    # Observed-only column; nullable so the permuted rows' NaN doesn't turn counts into floats.
    out["n_kept"] = out["n_kept"].astype("Int64")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="analysis/analysis-configs/<id>.yaml")
    args = parser.parse_args(argv)

    cfg = load_decision_threshold_config(args.config)
    params = cfg["params"]
    inputs = resolve_inputs(cfg)
    with open(inputs["predictions_path"], "rb") as f:
        predictions = pickle.load(f)

    out_dir = RESULTS_ROOT / cfg["id"]
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"[decision_threshold] {cfg['id']} | calibration {params['calibration_config_id']} | "
          f"{len(params['thresholds'])} thresholds | {params['n_permutations']} permutations | out {out_dir}")

    results = []
    for c in params["cells"]:
        train, test = c["train"], c["test"]
        if test in params["invalid_datasets"]:
            print(f"  WARNING {train}->{test}: flagged invalid -- {params['invalid_datasets'][test]}")
        df = score_cell(cfg, inputs, predictions, train, test)
        results.append(df)
        for method, _column in METHODS:
            sel = df[df["method"] == method]
            observed, permuted = sel[sel["curve"] == "observed"], sel[sel["curve"] == "permuted_mean"]
            out_path = figures_dir / f"{train}-{test}-{method}.pdf"
            plot_decision_threshold_curve(observed, permuted, params["highlight_threshold"], out_path)
            print(f"    {method}: " + "  ".join(
                f"t={r.threshold:.2f} n={r.n_kept} R={r.recovery:.3f} V={r.validity:.3f}"
                for r in observed.itertuples()))
            print(f"    wrote {out_path}")

    curves = pd.concat(results, ignore_index=True)
    curves.to_csv(out_dir / "curves.csv", index=False)
    save_decision_threshold_colorbar(figures_dir / "decision_threshold_colorbar.pdf")
    save_decision_threshold_legend(figures_dir / "decision_threshold_legend.pdf")
    print(f"\nWrote {len(curves)} rows to {out_dir / 'curves.csv'}")


if __name__ == "__main__":
    main()
