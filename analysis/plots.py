"""Standard plot functions for ScholarlM analysis notebooks.

All functions return a ``matplotlib.figure.Figure`` so callers can save or
display as needed.

Typical usage
-------------
    from analysis.plots import recovery_bar, calibration_curve

    fig = recovery_bar(metrics_df, title="Pond — gemma-3-27b")
    fig.savefig("figures/recovery.pdf", bbox_inches="tight")
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def recovery_bar(metrics_df: pd.DataFrame, title: str = "") -> Figure:
    """Bar chart of recall and precision, one bar group per row of ``metrics_df``.

    Args:
        metrics_df: DataFrame with columns ``recall`` and ``precision`` and an
            index used as the x-axis labels (e.g. ablation names or model names).
        title: Optional plot title.

    Returns:
        ``Figure`` with a single Axes.
    """
    labels = list(metrics_df.index)
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
    ax.bar(x - width / 2, metrics_df["recall"], width, label="Recall")
    ax.bar(x + width / 2, metrics_df["precision"], width, label="Precision")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def calibration_curve(
    diag_dict: dict,
    judge_labels: list[str] | None = None,
) -> Figure:
    """Reliability diagram (calibration curve).

    Args:
        diag_dict: Single reliability diagram dict *or* mapping
            ``judge_label -> reliability_diagram_data_dict``.  If a single dict
            (has key ``"bin_centers"``), it is wrapped into a one-item mapping.
        judge_labels: Display labels for each judge; only used when ``diag_dict``
            is a mapping.

    Returns:
        ``Figure`` with reliability diagram(s).
    """
    if "bin_centers" in diag_dict:
        diag_dict = {"model": diag_dict}

    judges = list(diag_dict.keys())
    if judge_labels is None:
        judge_labels = judges

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for label, (judge, diag) in zip(judge_labels, diag_dict.items()):
        ece = diag.get("ece", float("nan"))
        valid = ~np.isnan(diag["bin_accuracy"])
        ax.plot(
            diag["bin_centers"][valid],
            diag["bin_accuracy"][valid],
            "o-",
            label=f"{label} (ECE={ece:.3f})",
        )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("Calibration curve")
    fig.tight_layout()
    return fig


def probe_accuracy_heatmap(df: pd.DataFrame) -> Figure:
    """Heatmap of probe accuracy (extraction models × judge models).

    Args:
        df: DataFrame with extraction models as rows and judge models as
            columns, values in [0, 1].

    Returns:
        ``Figure`` with a single heatmap Axes.
    """
    fig, ax = plt.subplots(figsize=(max(4, len(df.columns) * 1.0), max(3, len(df) * 0.8)))
    values = df.values.astype(float)
    im = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(list(df.columns), rotation=30, ha="right")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(list(df.index))
    ax.set_xlabel("Judge model")
    ax.set_ylabel("Extraction model")
    ax.set_title("Probe accuracy")
    for i in range(len(df)):
        for j in range(len(df.columns)):
            v = values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    fig.tight_layout()
    return fig


def probability_distribution(
    judged_df: pd.DataFrame,
    prob_col: str,
    label_col: str,
) -> Figure:
    """Histogram of predicted probabilities split by ground-truth label.

    Args:
        judged_df: DataFrame with ``prob_col`` (float [0,1]) and ``label_col``
            (bool) columns.
        prob_col: Column containing predicted probabilities.
        label_col: Column containing boolean ground-truth labels.

    Returns:
        ``Figure`` with overlapping histograms.
    """
    valid = judged_df[judged_df[label_col].astype(bool)][prob_col].dropna()
    invalid = judged_df[~judged_df[label_col].astype(bool)][prob_col].dropna()

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(invalid, bins=bins, alpha=0.6, label="Invalid", color="tomato", density=True)
    ax.hist(valid, bins=bins, alpha=0.6, label="Valid", color="steelblue", density=True)
    ax.set_xlabel(prob_col)
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("Predicted probability distribution")
    fig.tight_layout()
    return fig


def cross_dataset_matrix(df: pd.DataFrame) -> Figure:
    """Heatmap of cross-dataset probe accuracy (train dataset × test dataset).

    Args:
        df: DataFrame produced by ``cross_dataset_probe_matrix``, with dataset
            names as both index and columns.

    Returns:
        ``Figure`` with a single heatmap Axes.
    """
    fig, ax = plt.subplots(figsize=(max(4, len(df.columns) * 0.9), max(3, len(df) * 0.9)))
    values = df.values.astype(float)
    im = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(list(df.columns), rotation=30, ha="right")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(list(df.index))
    ax.set_xlabel("Test dataset")
    ax.set_ylabel("Train dataset")
    ax.set_title("Cross-dataset probe accuracy")
    for i in range(len(df)):
        for j in range(len(df.columns)):
            v = values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    return fig
