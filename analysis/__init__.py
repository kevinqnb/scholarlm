"""
Experiment-specific analysis code for the ScholarlM project.

This package depends on both ``scholarlm`` (the core library) and
``experiments/paths.py`` (the path-construction module).  It is intentionally
kept outside ``src/scholarlm/`` so that the core library has no dependency on
experiment scaffolding.

Modules
-------
loaders
    Load experiment outputs by (dataset, model, date).
metrics
    Recovery rate, hallucination rate, per-paper summaries.
plots
    Standard plot functions (all return ``matplotlib.Figure``).
cross_dataset
    Cross-dataset probe transferability.
"""
from .loaders import (
    load_extraction,
    load_ablation,
    load_combined_judgements,
    load_ground_truth,
    load_activations,
    cached_match,
)
from .metrics import recovery_rate, validity_rate, per_paper_metrics
from .plots import (
    recovery_bar,
    calibration_curve,
    probe_accuracy_heatmap,
    probability_distribution,
    cross_dataset_matrix,
)

__all__ = [
    "load_extraction",
    "load_ablation",
    "load_combined_judgements",
    "load_ground_truth",
    "load_activations",
    "cached_match",
    "recovery_rate",
    "hallucination_rate",
    "per_paper_metrics",
    "recovery_bar",
    "calibration_curve",
    "probe_accuracy_heatmap",
    "probability_distribution",
    "cross_dataset_matrix",
    "cross_dataset_probe_matrix",
    "load_activations_and_labels",
]
