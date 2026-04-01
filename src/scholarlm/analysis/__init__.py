"""
Analysis utilities for the ScholarlM experiment framework.

Modules
-------
probe
    Train and evaluate logistic-regression probes on JudgementLM attention
    activations to predict valid/invalid labels.
calibration
    Compute ECE and reliability-diagram data for judge model probabilities.
cross_dataset
    Train probes on one dataset's activations and evaluate on another's.
"""
from .probe import train_probe, eval_probe, build_feature_matrix
from .calibration import compute_ece, reliability_diagram_data
from .cross_dataset import cross_dataset_probe_matrix

__all__ = [
    "train_probe",
    "eval_probe",
    "build_feature_matrix",
    "compute_ece",
    "reliability_diagram_data",
    "cross_dataset_probe_matrix",
]
