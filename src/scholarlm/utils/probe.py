"""
Logistic-regression probe utilities for JudgementLM attention activations.

Activations are stored by ``run_judge.py`` as ``attention_outputs.npz`` files
where each key is a ``str(measurement_id)`` and each value is a NumPy array of
shape ``(n_layers, n_heads, head_dim)``.  The probe operates on a flattened
feature representation of these arrays.

Typical usage
-------------
    from scholarlm.utils.probe import build_feature_matrix, train_probe, eval_probe
    import numpy as np

    activations = np.load("attention_outputs.npz")
    labels = ...  # boolean array aligned with activations

    X = build_feature_matrix(activations, measurement_ids)
    probe = train_probe(X_train, y_train)
    acc = eval_probe(probe, X_test, y_test)
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_feature_matrix(
    activations: dict[str, np.ndarray] | np.lib.npyio.NpzFile,
    measurement_ids: list[int | str],
    layer_reduction: str = "mean",
) -> np.ndarray:
    """Build a 2-D feature matrix from a set of activation arrays.

    Each activation is an array of shape ``(n_layers, n_heads, head_dim)``.
    The function reduces the head dimension (averaging over heads per layer)
    then concatenates across layers to produce a 1-D feature vector per
    measurement.

    Args:
        activations: Mapping from ``str(measurement_id)`` to an array of shape
            ``(n_layers, n_heads, head_dim)``.  Accepts either a plain ``dict``
            or a ``numpy.lib.npyio.NpzFile`` (the return value of
            ``np.load("attention_outputs.npz")``).
        measurement_ids: Ordered list of measurement IDs.  The output rows are
            aligned with this order.  IDs not present in ``activations`` are
            filled with zeros.
        layer_reduction: How to reduce across layers.  ``"mean"`` averages all
            layers into a single ``(n_heads * head_dim,)`` vector (compact but
            loses layer-specific structure).  ``"concat"`` concatenates layer
            vectors into a ``(n_layers * n_heads * head_dim,)`` vector
            (preserves layer structure, larger).  Default: ``"mean"``.

    Returns:
        Float32 array of shape ``(len(measurement_ids), n_features)``.

    Raises:
        ValueError: If ``layer_reduction`` is not one of ``"mean"`` or
            ``"concat"``.
    """
    if layer_reduction not in ("mean", "concat"):
        raise ValueError(f"layer_reduction must be 'mean' or 'concat', got '{layer_reduction}'")

    rows: list[np.ndarray] = []
    ref_shape: tuple[int, int, int] | None = None

    for mid in measurement_ids:
        key = str(mid)
        if key not in activations:
            # Pad missing entries with zeros (same shape as the reference)
            if ref_shape is None:
                rows.append(None)  # type: ignore[arg-type]
            else:
                n_layers, n_heads, head_dim = ref_shape
                if layer_reduction == "mean":
                    rows.append(np.zeros(n_heads * head_dim, dtype=np.float32))
                else:
                    rows.append(np.zeros(n_layers * n_heads * head_dim, dtype=np.float32))
            continue

        arr = np.array(activations[key], dtype=np.float32)  # (n_layers, n_heads, head_dim)
        if arr.ndim != 3:
            raise ValueError(
                f"Expected activation shape (n_layers, n_heads, head_dim) for id={mid}, "
                f"got shape {arr.shape}."
            )
        if ref_shape is None:
            ref_shape = arr.shape  # type: ignore[assignment]

        n_layers, n_heads, head_dim = arr.shape
        if layer_reduction == "mean":
            # Average over layers → (n_heads, head_dim) → flatten
            vec = arr.mean(axis=0).reshape(-1)
        else:
            # Concatenate layers → (n_layers * n_heads * head_dim,)
            vec = arr.reshape(-1)
        rows.append(vec)

    # Back-fill any leading None entries (missing activations before ref_shape was set)
    if ref_shape is not None:
        n_layers, n_heads, head_dim = ref_shape
        fill_size = n_heads * head_dim if layer_reduction == "mean" else n_layers * n_heads * head_dim
        rows = [r if r is not None else np.zeros(fill_size, dtype=np.float32) for r in rows]

    return np.stack(rows, axis=0) if rows else np.empty((0, 0), dtype=np.float32)


def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> Pipeline:
    """Train a logistic-regression probe on activation features.

    The probe is a scikit-learn ``Pipeline`` that standardizes features before
    fitting logistic regression, so it is robust to different activation scales
    across models.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)`` produced by
            ``build_feature_matrix``.
        y: Binary label array of shape ``(n_samples,)``.  ``True`` / ``1``
            means valid; ``False`` / ``0`` means invalid.
        C: Inverse regularization strength for logistic regression.
        max_iter: Maximum number of solver iterations.
        random_state: Random seed for the solver.

    Returns:
        Fitted scikit-learn ``Pipeline(StandardScaler, LogisticRegression)``.
    """
    probe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",
        )),
    ])
    probe.fit(X, y)
    return probe


def eval_probe(
    probe: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Evaluate a trained probe on held-out features and labels.

    Args:
        probe: Fitted probe returned by ``train_probe``.
        X: Feature matrix of shape ``(n_samples, n_features)``.
        y: Binary label array of shape ``(n_samples,)``.

    Returns:
        Dict with keys:
        - ``"accuracy"``  — fraction of correct predictions.
        - ``"n_samples"`` — number of evaluation examples.
    """
    preds = probe.predict(X)
    accuracy = float((preds == y).mean())
    return {"accuracy": accuracy, "n_samples": len(y)}
