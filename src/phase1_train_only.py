"""Train-only preprocessing primitives for Phase 1 evaluations.

Every learned quantity in this module is fit from arrays passed explicitly by
the caller.  The API intentionally has no access to repository-global labels,
test indices, or validation/test feature arrays.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def balanced_class_weights(train_targets, n_classes: int) -> np.ndarray:
    """Return inverse-frequency weights learned only from ``train_targets``."""
    targets = np.asarray(train_targets, dtype=np.int64)
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("train_targets must be a non-empty one-dimensional array")
    if targets.min() < 0 or targets.max() >= n_classes:
        raise ValueError("train_targets contains a label outside n_classes")
    counts = np.bincount(targets, minlength=n_classes).astype(np.float64)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"training partition has no samples for classes {missing}")
    return counts.sum() / (n_classes * counts)


def fit_train_scaler(train_features) -> StandardScaler:
    """Fit a standardizer from a training feature matrix only."""
    features = np.asarray(train_features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("train_features must be a non-empty two-dimensional array")
    return StandardScaler().fit(features)


@dataclass
class DurationResidualizer:
    """Per-feature linear duration adjustment fit on one training partition."""

    model: LinearRegression

    @classmethod
    def fit(cls, train_features, train_duration) -> "DurationResidualizer":
        features = np.asarray(train_features, dtype=np.float64)
        duration = np.asarray(train_duration, dtype=np.float64).reshape(-1, 1)
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("train_features must be a non-empty two-dimensional array")
        if duration.shape[0] != features.shape[0]:
            raise ValueError("train_duration and train_features must have equal rows")
        return cls(LinearRegression().fit(duration, features))

    def transform(self, features, duration) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        duration = np.asarray(duration, dtype=np.float64).reshape(-1, 1)
        if features.ndim != 2 or duration.shape[0] != features.shape[0]:
            raise ValueError("duration and features must have compatible rows")
        return features - self.model.predict(duration)


def array_hash(*arrays) -> str:
    """Stable SHA-256 for numeric training artifacts used by audit manifests."""
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(np.asarray(array))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def scaler_hash(scaler: StandardScaler) -> str:
    return array_hash(scaler.mean_, scaler.var_, scaler.scale_)


def residualizer_hash(residualizer: DurationResidualizer) -> str:
    model = residualizer.model
    return array_hash(model.coef_, model.intercept_)
