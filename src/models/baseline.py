"""Ridge-regression lick decoder — the baseline and interpretability anchor.

Every deeper model (LSTM, transformer) must beat this to justify its complexity. The
decoder is a plain linear map from the (optionally lag-augmented) neural population vector
to the continuous lick value, so its weights are directly inspectable per unit and lag.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from src.data.preprocessing import DEFAULT_LAGS, build_design_matrix


class RidgeDecoder:
    """Thin wrapper around :class:`sklearn.linear_model.Ridge` for lick decoding."""

    def __init__(self, alpha: float = 1.0, lags: int = DEFAULT_LAGS):
        self.alpha = alpha
        self.lags = lags
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeDecoder:
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    @property
    def coef_(self) -> np.ndarray:
        """Learned weights, one per (unit × lag) feature — the interpretability handle."""
        return self.model.coef_

    def predict_trace(self, neural_trial: np.ndarray) -> np.ndarray:
        """Decode a single trial's lick trace from its neural activity.

        Mirrors the API contract: takes one trial's neural activity and returns a lick
        trace of length ``n_time``.

        Args:
            neural_trial: ``(n_time, n_units)`` or ``(n_units, n_time)`` activity for one
                trial. The ``n_units`` axis is inferred by matching the trained feature count.

        Returns:
            ``(n_time,)`` predicted lick trace.
        """
        arr = np.asarray(neural_trial)
        n_units = self.coef_.shape[0] // (self.lags + 1)
        if arr.shape[0] == n_units and arr.shape[1] != n_units:
            arr = arr.T  # accept (n_units, n_time) and transpose to (n_time, n_units)
        X, _, _ = build_design_matrix(arr[None, :, :], np.zeros(arr.shape[:1])[None, :], self.lags)
        return self.predict(X)
