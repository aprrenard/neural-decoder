"""Unit tests for the RidgeDecoder baseline."""

import numpy as np

from src.data.preprocessing import build_design_matrix
from src.models.baseline import RidgeDecoder


def _linear_dataset(n_trials=30, n_time=25, n_units=6, seed=0):
    """Neural activity with a known linear map to the lick trace (plus small noise)."""
    rng = np.random.default_rng(seed)
    neural = rng.standard_normal((n_trials, n_time, n_units))
    weights = rng.standard_normal(n_units)
    target = neural @ weights + 0.01 * rng.standard_normal((n_trials, n_time))
    return neural, target


def test_fit_predict_shapes_and_determinism():
    neural, target = _linear_dataset()
    X, y, _ = build_design_matrix(neural, target, lags=0)
    model = RidgeDecoder(alpha=1.0, lags=0).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    # deterministic: refitting gives identical predictions
    np.testing.assert_allclose(pred, RidgeDecoder(alpha=1.0, lags=0).fit(X, y).predict(X))


def test_coef_exposes_one_weight_per_feature():
    neural, target = _linear_dataset()
    lags = 2
    X, y, _ = build_design_matrix(neural, target, lags=lags)
    model = RidgeDecoder(alpha=1.0, lags=lags).fit(X, y)
    assert model.coef_.shape == (neural.shape[2] * (lags + 1),)


def test_beats_mean_predictor():
    neural, target = _linear_dataset()
    X, y, _ = build_design_matrix(neural, target, lags=0)
    model = RidgeDecoder(alpha=1.0, lags=0).fit(X, y)
    mse_model = np.mean((y - model.predict(X)) ** 2)
    mse_mean = np.mean((y - y.mean()) ** 2)
    assert mse_model < mse_mean


def test_predict_trace_returns_time_series():
    neural, target = _linear_dataset()
    lags = 3
    X, y, _ = build_design_matrix(neural, target, lags=lags)
    model = RidgeDecoder(alpha=1.0, lags=lags).fit(X, y)
    n_time = neural.shape[1]
    trace = model.predict_trace(neural[0])  # (n_time, n_units)
    assert trace.shape == (n_time,)
    # accepts the transposed (n_units, n_time) orientation too
    assert model.predict_trace(neural[0].T).shape == (n_time,)
