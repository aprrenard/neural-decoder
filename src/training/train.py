"""Fit a ridge decoder for one split and log everything to MLflow.

No undocumented runs: every fit records its hyperparameters, metrics, and the model +
scaler artifacts under a run named ``{session_id}_{split}`` (or ``{loao}_{test_subject}``
for cross-animal folds), per the project naming convention.
"""

from __future__ import annotations

import mlflow
import mlflow.sklearn
import numpy as np

from src.data.preprocessing import fit_scaler
from src.models.baseline import RidgeDecoder

EXPERIMENT_NAME = "lick-decoder"


def train_ridge(X_train: np.ndarray, y_train: np.ndarray, alpha: float, lags: int) -> RidgeDecoder:
    """Fit a :class:`RidgeDecoder` on already-scaled training samples."""
    return RidgeDecoder(alpha=alpha, lags=lags).fit(X_train, y_train)


def run_and_log(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    run_name: str,
    params: dict,
    metric_fn,
) -> dict:
    """Fit + evaluate one split inside an MLflow run and return the metrics.

    Args:
        metric_fn: callable ``(y_true, y_pred) -> dict`` (see ``evaluate.regression_metrics``).
    """
    mlflow.set_experiment(EXPERIMENT_NAME)
    scaler = fit_scaler(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        model = train_ridge(X_train_s, y_train, params["alpha"], params["lags"])
        y_pred = model.predict(X_test_s)
        metrics = metric_fn(y_test, y_pred)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model.model, name="ridge")
        mlflow.sklearn.log_model(scaler, name="scaler")
    return metrics
