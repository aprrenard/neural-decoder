"""Validate the ridge decoder: within-session now (Q2), cross-animal LOAO scaffolded (Q1).

Within-session evaluation runs today on any ingested session (trial holdout + early→mid/late
epoch split). LOAO is fully implemented but needs ≥2 animals; with a single animal it logs a
clear skip message. Every split is logged as its own MLflow run via ``train.run_and_log``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

from src.data.preprocessing import (
    DEFAULT_LAGS,
    DEFAULT_TARGET,
    build_design_matrix,
    epoch_split,
    load_session,
    resample_target,
    select_samples,
    trial_holdout,
)
from src.training.train import run_and_log

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_ALPHA = 1.0


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """R², Pearson correlation, and MSE between decoded and true lick traces."""
    corr = float(pearsonr(y_true, y_pred)[0]) if np.std(y_pred) > 0 else 0.0
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": corr,
        "mse": float(mean_squared_error(y_true, y_pred)),
    }


def prepare_session(session: dict, target: str = DEFAULT_TARGET, lags: int = DEFAULT_LAGS):
    """Build the (X, y, trial_index) design matrix for one loaded session."""
    lick = resample_target(
        session["dlc"],
        session["time_dlc"],
        session["time_neural"],
        session["bodyparts"],
        bodypart=target,
    )
    return build_design_matrix(session["neural"], lick, lags=lags)


def evaluate_within_session(
    session: dict,
    *,
    alpha: float = DEFAULT_ALPHA,
    lags: int = DEFAULT_LAGS,
    target: str = DEFAULT_TARGET,
) -> dict:
    """Run trial-holdout and epoch-split evaluation for a single session."""
    session_id = str(session["session_id"])
    subject_id = str(session["subject_id"])
    X, y, trial_index = prepare_session(session, target=target, lags=lags)
    n_trials = session["neural"].shape[0]
    base_params = {
        "alpha": alpha,
        "lags": lags,
        "target": target,
        "subject_id": subject_id,
        "session_id": session_id,
    }
    results: dict[str, dict] = {}

    # (a) random trial holdout — an unbiased within-session decoding sanity check
    train_idx, test_idx = trial_holdout(n_trials)
    Xtr, ytr = select_samples(X, y, trial_index, train_idx)
    Xte, yte = select_samples(X, y, trial_index, test_idx)
    results["holdout"] = run_and_log(
        Xtr, ytr, Xte, yte,
        run_name=f"{session_id}_holdout",
        params={**base_params, "split": "holdout"},
        metric_fn=regression_metrics,
    )

    # (b) train on early epoch, evaluate on mid and late (Q2 within-session plasticity)
    epochs = epoch_split(n_trials)
    Xtr, ytr = select_samples(X, y, trial_index, epochs["early"])
    for eval_epoch in ("mid", "late"):
        Xte, yte = select_samples(X, y, trial_index, epochs[eval_epoch])
        results[f"early_to_{eval_epoch}"] = run_and_log(
            Xtr, ytr, Xte, yte,
            run_name=f"{session_id}_early_to_{eval_epoch}",
            params={**base_params, "split": f"early_to_{eval_epoch}"},
            metric_fn=regression_metrics,
        )
    return results


def evaluate_loao(
    sessions: list[dict],
    *,
    alpha: float = DEFAULT_ALPHA,
    lags: int = DEFAULT_LAGS,
    target: str = DEFAULT_TARGET,
) -> dict:
    """Leave-one-animal-out cross-validation across sessions (Q1).

    Requires ≥2 distinct subjects. With fewer, returns an empty result after logging why.
    """
    subject_ids = [str(s["subject_id"]) for s in sessions]
    n_subjects = len(set(subject_ids))
    if n_subjects < 2:
        print(
            f"[LOAO] skipped: need ≥2 animals, have {n_subjects} "
            f"({sorted(set(subject_ids))}). Ingest more sessions to run cross-animal folds."
        )
        return {}

    # Pool per-session design matrices, tagging each sample with its subject.
    Xs, ys, subj_of_sample = [], [], []
    for sess, subj in zip(sessions, subject_ids):
        Xi, yi, _ = prepare_session(sess, target=target, lags=lags)
        Xs.append(Xi)
        ys.append(yi)
        subj_of_sample.append(np.full(len(yi), subj))
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    subj_of_sample = np.concatenate(subj_of_sample)

    results = {}
    for test_subject in sorted(set(subject_ids)):
        test_mask = subj_of_sample == test_subject
        results[test_subject] = run_and_log(
            X[~test_mask], y[~test_mask], X[test_mask], y[test_mask],
            run_name=f"loao_{test_subject}",
            params={
                "alpha": alpha, "lags": lags, "target": target,
                "split": "loao", "test_subject": test_subject,
            },
            metric_fn=regression_metrics,
        )
    return results


def main() -> None:
    paths = sorted(PROCESSED_DIR.glob("*_aligned.npz"))
    if not paths:
        print(f"No aligned sessions found in {PROCESSED_DIR}. Run ingestion first.")
        return

    sessions = [load_session(p) for p in paths]
    for path, session in zip(paths, sessions):
        print(f"\n=== within-session: {path.name} ===")
        for split, metrics in evaluate_within_session(session).items():
            print(
                f"  {split:16s} "
                f"R²={metrics['r2']:+.3f}  r={metrics['pearson_r']:+.3f}  MSE={metrics['mse']:.4f}"
            )

    print("\n=== cross-animal LOAO ===")
    loao = evaluate_loao(sessions)
    for subject, metrics in loao.items():
        print(f"  held-out {subject}: R²={metrics['r2']:+.3f}  r={metrics['pearson_r']:+.3f}")


if __name__ == "__main__":
    main()
