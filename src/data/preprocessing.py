"""Turn ingested ``*_aligned.npz`` sessions into decoder-ready design matrices.

The ridge decoder predicts a continuous lick trace (``jaw_distance``) from the neural
population vector at each time bin, optionally augmented with a causal history window.
All splitting happens at the *trial* level so no time bin of a trial ever straddles the
train/test boundary, and the neural scaler is fit on training trials only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

DEFAULT_TARGET = "jaw_distance"
DEFAULT_LAGS = 10  # causal history bins added to each sample (10 * 10 ms = 100 ms)


def load_session(npz_path: str | Path) -> dict:
    """Load one aligned session ``.npz`` into a plain dict of arrays."""
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def resample_target(
    dlc: np.ndarray,
    time_dlc: np.ndarray,
    time_neural: np.ndarray,
    bodyparts: np.ndarray,
    bodypart: str = DEFAULT_TARGET,
) -> np.ndarray:
    """Resample one DLC bodypart onto the neural time grid.

    Args:
        dlc: ``(n_trials, n_time_dlc, n_bodyparts)`` aligned DLC traces.
        time_dlc: ``(n_time_dlc,)`` DLC time axis (seconds, relative to event).
        time_neural: ``(n_time,)`` neural time axis to resample onto.
        bodyparts: ``(n_bodyparts,)`` bodypart names matching ``dlc``'s last axis.
        bodypart: which bodypart to extract as the lick trace.

    Returns:
        ``(n_trials, n_time)`` target trace on the neural grid.
    """
    names = [str(b) for b in bodyparts]
    if bodypart not in names:
        raise ValueError(f"bodypart {bodypart!r} not in {names}")
    part_idx = names.index(bodypart)
    trace = dlc[:, :, part_idx]  # (n_trials, n_time_dlc)
    return np.vstack(
        [np.interp(time_neural, time_dlc, trace[trial]) for trial in range(trace.shape[0])]
    )


def build_design_matrix(
    neural: np.ndarray,
    target: np.ndarray,
    lags: int = DEFAULT_LAGS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten trials into per-timepoint samples with a causal lag window.

    Each sample is one ``(trial, time_bin)`` pair. Its features are the population vector
    at that bin concatenated with the vectors from the ``lags`` preceding bins (padded with
    the first bin at the trial start so every trial keeps all its time points).

    Args:
        neural: ``(n_trials, n_time, n_units)`` neural PSTH.
        target: ``(n_trials, n_time)`` lick trace on the same grid.
        lags: number of causal history bins to append (0 = instantaneous decoding).

    Returns:
        ``X`` ``(n_samples, n_units * (lags + 1))``, ``y`` ``(n_samples,)``, and
        ``trial_index`` ``(n_samples,)`` mapping each sample back to its trial.
    """
    if neural.shape[:2] != target.shape:
        raise ValueError(
            f"neural {neural.shape[:2]} and target {target.shape} trial/time axes disagree"
        )
    n_trials, n_time, _ = neural.shape

    # Build lagged copies along the time axis; lag k shifts the window k bins into the past,
    # edge-padding with the first bin so no time points are dropped.
    lagged = [neural]
    for k in range(1, lags + 1):
        shifted = np.empty_like(neural)
        shifted[:, k:, :] = neural[:, :-k, :]
        shifted[:, :k, :] = neural[:, :1, :]
        lagged.append(shifted)
    stacked = np.concatenate(lagged, axis=2)  # (n_trials, n_time, n_units * (lags + 1))

    X = stacked.reshape(n_trials * n_time, -1)
    y = target.reshape(n_trials * n_time)
    trial_index = np.repeat(np.arange(n_trials), n_time)
    return X, y, trial_index


def epoch_split(
    n_trials: int, fractions: tuple[float, float, float] = (0.34, 0.33, 0.33)
) -> dict[str, np.ndarray]:
    """Split trials into early / mid / late epochs by acquisition order (Q2).

    Trials are assumed to be in chronological order (as ingested from the trial table).
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    idx = np.arange(n_trials)
    b1 = int(round(fractions[0] * n_trials))
    b2 = b1 + int(round(fractions[1] * n_trials))
    return {"early": idx[:b1], "mid": idx[b1:b2], "late": idx[b2:]}


def trial_holdout(
    n_trials: int, test_frac: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Random within-session train/test split at the trial level."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_trials)
    n_test = int(round(test_frac * n_trials))
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    return train_idx, test_idx


def loao_splits(subject_ids: list[str]):
    """Leave-one-animal-out iterator over session indices grouped by subject (Q1).

    Yields ``(train_session_idx, test_session_idx)`` where each is an array of indices into
    the ``subject_ids`` list. Requires ≥2 distinct subjects to produce any fold.

    Args:
        subject_ids: subject id for each session, in session order.
    """
    from sklearn.model_selection import LeaveOneGroupOut

    subject_ids = np.asarray(subject_ids)
    n_subjects = len(np.unique(subject_ids))
    if n_subjects < 2:
        return  # no folds possible; caller handles the single-animal case
    logo = LeaveOneGroupOut()
    dummy = np.zeros((len(subject_ids), 1))
    for train_idx, test_idx in logo.split(dummy, groups=subject_ids):
        yield train_idx, test_idx


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a per-feature ``StandardScaler`` on training samples only."""
    return StandardScaler().fit(X_train)


def select_samples(
    X: np.ndarray, y: np.ndarray, trial_index: np.ndarray, trials: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Slice out the samples whose trial belongs to ``trials``."""
    mask = np.isin(trial_index, trials)
    return X[mask], y[mask]
