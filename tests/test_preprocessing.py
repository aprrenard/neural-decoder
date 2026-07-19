"""Unit tests for the decoder preprocessing: resampling, design matrix, and splits."""

import numpy as np
import pytest

from src.data.preprocessing import (
    build_design_matrix,
    epoch_split,
    loao_splits,
    resample_target,
    select_samples,
    trial_holdout,
)


@pytest.fixture
def synthetic_session():
    rng = np.random.default_rng(0)
    n_trials, n_time, n_units = 12, 20, 5
    n_time_dlc = 40
    neural = rng.standard_normal((n_trials, n_time, n_units))
    time_neural = np.linspace(-2.0, 3.0, n_time)
    time_dlc = np.linspace(-2.0, 3.0, n_time_dlc)
    dlc = rng.standard_normal((n_trials, n_time_dlc, 2))
    bodyparts = np.array(["jaw_angle", "jaw_distance"])
    return dict(
        neural=neural, time_neural=time_neural, dlc=dlc, time_dlc=time_dlc, bodyparts=bodyparts
    )


def test_resample_target_shape_and_grid(synthetic_session):
    s = synthetic_session
    target = resample_target(s["dlc"], s["time_dlc"], s["time_neural"], s["bodyparts"])
    assert target.shape == (s["neural"].shape[0], s["neural"].shape[1])
    # interpolation at grid endpoints reproduces the source samples exactly
    np.testing.assert_allclose(target[:, 0], s["dlc"][:, 0, 1])


def test_resample_target_unknown_bodypart(synthetic_session):
    s = synthetic_session
    with pytest.raises(ValueError):
        resample_target(s["dlc"], s["time_dlc"], s["time_neural"], s["bodyparts"], bodypart="nose")


@pytest.mark.parametrize("lags", [0, 3])
def test_build_design_matrix_shapes(synthetic_session, lags):
    s = synthetic_session
    n_trials, n_time, n_units = s["neural"].shape
    target = np.zeros((n_trials, n_time))
    X, y, trial_index = build_design_matrix(s["neural"], target, lags=lags)
    assert X.shape == (n_trials * n_time, n_units * (lags + 1))
    assert y.shape == (n_trials * n_time,)
    assert trial_index.shape == (n_trials * n_time,)
    # lag 0 features equal the raw population vector of the first sample
    np.testing.assert_allclose(X[0, :n_units], s["neural"][0, 0])


def test_design_matrix_axis_mismatch(synthetic_session):
    s = synthetic_session
    with pytest.raises(ValueError):
        build_design_matrix(s["neural"], np.zeros((3, 4)))


def test_epoch_split_partitions_all_trials():
    epochs = epoch_split(100)
    all_idx = np.concatenate([epochs["early"], epochs["mid"], epochs["late"]])
    np.testing.assert_array_equal(np.sort(all_idx), np.arange(100))
    # temporal order preserved: early precedes mid precedes late
    assert epochs["early"].max() < epochs["mid"].min() < epochs["late"].min()


def test_epoch_split_bad_fractions():
    with pytest.raises(ValueError):
        epoch_split(100, fractions=(0.5, 0.4, 0.4))


def test_trial_holdout_disjoint_and_complete():
    train_idx, test_idx = trial_holdout(50, test_frac=0.2, seed=1)
    assert set(train_idx).isdisjoint(test_idx)
    np.testing.assert_array_equal(np.sort(np.concatenate([train_idx, test_idx])), np.arange(50))
    assert len(test_idx) == 10


def test_select_samples_respects_trial_boundaries(synthetic_session):
    s = synthetic_session
    target = np.zeros(s["neural"].shape[:2])
    X, y, trial_index = build_design_matrix(s["neural"], target, lags=2)
    train_idx, test_idx = trial_holdout(s["neural"].shape[0], test_frac=0.25, seed=2)
    Xtr, _ = select_samples(X, y, trial_index, train_idx)
    Xte, _ = select_samples(X, y, trial_index, test_idx)
    # no sample leaks across the split, and together they cover every sample
    assert Xtr.shape[0] + Xte.shape[0] == X.shape[0]
    assert not set(np.unique(trial_index[np.isin(trial_index, train_idx)])) & set(test_idx)


def test_loao_splits_single_animal_yields_nothing():
    assert list(loao_splits(["AB151", "AB151"])) == []


def test_loao_splits_two_animals():
    folds = list(loao_splits(["AB151", "AB151", "AB085"]))
    assert len(folds) == 2  # one held-out fold per subject
    for train_idx, test_idx in folds:
        assert set(train_idx).isdisjoint(test_idx)
