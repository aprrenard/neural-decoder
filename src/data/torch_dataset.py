"""Trial-structured PyTorch datasets for the sequence models (LSTM / transformer).

The ridge pipeline flattens trials into per-bin samples. Sequence models instead consume
whole trials: one item = one trial, ``x`` of shape ``(T, n_neurons)``.

Two representations of the same spikes are kept apart on purpose:

* ``counts`` — raw spike counts per bin. This is the **Poisson reconstruction target** and
  must never be normalised (Poisson NLL is only meaningful on non-negative counts).
* ``x`` — the model **input**: ``log1p(counts)`` z-scored per neuron, with the statistics
  fit on training trials only (see :class:`InputNormalizer`).

``compute_psth`` (CICADA) returns firing rates in **Hz**, not counts: with 10 ms bins the
stored values are exact multiples of 100. :func:`hz_to_counts` converts back.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import DEFAULT_TARGET, load_session, resample_target

BIN_SIZE_S = 0.01  # neural bin width used at ingestion (seconds)


def hz_to_counts(neural_hz: np.ndarray, bin_size: float = BIN_SIZE_S) -> np.ndarray:
    """Convert a rate PSTH (Hz) back to integer spike counts per bin.

    Args:
        neural_hz: ``(n_trials, T, n_neurons)`` firing rate in Hz.
        bin_size: bin width in seconds.

    Returns:
        ``(n_trials, T, n_neurons)`` ``uint8`` counts (a 10 ms bin never holds > 255 spikes).
    """
    out = np.empty(neural_hz.shape, dtype=np.uint8)
    chunk = 32  # convert a few trials at a time: float64 temporaries of the full array are GBs
    for i in range(0, neural_hz.shape[0], chunk):
        scaled = neural_hz[i : i + chunk] * bin_size
        counts = np.rint(scaled)
        if not np.allclose(counts, scaled, atol=1e-6):
            raise ValueError("neural values are not integer counts × 1/bin_size — wrong bin_size?")
        if counts.max(initial=0) > 255:
            raise ValueError(f"max count per bin {counts.max()} does not fit in uint8")
        out[i : i + chunk] = counts
    return out


def rebin(counts: np.ndarray, lick: np.ndarray, time: np.ndarray, factor: int):
    """Merge ``factor`` adjacent time bins: counts are summed, the lick trace averaged.

    Trailing bins that do not fill a whole new bin are dropped.

    Args:
        counts: ``(n_trials, T, N)`` spike counts.
        lick: ``(n_trials, T)`` lick trace.
        time: ``(T,)`` bin times.
        factor: number of bins to merge (1 = no-op).

    Returns:
        ``counts`` ``(n_trials, T // factor, N)``, ``lick`` ``(n_trials, T // factor)``,
        ``time`` ``(T // factor,)`` (centre of each merged bin).
    """
    if factor == 1:
        return counts, lick, time
    n, T, N = counts.shape
    T_new = T // factor
    c = counts[:, : T_new * factor].reshape(n, T_new, factor, N).sum(axis=2)
    if c.max() > 255:
        raise ValueError("rebinned counts overflow uint8; use a smaller factor")
    lk = lick[:, : T_new * factor].reshape(n, T_new, factor).mean(axis=2)
    t = time[: T_new * factor].reshape(T_new, factor).mean(axis=1)
    return c.astype(np.uint8), lk, t


@dataclass
class TrialData:
    """One animal's trial-aligned data, ready for a sequence model.

    Attributes:
        counts: ``(n_trials, T, n_neurons)`` raw spike counts, ``uint8``.
        lick: ``(n_trials, T)`` lick trace (raw units, e.g. ``jaw_distance`` pixels).
        time: ``(T,)`` bin times relative to stimulus onset (s).
        subject_id: e.g. ``"AB151"``.
        session_id: e.g. ``"AB151_20250128_123409"``.
    """

    counts: np.ndarray
    lick: np.ndarray
    time: np.ndarray
    subject_id: str
    session_id: str

    @property
    def n_trials(self) -> int:
        return self.counts.shape[0]

    @property
    def n_neurons(self) -> int:
        return self.counts.shape[2]

    def subset(self, trials: np.ndarray) -> TrialData:
        """Return a view on a subset of trials (same animal)."""
        return TrialData(
            self.counts[trials], self.lick[trials], self.time, self.subject_id, self.session_id
        )


def load_trial_data(
    npz_path: str | Path, target: str = DEFAULT_TARGET, rebin_factor: int = 1
) -> TrialData:
    """Load an ``*_aligned.npz`` session as :class:`TrialData` (counts + lick on one grid)."""
    session = load_session(npz_path)
    lick = resample_target(
        session["dlc"], session["time_dlc"], session["time_neural"], session["bodyparts"],
        bodypart=target,
    )
    counts = hz_to_counts(session.pop("neural"))
    counts, lick, time = rebin(counts, lick, session["time_neural"], rebin_factor)
    return TrialData(
        counts=counts,
        lick=lick.astype(np.float32),
        time=time.astype(np.float32),
        subject_id=str(session["subject_id"]),
        session_id=str(session["session_id"]),
    )


class InputNormalizer:
    """``log1p`` then per-neuron z-score. Fit on **training trials only**.

    ``log1p`` compresses the heavy right tail of spike counts; the z-score puts every neuron
    on a comparable scale so no high-rate unit dominates the read-in layer.
    """

    def __init__(self, eps: float = 1e-3):
        self.eps = eps
        self.mean: np.ndarray | None = None  # (n_neurons,)
        self.std: np.ndarray | None = None  # (n_neurons,)

    def fit(self, counts: np.ndarray, chunk: int = 64) -> InputNormalizer:
        """Fit on ``(n_trials, T, N)`` counts, in trial chunks to bound memory."""
        n_neurons = counts.shape[2]
        s = np.zeros(n_neurons)
        s2 = np.zeros(n_neurons)
        n = 0
        for i in range(0, counts.shape[0], chunk):
            v = np.log1p(counts[i : i + chunk].astype(np.float32)).reshape(-1, n_neurons)
            s += v.sum(0)
            s2 += (v.astype(np.float64) ** 2).sum(0)
            n += v.shape[0]
        self.mean = (s / n).astype(np.float32)
        var = s2 / n - (s / n) ** 2
        self.std = np.sqrt(np.maximum(var, 0)).astype(np.float32) + self.eps
        return self

    def transform(self, counts: np.ndarray) -> np.ndarray:
        """``(..., N)`` counts → ``(..., N)`` float32 normalised input."""
        if self.mean is None:
            raise RuntimeError("InputNormalizer.transform called before fit")
        return (np.log1p(counts.astype(np.float32)) - self.mean) / self.std


@dataclass
class TargetScaler:
    """Scalar z-score for the lick trace (fit on training trials only)."""

    mean: float = 0.0
    std: float = 1.0

    @classmethod
    def fit(cls, lick: np.ndarray) -> TargetScaler:
        return cls(float(lick.mean()), float(lick.std()) + 1e-8)

    def transform(self, lick: np.ndarray) -> np.ndarray:
        return ((lick - self.mean) / self.std).astype(np.float32)

    def inverse(self, lick_z: np.ndarray) -> np.ndarray:
        return lick_z * self.std + self.mean


class TrialDataset(Dataset):
    """One item = one trial.

    Each item is a dict of tensors:
        ``x``      ``(T, N)`` float32 — normalised input.
        ``counts`` ``(T, N)`` float32 — raw counts (Poisson target).
        ``lick``   ``(T,)``   float32 — z-scored lick trace (MSE target).

    Normalisation happens lazily in ``__getitem__`` so only the compact ``uint8`` counts
    are held in memory (the full session as float32 would be ~1 GB).
    """

    def __init__(self, data: TrialData, normalizer: InputNormalizer, target_scaler: TargetScaler):
        self.data = data
        self.normalizer = normalizer
        self.target_scaler = target_scaler

    def __len__(self) -> int:
        return self.data.n_trials

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        counts = self.data.counts[i]
        return {
            "x": torch.from_numpy(self.normalizer.transform(counts)),
            "counts": torch.from_numpy(counts.astype(np.float32)),
            "lick": torch.from_numpy(self.target_scaler.transform(self.data.lick[i])),
        }


def train_val_test_split(
    n_trials: int, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random trial-level split into sorted train / val / test indices."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_trials)
    n_test = int(round(test_frac * n_trials))
    n_val = int(round(val_frac * n_trials))
    test = np.sort(perm[:n_test])
    val = np.sort(perm[n_test : n_test + n_val])
    train = np.sort(perm[n_test + n_val :])
    return train, val, test
