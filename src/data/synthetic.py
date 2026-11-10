"""Synthetic multi-animal spiking data for tests and the LOAO demo.

Only one real animal is ingested so far, so cross-animal code paths (shared encoder,
per-animal read-in, calibration of a held-out animal) are exercised on simulated data:

* every animal shares the same **latent dynamics** (a few smooth trial-locked factors),
* each animal has its **own neurons** (different count, random loading matrix),
* the lick trace is a fixed nonlinear function of the latents.

So a decoder that learns the latent → lick map in a shared space should transfer across
animals once a per-animal read-in is calibrated. That mirrors the NDT1 hypothesis.
"""

from __future__ import annotations

import numpy as np

from src.data.torch_dataset import TrialData


def _latents(n_trials: int, T: int, n_latents: int, rng: np.random.Generator) -> np.ndarray:
    """``(n_trials, T, n_latents)`` smooth latents: trial-locked bump + trial-varying wobble."""
    t = np.linspace(0, 1, T)
    onset = 0.4  # "stimulus" at 40 % of the trial
    lat = np.empty((n_trials, T, n_latents))
    for k in range(n_latents):
        freq = 1.5 + k
        phase = rng.uniform(0, 2 * np.pi, size=(n_trials, 1))
        gain = rng.uniform(0.5, 1.5, size=(n_trials, 1))  # trial-to-trial variability
        bump = np.exp(-((t - onset - 0.1 * k) ** 2) / 0.005)
        lat[:, :, k] = gain * bump + 0.5 * np.sin(2 * np.pi * freq * t + phase)
    return lat


def make_synthetic_animal(
    subject_id: str,
    n_neurons: int,
    n_trials: int = 64,
    T: int = 100,
    n_latents: int = 4,
    base_rate: float = 0.1,
    seed: int = 0,
) -> TrialData:
    """Simulate one animal: Poisson spikes from shared latent dynamics.

    Args:
        subject_id: animal name (used as the dict key in multi-animal code).
        n_neurons: neurons for this animal (may differ between animals).
        base_rate: mean spike count per bin (≈ 10 Hz at 10 ms bins).
        seed: controls the loading matrix, latents and spikes of this animal.
    """
    rng = np.random.default_rng(seed)
    lat = _latents(n_trials, T, n_latents, rng)
    W = rng.normal(0, 0.6, size=(n_latents, n_neurons))  # animal-specific loadings
    log_rate = np.log(base_rate) + lat @ W
    counts = rng.poisson(np.exp(np.clip(log_rate, -10, 3))).astype(np.uint8)
    lick = np.tanh(lat[:, :, 0] + 0.5 * lat[:, :, 1] ** 2)  # same map for every animal
    return TrialData(
        counts=counts,
        lick=lick.astype(np.float32),
        time=np.linspace(-2, 3, T, dtype=np.float32),
        subject_id=subject_id,
        session_id=f"{subject_id}_synthetic",
    )
