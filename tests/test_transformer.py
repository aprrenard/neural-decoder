"""Unit tests for the NDT model, its losses, the trial dataset and the training pipeline."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data.synthetic import make_synthetic_animal
from src.data.torch_dataset import InputNormalizer, hz_to_counts, rebin
from src.models.transformer import (
    NDTEncoder,
    NeuralDataTransformer,
    count_parameters,
    get_attention_maps,
    masked_poisson_loss,
    random_time_mask,
)
from src.training.train_torch import NDTConfig, pretrain_loss, run_loao, run_within_session

B, T, D = 3, 20, 16


def _model(n_neurons=7, encoder=None):
    torch.manual_seed(0)
    encoder = encoder or NDTEncoder(d_model=D, nhead=2, num_layers=2, dropout=0.1)
    return NeuralDataTransformer(encoder, n_neurons)


# ---------------------------------------------------------------- model


def test_output_shapes():
    model = _model(7)
    x = torch.randn(B, T, 7)
    assert model.decode(x).shape == (B, T)
    assert model.reconstruct(x, random_time_mask(B, T)).shape == (B, T, 7)
    assert model.encode(x).shape == (B, T, D)


def test_wrong_neuron_count_is_rejected():
    with pytest.raises(AssertionError):
        _model(7).decode(torch.randn(B, T, 8))


def test_masked_bins_ignore_their_input():
    """A masked bin's input is replaced by mask_token, so changing it changes nothing."""
    model = _model(7).eval()
    x = torch.randn(1, T, 7)
    mask = torch.zeros(1, T, dtype=torch.bool)
    mask[0, 5] = True
    x2 = x.clone()
    x2[0, 5] += 10.0
    with torch.no_grad():
        torch.testing.assert_close(model.reconstruct(x, mask), model.reconstruct(x2, mask))
        # ...whereas without the mask the perturbation propagates
        assert not torch.allclose(model.decode(x), model.decode(x2))


def test_shared_encoder_gets_gradients_from_both_animals():
    encoder = NDTEncoder(d_model=D, nhead=2, num_layers=2, dropout=0.0)
    models = nn.ModuleDict({"A": _model(5, encoder), "B": _model(9, encoder)})
    # ModuleDict deduplicates the shared encoder: params = encoder + two heads
    n_heads = sum(count_parameters(m.head) for m in models.values())
    assert count_parameters(models) == count_parameters(encoder) + n_heads

    probe = models["A"].encoder.transformer.layers[0].linear1.weight
    grads = {}
    for name, n in (("A", 5), ("B", 9)):
        models.zero_grad()
        batch = {"x": torch.randn(B, T, n), "counts": torch.poisson(torch.ones(B, T, n))}
        pretrain_loss(models[name], batch, mask_ratio=0.3).backward()
        grads[name] = probe.grad.clone()
        assert probe.grad.abs().sum() > 0
    assert not torch.allclose(grads["A"], grads["B"])


def test_attention_maps_are_row_stochastic():
    model = _model(7)
    maps = get_attention_maps(model, torch.randn(2, T, 7))
    assert len(maps) == 2
    assert maps[0].shape == (2, 2, T, T)  # (B, nhead, T, T)
    torch.testing.assert_close(maps[0].sum(-1), torch.ones(2, 2, T))


# ---------------------------------------------------------------- masking / loss


def test_random_time_mask_has_at_least_one_bin_and_is_reproducible():
    mask = random_time_mask(50, 4, ratio=0.0)
    assert mask.sum(1).min() == 1
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    assert torch.equal(random_time_mask(5, T, generator=g1), random_time_mask(5, T, generator=g2))


def test_masked_poisson_loss_only_counts_masked_bins():
    counts = torch.poisson(torch.full((B, T, 4), 2.0))
    mask = random_time_mask(B, T, 0.3)
    log_rate = torch.zeros(B, T, 4)
    perturbed = log_rate.clone()
    perturbed[~mask] = 5.0  # change predictions only at unmasked bins
    assert torch.isclose(
        masked_poisson_loss(log_rate, counts, mask), masked_poisson_loss(perturbed, counts, mask)
    )


def test_poisson_loss_is_minimised_at_true_rate():
    torch.manual_seed(0)
    counts = torch.poisson(torch.full((8, 200, 3), 0.5))
    mask = torch.ones(8, 200, dtype=torch.bool)
    losses = {r: masked_poisson_loss(torch.full_like(counts, np.log(r)), counts, mask).item()
              for r in (0.1, 0.5, 2.0)}
    assert min(losses, key=losses.get) == 0.5


# ---------------------------------------------------------------- data


def test_hz_to_counts_roundtrip_and_rebin():
    counts = np.random.default_rng(0).poisson(0.3, size=(4, 10, 3)).astype(np.uint8)
    np.testing.assert_array_equal(hz_to_counts(counts / 0.01), counts)
    with pytest.raises(ValueError):
        hz_to_counts(counts / 0.01 + 1.0)  # not integer counts
    c2, lick2, t2 = rebin(counts, np.ones((4, 10)), np.arange(10.0), 2)
    assert c2.shape == (4, 5, 3) and lick2.shape == (4, 5) and t2.shape == (5,)
    np.testing.assert_array_equal(c2.sum(1), counts.sum(1))  # spikes are conserved


def test_input_normalizer_zero_mean_unit_std_on_fit_data():
    counts = np.random.default_rng(0).poisson(1.0, size=(30, 20, 5)).astype(np.uint8)
    z = InputNormalizer().fit(counts).transform(counts).reshape(-1, 5)
    np.testing.assert_allclose(z.mean(0), 0, atol=1e-4)
    np.testing.assert_allclose(z.std(0), 1, atol=1e-2)


# ---------------------------------------------------------------- pipeline smoke tests

TINY = dict(d_model=16, nhead=2, num_layers=1, batch_size=8, pretrain_epochs=2,
            finetune_epochs=2, calib_epochs=2, n_calib_trials=10, device="cpu")


@pytest.fixture
def tmp_mlflow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    return tmp_path


def test_within_session_pipeline_runs_and_logs(tmp_mlflow):
    import mlflow

    data = make_synthetic_animal("SIM01", n_neurons=12, n_trials=40, T=30)
    metrics = run_within_session(data, NDTConfig(**TINY), experiment="test")
    assert set(metrics) == {"test_r2", "test_pearson_r", "test_mse"}
    runs = mlflow.search_runs(experiment_names=["test"])
    assert set(runs["tags.stage"].dropna()) == {"pretrain", "finetune"}


def test_loao_pipeline_runs_with_different_neuron_counts(tmp_mlflow):
    animals = {s: make_synthetic_animal(s, n, n_trials=30, T=30, seed=i)
               for i, (s, n) in enumerate([("SIM01", 8), ("SIM02", 13)])}
    results = run_loao(animals, NDTConfig(**TINY), experiment="test")
    assert set(results) == {"SIM01", "SIM02"}
