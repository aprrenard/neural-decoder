"""PyTorch training for the NDT: pretrain → fine-tune → (LOAO) calibrate, logged to MLflow.

MLflow layout (one experiment, ``ndt1``)::

    ndt_{session_id}_{split}            parent run: config, test metrics, final model
    ├── ndt_..._pretrain                stage=pretrain   per-epoch train/val Poisson NLL
    └── ndt_..._finetune                stage=finetune   per-epoch train/val MSE, test R²

    ndt_loao                            parent run for a leave-one-animal-out sweep
    └── loao_{test_subject}             one child per fold
        ├── ..._pretrain / ..._finetune  on the training animals only
        └── ..._calibrate                frozen encoder, held-out animal's read-in + decoder

Every animal is handled through an ``nn.ModuleDict`` ``{subject_id: NeuralDataTransformer}``
whose entries share one encoder. A within-session run is simply the one-animal case.

Run from the project root:
    uv run python -m src.training.train_torch --split holdout
    uv run python -m src.training.train_torch --synthetic-loao
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from src.data.preprocessing import DEFAULT_TARGET, epoch_split
from src.data.torch_dataset import (
    InputNormalizer,
    TargetScaler,
    TrialData,
    TrialDataset,
    load_trial_data,
    train_val_test_split,
)
from src.models.transformer import (
    NDTEncoder,
    NeuralDataTransformer,
    count_parameters,
    get_attention_maps,
    masked_poisson_loss,
    random_time_mask,
)
from src.training.evaluate import regression_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXPERIMENT_NAME = "ndt1"


# --------------------------------------------------------------------------- config / setup


@dataclass
class NDTConfig:
    """Every knob of a run. Logged to MLflow as params and as ``config.json``."""

    # data
    target: str = DEFAULT_TARGET
    rebin_factor: int = 2  # 2 → 20 ms bins, 250 tokens per trial
    split: str = "holdout"  # "holdout" (random trials) | "epoch" (train early → test mid/late)
    val_frac: float = 0.15
    test_frac: float = 0.15
    # model
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 3
    dropout: float = 0.2
    # optimisation
    batch_size: int = 32
    weight_decay: float = 1e-2
    warmup_epochs: int = 2
    grad_clip: float = 1.0
    patience: int = 8  # early stopping on the validation loss
    # stage 1 — masked pretraining
    mask_ratio: float = 0.2
    pretrain_epochs: int = 40
    pretrain_lr: float = 1e-3
    # stage 2 — supervised fine-tuning
    finetune_epochs: int = 40
    finetune_lr: float = 5e-4
    freeze_encoder_in_finetune: bool = False
    # LOAO calibration of the held-out animal
    n_calib_trials: int = 20
    calib_epochs: int = 60
    calib_lr: float = 3e-3
    # reproducibility / hardware
    seed: int = 0
    device: str = "auto"  # "auto" | "cpu" | "mps" | "cuda"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (covers CPU, CUDA and MPS generators)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(preference: str = "auto") -> torch.device:
    """``auto`` → CUDA if present, else Apple MPS, else CPU."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_mlflow(experiment: str = EXPERIMENT_NAME) -> None:
    """Point MLflow at the project's SQLite store regardless of the current directory.

    Absolute paths matter when running from ``notebooks/`` in an interactive window:
    the default ``sqlite:///mlflow.db`` is resolved against the *current* directory.
    An explicit ``MLFLOW_TRACKING_URI`` environment variable (used by the tests) wins.
    """
    if os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        artifact_location = None  # let MLflow pick the default next to the store
    else:
        mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
        artifact_location = (PROJECT_ROOT / "mlruns" / experiment).as_uri()
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=artifact_location)
    mlflow.set_experiment(experiment)


# --------------------------------------------------------------------------- data loading


def make_loader(
    data: TrialData,
    normalizer: InputNormalizer,
    scaler: TargetScaler,
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
) -> DataLoader:
    """DataLoader over trials. A dedicated generator makes the shuffle order reproducible."""
    return DataLoader(
        TrialDataset(data, normalizer, scaler),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )


class MultiAnimalLoader:
    """Interleave batches from one loader per animal; yields ``(subject_id, batch)``.

    A batch never mixes animals (they have different neuron counts), but consecutive
    optimiser steps alternate between animals in random order, so the shared encoder
    is pulled towards all of them at once.
    """

    def __init__(self, loaders: dict[str, DataLoader], shuffle: bool = True, seed: int = 0):
        self.loaders = loaders
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return sum(len(dl) for dl in self.loaders.values())

    def __iter__(self):
        order = [s for s, dl in self.loaders.items() for _ in range(len(dl))]
        if self.shuffle:
            self.rng.shuffle(order)
        iters = {s: iter(dl) for s, dl in self.loaders.items()}
        for s in order:
            yield s, next(iters[s])


# --------------------------------------------------------------------------- losses


def pretrain_loss(
    model: NeuralDataTransformer, batch: dict, mask_ratio: float, generator=None
) -> torch.Tensor:
    """Stage 1: hide random time bins, reconstruct their raw counts (masked Poisson NLL)."""
    x = batch["x"]  # (B, T, N)
    mask = random_time_mask(x.size(0), x.size(1), mask_ratio, x.device, generator)
    return masked_poisson_loss(model.reconstruct(x, mask), batch["counts"], mask)


def finetune_loss(model: NeuralDataTransformer, batch: dict, generator=None) -> torch.Tensor:
    """Stage 2: MSE between decoded and true (z-scored) lick trace, all bins."""
    return F.mse_loss(model.decode(batch["x"]), batch["lick"])


# --------------------------------------------------------------------------- training loop


def warmup_cosine(optimizer, warmup_steps: int, total_steps: int):
    """Linear warm-up to the base LR, then cosine decay to 0. Stepped once per batch."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def run_epoch(
    models: nn.ModuleDict,
    loader,
    loss_fn,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    grad_clip: float | None = None,
    generator: torch.Generator | None = None,
) -> float:
    """One pass over ``loader``. Trains if ``optimizer`` is given, else evaluates.

    Returns the trial-weighted mean loss.
    """
    train = optimizer is not None
    models.train(train)  # toggles dropout; eval mode = deterministic
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for subject, batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = loss_fn(models[subject], batch, generator=generator)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(models.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            bs = batch["x"].size(0)
            total += loss.item() * bs  # .item() syncs with the device: once per batch is fine
            n += bs
    return total / n


def fit(
    models: nn.ModuleDict,
    train_loader,
    val_loader,
    loss_fn,
    *,
    epochs: int,
    lr: float,
    cfg: NDTConfig,
    device: torch.device,
) -> dict[str, list[float]]:
    """AdamW + warm-up/cosine, early stopping on val loss, restore the best weights.

    Only parameters with ``requires_grad=True`` are optimised, which is how freezing the
    encoder works. Per-epoch metrics go to the *active* MLflow run with ``step=epoch``.
    """
    params = [p for p in models.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)
    steps = len(train_loader)
    scheduler = warmup_cosine(optimizer, cfg.warmup_epochs * steps, epochs * steps)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val, best_epoch, best_state, bad = math.inf, -1, None, 0
    for epoch in range(epochs):
        train_loss = run_epoch(
            models, train_loader, loss_fn, device, optimizer, scheduler, cfg.grad_clip
        )
        # Fixed generator → identical validation masks every epoch → comparable val loss.
        val_gen = torch.Generator().manual_seed(cfg.seed + 1)
        val_loss = run_epoch(models, val_loader, loss_fn, device, generator=val_gen)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        mlflow.log_metrics(
            {"train_loss": train_loss, "val_loss": val_loss, "lr": scheduler.get_last_lr()[0]},
            step=epoch,
        )
        if val_loss < best_val - 1e-4:
            best_val, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in models.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        models.load_state_dict(best_state)
    mlflow.log_metrics({"best_val_loss": best_val, "best_epoch": best_epoch, "epochs_run": epoch + 1})
    return history


@torch.no_grad()
def predict(
    model: NeuralDataTransformer, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Decode every trial in ``loader``. Returns ``(pred, true)``, each ``(n_trials, T)`` z-units."""
    model.eval()
    preds, trues = [], []
    for batch in loader:
        preds.append(model.decode(batch["x"].to(device)).cpu().numpy())
        trues.append(batch["lick"].numpy())
    return np.concatenate(preds), np.concatenate(trues)


@torch.no_grad()
def mean_rate_baseline_nll(
    loader, mean_counts: torch.Tensor, mask_ratio: float, seed: int
) -> float:
    """Masked Poisson NLL of a model that always predicts each neuron's mean rate.

    The pretrained NDT must beat this to have learned anything beyond firing rates.
    Uses the same fixed validation masks as :func:`fit`.
    """
    gen = torch.Generator().manual_seed(seed)
    log_rate_1 = torch.log(mean_counts.clamp(min=1e-3))
    total, n = 0.0, 0
    for _, batch in loader:
        counts = batch["counts"]
        mask = random_time_mask(counts.size(0), counts.size(1), mask_ratio, "cpu", gen)
        log_rate = log_rate_1.expand_as(counts)
        total += masked_poisson_loss(log_rate, counts, mask).item() * counts.size(0)
        n += counts.size(0)
    return total / n


# --------------------------------------------------------------------------- figures


def plot_loss_curves(history: dict[str, list[float]], title: str, baseline: float | None = None):
    fig = Figure(figsize=(5, 3.5))
    ax = fig.subplots()
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    if baseline is not None:
        ax.axhline(baseline, color="grey", ls="--", label="mean-rate baseline")
    ax.set(xlabel="epoch", ylabel="loss", title=title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_predictions(pred: np.ndarray, true: np.ndarray, time: np.ndarray, n: int = 4):
    fig = Figure(figsize=(8, 1.8 * n))
    axes = fig.subplots(n, 1, sharex=True)
    for ax, i in zip(np.atleast_1d(axes), np.linspace(0, len(pred) - 1, n).astype(int)):
        ax.plot(time, true[i], color="k", lw=1, label="true")
        ax.plot(time, pred[i], color="C1", lw=1, label="decoded")
        ax.set_ylabel(f"trial {i}")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time from stimulus (s)")
    fig.tight_layout()
    return fig


def plot_attention(attn: torch.Tensor, time: np.ndarray, title: str = ""):
    """``attn`` ``(nhead, T, T)`` for one trial → one heatmap per head."""
    nhead = attn.size(0)
    fig = Figure(figsize=(3 * nhead, 3))
    axes = fig.subplots(1, nhead)
    ext = [time[0], time[-1], time[-1], time[0]]
    for h, ax in enumerate(np.atleast_1d(axes)):
        ax.imshow(attn[h].cpu().numpy(), extent=ext, cmap="viridis")
        ax.set(title=f"head {h}", xlabel="key time (s)")
    np.atleast_1d(axes)[0].set_ylabel("query time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- helpers


def _prepare_animal(
    data: TrialData, train_idx: np.ndarray, cfg: NDTConfig
) -> tuple[InputNormalizer, TargetScaler, torch.Tensor]:
    """Fit input/target scalers and mean counts on this animal's *training* trials only."""
    train = data.subset(train_idx)
    normalizer = InputNormalizer().fit(train.counts)
    scaler = TargetScaler.fit(train.lick)
    mean_counts = torch.tensor(train.counts.mean(axis=(0, 1)), dtype=torch.float32)
    return normalizer, scaler, mean_counts


def _build_encoder(cfg: NDTConfig, T: int) -> NDTEncoder:
    return NDTEncoder(cfg.d_model, cfg.nhead, cfg.num_layers, cfg.dropout, max_len=max(T, 1000))


def _stage_run(name: str, stage: str, **tags):
    """Nested MLflow run for one training stage."""
    run = mlflow.start_run(run_name=name, nested=True)
    mlflow.set_tags({"model": "ndt1", "stage": stage, **tags})
    return run


def _log_preprocessing(normalizers: dict, scalers: dict) -> None:
    """Save what's needed to reproduce the inputs at inference time."""
    mlflow.log_dict(
        {
            s: {
                "input_log1p_mean": normalizers[s].mean.tolist(),
                "input_log1p_std": normalizers[s].std.tolist(),
                "lick_mean": scalers[s].mean,
                "lick_std": scalers[s].std,
            }
            for s in normalizers
        },
        "preprocessing.json",
    )


# --------------------------------------------------------------------------- within-session


def run_within_session(
    data: TrialData, cfg: NDTConfig, experiment: str = EXPERIMENT_NAME
) -> dict[str, float]:
    """Pretrain + fine-tune on one session, evaluate on held-out trials (Q2-style splits).

    ``split="holdout"``: random train/val/test trials.
    ``split="epoch"``: train/val drawn from the early epoch, test on mid and late epochs.
    Pretraining sees only training trials, so test trials never enter any stage.
    """
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    setup_mlflow(experiment)
    subject = data.subject_id
    T = data.counts.shape[1]

    if cfg.split == "holdout":
        tr, va, te = train_val_test_split(data.n_trials, cfg.val_frac, cfg.test_frac, cfg.seed)
        test_sets = {"test": te}
    elif cfg.split == "epoch":
        ep = epoch_split(data.n_trials)
        early = np.random.default_rng(cfg.seed).permutation(ep["early"])
        n_val = int(round(cfg.val_frac * len(early)))
        va, tr = np.sort(early[:n_val]), np.sort(early[n_val:])
        test_sets = {"mid": ep["mid"], "late": ep["late"]}
    else:
        raise ValueError(f"unknown split {cfg.split!r}")

    normalizer, scaler, mean_counts = _prepare_animal(data, tr, cfg)
    encoder = _build_encoder(cfg, T)
    model = NeuralDataTransformer(encoder, data.n_neurons)
    model.head.init_recon_bias(mean_counts)
    models = nn.ModuleDict({subject: model}).to(device)

    def loader(idx, shuffle):
        return make_loader(data.subset(idx), normalizer, scaler, cfg.batch_size, shuffle, cfg.seed)

    train_loader = MultiAnimalLoader({subject: loader(tr, True)}, seed=cfg.seed)
    val_loader = MultiAnimalLoader({subject: loader(va, False)}, shuffle=False)

    run_name = f"ndt_{data.session_id}_{cfg.split}"
    results: dict[str, float] = {}
    with mlflow.start_run(run_name=run_name):
        tags = {"split": cfg.split, "subject_id": subject, "session_id": data.session_id}
        mlflow.set_tags({"model": "ndt1", **tags})
        mlflow.log_params(asdict(cfg))
        mlflow.log_params({
            "n_neurons": data.n_neurons, "T": T, "n_train": len(tr), "n_val": len(va),
            "n_params": count_parameters(models), "device_used": str(device),
        })
        mlflow.log_dict(asdict(cfg), "config.json")
        _log_preprocessing({subject: normalizer}, {subject: scaler})

        # ---- stage 1: masked pretraining (no labels)
        if cfg.pretrain_epochs > 0:
            with _stage_run(f"{run_name}_pretrain", "pretrain", **tags):

                def pt_loss(m, b, generator=None):
                    return pretrain_loss(m, b, cfg.mask_ratio, generator)

                hist = fit(models, train_loader, val_loader, pt_loss,
                           epochs=cfg.pretrain_epochs, lr=cfg.pretrain_lr, cfg=cfg, device=device)
                baseline = mean_rate_baseline_nll(val_loader, mean_counts, cfg.mask_ratio, cfg.seed + 1)
                mlflow.log_metric("val_loss_mean_rate_baseline", baseline)
                mlflow.log_figure(plot_loss_curves(hist, "pretrain (masked Poisson NLL)", baseline),
                                  "pretrain_loss.png")

        # ---- stage 2: supervised fine-tuning on the lick trace
        with _stage_run(f"{run_name}_finetune", "finetune", **tags):
            if cfg.freeze_encoder_in_finetune:
                encoder.requires_grad_(False)
            hist = fit(models, train_loader, val_loader, finetune_loss,
                       epochs=cfg.finetune_epochs, lr=cfg.finetune_lr, cfg=cfg, device=device)
            mlflow.log_figure(plot_loss_curves(hist, "finetune (MSE, z-scored lick)"),
                              "finetune_loss.png")
            for name, idx in test_sets.items():
                pred, true = predict(model, loader(idx, False), device)
                m = regression_metrics(true.ravel(), pred.ravel())
                results.update({f"{name}_{k}": v for k, v in m.items()})
                mlflow.log_figure(plot_predictions(pred, true, data.time), f"pred_{name}.png")
            mlflow.log_metrics(results)
            x0 = next(iter(loader(test_sets[next(iter(test_sets))], False)))["x"][:1].to(device)
            attn = get_attention_maps(model, x0)
            mlflow.log_figure(plot_attention(attn[-1][0], data.time, "last layer, one test trial"),
                              "attention_last_layer.png")

        mlflow.log_metrics(results)  # also on the parent for easy comparison across runs
        model.to("cpu")
        mlflow.pytorch.log_model(
            model, name="model", input_example=np.zeros((1, T, data.n_neurons), dtype=np.float32)
        )
    return results


# --------------------------------------------------------------------------- LOAO


def run_loao(
    animals: dict[str, TrialData], cfg: NDTConfig, experiment: str = EXPERIMENT_NAME
) -> dict[str, dict[str, float]]:
    """Leave-one-animal-out sweep (Q1): one nested child run per held-out animal."""
    if len(animals) < 2:
        raise ValueError(f"LOAO needs ≥ 2 animals, got {list(animals)}")
    setup_mlflow(experiment)
    results = {}
    with mlflow.start_run(run_name="ndt_loao"):
        mlflow.set_tags({"model": "ndt1", "split": "loao"})
        mlflow.log_params(asdict(cfg))
        mlflow.log_param("subjects", ",".join(sorted(animals)))
        mlflow.log_dict(asdict(cfg), "config.json")
        for test_subject in sorted(animals):
            with mlflow.start_run(run_name=f"loao_{test_subject}", nested=True):
                mlflow.set_tags({"model": "ndt1", "split": "loao", "test_subject": test_subject})
                results[test_subject] = _run_loao_fold(animals, test_subject, cfg)
                mlflow.log_metrics(results[test_subject])
    return results


def _run_loao_fold(
    animals: dict[str, TrialData], test_subject: str, cfg: NDTConfig
) -> dict[str, float]:
    """Pretrain + fine-tune on the training animals; calibrate the held-out animal.

    The held-out animal never enters pretraining or fine-tuning, even unlabelled.
    Calibration: encoder frozen, only the held-out ``read_in`` + ``decode_head`` are fit on
    ``n_calib_trials`` trials; the remaining trials are the test set. This is *calibrated*
    transfer, not zero-shot.
    """
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    T = next(iter(animals.values())).counts.shape[1]
    tags = {"split": "loao", "test_subject": test_subject}

    encoder = _build_encoder(cfg, T)
    models = nn.ModuleDict()
    train_loaders, val_loaders, mean_counts, norms, scalers = {}, {}, {}, {}, {}
    for s, d in animals.items():
        if s == test_subject:
            continue
        tr, va, _ = train_val_test_split(d.n_trials, cfg.val_frac, 0.0, cfg.seed)
        norms[s], scalers[s], mean_counts[s] = _prepare_animal(d, tr, cfg)
        models[s] = NeuralDataTransformer(encoder, d.n_neurons)  # same encoder object
        models[s].head.init_recon_bias(mean_counts[s])
        train_loaders[s] = make_loader(d.subset(tr), norms[s], scalers[s], cfg.batch_size, True, cfg.seed)
        val_loaders[s] = make_loader(d.subset(va), norms[s], scalers[s], cfg.batch_size, False)
    models.to(device)
    train_loader = MultiAnimalLoader(train_loaders, seed=cfg.seed)
    val_loader = MultiAnimalLoader(val_loaders, shuffle=False)
    mlflow.log_param("n_params_fold", count_parameters(models))
    _log_preprocessing(norms, scalers)

    if cfg.pretrain_epochs > 0:
        with _stage_run(f"loao_{test_subject}_pretrain", "pretrain", **tags):

            def pt_loss(m, b, generator=None):
                return pretrain_loss(m, b, cfg.mask_ratio, generator)

            hist = fit(models, train_loader, val_loader, pt_loss,
                       epochs=cfg.pretrain_epochs, lr=cfg.pretrain_lr, cfg=cfg, device=device)
            mlflow.log_figure(plot_loss_curves(hist, "pretrain"), "pretrain_loss.png")

    with _stage_run(f"loao_{test_subject}_finetune", "finetune", **tags):
        hist = fit(models, train_loader, val_loader, finetune_loss,
                   epochs=cfg.finetune_epochs, lr=cfg.finetune_lr, cfg=cfg, device=device)
        mlflow.log_figure(plot_loss_curves(hist, "finetune"), "finetune_loss.png")

    # ---- calibration on the held-out animal
    held = animals[test_subject]
    perm = np.random.default_rng(cfg.seed).permutation(held.n_trials)
    calib, test = perm[: cfg.n_calib_trials], np.sort(perm[cfg.n_calib_trials :])
    n_cval = max(1, int(round(cfg.val_frac * len(calib))))
    c_val, c_tr = np.sort(calib[:n_cval]), np.sort(calib[n_cval:])
    norm, scaler, _ = _prepare_animal(held, calib, cfg)  # scalers fit on calibration trials

    encoder.requires_grad_(False)
    held_model = NeuralDataTransformer(encoder, held.n_neurons).to(device)
    held_model.head.recon_head.requires_grad_(False)  # unused during calibration
    held_models = nn.ModuleDict({test_subject: held_model})

    def held_loader(idx, shuffle):
        return make_loader(held.subset(idx), norm, scaler, cfg.batch_size, shuffle, cfg.seed)

    with _stage_run(f"loao_{test_subject}_calibrate", "calibrate", **tags):
        mlflow.log_params({"n_calib_trials": len(calib), "n_test_trials": len(test),
                           "n_params_calibrated": count_parameters(held_models)})
        hist = fit(held_models,
                   MultiAnimalLoader({test_subject: held_loader(c_tr, True)}, seed=cfg.seed),
                   MultiAnimalLoader({test_subject: held_loader(c_val, False)}, shuffle=False),
                   finetune_loss, epochs=cfg.calib_epochs, lr=cfg.calib_lr, cfg=cfg, device=device)
        mlflow.log_figure(plot_loss_curves(hist, "calibrate"), "calibrate_loss.png")
        pred, true = predict(held_model, held_loader(test, False), device)
        metrics = {f"test_{k}": v for k, v in regression_metrics(true.ravel(), pred.ravel()).items()}
        mlflow.log_metrics(metrics)
        mlflow.log_figure(plot_predictions(pred, true, held.time), "pred_test.png")
    encoder.requires_grad_(True)
    return metrics


# --------------------------------------------------------------------------- CLI


def _parse_args() -> tuple[NDTConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for f in fields(NDTConfig):
        kind = type(f.default)
        conv = (lambda s: s.lower() in ("1", "true", "yes")) if kind is bool else kind
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=conv, default=f.default)
    parser.add_argument("--synthetic-loao", action="store_true",
                        help="run the LOAO pipeline on 3 simulated animals")
    args = parser.parse_args()
    cfg = NDTConfig(**{f.name: getattr(args, f.name) for f in fields(NDTConfig)})
    return cfg, args


def main() -> None:
    cfg, args = _parse_args()
    if args.synthetic_loao:
        from src.data.synthetic import make_synthetic_animal

        animals = {
            s: make_synthetic_animal(s, n, n_trials=120, T=100, seed=i)
            for i, (s, n) in enumerate([("SIM01", 40), ("SIM02", 55), ("SIM03", 70)])
        }
        for s, m in run_loao(animals, cfg, experiment="ndt1-synthetic").items():
            print(f"held-out {s}: R²={m['test_r2']:+.3f}  r={m['test_pearson_r']:+.3f}")
        return

    paths = sorted(PROCESSED_DIR.glob("*_aligned.npz"))
    if not paths:
        print(f"No aligned sessions in {PROCESSED_DIR}. Run ingestion first.")
        return
    for path in paths:
        data = load_trial_data(path, cfg.target, cfg.rebin_factor)
        print(f"=== {data.session_id}: {data.n_trials} trials, T={data.counts.shape[1]}, "
              f"N={data.n_neurons} ===")
        for k, v in run_within_session(data, cfg).items():
            print(f"  {k:20s} {v:+.4f}")


if __name__ == "__main__":
    main()
