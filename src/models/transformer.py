"""NDT1-style Neural Data Transformer (Ye & Pandarinath 2021), kept deliberately small.

Tokens are **time bins**: one trial ``(T, n_neurons)`` is a sequence of ``T`` tokens, each
the population vector at that bin. Attention is bidirectional (BERT-like, no causal mask).

Module layout — what is shared across animals and what is not::

    x (B, T, N_animal) ──read_in──▶ (B, T, d) ──[mask_token]──▶ +pos ──▶ TransformerEncoder ──▶ z (B, T, d)
         per animal                     shared          shared            shared
    z ──recon_head──▶ log-rate (B, T, N_animal)   per animal   (stage 1: masked Poisson NLL)
    z ──decode_head─▶ lick     (B, T)             per animal   (stage 2: MSE)

``NDTEncoder`` only ever sees ``d_model``-dim tokens, so one encoder instance can be shared
by animals with different neuron counts. Each animal gets an ``AnimalHead``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal encoding on the **time axis only** (neuron order is arbitrary).

    Stored as a buffer: it moves with ``.to(device)`` and is saved in ``state_dict`` but is
    not a trainable parameter.
    """

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        if d_model % 2:
            raise ValueError(f"d_model must be even for sin/cos pairs, got {d_model}")
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()  # (max_len, 1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, d_model)`` → ``(B, T, d_model)``."""
        T = x.size(1)
        assert T <= self.pe.size(1), f"sequence length {T} > max_len {self.pe.size(1)}"
        return x + self.pe[:, :T]


class NDTEncoder(nn.Module):
    """The shared part: mask token, positional encoding and the transformer stack.

    Uses pre-LayerNorm (``norm_first=True``) blocks, which train more stably than the
    original post-LN without needing a long warm-up; a final LayerNorm closes the stack.
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        max_len: int = 1000,
    ):
        super().__init__()
        self.d_model = d_model
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.pos = PositionalEncoding(d_model, max_len)
        self.embed_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode a batch of token sequences.

        Args:
            tokens: ``(B, T, d_model)`` read-in population vectors.
            mask: optional ``(B, T)`` bool, ``True`` = this bin is hidden and replaced by
                the learned ``mask_token`` (masked-modelling pretraining).

        Returns:
            ``(B, T, d_model)`` contextual latents.
        """
        assert tokens.dim() == 3 and tokens.size(-1) == self.d_model, tokens.shape
        if mask is not None:
            assert mask.shape == tokens.shape[:2], (mask.shape, tokens.shape)
            tokens = torch.where(mask.unsqueeze(-1), self.mask_token, tokens)
        return self.transformer(self.embed_dropout(self.pos(tokens)))


class AnimalHead(nn.Module):
    """Per-animal input/output layers around the shared encoder."""

    def __init__(self, n_neurons: int, d_model: int):
        super().__init__()
        self.n_neurons = n_neurons
        self.read_in = nn.Linear(n_neurons, d_model)
        self.recon_head = nn.Linear(d_model, n_neurons)  # outputs log-rate per neuron
        self.decode_head = nn.Linear(d_model, 1)  # lick value per bin

    @torch.no_grad()
    def init_recon_bias(self, mean_counts: torch.Tensor) -> None:
        """Start the reconstruction at each neuron's mean rate (log of mean count per bin).

        Without this the initial prediction is ``exp(0) = 1`` spike/bin for every neuron —
        ~10× too high for 10 Hz cells — and the first epochs are spent just learning offsets.
        """
        self.recon_head.bias.copy_(torch.log(mean_counts.clamp(min=1e-3)))


class NeuralDataTransformer(nn.Module):
    """One animal's model: its own ``AnimalHead`` around a (possibly shared) encoder.

    Several ``NeuralDataTransformer`` instances can hold the *same* ``NDTEncoder`` object;
    gradients from every animal then accumulate into the one set of encoder weights.
    """

    def __init__(self, encoder: NDTEncoder, n_neurons: int):
        super().__init__()
        self.encoder = encoder
        self.head = AnimalHead(n_neurons, encoder.d_model)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, T, N)`` input → ``(B, T, d_model)`` latents."""
        assert x.dim() == 3 and x.size(-1) == self.head.n_neurons, (
            f"expected (B, T, {self.head.n_neurons}), got {tuple(x.shape)}"
        )
        return self.encoder(self.head.read_in(x), mask)

    def reconstruct(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked reconstruction: ``(B, T, N)`` → log-rate ``(B, T, N)``."""
        return self.head.recon_head(self.encode(x, mask))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Lick decoding: ``(B, T, N)`` → ``(B, T)``."""
        return self.head.decode_head(self.encode(x)).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward = decoding (what the serving API and ``mlflow.pytorch`` call)."""
        return self.decode(x)


def random_time_mask(
    B: int,
    T: int,
    ratio: float = 0.2,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Bernoulli(``ratio``) mask over time bins, ``(B, T)`` bool, ≥ 1 masked bin per row.

    Pass a seeded CPU ``generator`` to get the *same* mask every call (used for validation
    so the val loss is not noisy from epoch to epoch).
    """
    mask = torch.rand(B, T, generator=generator) < ratio
    empty = ~mask.any(dim=1)
    if empty.any():
        idx = torch.randint(0, T, (int(empty.sum()),), generator=generator)
        mask[empty, idx] = True
    return mask.to(device)


def masked_poisson_loss(
    log_rate: torch.Tensor, counts: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Poisson NLL averaged over masked (bin, neuron) entries only.

    Args:
        log_rate: ``(B, T, N)`` predicted log-rate (``log_input=True``).
        counts: ``(B, T, N)`` **raw** spike counts (never normalised).
        mask: ``(B, T)`` bool, ``True`` = bin was masked → contributes to the loss.

    Returns:
        Scalar loss. With ``full=False`` the constant ``log(k!)`` term is omitted, so the
        value can be negative; only differences between models are meaningful.
    """
    nll = F.poisson_nll_loss(log_rate, counts, log_input=True, full=False, reduction="none")
    m = mask.unsqueeze(-1).to(nll.dtype)
    return (nll * m).sum() / (m.sum() * counts.size(-1)).clamp(min=1)


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def get_attention_maps(
    model: NeuralDataTransformer, x: torch.Tensor, mask: torch.Tensor | None = None
) -> list[torch.Tensor]:
    """Return per-layer attention weights ``[(B, nhead, T, T), ...]`` for one batch.

    ``nn.TransformerEncoderLayer`` calls its attention with ``need_weights=False``, so the
    weights are normally thrown away. Forward hooks flip that flag and catch the output.
    PyTorch also has a fused inference "fast path" that skips the attention submodule (and
    therefore the hooks) in eval mode without grad; making the input require grad disables it.
    """
    maps: list[torch.Tensor] = []
    handles = []

    def pre_hook(module, args, kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return args, kwargs

    def post_hook(module, args, kwargs, output):
        maps.append(output[1].detach())

    for layer in model.encoder.transformer.layers:
        handles.append(layer.self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(layer.self_attn.register_forward_hook(post_hook, with_kwargs=True))
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            model.encode(x.detach().clone().requires_grad_(True), mask)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    if len(maps) != len(model.encoder.transformer.layers):
        raise RuntimeError("attention hooks did not fire (fused fast path?)")
    return maps
