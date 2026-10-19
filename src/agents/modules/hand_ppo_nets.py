"""Networks of the per-hand PPO agent (agents.hand_ppo): one actor and one critic per hand.

Both are rsl_rl's MLPModel layout with the 1D-conv FutureTrajEncoder in front: a running observation normalizer
(rsl_rl EmpiricalNormalization) over every input dim, then the encoder over the future-trajectory window at the obs
tail (each network has its own encoder; the window is replaced by its latent), then the trunk: rsl_rl's MLP (``arch:
mlp``; PyTorch default init, as rsl_rl's MLPModel) or the SimBa residual trunk of the alphanist paper (``arch: simba``,
SimbaTrunk). The actor's trunk outputs the means of its hand's action dims; its std is rsl_rl's GaussianDistribution
with std_type "scalar" (a learned state-independent std per dim, clamped to [1e-6, 1e6]). The critic's trunk outputs
one value (in normalized units when agents.hand_ppo normalizes values).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.modules.mlp import MLP
from rsl_rl.modules.normalization import EmpiricalNormalization
from torch.distributions import Normal

from .traj_encoder import FutureTrajEncoder

_STD_RANGE = (1e-6, 1e6)   # rsl_rl GaussianDistribution's default clamp for std_type "scalar"
_SIMBA_ACTIVATIONS = {"swish": nn.SiLU, "relu": nn.ReLU}


def _he_normal_(w: torch.Tensor) -> None:
    """flax ``he_normal`` (SimBa's block init): variance_scaling(2, fan_in, truncated_normal) = a standard normal
    truncated to [-2, 2], scaled by sqrt(2 / fan_in) / 0.8796... (the std of that truncated normal)."""
    std = math.sqrt(2.0 / w.shape[1]) / 0.87962566103423978
    nn.init.trunc_normal_(w, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


class SimbaBlock(nn.Module):
    """SimBa ResidualBlock (scale_rl/networks/layers.py): x + Dense(h)(act(Dense(e * h)(LayerNorm(x)))), He-normal
    kernels, zero biases. alphanist (Fig. 3) uses Swish as the activation; SimBa uses ReLU."""

    def __init__(self, hidden: int, expansion: int, activation: str, ln_eps: float) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(hidden, eps=ln_eps)
        self.fc1 = nn.Linear(hidden, expansion * hidden)
        self.act = _SIMBA_ACTIVATIONS[activation]()
        self.fc2 = nn.Linear(expansion * hidden, hidden)
        for fc in (self.fc1, self.fc2):
            _he_normal_(fc.weight)
            nn.init.zeros_(fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(self.ln(x))))


class SimbaTrunk(nn.Module):
    """The alphanist actor / critic trunk (Fig. 3) = SimBa's encoder (sac_network.SACEncoder, block_type residual) +
    a linear head: linear embedding to ``hidden_dim`` (orthogonal, gain 1), ``num_blocks`` SimbaBlocks, LayerNorm,
    linear head (orthogonal, gain ``head_init_scale``; SimBa's heads use 1). Zero biases; LayerNorm epsilon 1e-6 =
    flax's default."""

    def __init__(self, d_in: int, out_dim: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden, eps = int(cfg["hidden_dim"]), float(cfg["ln_eps"])
        if str(cfg["activation"]) not in _SIMBA_ACTIVATIONS:
            raise ValueError(f"simba.activation must be one of {sorted(_SIMBA_ACTIVATIONS)}, got {cfg['activation']!r}")
        self.embed = nn.Linear(d_in, hidden)
        self.blocks = nn.Sequential(*(SimbaBlock(hidden, int(cfg["expansion"]), str(cfg["activation"]), eps)
                                      for _ in range(int(cfg["num_blocks"]))))
        self.post_ln = nn.LayerNorm(hidden, eps=eps)
        self.head = nn.Linear(hidden, out_dim)
        nn.init.orthogonal_(self.embed.weight, gain=1.0)
        nn.init.orthogonal_(self.head.weight, gain=float(cfg["head_init_scale"]))
        nn.init.zeros_(self.embed.bias)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.post_ln(self.blocks(self.embed(x))))


class _EncodedMLP(nn.Module):
    """obs normalizer -> FutureTrajEncoder (window -> latent) -> trunk (rsl_rl MLP or SimbaTrunk; the attribute is
    ``mlp`` for both so the ``arch: mlp`` checkpoints keep their keys)."""

    def __init__(self, obs_dim: int, out_dim: int, traj_cfg: dict[str, Any] | None, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.obs_norm = EmpiricalNormalization(obs_dim) if bool(cfg.get("obs_normalization", True)) else nn.Identity()
        self.traj = FutureTrajEncoder(traj_cfg) if traj_cfg else None
        d_in = obs_dim if self.traj is None else obs_dim - self.traj.window_width + self.traj.latent_dim
        arch = str(cfg.get("arch", "mlp"))
        if arch == "mlp":
            self.mlp = MLP(d_in, out_dim, [int(h) for h in cfg.get("hidden_dims", [256, 256, 256])],
                           str(cfg.get("activation", "elu")))
        elif arch == "simba":
            self.mlp = SimbaTrunk(d_in, out_dim, cfg["simba"])
        else:
            raise ValueError(f"arch must be 'mlp' or 'simba' here, got {arch!r}")

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.obs_norm(obs)
        return x if self.traj is None else self.traj(x)

    def update_normalization(self, obs: torch.Tensor) -> None:
        if isinstance(self.obs_norm, EmpiricalNormalization):
            self.obs_norm.update(obs)


class HandActor(_EncodedMLP):
    """One hand's policy: means from the encoded MLP, learned state-independent std (rsl_rl scalar std)."""

    def __init__(self, obs_dim: int, action_dim: int, traj_cfg: dict[str, Any] | None, cfg: dict[str, Any]) -> None:
        super().__init__(obs_dim, action_dim, traj_cfg, cfg)
        self.std_param = nn.Parameter(float(cfg.get("init_std", 1.0)) * torch.ones(action_dim))

    def mean(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.features(obs))

    def std(self) -> torch.Tensor:
        return self.std_param.clamp(*_STD_RANGE)

    def distribution(self, obs: torch.Tensor) -> Normal:
        return Normal(self.mean(obs), self.std())


class HandCritic(_EncodedMLP):
    """One hand's value function: the encoded trunk with one output (``critic.arch`` mlp or simba)."""

    def __init__(self, obs_dim: int, traj_cfg: dict[str, Any] | None, cfg: dict[str, Any]) -> None:
        super().__init__(obs_dim, 1, traj_cfg, cfg)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.features(obs)).squeeze(-1)


class FlashSACValueCritic(nn.Module):
    """FlashSAC's critic member as a state-value function V(s) (``critic.arch: flashsac``): the same layers as one
    member of rsl_rl_flashsac's FlashSACDoubleCritic behind the FutureTrajEncoder, without the action input —
    FlashSACEmbedder (UnitBatchNorm over the raw [obs | trajectory latent], then a unit-norm linear to ``hidden_dim``),
    ``num_blocks`` FlashSACBlocks (unit-norm linears hidden -> 4 hidden -> hidden, UnitBatchNorm, ReLU, residual),
    UnitRMSNorm, and the categorical head (unit-norm linear + bias -> ``num_bins`` logits over [min_v, max_v]; value =
    expected bin). No separate observation normalizer (the BatchNorm normalizes the inputs, as in FlashSAC).
    BatchNorm uses batch statistics in the update (``logits(..., training=True)``) and running statistics everywhere
    else (``forward``: rollout values, bootstrap values, RAgger re-valuation). Trained by cross-entropy against the
    two-hot projection of the normalized return (``ce_loss``); ``normalize_parameters`` re-projects the unit-norm
    weights after every optimizer step (FlashSAC)."""

    def __init__(self, obs_dim: int, traj_cfg: dict[str, Any] | None, cfg: dict[str, Any]) -> None:
        super().__init__()
        from rsl_rl_flashsac.modules.flash_sac_layers import (
            FlashSACBlock,
            FlashSACEmbedder,
            UnitLinear,
            UnitRMSNorm,
        )

        self.traj = FutureTrajEncoder(traj_cfg) if traj_cfg else None
        d_in = obs_dim if self.traj is None else obs_dim - self.traj.window_width + self.traj.latent_dim
        hidden = int(cfg.get("hidden_dim", 1024))
        self.num_bins = int(cfg.get("num_bins", 101))
        self.min_v, self.max_v = float(cfg.get("min_v", -5.0)), float(cfg.get("max_v", 5.0))
        self.embedder = FlashSACEmbedder(d_in, hidden)
        self.blocks = nn.ModuleList(FlashSACBlock(hidden) for _ in range(int(cfg.get("num_blocks", 2))))
        self.post_norm = UnitRMSNorm(hidden)
        self.head = UnitLinear(hidden, self.num_bins)
        self.head_bias = nn.Parameter(torch.zeros(self.num_bins))
        self.register_buffer("bin_values", torch.linspace(self.min_v, self.max_v, self.num_bins))
        self.normalize_parameters()

    def logits(self, obs: torch.Tensor, training: bool) -> torch.Tensor:
        x = obs if self.traj is None else self.traj(obs)
        x = self.embedder(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        return self.head(self.post_norm(x)) + self.head_bias

    def expectation(self, logits: torch.Tensor) -> torch.Tensor:
        return (torch.softmax(logits, dim=-1) * self.bin_values).sum(-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.expectation(self.logits(obs, training=False))

    def two_hot(self, y: torch.Tensor) -> torch.Tensor:
        """(B,) targets -> (B, num_bins) weights on the two nearest bins; their expectation equals the clamped target."""
        pos = (y.clamp(self.min_v, self.max_v) - self.min_v) / (self.max_v - self.min_v) * (self.num_bins - 1)
        lo = pos.floor().long().clamp(0, self.num_bins - 1)
        hi = (lo + 1).clamp(max=self.num_bins - 1)
        w_hi = pos - lo.to(pos.dtype)
        t = torch.zeros(y.shape[0], self.num_bins, device=y.device, dtype=pos.dtype)
        t.scatter_add_(1, lo[:, None], (1.0 - w_hi)[:, None])
        t.scatter_add_(1, hi[:, None], w_hi[:, None])
        return t

    def ce_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -(self.two_hot(y) * torch.log_softmax(logits, dim=-1)).sum(-1).mean()

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        for m in self.modules():
            if m is not self and hasattr(m, "normalize_parameters"):
                m.normalize_parameters()

    def update_normalization(self, obs: torch.Tensor) -> None:
        """No running observation normalizer (the embedder's BatchNorm normalizes the inputs)."""


class ValueRMS(nn.Module):
    """rl_games' RunningMeanStd for one value head (DexMachina's ``normalize_value``): float64 running mean/var
    with count initialized to 1, unbiased batch variance, epsilon 1e-5, outputs clamped to [-5, 5]; ``denorm``
    clamps the input to [-5, 5] first. ``update`` = rl_games' forward in train mode (stats only)."""

    def __init__(self, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros((), dtype=torch.float64))
        self.register_buffer("running_var", torch.ones((), dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1)
        b_mean, b_var, b_count = x.mean().double(), x.var().double(), float(x.numel())
        delta = b_mean - self.running_mean
        tot = self.count + b_count
        new_mean = self.running_mean + delta * b_count / tot
        m2 = self.running_var * self.count + b_var * b_count + delta ** 2 * self.count * b_count / tot
        self.running_mean.copy_(new_mean)
        self.running_var.copy_(m2 / tot)
        self.count.copy_(tot)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        y = (x - self.running_mean.float()) / torch.sqrt(self.running_var.float() + self.epsilon)
        return y.clamp(-5.0, 5.0)

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.running_var.float() + self.epsilon) * y.clamp(-5.0, 5.0) + self.running_mean.float()
