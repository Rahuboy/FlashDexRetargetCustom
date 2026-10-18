"""Networks of the per-hand PPO agent (agents.hand_ppo): one actor and one critic per hand.

Both read the full flat env observation: a running observation normalizer (rsl_rl EmpiricalNormalization) over
every input dim, then the 1D-conv FutureTrajEncoder over the future-trajectory window at the obs tail (each network
has its own encoder), then the trunk.

Actor trunk = FlashSAC's actor trunk without BatchNorm: a bias-free embedding linear, ``num_blocks`` residual blocks
(linear -> ReLU -> linear -> ReLU, plus skip, width ``hidden_dim`` expanded ``expansion``x inside the block) and an
RMSNorm. ``trunk: unit`` keeps FlashSAC's weight constraint (every linear row re-normalized to unit norm and the
RMSNorm scale to norm sqrt(d) after each optimizer step, via ``normalize_parameters``); ``trunk: plain`` uses
ordinary biased linears and a plain RMSNorm. Head: PPO's Gaussian (rsl_rl GaussianDistribution, ``std_type
scalar``): an unsquashed mean (unit-norm linear + bias, or a plain linear) and a learned state-independent std per
action dim, clamped to [1e-6, 1e6].

Critic = rsl_rl's MLP critic (hidden dims, activation) on [flat obs | trajectory latent], one scalar output (the
value in normalized units; agents.hand_ppo de-normalizes it).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules.normalization import EmpiricalNormalization
from torch.distributions import Normal

from .traj_encoder import FutureTrajEncoder

_STD_RANGE = (1e-6, 1e6)   # rsl_rl GaussianDistribution's default clamp for std_type "scalar"


class _UnitLinear(nn.Module):
    """FlashSAC's UnitLinear: bias-free, orthogonal init, rows re-normalized to unit norm after every step."""

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.w = nn.Linear(d_in, d_out, bias=False)
        nn.init.orthogonal_(self.w.weight, gain=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        self.w.weight.copy_(F.normalize(self.w.weight, dim=-1, eps=1e-8))


class _RMSNorm(nn.Module):
    """FlashSAC's UnitRMSNorm (``unit``: scale re-normalized to norm sqrt(d) after every step) or a plain RMSNorm."""

    def __init__(self, d: int, unit: bool, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
        self.unit = unit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps) * self.weight

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        if self.unit:
            d = self.weight.shape[-1]
            self.weight.mul_(math.sqrt(d) * torch.rsqrt(torch.sum(self.weight * self.weight) + 1e-8))


def _linear(d_in: int, d_out: int, unit: bool) -> nn.Module:
    return _UnitLinear(d_in, d_out) if unit else nn.Linear(d_in, d_out)


class _Block(nn.Module):
    """FlashSACBlock without its two BatchNorms: x + relu(W2 relu(W1 x))."""

    def __init__(self, hidden_dim: int, expansion: int, unit: bool) -> None:
        super().__init__()
        self.w1 = _linear(hidden_dim, hidden_dim * expansion, unit)
        self.w2 = _linear(hidden_dim * expansion, hidden_dim, unit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.relu(self.w2(F.relu(self.w1(x))))


def _make_obs_norm(obs_dim: int, enabled: bool) -> nn.Module:
    return EmpiricalNormalization(obs_dim) if enabled else nn.Identity()


class HandActor(nn.Module):
    """One hand's policy: obs normalizer -> traj encoder -> FlashSAC trunk (no BatchNorm) -> PPO Gaussian head."""

    def __init__(self, obs_dim: int, action_dim: int, traj_cfg: dict[str, Any] | None, cfg: dict[str, Any]) -> None:
        super().__init__()
        unit = str(cfg.get("trunk", "unit")) == "unit"
        if str(cfg.get("trunk", "unit")) not in ("unit", "plain"):
            raise ValueError(f"actor.trunk must be 'unit' or 'plain', got {cfg.get('trunk')!r}")
        hidden = int(cfg.get("hidden_dim", 128))
        self.obs_norm = _make_obs_norm(obs_dim, bool(cfg.get("obs_normalization", True)))
        self.traj = FutureTrajEncoder(traj_cfg) if traj_cfg else None
        d_in = obs_dim if self.traj is None else obs_dim - self.traj.window_width + self.traj.latent_dim
        self.embed = _linear(d_in, hidden, unit)
        self.blocks = nn.ModuleList(
            _Block(hidden, int(cfg.get("expansion", 4)), unit) for _ in range(int(cfg.get("num_blocks", 2)))
        )
        self.post_norm = _RMSNorm(hidden, unit)
        self.mean_w = _linear(hidden, action_dim, unit)
        self.mean_bias = nn.Parameter(torch.zeros(action_dim)) if unit else None
        self.std_param = nn.Parameter(float(cfg.get("init_std", 1.0)) * torch.ones(action_dim))
        self.normalize_parameters()

    def mean(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.obs_norm(obs)
        if self.traj is not None:
            x = self.traj(x)
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.mean_w(self.post_norm(x))
        return x if self.mean_bias is None else x + self.mean_bias

    def std(self) -> torch.Tensor:
        return self.std_param.clamp(*_STD_RANGE)

    def distribution(self, obs: torch.Tensor) -> Normal:
        return Normal(self.mean(obs), self.std())

    def normalize_parameters(self) -> None:
        for m in self.modules():
            if m is not self and hasattr(m, "normalize_parameters"):
                m.normalize_parameters()

    def update_normalization(self, obs: torch.Tensor) -> None:
        if isinstance(self.obs_norm, EmpiricalNormalization):
            self.obs_norm.update(obs)


class HandCritic(nn.Module):
    """One hand's value function: obs normalizer -> traj encoder -> rsl_rl-style MLP -> 1 (normalized value)."""

    _ACTS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU}

    def __init__(self, obs_dim: int, traj_cfg: dict[str, Any] | None, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.obs_norm = _make_obs_norm(obs_dim, bool(cfg.get("obs_normalization", True)))
        self.traj = FutureTrajEncoder(traj_cfg) if traj_cfg else None
        d = obs_dim if self.traj is None else obs_dim - self.traj.window_width + self.traj.latent_dim
        act = self._ACTS[str(cfg.get("activation", "elu"))]
        layers: list[nn.Module] = []
        for h in [int(x) for x in cfg.get("hidden_dims", [256, 256, 256])]:
            layers += [nn.Linear(d, h), act()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.obs_norm(obs)
        if self.traj is not None:
            x = self.traj(x)
        return self.mlp(x).squeeze(-1)

    def update_normalization(self, obs: torch.Tensor) -> None:
        if isinstance(self.obs_norm, EmpiricalNormalization):
            self.obs_norm.update(obs)


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
