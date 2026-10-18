"""Networks of the per-hand PPO agent (agents.hand_ppo): one actor and one critic per hand.

Both are rsl_rl's MLPModel layout with the 1D-conv FutureTrajEncoder in front: a running observation normalizer
(rsl_rl EmpiricalNormalization) over every input dim, then the encoder over the future-trajectory window at the obs
tail (each network has its own encoder; the window is replaced by its latent), then rsl_rl's MLP (PyTorch default
init, as rsl_rl's MLPModel). The actor's MLP outputs the means of its hand's action dims; its std is rsl_rl's
GaussianDistribution with std_type "scalar" (a learned state-independent std per dim, clamped to [1e-6, 1e6]). The
critic's MLP outputs one value (in normalized units when agents.hand_ppo normalizes values).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from rsl_rl.modules.mlp import MLP
from rsl_rl.modules.normalization import EmpiricalNormalization
from torch.distributions import Normal

from .traj_encoder import FutureTrajEncoder

_STD_RANGE = (1e-6, 1e6)   # rsl_rl GaussianDistribution's default clamp for std_type "scalar"


class _EncodedMLP(nn.Module):
    """obs normalizer -> FutureTrajEncoder (window -> latent) -> rsl_rl MLP."""

    def __init__(self, obs_dim: int, out_dim: int, traj_cfg: dict[str, Any] | None, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.obs_norm = EmpiricalNormalization(obs_dim) if bool(cfg.get("obs_normalization", True)) else nn.Identity()
        self.traj = FutureTrajEncoder(traj_cfg) if traj_cfg else None
        d_in = obs_dim if self.traj is None else obs_dim - self.traj.window_width + self.traj.latent_dim
        self.mlp = MLP(d_in, out_dim, [int(h) for h in cfg.get("hidden_dims", [256, 256, 256])],
                       str(cfg.get("activation", "elu")))

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
    """One hand's value function: the encoded MLP with one output."""

    def __init__(self, obs_dim: int, traj_cfg: dict[str, Any] | None, cfg: dict[str, Any]) -> None:
        super().__init__(obs_dim, 1, traj_cfg, cfg)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.features(obs)).squeeze(-1)


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
