"""Per-hand PPO as an agents.Agent (config/runner/hand_ppo.yaml), driven by runner.on_policy_runner.OnPolicyRunner.

One actor and one critic per hand (agents.modules.hand_ppo_nets), each hand trained on its own reward head, as in
the FlashSAC recipe's side actors and reward heads: the env splits every two-hand reward term into per-hand twins
(rewards.split_sides, via per_hand_actor_critic=true) and the vec-env wrapper returns (B, H) head rewards. The hands
share the episode ends (dones) and the observation.

Per hand h, each iteration:
- rewards r_h are scaled by ``reward_scale`` (DexMachina's rl_games reward_shaper.scale_value); a time-out adds
  gamma * V_h(s_t) (rl_games value_bootstrap / rsl_rl time-out bootstrapping), using the stored values;
- GAE(gamma, lam) over the window with the shared dones gives returns R_h and advantages A_h = R_h - V_h;
  A_h is normalized over the whole batch;
- with ``normalize_value`` (DexMachina's rl_games normalize_value) the critic outputs normalized values: its
  rl_games RunningMeanStd is updated with this batch's values and then its returns, the targets are the normalized
  returns, and values are de-normalized for GAE and bootstrapping;
- the loss of hand h is its clipped surrogate (its own log-prob ratio over its own action dims, its own A_h)
  + value_loss_coef * MSE(normalized value) + bounds_loss_coef * rl_games' bound loss on its action means
  - entropy_coef * entropy; hand h's actor and critic have their own Adam optimizer and their own adaptive learning
  rate (rsl_rl's rule on hand h's KL); gradients are clipped per network (rsl_rl).
Observation normalizers are updated after the update from the fresh rows only (rsl_rl updates them after the
update; the RAgger carried rows were counted when they were fresh).

Optional RAgger window (``carry_steps`` > 0), as agents.ppo.PPOAgent: the last ``carry_steps`` rows of the previous
window move to the front and are re-valued by the current critics; the minibatches grow with the window.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from .base import Agent
from .modules.hand_ppo_nets import HandActor, HandCritic, ValueRMS


class HandPPOAgent(Agent):
    def __init__(
        self,
        cfg: DictConfig,
        obs_dim: int,
        side_names: list[str],
        side_action_ids: list[list[int]],
        num_envs: int,
        traj_cfg: dict[str, Any] | None,
        device: str,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.num_envs = int(num_envs)
        self.sides = list(side_names)
        self.H = len(self.sides)
        self.ids = [torch.as_tensor(i, dtype=torch.long, device=device) for i in side_action_ids]
        self.action_dim = sum(len(i) for i in side_action_ids)
        order = torch.sort(torch.cat(self.ids)).values
        if not torch.equal(order, torch.arange(self.action_dim, device=device)):
            raise ValueError(f"side action ids must partition range({self.action_dim}): {side_action_ids}")
        self.fresh_steps = int(cfg.num_steps_per_env)
        self.carry_steps = int(cfg.get("carry_steps", 0))
        if not 0 <= self.carry_steps <= self.fresh_steps:
            raise ValueError(f"carry_steps must be in [0, num_steps_per_env], got {self.carry_steps}")
        self.window = self.fresh_steps + self.carry_steps
        self.clip_actions = float(cfg.get("clip_actions", 1.0))
        self.empty_cache = bool(cfg.get("empty_cache_after_update", False))

        a = OmegaConf.to_container(cfg.algorithm, resolve=True)
        self.epochs = int(a["num_learning_epochs"])
        self.num_mb = int(a["num_mini_batches"])
        self.clip_param = float(a["clip_param"])
        self.gamma = float(a["gamma"])
        self.lam = float(a["lam"])
        self.vf_coef = float(a["value_loss_coef"])
        self.ent_coef = float(a["entropy_coef"])
        self.bounds_coef = float(a["bounds_loss_coef"])
        self.bounds_soft = float(a["bounds_soft_bound"])
        self.reward_scale = float(a["reward_scale"])
        self.normalize_value = bool(a["normalize_value"])
        self.max_grad_norm = float(a["max_grad_norm"])
        self.schedule = str(a["schedule"])
        self.desired_kl = float(a["desired_kl"])
        if self.schedule not in ("adaptive", "fixed"):
            raise ValueError(f"algorithm.schedule must be 'adaptive' or 'fixed', got {self.schedule!r}")

        actor_cfg = OmegaConf.to_container(cfg.actor, resolve=True)
        critic_cfg = OmegaConf.to_container(cfg.critic, resolve=True)
        self.actors = nn.ModuleList(HandActor(obs_dim, len(i), traj_cfg, actor_cfg) for i in side_action_ids).to(device)
        self.critics = nn.ModuleList(HandCritic(obs_dim, traj_cfg, critic_cfg) for _ in self.sides).to(device)
        self.value_rms = nn.ModuleList(ValueRMS() for _ in self.sides).to(device)
        self.lrs = [float(a["learning_rate"])] * self.H
        self.optimizers = [
            torch.optim.Adam(list(self.actors[h].parameters()) + list(self.critics[h].parameters()), lr=self.lrs[h])
            for h in range(self.H)
        ]
        self.actors.train()
        self.critics.train()

        T, N, A, H, D = self.window, self.num_envs, self.action_dim, self.H, obs_dim
        z = dict(device=device)
        self.obs = torch.zeros(T, N, D, **z)
        self.actions = torch.zeros(T, N, A, **z)
        self.mu = torch.zeros(T, N, A, **z)
        self.sigma = torch.zeros(T, N, A, **z)
        self.rewards = torch.zeros(T, N, H, **z)
        self.values = torch.zeros(T, N, H, **z)
        self.logp = torch.zeros(T, N, H, **z)
        self.dones = torch.zeros(T, N, dtype=torch.bool, **z)
        self.timeouts = torch.zeros(T, N, dtype=torch.bool, **z)
        self.step = 0
        self._pending: dict[str, torch.Tensor] | None = None
        self._last_obs: torch.Tensor | None = None
        self._have_prev = False
        self._carried_now = 0

        n_a = sum(p.numel() for p in self.actors[0].parameters())
        n_c = sum(p.numel() for p in self.critics[0].parameters())
        print(f"[hand_ppo] sides {self.sides}, action dims {[len(i) for i in side_action_ids]}; per hand: actor "
              f"{n_a:,} params, critic {n_c:,} params\n[hand_ppo] actor {self.actors[0]}\n[hand_ppo] critic "
              f"{self.critics[0]}", flush=True)
        print(f"[hand_ppo] window {T} rows = carry {self.carry_steps} + fresh {self.fresh_steps}; algorithm {a}; "
              f"empty_cache_after_update={self.empty_cache}", flush=True)

    # ─── construction ────────────────────────────────────────────────────────────
    @classmethod
    def build(cls, cfg: DictConfig, env: Any, env_info: dict[str, Any], command_name: str) -> "HandPPOAgent":
        names = list(env_info.get("reward_heads") or [])
        if len(names) < 2:
            raise ValueError("hand_ppo needs per-hand reward heads: set per_hand_actor_critic=true "
                             "(runner.agent.reward_heads.enable)")
        action_sides = dict(env_info.get("action_sides") or {})
        missing = [n for n in names if not action_sides.get(n)]
        if missing:
            raise ValueError(f"hand_ppo: the env has no action dims for {missing}")
        traj_cfg = None
        te = OmegaConf.to_container(cfg.traj_encoder, resolve=True) if cfg.get("traj_encoder") else {}
        if te.get("enable", False):
            encoded = list(env_info.get("encoded_obs_dims") or [])
            if not encoded:
                raise ValueError("traj_encoder.enable=true but the env has no encoded obs groups")
            traj_cfg = {**te, "window_dims": [int(d) for _, d in encoded]}
        obs_dim = int(env.single_observation_space.shape[0])
        return cls(cfg, obs_dim, names, [[int(i) for i in action_sides[n]] for n in names], env.num_envs, traj_cfg,
                   str(cfg.get("device", "cuda")))

    def _to_tensor(self, x: Any) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.device, dtype=torch.float32)
        return torch.as_tensor(x, device=self.device, dtype=torch.float32)

    def _values(self, obs: torch.Tensor) -> torch.Tensor:
        """(B, H) values in reward-scaled units (de-normalized when normalize_value)."""
        vs = []
        for h in range(self.H):
            v = self.critics[h](obs)
            vs.append(self.value_rms[h].denormalize(v) if self.normalize_value else v)
        return torch.stack(vs, dim=-1)

    # ─── rollout ─────────────────────────────────────────────────────────────────
    def begin_rollout(self) -> int:
        """Set up the window for the next rollout; returns the number of env steps to collect."""
        self.step = 0
        self._carried_now = 0
        if self.carry_steps > 0 and self._have_prev:
            c, T = self.carry_steps, self.window
            with torch.no_grad():
                for buf in (self.obs, self.actions, self.mu, self.sigma, self.rewards, self.logp, self.dones,
                            self.timeouts):
                    buf[:c] = buf[T - c:].clone()
                # re-value the carried rows with the current critics (normalizers as they are now, no update)
                flat = self._values(self.obs[:c].flatten(0, 1))
                self.values[:c] = flat.view(c, self.num_envs, self.H)
            self.step = c
            self._carried_now = c
        return self.window - self.step

    def act(self, obs: Any, deterministic: bool = False) -> np.ndarray:
        obs_t = self._to_tensor(obs)
        with torch.no_grad():
            action = torch.zeros(obs_t.shape[0], self.action_dim, device=self.device)
            if deterministic:
                for h in range(self.H):
                    action[:, self.ids[h]] = self.actors[h].mean(obs_t)
                return action.clamp(-self.clip_actions, self.clip_actions).cpu().numpy()
            mu = torch.zeros_like(action)
            sigma = torch.zeros_like(action)
            logp = torch.zeros(obs_t.shape[0], self.H, device=self.device)
            for h in range(self.H):
                dist = self.actors[h].distribution(obs_t)
                a_h = dist.sample()
                action[:, self.ids[h]] = a_h
                mu[:, self.ids[h]] = dist.mean
                sigma[:, self.ids[h]] = dist.stddev
                logp[:, h] = dist.log_prob(a_h).sum(-1)
            self._pending = {"obs": obs_t, "actions": action, "mu": mu, "sigma": sigma, "logp": logp,
                             "values": self._values(obs_t)}
        return action.clamp(-self.clip_actions, self.clip_actions).cpu().numpy()

    def observe(self, transition: Mapping[str, Any]) -> None:
        """One env step of the rollout (after act). ``reward`` is (B, H) head rewards; ``next_observation`` is the
        env's current (post-reset) observation, the bootstrap state at the end of the rollout."""
        if self._pending is None:
            raise RuntimeError("observe() without a preceding act()")
        if self.step >= self.window:
            raise RuntimeError(f"rollout overflow: window is {self.window} rows")
        t, p = self.step, self._pending
        rewards = self._to_tensor(transition["reward"]).view(self.num_envs, self.H)
        terminated = torch.as_tensor(np.asarray(transition["terminated"]), device=self.device).view(-1).bool()
        truncated = torch.as_tensor(np.asarray(transition["truncated"]), device=self.device).view(-1).bool()
        self.obs[t] = p["obs"]
        self.actions[t] = p["actions"]
        self.mu[t] = p["mu"]
        self.sigma[t] = p["sigma"]
        self.logp[t] = p["logp"]
        self.values[t] = p["values"]
        self.rewards[t] = rewards * self.reward_scale
        self.dones[t] = terminated | truncated
        self.timeouts[t] = truncated
        self.step += 1
        self._pending = None
        self._last_obs = self._to_tensor(transition["next_observation"])

    def ready(self) -> bool:
        return True

    # ─── update ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _returns(self) -> tuple[torch.Tensor, torch.Tensor]:
        """GAE per head over the window; returns (returns, advantages), both (T, N, H), reward-scaled units."""
        last_values = self._values(self._last_obs)
        rewards = self.rewards + self.gamma * self.values * self.timeouts.unsqueeze(-1)
        not_done = (~self.dones).float().unsqueeze(-1)
        returns = torch.zeros_like(self.values)
        adv = torch.zeros_like(last_values)
        for t in reversed(range(self.window)):
            next_values = last_values if t == self.window - 1 else self.values[t + 1]
            delta = rewards[t] + not_done[t] * self.gamma * next_values - self.values[t]
            adv = delta + not_done[t] * self.gamma * self.lam * adv
            returns[t] = adv + self.values[t]
        return returns, returns - self.values

    def update(self) -> dict[str, float]:
        if self.step != self.window:
            raise RuntimeError(f"update with an incomplete window ({self.step}/{self.window} rows)")
        t0 = time.perf_counter()
        returns, adv = self._returns()
        B = self.window * self.num_envs
        obs = self.obs.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        mu_old = self.mu.flatten(0, 1)
        sigma_old = self.sigma.flatten(0, 1)
        logp_old = self.logp.flatten(0, 1)
        returns = returns.flatten(0, 1)
        adv = adv.flatten(0, 1)
        adv = (adv - adv.mean(0, keepdim=True)) / (adv.std(0, keepdim=True) + 1e-8)
        targets = returns
        if self.normalize_value:
            values = self.values.flatten(0, 1)
            cols = []
            for h in range(self.H):
                self.value_rms[h].update(values[:, h])     # rl_games prepare_dataset: values first, then returns
                self.value_rms[h].update(returns[:, h])
                cols.append(self.value_rms[h].normalize(returns[:, h]))
            targets = torch.stack(cols, dim=-1)

        stats = {k: [0.0] * self.H for k in ("surrogate", "value", "bounds", "entropy", "kl", "clip_frac")}
        mb = B // self.num_mb
        n_updates = 0
        for _ in range(self.epochs):
            perm = torch.randperm(self.num_mb * mb, device=self.device)
            for i in range(self.num_mb):
                rows = perm[i * mb:(i + 1) * mb]
                o = obs[rows]
                for h in range(self.H):
                    ids = self.ids[h]
                    dist = self.actors[h].distribution(o)
                    mu_new = dist.mean
                    if self.schedule == "adaptive":
                        with torch.no_grad():
                            s_old, s_new = sigma_old[rows][:, ids], dist.stddev.expand_as(mu_new)
                            kl = (torch.log(s_new / s_old)
                                  + (s_old.square() + (mu_old[rows][:, ids] - mu_new).square()) / (2 * s_new.square())
                                  - 0.5).sum(-1).mean()
                            if kl > self.desired_kl * 2.0:
                                self.lrs[h] = max(1e-5, self.lrs[h] / 1.5)
                            elif self.desired_kl / 2.0 > kl > 0.0:
                                self.lrs[h] = min(1e-2, self.lrs[h] * 1.5)
                            for g in self.optimizers[h].param_groups:
                                g["lr"] = self.lrs[h]
                            stats["kl"][h] += float(kl)
                    logp = dist.log_prob(actions[rows][:, ids]).sum(-1)
                    ratio = torch.exp(logp - logp_old[rows, h])
                    a_h = adv[rows, h]
                    surrogate = torch.max(-a_h * ratio,
                                          -a_h * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)).mean()
                    v = self.critics[h](o)
                    value_loss = (targets[rows, h] - v).square().mean()
                    s = self.bounds_soft
                    bounds = ((mu_new - s).clamp_min(0.0).square() + (mu_new + s).clamp_max(0.0).square()).sum(-1).mean()
                    entropy = dist.entropy().sum(-1).mean()
                    loss = surrogate + self.vf_coef * value_loss + self.bounds_coef * bounds - self.ent_coef * entropy
                    self.optimizers[h].zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actors[h].parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critics[h].parameters(), self.max_grad_norm)
                    self.optimizers[h].step()
                    self.actors[h].normalize_parameters()
                    stats["surrogate"][h] += float(surrogate)
                    stats["value"][h] += float(value_loss)
                    stats["bounds"][h] += float(bounds)
                    stats["entropy"][h] += float(entropy)
                    stats["clip_frac"][h] += float(((ratio - 1.0).abs() > self.clip_param).float().mean())
                n_updates += 1

        # observation normalizers: the fresh rows only (the carried rows were counted when they were fresh)
        fresh = self.obs[self._carried_now:].flatten(0, 1)
        with torch.no_grad():
            for h in range(self.H):
                self.actors[h].update_normalization(fresh)
                self.critics[h].update_normalization(fresh)
        self._have_prev = True
        t_update = time.perf_counter() - t0
        t_cache = 0.0
        if self.empty_cache and torch.cuda.is_available():
            t1 = time.perf_counter()
            torch.cuda.empty_cache()
            t_cache = time.perf_counter() - t1

        out: dict[str, float] = {}
        for h, name in enumerate(self.sides):
            for k, v in stats.items():
                if k == "kl" and self.schedule != "adaptive":
                    continue
                out[f"loss/{k}_{name}" if k in ("surrogate", "value", "bounds", "entropy") else f"train/{k}_{name}"] = \
                    v[h] / n_updates
            out[f"train/learning_rate_{name}"] = self.lrs[h]
            out[f"train/action_std_{name}"] = float(self.actors[h].std().mean())
            out[f"train/return_mean_{name}"] = float(returns[:, h].mean())
            if self.normalize_value:
                out[f"train/value_rms_mean_{name}"] = float(self.value_rms[h].running_mean)
                out[f"train/value_rms_std_{name}"] = float(self.value_rms[h].running_var.sqrt())
        out.update({
            "train/learning_rate": float(np.mean(self.lrs)),
            "train/action_std": float(np.mean([float(self.actors[h].std().mean()) for h in range(self.H)])),
            "train/carried_rows": float(self._carried_now),
            "train/window_rows": float(self.window),
            "train/update_time_s": t_update,
            "train/empty_cache_s": t_cache,
        })
        return out

    def on_env_reset(self) -> None:
        """The runner reset the train envs (eval drove them): the previous window is no longer contiguous."""
        self._have_prev = False

    # ─── checkpoints ─────────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save({
            "sides": self.sides,
            "actors": self.actors.state_dict(),
            "critics": self.critics.state_dict(),
            "value_rms": self.value_rms.state_dict(),
            "optimizers": [o.state_dict() for o in self.optimizers],
            "lrs": list(self.lrs),
        }, os.path.join(path, "hand_ppo.pt"))

    def load(self, path: str) -> None:
        state = torch.load(os.path.join(path, "hand_ppo.pt"), map_location=self.device, weights_only=False)
        if list(state["sides"]) != self.sides:
            raise ValueError(f"checkpoint sides {state['sides']} != env sides {self.sides}")
        self.actors.load_state_dict(state["actors"])
        self.critics.load_state_dict(state["critics"])
        self.value_rms.load_state_dict(state["value_rms"])
        for o, s in zip(self.optimizers, state["optimizers"]):
            o.load_state_dict(s)
        self.lrs = [float(x) for x in state["lrs"]]
        for o, lr in zip(self.optimizers, self.lrs):
            for g in o.param_groups:
                g["lr"] = lr
