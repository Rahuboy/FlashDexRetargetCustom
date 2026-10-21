"""Per-hand PPO as an agents.Agent (config/runner/hand_ppo.yaml), driven by runner.on_policy_runner.OnPolicyRunner.

One actor and one critic per hand (agents.modules.hand_ppo_nets: rsl_rl's MLP actor / critic with the 1D-conv
future-trajectory encoder in front), each hand trained on its own reward head, as in the FlashSAC recipe's side
actors and reward heads: the env splits every two-hand reward term into per-hand twins
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

``algorithm.impl`` selects the update: rsl_rl (the above; default) or rl_games = DexMachina's rl_games PPO epoch per hand
(see ``_update_rl_games``): fixed env-block minibatches in the same order every pass, the KL reference refreshed with
each minibatch's pre-step means / stds (rl_games' dataset.update_mu_sigma), rl_games' KL(new || ref), the adaptive LR
stepped once per pass on the pass's mean KL (rl_games' AdaptiveScheduler, LR in [1e-6, 1e-2]), one gradient-norm clip
over the hand's actor + critic, the time-out bootstrap added to the stored reward at collection (rl_games'
value_bootstrap), and the value loss clipped only with ``clip_value``.

``actor.arch`` = ``critic.arch`` = lstm (FDR-35): separate LSTM actor and LSTM critic per hand (hand_ppo_nets, rsl_rl
5.5.1's RNNModel layout), trained as rsl_rl 5.5.1's recurrent PPO: the LSTM states are carried from step to step, zeroed
for the envs that are done after a step, and recorded at the rollout start; the time-out bootstrap uses the stored
values (as above); the last value comes from the critic state without advancing it; the update (``_update_recurrent``)
splits the window into trajectories at the dones (rsl_rl split_and_pad_trajectories), walks env-block minibatches in a
fixed order (rsl_rl recurrent_mini_batch_generator), starts each trajectory from the recorded state (t = 0) or zeros
(after a done), with the rsl_rl-style per-minibatch adaptive LR and per-network gradient clipping. Deterministic
actions (eval) use their own LSTM state (``reset_eval_memory`` / ``on_eval_dones``). No RAgger carry.
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
from .modules.hand_ppo_nets import FlashSACValueCritic, HandActor, HandCritic, ValueRMS


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
        self.impl = str(a.get("impl", "rsl_rl"))
        if self.impl not in ("rsl_rl", "rl_games"):
            raise ValueError(f"algorithm.impl must be 'rsl_rl' or 'rl_games', got {self.impl!r}")
        self.clip_value = bool(a.get("clip_value", False))
        if self.clip_value and self.impl != "rl_games":
            raise ValueError("algorithm.clip_value is implemented for algorithm.impl=rl_games only")
        if self.impl == "rl_games" and self.num_envs % self.num_mb:
            raise ValueError(f"impl=rl_games: num_envs ({self.num_envs}) must be divisible by num_mini_batches "
                             f"({self.num_mb}); each minibatch is a block of envs with all its window rows")

        actor_cfg = OmegaConf.to_container(cfg.actor, resolve=True)
        critic_cfg = OmegaConf.to_container(cfg.critic, resolve=True)
        self.actors = nn.ModuleList(HandActor(obs_dim, len(i), traj_cfg, actor_cfg) for i in side_action_ids).to(device)
        self.critic_arch = str(critic_cfg.get("arch", "mlp"))
        if self.critic_arch not in ("mlp", "flashsac", "simba", "lstm"):
            raise ValueError(f"critic.arch must be 'mlp', 'flashsac', 'simba' or 'lstm', got {self.critic_arch!r}")
        if self.critic_arch == "flashsac" and not self.normalize_value:
            raise ValueError("critic.arch=flashsac predicts normalized values on [min_v, max_v]: set normalize_value")
        if self.critic_arch == "flashsac" and self.impl == "rl_games":
            raise ValueError("algorithm.impl=rl_games supports critic.arch mlp / simba (joint optimizer) only")
        critic_cls = FlashSACValueCritic if self.critic_arch == "flashsac" else HandCritic
        self.critics = nn.ModuleList(critic_cls(obs_dim, traj_cfg, critic_cfg) for _ in self.sides).to(device)
        self.value_rms = nn.ModuleList(ValueRMS() for _ in self.sides).to(device)
        self.recurrent = bool(getattr(self.actors[0], "is_recurrent", False))
        if self.recurrent or bool(getattr(self.critics[0], "is_recurrent", False)):
            if not (self.recurrent and bool(getattr(self.critics[0], "is_recurrent", False))):
                raise ValueError("arch=lstm needs actor.arch=lstm and critic.arch=lstm (separate recurrent actor / critic)")
            if self.impl != "rsl_rl":
                raise ValueError("arch=lstm is implemented for algorithm.impl=rsl_rl only")
            if self.carry_steps:
                raise ValueError("arch=lstm does not support carry_steps > 0 (carried rows would need their LSTM states)")
            if self.num_envs % self.num_mb:
                raise ValueError(f"arch=lstm: num_envs ({self.num_envs}) must be divisible by num_mini_batches ({self.num_mb})")
        # arch=lstm: per-hand (h, c) states of the training rollout (None = zeros), their rollout-start snapshots,
        # and the deterministic (eval) actor states
        self._hid_a: list = [None] * self.H
        self._hid_c: list = [None] * self.H
        self._h0_a: list = [None] * self.H
        self._h0_c: list = [None] * self.H
        self._eval_hid_a: list = [None] * self.H
        self.lrs = [float(a["learning_rate"])] * self.H
        self.critic_optimizers: list[torch.optim.Optimizer] = []
        self.critic_schedulers: list[torch.optim.lr_scheduler.LambdaLR] = []
        if self.critic_arch != "flashsac":   # mlp, simba or lstm: HandCritic, joint optimizer, MSE
            # one optimizer per hand over its actor + critic; the adaptive LR (hand h's KL) drives both
            self.optimizers = [
                torch.optim.Adam(list(self.actors[h].parameters()) + list(self.critics[h].parameters()), lr=self.lrs[h])
                for h in range(self.H)
            ]
        else:
            # actor: per-hand Adam with the adaptive LR; critic: its own Adam with FlashSAC's warmup-cosine LR
            # (rsl_rl_flashsac FlashSAC: Adam defaults, base LR = peak, LambdaLR(schedule / peak), no gradient clipping)
            from rsl_rl_flashsac.utils import warmup_cosine_decay_scheduler

            self.optimizers = [torch.optim.Adam(self.actors[h].parameters(), lr=self.lrs[h]) for h in range(self.H)]
            peak = float(critic_cfg["lr_peak"])
            sched = warmup_cosine_decay_scheduler(
                init_value=float(critic_cfg["lr_init"]), peak_value=peak, end_value=float(critic_cfg["lr_end"]),
                warmup_steps=int(critic_cfg["lr_warmup_updates"]), decay_steps=int(critic_cfg["lr_decay_updates"]))
            for h in range(self.H):
                opt = torch.optim.Adam(self.critics[h].parameters(), lr=peak)
                self.critic_optimizers.append(opt)
                self.critic_schedulers.append(
                    torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step, _s=sched, _p=peak: _s(step) / _p))
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
        if self.recurrent:
            print(f"[hand_ppo] arch=lstm: separate LSTM actor / critic per hand (LSTM {self.actors[0].rnn.hidden_size} x "
                  f"{self.actors[0].rnn.num_layers} layer(s) -> MLP); rsl_rl recurrent PPO: BPTT over the {T}-row window, "
                  f"trajectories split at dones, {self.num_mb} env-block minibatches of {self.num_envs // self.num_mb} envs "
                  f"in fixed order, states recorded at the rollout start / zeroed after dones; eval uses its own state",
                  flush=True)
        if self.impl == "rl_games":
            print(f"[hand_ppo] impl=rl_games: {self.epochs} passes x {self.num_mb} env-block minibatches of "
                  f"{self.num_envs // self.num_mb} envs x {T} rows; KL reference refreshed per minibatch; LR stepped per "
                  f"pass (desired_kl {self.desired_kl}, LR in [1e-6, 1e-2]); joint grad clip {self.max_grad_norm}; "
                  f"clip_value={self.clip_value}; time-out bootstrap at collection", flush=True)

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
        """(B, H) values in reward-scaled units (de-normalized when normalize_value). arch=lstm: from the critics'
        current training states without advancing them (rsl_rl compute_returns restores the critic state)."""
        vs = []
        for h in range(self.H):
            if self.recurrent:
                lat, _ = self.critics[h].latent_step(obs, self._state(self._hid_c[h], self.critics[h], obs.shape[0]))
                v = self.critics[h].value_from_latent(lat)
            else:
                v = self.critics[h](obs)
            vs.append(self.value_rms[h].denormalize(v) if self.normalize_value else v)
        return torch.stack(vs, dim=-1)

    def _state(self, hidden, net: nn.Module, n: int):
        """arch=lstm: a stored (h, c) state, or zeros for ``n`` envs when there is none yet."""
        return hidden if hidden is not None else net.zero_hidden(n, self.device)

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
        if self.recurrent:   # the states every trajectory that starts at t = 0 begins from (rsl_rl saved hidden states)
            self._h0_a = [tuple(x.clone() for x in self._state(self._hid_a[h], self.actors[h], self.num_envs))
                          for h in range(self.H)]
            self._h0_c = [tuple(x.clone() for x in self._state(self._hid_c[h], self.critics[h], self.num_envs))
                          for h in range(self.H)]
        return self.window - self.step

    def act(self, obs: Any, deterministic: bool = False) -> np.ndarray:
        obs_t = self._to_tensor(obs)
        if self.recurrent:
            return self._act_recurrent(obs_t, deterministic)
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

    @torch.no_grad()
    def _act_recurrent(self, obs_t: torch.Tensor, deterministic: bool) -> np.ndarray:
        """arch=lstm act: deterministic = eval (its own actor states); else sample, advance the training actor and
        critic states (rsl_rl PPO.act records the states before this step: begin_rollout's snapshot + zeros after dones)."""
        n = obs_t.shape[0]
        action = torch.zeros(n, self.action_dim, device=self.device)
        if deterministic:
            for h in range(self.H):
                lat, self._eval_hid_a[h] = self.actors[h].latent_step(
                    obs_t, self._state(self._eval_hid_a[h], self.actors[h], n))
                action[:, self.ids[h]] = self.actors[h].mlp(lat)
            return action.clamp(-self.clip_actions, self.clip_actions).cpu().numpy()
        mu = torch.zeros_like(action)
        sigma = torch.zeros_like(action)
        logp = torch.zeros(n, self.H, device=self.device)
        values = torch.zeros(n, self.H, device=self.device)
        for h in range(self.H):
            lat, self._hid_a[h] = self.actors[h].latent_step(obs_t, self._state(self._hid_a[h], self.actors[h], n))
            dist = self.actors[h].distribution_from_latent(lat)
            a_h = dist.sample()
            action[:, self.ids[h]] = a_h
            mu[:, self.ids[h]] = dist.mean
            sigma[:, self.ids[h]] = dist.stddev
            logp[:, h] = dist.log_prob(a_h).sum(-1)
            lat_c, self._hid_c[h] = self.critics[h].latent_step(obs_t, self._state(self._hid_c[h], self.critics[h], n))
            v = self.critics[h].value_from_latent(lat_c)
            values[:, h] = self.value_rms[h].denormalize(v) if self.normalize_value else v
        self._pending = {"obs": obs_t, "actions": action, "mu": mu, "sigma": sigma, "logp": logp, "values": values}
        return action.clamp(-self.clip_actions, self.clip_actions).cpu().numpy()

    @staticmethod
    def _zero_done(states: list, done: torch.Tensor) -> None:
        """Zero the (h, c) states of the done envs in place (rsl_rl RNN.reset(dones))."""
        for hid in states:
            if hid is not None:
                for x in hid:
                    x[:, done] = 0.0

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
        if self.impl == "rl_games":   # rl_games value_bootstrap: gamma * V(s_t) joins the stored (shaped) reward
            self.rewards[t] = rewards * self.reward_scale + self.gamma * p["values"] * truncated.unsqueeze(-1).float()
        else:
            self.rewards[t] = rewards * self.reward_scale
        self.dones[t] = terminated | truncated
        self.timeouts[t] = truncated
        if self.recurrent and bool(self.dones[t].any()):   # rsl_rl process_env_step: actor / critic reset(dones)
            self._zero_done(self._hid_a + self._hid_c, self.dones[t])
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
        if self.impl == "rl_games":   # the time-out bootstrap is already in the stored reward (observe)
            rewards = self.rewards
        else:
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
        old_values = self.values.flatten(0, 1)   # impl=rl_games clip_value: the rollout's values (normalized below)
        if self.normalize_value:
            values = self.values.flatten(0, 1)
            cols = []
            old_cols = []
            for h in range(self.H):
                self.value_rms[h].update(values[:, h])     # rl_games prepare_dataset: values first, then returns
                if self.impl == "rl_games":   # rl_games normalizes the old values with the stats after their update
                    old_cols.append(self.value_rms[h].normalize(values[:, h]))
                self.value_rms[h].update(returns[:, h])
                cols.append(self.value_rms[h].normalize(returns[:, h]))
            targets = torch.stack(cols, dim=-1)
            if old_cols:
                old_values = torch.stack(old_cols, dim=-1)

        keys = ("surrogate", "value", "bounds", "entropy", "kl", "clip_frac")
        if self.critic_arch == "flashsac":
            keys += ("value_mse",)
        stats = {k: [0.0] * self.H for k in keys}
        kl_rollout = [0.0] * self.H
        if self.impl == "rl_games":
            n_updates = self._update_rl_games(stats, kl_rollout, adv, targets, old_values)
        elif self.recurrent:
            n_updates = self._update_recurrent(stats, adv, targets)
        else:
            n_updates = self._update_rsl_rl(stats, B, obs, actions, mu_old, sigma_old, logp_old, adv, targets)

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
                if k == "kl" and self.schedule != "adaptive" and self.impl != "rl_games":
                    continue
                out[f"loss/{k}_{name}" if k in ("surrogate", "value", "bounds", "entropy") else f"train/{k}_{name}"] = \
                    v[h] / n_updates
            if self.impl == "rl_games":   # last pass, pre-step policy vs the rows' collecting policy
                out[f"train/kl_rollout_{name}"] = kl_rollout[h] / self.num_mb
            out[f"train/learning_rate_{name}"] = self.lrs[h]
            if self.critic_optimizers:
                out[f"train/critic_lr_{name}"] = float(self.critic_optimizers[h].param_groups[0]["lr"])
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

    def _update_rl_games(self, stats: dict[str, list[float]], kl_rollout: list[float], adv: torch.Tensor,
                         targets: torch.Tensor, old_values: torch.Tensor) -> int:
        """rl_games' PPO epoch per hand (DexMachina's rl_games: a2c_common.ContinuousA2CBase.train_epoch +
        a2c_continuous.calc_gradients, schedule_type standard, permute off). ``num_learning_epochs`` passes; each
        pass walks ``num_mini_batches`` minibatches in a fixed order, minibatch i = envs [i E, (i + 1) E) with all
        their window rows (rl_games' contiguous slices of the env-major flattened batch). Per minibatch and hand:
        loss = clipped surrogate + value_loss_coef * value loss - entropy_coef * entropy + bounds_loss_coef * bound
        loss (rl_games' order; value_loss_coef 2 = 0.5 * critic_coef 4), the value loss clipped around the rollout's
        normalized values only with ``clip_value``; one gradient-norm clip over the hand's actor and critic, one Adam
        step; KL = rl_games' policy_kl(new = the pre-step means / stds, ref = the rows' recorded means / stds), then the
        rows' reference is overwritten with the pre-step means / stds (dataset.update_mu_sigma). After each pass the
        hand's LR follows AdaptiveScheduler on the mean KL of the pass. Returns the number of minibatch steps."""
        T, N, H, D = self.window, self.num_envs, self.H, self.obs.shape[-1]
        E = N // self.num_mb
        adv = adv.view(T, N, H)
        targets = targets.view(T, N, H)
        old_values = old_values.view(T, N, H)
        mu_ref = self.mu.clone()
        sigma_ref = self.sigma.clone()
        n_updates = 0
        for ep in range(self.epochs):
            pass_kls: list[list[torch.Tensor]] = [[] for _ in range(H)]
            for i in range(self.num_mb):
                sl = slice(i * E, (i + 1) * E)
                o = self.obs[:, sl].reshape(T * E, D)
                act = self.actions[:, sl].reshape(T * E, self.action_dim)
                for h in range(H):
                    ids = self.ids[h]
                    k = len(ids)
                    dist = self.actors[h].distribution(o)
                    mu_new = dist.mean
                    sigma_new = dist.stddev.expand_as(mu_new)
                    logp = dist.log_prob(act[:, ids]).sum(-1)
                    ratio = torch.exp(logp - self.logp[:, sl, h].reshape(-1))
                    a_h = adv[:, sl, h].reshape(-1)
                    surrogate = torch.max(-a_h * ratio,
                                          -a_h * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)).mean()
                    v = self.critics[h](o)
                    ret = targets[:, sl, h].reshape(-1)
                    if self.clip_value:   # rl_games common_losses.default_critic_loss
                        v0 = old_values[:, sl, h].reshape(-1)
                        v_clip = v0 + (v - v0).clamp(-self.clip_param, self.clip_param)
                        value_loss = torch.max((v - ret).square(), (v_clip - ret).square()).mean()
                    else:
                        value_loss = (ret - v).square().mean()
                    s = self.bounds_soft
                    bounds = ((mu_new - s).clamp_min(0.0).square() + (mu_new + s).clamp_max(0.0).square()).sum(-1).mean()
                    entropy = dist.entropy().sum(-1).mean()
                    loss = surrogate + self.vf_coef * value_loss - self.ent_coef * entropy + self.bounds_coef * bounds
                    self.optimizers[h].zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(self.actors[h].parameters()) + list(self.critics[h].parameters()),
                                             self.max_grad_norm)
                    self.optimizers[h].step()
                    with torch.no_grad():
                        m_new, s_new = mu_new.detach(), sigma_new.detach()
                        m_ref = mu_ref[:, sl, ids].reshape(T * E, k)
                        s_ref = sigma_ref[:, sl, ids].reshape(T * E, k)
                        # rl_games torch_ext.policy_kl(p0 = new, p1 = ref): KL(new || ref)
                        kl = (torch.log(s_ref / s_new + 1e-5)
                              + (s_new.square() + (m_ref - m_new).square()) / (2.0 * (s_ref.square() + 1e-5))
                              - 0.5).sum(-1).mean()
                        pass_kls[h].append(kl)
                        mu_ref[:, sl, ids] = m_new.view(T, E, k)
                        sigma_ref[:, sl, ids] = s_new.view(T, E, k)
                        if ep == self.epochs - 1:   # diagnostic: vs the rows' collecting policy
                            m0 = self.mu[:, sl, ids].reshape(T * E, k)
                            s0 = self.sigma[:, sl, ids].reshape(T * E, k)
                            kl_rollout[h] += float((torch.log(s0 / s_new + 1e-5)
                                                    + (s_new.square() + (m0 - m_new).square())
                                                    / (2.0 * (s0.square() + 1e-5)) - 0.5).sum(-1).mean())
                    stats["surrogate"][h] += float(surrogate)
                    stats["value"][h] += float(value_loss)
                    stats["bounds"][h] += float(bounds)
                    stats["entropy"][h] += float(entropy)
                    stats["kl"][h] += float(kl)
                    stats["clip_frac"][h] += float(((ratio - 1.0).abs() > self.clip_param).float().mean())
                n_updates += 1
            if self.schedule == "adaptive":   # rl_games AdaptiveScheduler, once per pass on the pass's mean KL
                for h in range(H):
                    av = torch.mean(torch.stack(pass_kls[h])).item()   # torch_ext.mean_list
                    lr = self.lrs[h]
                    if av > 2.0 * self.desired_kl:
                        lr = max(lr / 1.5, 1e-6)
                    if av < 0.5 * self.desired_kl:
                        lr = min(lr * 1.5, 1e-2)
                    self.lrs[h] = lr
                    for g in self.optimizers[h].param_groups:
                        g["lr"] = lr
        return n_updates

    def _update_rsl_rl(self, stats: dict[str, list[float]], B: int, obs: torch.Tensor, actions: torch.Tensor,
                       mu_old: torch.Tensor, sigma_old: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor,
                       targets: torch.Tensor) -> int:
        """The rsl_rl-style epoch (impl=rsl_rl): random row minibatches, per-minibatch adaptive LR on the KL to the
        rollout policy, gradient clipping per network. Returns the number of minibatch steps."""
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
                    if self.critic_arch == "flashsac":
                        logits = self.critics[h].logits(o, training=True)
                        value_loss = self.critics[h].ce_loss(logits, targets[rows, h])
                        with torch.no_grad():
                            stats["value_mse"][h] += float(
                                (self.critics[h].expectation(logits) - targets[rows, h]).square().mean())
                    else:
                        v = self.critics[h](o)
                        value_loss = (targets[rows, h] - v).square().mean()
                    s = self.bounds_soft
                    bounds = ((mu_new - s).clamp_min(0.0).square() + (mu_new + s).clamp_max(0.0).square()).sum(-1).mean()
                    entropy = dist.entropy().sum(-1).mean()
                    loss = surrogate + self.vf_coef * value_loss + self.bounds_coef * bounds - self.ent_coef * entropy
                    self.optimizers[h].zero_grad(set_to_none=True)
                    if self.critic_arch != "flashsac":
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.actors[h].parameters(), self.max_grad_norm)
                        nn.utils.clip_grad_norm_(self.critics[h].parameters(), self.max_grad_norm)
                        self.optimizers[h].step()
                    else:
                        self.critic_optimizers[h].zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.actors[h].parameters(), self.max_grad_norm)
                        self.optimizers[h].step()
                        self.critic_optimizers[h].step()
                        self.critic_schedulers[h].step()
                        self.critics[h].normalize_parameters()
                    stats["surrogate"][h] += float(surrogate)
                    stats["value"][h] += float(value_loss)
                    stats["bounds"][h] += float(bounds)
                    stats["entropy"][h] += float(entropy)
                    stats["clip_frac"][h] += float(((ratio - 1.0).abs() > self.clip_param).float().mean())
                n_updates += 1
        return n_updates

    def _trajectories(self):
        """arch=lstm: the window split into trajectories at the dones (rsl_rl split_and_pad_trajectories; env-major
        order), their masks, each trajectory's (env, start step), and the per-env offsets into the trajectory list."""
        from rsl_rl.utils import split_and_pad_trajectories

        padded, masks = split_and_pad_trajectories(self.obs, self.dones.unsqueeze(-1))
        starts = torch.zeros_like(self.dones)
        starts[0] = True
        starts[1:] = self.dones[:-1]   # rsl_rl last_was_done
        env_idx, t_idx = starts.transpose(0, 1).nonzero(as_tuple=True)   # env-major = the padded order
        offsets = torch.cat([starts.new_zeros(1, dtype=torch.long), starts.sum(0).cumsum(0)])
        return padded, masks, env_idx, t_idx, offsets

    @staticmethod
    def _start_states(h0: tuple, env_idx: torch.Tensor, t_idx: torch.Tensor) -> tuple:
        """The (h, c) state each trajectory starts from: the rollout-start state of its env if it starts at t = 0,
        zeros if it starts after a done (rsl_rl: the saved state at a trajectory start)."""
        first = t_idx == 0
        out = []
        for x in h0:
            st = x.new_zeros(x.shape[0], env_idx.shape[0], x.shape[2])
            st[:, first] = x[:, env_idx[first]]
            out.append(st)
        return tuple(out)

    @torch.no_grad()
    def recurrent_replay(self) -> tuple[torch.Tensor, torch.Tensor]:
        """arch=lstm diagnostic: the window's (T, N, H) log-probs and values recomputed through the update's
        sequence path with the current parameters (equal to the stored ones before any optimizer step)."""
        padded, masks, env_idx, t_idx, _ = self._trajectories()
        logp = torch.zeros_like(self.logp)
        values = torch.zeros_like(self.values)
        for h in range(self.H):
            lat = self.actors[h].latent_seq(padded, masks, self._start_states(self._h0_a[h], env_idx, t_idx))
            logp[..., h] = self.actors[h].distribution_from_latent(lat).log_prob(self.actions[..., self.ids[h]]).sum(-1)
            lat_c = self.critics[h].latent_seq(padded, masks, self._start_states(self._h0_c[h], env_idx, t_idx))
            v = self.critics[h].value_from_latent(lat_c)
            values[..., h] = self.value_rms[h].denormalize(v) if self.normalize_value else v
        return logp, values

    def _update_recurrent(self, stats: dict[str, list[float]], adv: torch.Tensor, targets: torch.Tensor) -> int:
        """arch=lstm (rsl_rl 5.5.1 recurrent PPO): minibatch i = envs [i E, (i + 1) E) with all their trajectories
        (rsl_rl recurrent_mini_batch_generator: the same order every epoch); per minibatch and hand the actor and critic
        run over the padded trajectories from their start states and return to the (T, E) layout; then the losses,
        the adaptive LR on the KL to the rollout policy, and the per-network gradient clipping of _update_rsl_rl.
        Memory (FDR-35's first launch died of CUDA OOM at its first update): each minibatch's trajectories are padded
        when it is used (split_and_pad_trajectories of the env block = the block's slice of the whole-window padding:
        trajectories are per env, both pad to T), and the actor and critic losses are backpropagated one after the other
        (separate networks, no shared parameters: the same gradients as one backward of their sum), so only one
        network's graph is alive at a time. Returns the number of minibatch steps."""
        from rsl_rl.utils import split_and_pad_trajectories

        T, N, H = self.window, self.num_envs, self.H
        E = N // self.num_mb
        adv = adv.view(T, N, H)
        targets = targets.view(T, N, H)
        starts = torch.zeros_like(self.dones)
        starts[0] = True
        starts[1:] = self.dones[:-1]
        env_idx, t_idx = starts.transpose(0, 1).nonzero(as_tuple=True)
        offsets = torch.cat([starts.new_zeros(1, dtype=torch.long), starts.sum(0).cumsum(0)])
        n_updates = 0
        for _ in range(self.epochs):
            for i in range(self.num_mb):
                tr = slice(int(offsets[i * E]), int(offsets[(i + 1) * E]))
                sl = slice(i * E, (i + 1) * E)
                o, mk = split_and_pad_trajectories(self.obs[:, sl], self.dones[:, sl].unsqueeze(-1))
                ei, ti = env_idx[tr], t_idx[tr]
                for h in range(H):
                    ids = self.ids[h]
                    lat = self.actors[h].latent_seq(o, mk, self._start_states(self._h0_a[h], ei, ti)).flatten(0, 1)
                    dist = self.actors[h].distribution_from_latent(lat)
                    mu_new = dist.mean
                    if self.schedule == "adaptive":
                        with torch.no_grad():
                            s_old = self.sigma[:, sl][..., ids].flatten(0, 1)
                            m_old = self.mu[:, sl][..., ids].flatten(0, 1)
                            s_new = dist.stddev.expand_as(mu_new)
                            kl = (torch.log(s_new / s_old)
                                  + (s_old.square() + (m_old - mu_new).square()) / (2 * s_new.square())
                                  - 0.5).sum(-1).mean()
                            if kl > self.desired_kl * 2.0:
                                self.lrs[h] = max(1e-5, self.lrs[h] / 1.5)
                            elif self.desired_kl / 2.0 > kl > 0.0:
                                self.lrs[h] = min(1e-2, self.lrs[h] * 1.5)
                            for g in self.optimizers[h].param_groups:
                                g["lr"] = self.lrs[h]
                            stats["kl"][h] += float(kl)
                    logp = dist.log_prob(self.actions[:, sl][..., ids].flatten(0, 1)).sum(-1)
                    ratio = torch.exp(logp - self.logp[:, sl, h].flatten())
                    a_h = adv[:, sl, h].flatten()
                    surrogate = torch.max(-a_h * ratio,
                                          -a_h * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)).mean()
                    s = self.bounds_soft
                    bounds = ((mu_new - s).clamp_min(0.0).square() + (mu_new + s).clamp_max(0.0).square()).sum(-1).mean()
                    entropy = dist.entropy().sum(-1).mean()
                    self.optimizers[h].zero_grad(set_to_none=True)
                    (surrogate + self.bounds_coef * bounds - self.ent_coef * entropy).backward()   # frees the actor graph
                    lat_c = self.critics[h].latent_seq(o, mk, self._start_states(self._h0_c[h], ei, ti)).flatten(0, 1)
                    v = self.critics[h].value_from_latent(lat_c)
                    value_loss = (targets[:, sl, h].flatten() - v).square().mean()
                    (self.vf_coef * value_loss).backward()
                    nn.utils.clip_grad_norm_(self.actors[h].parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critics[h].parameters(), self.max_grad_norm)
                    self.optimizers[h].step()
                    stats["surrogate"][h] += float(surrogate)
                    stats["value"][h] += float(value_loss)
                    stats["bounds"][h] += float(bounds)
                    stats["entropy"][h] += float(entropy)
                    stats["clip_frac"][h] += float(((ratio - 1.0).abs() > self.clip_param).float().mean())
                n_updates += 1
        return n_updates

    def on_env_reset(self) -> None:
        """The runner reset the train envs (eval drove them): the previous window is no longer contiguous; arch=lstm:
        the training LSTM states restart from zeros."""
        self._have_prev = False
        self._hid_a = [None] * self.H
        self._hid_c = [None] * self.H

    def reset_eval_memory(self) -> None:
        """evaluation.evaluate, at every eval batch reset: arch=lstm deterministic actions restart from zero states."""
        self._eval_hid_a = [None] * self.H

    def on_eval_dones(self, dones: Any) -> None:
        """evaluation.evaluate, after every eval step: zero the eval LSTM states of the envs that are done."""
        if not self.recurrent:
            return
        d = torch.as_tensor(np.asarray(dones) if not torch.is_tensor(dones) else dones, device=self.device).view(-1).bool()
        if bool(d.any()):
            self._zero_done(self._eval_hid_a, d)

    # ─── checkpoints ─────────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save({
            "sides": self.sides,
            "actors": self.actors.state_dict(),
            "critics": self.critics.state_dict(),
            "value_rms": self.value_rms.state_dict(),
            "optimizers": [o.state_dict() for o in self.optimizers],
            "critic_optimizers": [o.state_dict() for o in self.critic_optimizers],
            "critic_schedulers": [s.state_dict() for s in self.critic_schedulers],
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
        for o, s in zip(self.critic_optimizers, state.get("critic_optimizers", [])):
            o.load_state_dict(s)
        for sch, s in zip(self.critic_schedulers, state.get("critic_schedulers", [])):
            sch.load_state_dict(s)
        self.lrs = [float(x) for x in state["lrs"]]
        for o, lr in zip(self.optimizers, self.lrs):
            for g in o.param_groups:
                g["lr"] = lr
