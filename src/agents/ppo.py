"""PPO as an agents.Agent: rsl-rl-lib's PPO, RolloutStorage and MLPModel (library defaults unless set in
config/runner/ppo.yaml), driven by runner.on_policy_runner.OnPolicyRunner.

Optional RAgger window (``runner.agent.carry_steps`` > 0; 0 = plain PPO, code path unchanged): the window is
``carry_steps + num_steps_per_env`` rows. Before each rollout the last ``carry_steps`` rows of the previous window
(observations, actions, rewards, dones, log-probabilities, distribution parameters) move to the front and are
re-valued by the current critic; the rollout fills the remaining ``num_steps_per_env`` rows. GAE runs over the
whole window, every row is trained (the mini-batches grow with the window, so the optimizer steps per update are
unchanged), and the observation normalizers are updated from the fresh rows only. Without a previous window
(start of training, or after an eval reset the train envs) the rollout fills the whole window with fresh rows.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from .base import Agent

_OBS_GROUPS = {"actor": ["policy"], "critic": ["policy"]}


class PPOAgent(Agent):
    def __init__(self, cfg: DictConfig, obs_dim: int, action_dim: int, num_envs: int, device: str) -> None:
        self.cfg = cfg
        self.device = device
        self.num_envs = int(num_envs)
        self.fresh_steps = int(cfg.num_steps_per_env)
        self.carry_steps = int(cfg.get("carry_steps", 0))
        if not 0 <= self.carry_steps <= self.fresh_steps:
            raise ValueError(f"carry_steps must be in [0, num_steps_per_env], got {self.carry_steps}")
        self.window = self.fresh_steps + self.carry_steps
        self.clip_actions = float(cfg.get("clip_actions", 1.0))

        obs0 = self._td(torch.zeros(self.num_envs, obs_dim, device=device))
        actor_cfg = OmegaConf.to_container(cfg.actor, resolve=True)
        critic_cfg = OmegaConf.to_container(cfg.critic, resolve=True)
        actor = MLPModel(obs0, _OBS_GROUPS, "actor", action_dim, **actor_cfg).to(device)
        critic = MLPModel(obs0, _OBS_GROUPS, "critic", 1, **critic_cfg).to(device)
        storage = RolloutStorage("rl", self.num_envs, self.window, obs0, [action_dim], device)
        alg_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
        self.alg = PPO(actor, critic, storage, device=device, **alg_cfg)
        self.alg.train_mode()
        print(f"[ppo] actor {actor}\n[ppo] critic {critic}", flush=True)
        print(f"[ppo] window {self.window} rows = carry {self.carry_steps} + fresh {self.fresh_steps}; "
              f"algorithm {alg_cfg}", flush=True)

        self._have_prev = False      # a previous window exists whose last carry_steps rows can be carried
        self._carried_now = 0        # rows carried into the current window
        self._last_obs: torch.Tensor | None = None

    # ─── construction ────────────────────────────────────────────────────────────
    @classmethod
    def build(cls, cfg: DictConfig, env: Any, env_info: dict[str, Any], command_name: str) -> "PPOAgent":
        if env_info.get("reward_heads"):
            raise ValueError("PPO trains one actor-critic on the total reward: set per_hand_actor_critic=false")
        obs_dim = int(env.single_observation_space.shape[0])
        action_dim = int(env.single_action_space.shape[0])
        return cls(cfg, obs_dim, action_dim, env.num_envs, str(cfg.get("device", "cuda")))

    @staticmethod
    def _td(obs: torch.Tensor) -> TensorDict:
        return TensorDict({"policy": obs}, batch_size=[obs.shape[0]], device=obs.device)

    def _to_tensor(self, x: Any) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.device, dtype=torch.float32)
        return torch.as_tensor(x, device=self.device, dtype=torch.float32)

    # ─── rollout ─────────────────────────────────────────────────────────────────
    def begin_rollout(self) -> int:
        """Set up the window for the next rollout; returns the number of env steps to collect."""
        st = self.alg.storage
        st.clear()
        self._carried_now = 0
        if self.carry_steps > 0 and self._have_prev:
            c, T = self.carry_steps, self.window
            with torch.no_grad():
                st.observations[:c] = st.observations[T - c:].clone()
                for buf in (st.actions, st.rewards, st.dones, st.actions_log_prob):
                    buf[:c] = buf[T - c:].clone()
                for p in st.distribution_params:
                    p[:c] = p[T - c:].clone()
                # re-value the carried rows with the current critic (normalizer as it is now, no update)
                st.values[:c] = self.alg.critic(st.observations[:c].flatten(0, 1)).view(c, self.num_envs, 1)
            st.step = c
            self._carried_now = c
        return self.window - st.step

    def act(self, obs: Any, deterministic: bool = False) -> np.ndarray:
        obs_t = self._to_tensor(obs)
        if deterministic:
            with torch.no_grad():
                actions = self.alg.actor(self._td(obs_t))
            return actions.clamp(-self.clip_actions, self.clip_actions).cpu().numpy()
        with torch.no_grad():
            actions = self.alg.act(self._td(obs_t))
        return actions.clamp(-self.clip_actions, self.clip_actions).cpu().numpy()

    def observe(self, transition: Mapping[str, Any]) -> None:
        """One env step of the rollout. ``next_observation`` is the env's current (post-reset) observation:
        the bootstrap state at the end of the rollout (done rows are masked by GAE)."""
        self._last_obs = self._to_tensor(transition["next_observation"])
        rewards = self._to_tensor(transition["reward"]).view(-1)
        terminated = torch.as_tensor(np.asarray(transition["terminated"]), device=self.device).view(-1).bool()
        truncated = torch.as_tensor(np.asarray(transition["truncated"]), device=self.device).view(-1).bool()
        # rsl_rl: dones end the GAE chain; time-outs add gamma * V(s_t) to the reward
        self.alg.process_env_step(self._td(self._last_obs), rewards, terminated | truncated,
                                  {"time_outs": truncated.float()})

    def ready(self) -> bool:
        return True

    def update(self) -> dict[str, float]:
        st = self.alg.storage
        if st.step != self.window:
            raise RuntimeError(f"update with an incomplete window ({st.step}/{self.window} rows)")
        with torch.no_grad():
            self.alg.compute_returns(self._td(self._last_obs))
        skip = self._carried_now * self.num_envs
        patched = []
        if skip:
            # the carried rows were already counted by the normalizers when they were fresh
            for model in (self.alg.actor, self.alg.critic):
                orig = model.update_normalization
                model.update_normalization = (lambda o, _f=orig: _f(o[skip:]))
                patched.append((model, orig))
        try:
            info = self.alg.update()
        finally:
            for model, _orig in patched:
                del model.update_normalization   # back to the class method
        self._have_prev = True
        out = {f"loss/{k}": float(v) for k, v in info.items()}
        out.update({
            "train/learning_rate": float(self.alg.learning_rate),
            "train/action_std": float(self.alg.actor.output_std.mean()) if self._std_ready() else 0.0,
            "train/carried_rows": float(self._carried_now),
            "train/window_rows": float(self.window),
        })
        return out

    def _std_ready(self) -> bool:
        try:
            self.alg.actor.output_std
            return True
        except Exception:  # noqa: BLE001
            return False

    def on_env_reset(self) -> None:
        """The runner reset the train envs (eval drove them): the previous window is no longer contiguous."""
        self._have_prev = False

    # ─── checkpoints ─────────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.alg.save(), os.path.join(path, "ppo.pt"))

    def load(self, path: str) -> None:
        state = torch.load(os.path.join(path, "ppo.pt"), map_location=self.device, weights_only=False)
        self.alg.load(state, None, strict=True)
