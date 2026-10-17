"""On-policy loop (PPO, agents.ppo.PPOAgent) over the same envs, eval, checkpoints, resume state and wandb
logging as OffPolicyRunner. One iteration = one rollout (every action from the policy, including right after a
reset) + one update. Eval and checkpoints run at the end of the iteration in which their absolute env-step grid
point is crossed (at most one rollout late); after an eval drove the train envs they are reset and the agent
starts a fresh window."""

from __future__ import annotations

import numpy as np
import tqdm
from omegaconf import DictConfig

from common.average_meters import AverageMeter
from common.logger import log_metrics

from .off_policy_runner import OffPolicyRunner


class OnPolicyRunner(OffPolicyRunner):
    def __init__(self, env, runner_cfg: DictConfig, eval_cfg: DictConfig, eval_env, headless: bool = True) -> None:
        super().__init__(env, runner_cfg, eval_cfg, eval_env, headless=headless)
        self._max_iterations = int(runner_cfg.max_epochs)
        self._log_every_it = max(1, int(runner_cfg.get("log_every_iterations", 1)))

    @staticmethod
    def _crossed_range(prev_step: int, env_step: int, stride: int) -> bool:
        return stride > 0 and (env_step // stride) != (prev_step // stride)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        del num_learning_iterations, init_at_random_ep_len   # the run length is runner.max_epochs iterations
        agent = self._agent
        self._install_stop_signal()
        self._start_clock()
        meter = AverageMeter()
        env_step = self._step_offset
        solved_stop = False

        for it in tqdm.tqdm(range(self._max_iterations), smoothing=0.1, mininterval=1.0):
            prev_step = env_step
            for _ in range(agent.begin_rollout()):
                self._collect_on_policy(it, meter)
                env_step += self._n_envs
            meter.update(agent.update())

            if self._crossed_range(prev_step, env_step, self._eval_stride) and self._eval_supported():
                eval_metrics = self._run_eval(env_step)
                log_metrics({**eval_metrics, **self._cost_metrics(env_step), "train/iteration": it + 1},
                            step=env_step)
                if getattr(self._eval_env, "shares_train_env", False):
                    self._reset_train_env()   # eval drove the train env: restart its episodes
                    agent.on_env_reset()
                if self._all_solved(eval_metrics, env_step):
                    solved_stop = True
                    break

            if (it + 1) % self._log_every_it == 0:
                log_metrics({**meter.mean(), **self._cost_metrics(env_step), "train/iteration": it + 1},
                            step=env_step)
                meter.reset()

            if self._crossed_range(prev_step, env_step, self._save_stride):
                self._save_checkpoint(env_step)

            if self._stop_requested:
                self._graceful_stop(env_step)

        if self._eval_supported() and not solved_stop:
            log_metrics({**self._run_eval(env_step), **self._cost_metrics(env_step)}, step=env_step)
        self._finish()

    def _collect_on_policy(self, it: int, meter: AverageMeter) -> None:
        """Step every train env once with the policy's sampled actions and hand the step to the agent."""
        actions = self._agent.act(self._observations)
        next_obs, rewards, terminateds, truncateds, infos = self._train_env.step(np.asarray(actions))
        self._clamp_nonfinite(it, rewards, next_obs, next_obs)
        if "episode_info" in infos:
            meter.update(infos["episode_info"])
        self._agent.observe({
            "reward": rewards,
            "terminated": terminateds,
            "truncated": truncateds,
            "next_observation": next_obs,
        })
        self._observations = next_obs
        self._just_reset = False
