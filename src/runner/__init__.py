from __future__ import annotations

from mjlab.rl import RslRlVecEnvWrapper
from omegaconf import DictConfig

from .off_policy_runner import OffPolicyRunner
from .on_policy_runner import OnPolicyRunner


def create_runner(
    runner_cfg: DictConfig,
    eval_cfg: DictConfig,
    env: RslRlVecEnvWrapper,
    eval_env: RslRlVecEnvWrapper | None = None,
    headless: bool = True,
) -> OffPolicyRunner:
    """Build the runner from ``cfg.runner`` + ``cfg.eval``; ``cfg.runner.agent`` picks the algorithm
    (``cfg.runner.on_policy``: the PPO loop of on_policy_runner, default false)."""
    if eval_env is None:
        raise ValueError("the runner's eval needs the dedicated eval env from create_envs")
    cls = OnPolicyRunner if bool(runner_cfg.get("on_policy", False)) else OffPolicyRunner
    return cls(env, runner_cfg, eval_cfg, eval_env, headless=headless)
