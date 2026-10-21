"""The adv/ panel of agents.hand_ppo.HandPPOAgent (the raw GAE advantages before the batch normalization) against numpy on
the same advantages, on random data (no simulator): python tests/test_hand_ppo_adv_stats.py (repo root, FlashDexRetarget env)."""
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agents.hand_ppo import HandPPOAgent  # noqa: E402

N, T, K, FLAT, PER_STEP = 64, 6, 10, 21, 8
D = FLAT + K * PER_STEP
SIDES, IDS, H = ["right", "left"], [[0, 1, 2, 6], [3, 4, 5, 7]], 2
TRAJ = {"enable": True, "aggregation": "conv1d", "n_future": K, "n_sides": 1, "latent_dim": 16, "hidden_dim": 16,
        "num_layers": 2, "window_dims": [K * PER_STEP]}

cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "config/runner/hand_ppo.yaml").runner.agent
cfg = OmegaConf.merge(cfg, {"device": "cpu", "num_steps_per_env": T, "carry_steps": 0, "actor": {"hidden_dims": [32, 32]},
                            "critic": {"hidden_dims": [32, 32], "lr_decay_updates": 40}, "empty_cache_after_update": False})
torch.manual_seed(0)
agent = HandPPOAgent(cfg, D, SIDES, IDS, N, TRAJ, "cpu")
rng = np.random.default_rng(0)
for it in range(2):
    obs = rng.standard_normal((N, D)).astype(np.float32)
    for _ in range(agent.begin_rollout()):
        agent.act(obs)
        obs = rng.standard_normal((N, D)).astype(np.float32)
        agent.observe({"reward": 3.0 * rng.standard_normal((N, H)).astype(np.float32) + 0.5,
                       "terminated": rng.random(N) < 0.05, "truncated": rng.random(N) < 0.02, "next_observation": obs})
    _, adv = agent._returns()                       # the advantages update() is about to normalize
    a = adv.flatten(0, 1).double().numpy()
    info = agent.update()
    for h, name in enumerate(SIDES):
        x = a[:, h]
        c = x - x.mean()
        ref = {"mean_abs": np.abs(x).mean(), "mean": x.mean(), "var": x.var(ddof=1), "std": x.std(ddof=1),
               "min": x.min(), "max": x.max(), "skew": (c ** 3).mean() / (c ** 2).mean() ** 1.5,
               "kurtosis": (c ** 4).mean() / (c ** 2).mean() ** 2 - 3.0, "frac_pos": (x > 0).mean()}
        for k, v in ref.items():
            got = info[f"adv/{k}_{name}"]
            assert np.isclose(got, v, rtol=1e-9, atol=1e-12), (it, name, k, got, v)
assert sum(k.startswith("adv/") for k in info) == 9 * H
print(f"adv/ panel == numpy on the pre-normalization advantages (both hands, 2 updates): mean_abs "
      f"{info['adv/mean_abs_right']:.4f}, std {info['adv/std_right']:.4f}, skew {info['adv/skew_right']:+.3f}, "
      f"kurtosis {info['adv/kurtosis_right']:+.3f}  OK")
