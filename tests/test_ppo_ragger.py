"""Unit checks of agents.ppo.PPOAgent's RAgger window on random data (no simulator):
python tests/test_ppo_ragger.py  (from the repo root, FlashDexRetarget env)."""
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agents.ppo import PPOAgent  # noqa: E402

N, D, A, T = 64, 11, 5, 6


def make(carry):
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "config/runner/ppo.yaml").runner.agent
    cfg = OmegaConf.merge(cfg, {"device": "cpu", "num_steps_per_env": T, "carry_steps": carry})
    torch.manual_seed(0)
    return PPOAgent(cfg, D, A, N, "cpu")


def rollout(agent, rng, n):
    obs = rng.standard_normal((N, D)).astype(np.float32)
    for _ in range(n):
        agent.act(obs)
        obs = rng.standard_normal((N, D)).astype(np.float32)
        agent.observe({"reward": rng.standard_normal(N).astype(np.float32), "terminated": rng.random(N) < 0.05,
                       "truncated": rng.random(N) < 0.02, "next_observation": obs})


def count_steps(agent):
    calls = [0]
    orig = agent.alg.optimizer.step
    def step(*a, **k):
        calls[0] += 1
        return orig(*a, **k)
    agent.alg.optimizer.step = step
    return calls


def norm_count(agent):
    return int(agent.alg.actor.obs_normalizer.count), int(agent.alg.critic.obs_normalizer.count)


rng = np.random.default_rng(0)
# ── plain PPO: window = T fresh rows every iteration, 5 x 4 optimizer steps, normalizers see every row ──
ppo = make(0)
calls = count_steps(ppo)
for it in range(3):
    n = ppo.begin_rollout()
    assert n == T, n
    rollout(ppo, rng, n)
    c0 = norm_count(ppo)
    ppo.update()
    assert norm_count(ppo)[0] - c0[0] == T * N and norm_count(ppo)[1] - c0[1] == T * N
assert calls[0] == 3 * 5 * 4, calls
assert ppo.alg.storage.observations.batch_size[0] == T
print("PPO: window", T, "rows; 20 optimizer steps per update; normalizers +T*N per update  OK")

# ── RAgger, carry C = T: first window all fresh (2T steps); then C carried + T fresh ──
C = T
rg = make(C)
calls = count_steps(rg)
n = rg.begin_rollout()
assert n == C + T, n
rollout(rg, rng, n)
c0 = norm_count(rg)
rg.update()
assert norm_count(rg)[0] - c0[0] == (C + T) * N, "first window: every row is fresh"
st = rg.alg.storage
last = {k: v[-C:].clone() for k, v in (("obs", st.observations["policy"]), ("act", st.actions), ("rew", st.rewards),
                                        ("done", st.dones), ("logp", st.actions_log_prob))}
last_params = [p[-C:].clone() for p in st.distribution_params]
n = rg.begin_rollout()
assert n == T, n
assert rg._carried_now == C and st.step == C
for k, v in (("obs", st.observations["policy"]), ("act", st.actions), ("rew", st.rewards), ("done", st.dones),
             ("logp", st.actions_log_prob)):
    assert torch.equal(v[:C], last[k]), k
for p, q in zip(st.distribution_params, last_params):
    assert torch.equal(p[:C], q)
with torch.no_grad():
    v_now = rg.alg.critic(st.observations[:C].flatten(0, 1)).view(C, N, 1)
assert torch.equal(st.values[:C], v_now), "carried rows re-valued by the current critic"
rollout(rg, rng, n)
c0 = norm_count(rg)
rg.update()
assert norm_count(rg)[0] - c0[0] == T * N and norm_count(rg)[1] - c0[1] == T * N, "normalizers: fresh rows only"
assert calls[0] == 2 * 5 * 4, calls
print(f"RAgger: first window {C + T} fresh; then {C} carried (obs/actions/rewards/dones/logp/params identical, "
      f"values = current critic) + {T} fresh; 20 optimizer steps per update; normalizers +T*N  OK")

# ── after an env reset the next window is all fresh again ──
rg.on_env_reset()
assert rg.begin_rollout() == C + T and rg._carried_now == 0
print("RAgger: after on_env_reset the window is all fresh  OK")
assert "update_normalization" not in vars(rg.alg.actor) and "update_normalization" not in vars(rg.alg.critic)
print("ALL OK")
