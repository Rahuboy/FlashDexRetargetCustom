"""Unit checks of agents.hand_ppo.HandPPOAgent on random data (no simulator):
python tests/test_hand_ppo.py  (from the repo root, FlashDexRetarget env)."""
import copy
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.distributions import Normal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agents.hand_ppo import HandPPOAgent  # noqa: E402

N, T, K = 64, 6, 10                       # envs, fresh rows, future steps
FLAT, PER_STEP = 21, 8                    # flat obs dims, per-step window features (both hands)
D = FLAT + K * PER_STEP                   # obs = [flat | window], window at the tail as the env lays it out
SIDES, IDS = ["right", "left"], [[0, 1, 2, 6], [3, 4, 5, 7]]   # interleaved like the env's action layout
A, H = 8, 2
TRAJ = {"enable": True, "aggregation": "conv1d", "n_future": K, "n_sides": 1, "latent_dim": 16, "hidden_dim": 16,
        "num_layers": 2, "window_dims": [K * PER_STEP]}


def make(carry=0, seed=0, critic=None, actor=None):
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "config/runner/hand_ppo.yaml").runner.agent
    cfg = OmegaConf.merge(cfg, {"device": "cpu", "num_steps_per_env": T, "carry_steps": carry,
                                "actor": {"hidden_dims": [32, 32]}, "critic": {"hidden_dims": [32, 32], "lr_decay_updates": 40},
                                "empty_cache_after_update": False})
    if critic:
        cfg = OmegaConf.merge(cfg, {"critic": critic})
    if actor:
        cfg = OmegaConf.merge(cfg, {"actor": actor})
    torch.manual_seed(seed)
    return HandPPOAgent(cfg, D, SIDES, IDS, N, TRAJ, "cpu")


def rollout(agent, rng, n, p_term=0.05, p_trunc=0.02):
    obs = rng.standard_normal((N, D)).astype(np.float32)
    for _ in range(n):
        agent.act(obs)
        obs = rng.standard_normal((N, D)).astype(np.float32)
        agent.observe({"reward": rng.standard_normal((N, H)).astype(np.float32), "terminated": rng.random(N) < p_term,
                       "truncated": rng.random(N) < p_trunc, "next_observation": obs})
    return obs


rng = np.random.default_rng(0)

# ── 1. stored log-probs == the policy's log-probs before any update (PPO ratio exactly 1 at the first minibatch) ──
ag = make()
ag.begin_rollout()
rollout(ag, rng, T)
o = ag.obs.flatten(0, 1)
for h in range(H):
    lp = ag.actors[h].distribution(o).log_prob(ag.actions.flatten(0, 1)[:, ag.ids[h]]).sum(-1)
    assert torch.allclose(lp, ag.logp.flatten(0, 1)[:, h], atol=1e-5), h
print("1. ratio == 1 before the first optimizer step (both hands)  OK")

# ── 2. GAE per head == a reference loop (rewards x0.1 stored; time-outs bootstrap gamma * V(s_t); shared dones) ──
returns, adv = ag._returns()
g, lam = ag.gamma, ag.lam
V, R = ag.values.numpy(), ag.rewards.numpy()
dn, to = ag.dones.numpy(), ag.timeouts.numpy()
with torch.no_grad():
    Vl = ag._values(ag._last_obs).numpy()
ref = np.zeros_like(V)
a_ = np.zeros_like(Vl)
for t in reversed(range(T)):
    nv = Vl if t == T - 1 else V[t + 1]
    r = R[t] + g * V[t] * to[t][:, None]
    nd = 1.0 - dn[t][:, None]
    a_ = r + nd * g * nv - V[t] + nd * g * lam * a_
    ref[t] = a_ + V[t]
assert np.allclose(returns.numpy(), ref, atol=1e-5) and np.allclose(adv.numpy(), ref - V, atol=1e-5)
print("2. per-head GAE with shared dones and time-out bootstrap matches the reference  OK")

# ── 3. ValueRMS == rl_games RunningMeanStd (float64, count starts at 1, unbiased var, clamp +-5) ──
rms = ag.value_rms[0]
m, v, c = 0.0, 1.0, 1.0
for _ in range(3):
    x = rng.standard_normal(500) * 3 + 2
    bm, bv, bc = x.mean(), x.var(ddof=1), len(x)
    d = bm - m
    tot = c + bc
    m, v, c = m + d * bc / tot, (v * c + bv * bc + d * d * c * bc / tot) / tot, tot
    rms.update(torch.as_tensor(x, dtype=torch.float32))
assert abs(float(rms.running_mean) - m) < 1e-4 and abs(float(rms.running_var) - v) < 1e-3 and float(rms.count) == c
y = torch.tensor([-100.0, 0.5, 100.0])
assert torch.allclose(rms.normalize(y), ((y - m) / np.sqrt(v + 1e-5)).clamp(-5, 5), atol=1e-4)
assert torch.allclose(rms.denormalize(rms.normalize(torch.tensor([1.0, 2.0]))), torch.tensor([1.0, 2.0]), atol=1e-4)
rms.load_state_dict(make().value_rms[0].state_dict())   # reset
print("3. value normalization matches rl_games' RunningMeanStd  OK")

# ── 4. bound loss (rl_games, soft bound 1.1) on known means ──
mu = torch.tensor([[1.6, -0.5, -2.1, 1.0]])
s = ag.bounds_soft
b = ((mu - s).clamp_min(0).square() + (mu + s).clamp_max(0).square()).sum(-1).mean()
assert abs(float(b) - (0.5 ** 2 + 1.0 ** 2)) < 1e-6
print("4. bound loss: means 1.6 / -2.1 cost 0.25 + 1.0, means inside +-1.1 cost 0  OK")

# ── 5. hand h's loss only reaches hand h's networks; optimizer h holds exactly hand h's parameters ──
for h in range(H):
    own = {id(p) for p in list(ag.actors[h].parameters()) + list(ag.critics[h].parameters())}
    got = {id(p) for grp in ag.optimizers[h].param_groups for p in grp["params"]}
    assert own == got, h
before = [copy.deepcopy(x.state_dict()) for x in list(ag.actors) + list(ag.critics)]
info = ag.update()
after = [x.state_dict() for x in list(ag.actors) + list(ag.critics)]
assert all(any(not torch.equal(b_[k], a_[k]) for k in b_) for b_, a_ in zip(before, after)), "a network did not move"
for k in ("loss/surrogate_right", "loss/value_left", "loss/bounds_right", "train/kl_left", "train/learning_rate_right",
          "train/action_std_left", "train/clip_frac_right", "train/value_rms_std_left"):
    assert k in info and np.isfinite(info[k]), k
print("5. one optimizer per hand over its own actor + critic; all four networks updated; per-hand metrics finite  OK")

# ── 6. separate adaptive LR: freezing hand 'left' (LR 0, no KL change) leaves its LR at 0 while 'right' adapts ──
ag2 = make(seed=1)
ag2.begin_rollout()
rollout(ag2, rng, T)
ag2.lrs[1] = 0.0
for grp in ag2.optimizers[1].param_groups:
    grp["lr"] = 0.0
ag2.update()
assert ag2.lrs[1] == 0.0 and ag2.lrs[0] != 1e-3, ag2.lrs
print(f"6. per-hand learning rates are independent: right 1e-3 -> {ag2.lrs[0]:.2e}, frozen left stays 0  OK")

# ── 7. architecture: actor and critic = obs normalizer -> own traj encoder -> rsl_rl MLP; actor out = its hand's dims ──
from rsl_rl.modules.mlp import MLP  # noqa: E402
lat = TRAJ["latent_dim"]
for h in range(H):
    for net, out in ((ag.actors[h], len(IDS[h])), (ag.critics[h], 1)):
        assert isinstance(net.mlp, MLP)
        lin = [m for m in net.mlp if isinstance(m, torch.nn.Linear)]
        assert lin[0].in_features == FLAT + lat and lin[-1].out_features == out, (lin[0], lin[-1])
encs = [ag.actors[h].traj for h in range(H)] + [ag.critics[h].traj for h in range(H)]
assert len({id(e) for e in encs}) == 4
assert torch.allclose(ag.actors[0].std(), torch.ones(len(IDS[0])), atol=0.05)
print(f"7. actor / critic = normalizer -> own encoder ({K}x{PER_STEP} window -> {lat}) -> rsl_rl MLP; 4 separate encoders  OK")

# ── 8. RAgger window: carried rows moved to the front and re-valued; normalizers count fresh rows only ──
rag = make(carry=T // 2)
c = T // 2
for it in range(3):
    n = rag.begin_rollout()
    assert n == (T + c if it == 0 else T), (it, n)
    if it > 0:
        assert torch.equal(rag.obs[:c], prev_tail_obs) and torch.equal(rag.logp[:c], prev_tail_logp)
        with torch.no_grad():
            assert torch.allclose(rag.values[:c], rag._values(rag.obs[:c].flatten(0, 1)).view(c, N, H), atol=1e-6)
    rollout(rag, rng, n)
    cnt0 = int(rag.actors[0].obs_norm.count)
    rag.update()
    fresh = (T + c if it == 0 else T) * N
    assert int(rag.actors[0].obs_norm.count) - cnt0 == fresh, (it, int(rag.actors[0].obs_norm.count) - cnt0, fresh)
    prev_tail_obs, prev_tail_logp = rag.obs[-c:].clone(), rag.logp[-c:].clone()
rag.on_env_reset()
assert rag.begin_rollout() == T + c
print("8. RAgger: carried rows at the front, re-valued by the current critics; normalizers +fresh rows only; "
      "eval reset -> full fresh window  OK")

# ── 9. checkpoint round trip ──
with tempfile.TemporaryDirectory() as d:
    ag.save(d)
    ag3 = make(seed=7)
    ag3.load(d)
    x = torch.as_tensor(rng.standard_normal((5, D)).astype(np.float32))
    for h in range(H):
        assert torch.allclose(ag3.actors[h].mean(x), ag.actors[h].mean(x)) and \
            torch.allclose(ag3.critics[h](x), ag.critics[h](x))
    assert ag3.lrs == ag.lrs
print("9. save / load round trip  OK")

# ── 10. deterministic act = clipped means in env action order; stochastic act clipped to +-1 ──
x = rng.standard_normal((N, D)).astype(np.float32) * 50
det = ag.act(x, deterministic=True)
with torch.no_grad():
    want = torch.zeros(N, A)
    for h in range(H):
        want[:, ag.ids[h]] = ag.actors[h].mean(torch.as_tensor(x))
assert np.allclose(det, want.clamp(-1, 1).numpy(), atol=1e-6) and np.abs(ag.act(x)).max() <= 1.0
ag._pending = None
print("10. deterministic actions = per-hand means in env order, clipped; samples clipped  OK")
# ── 11. critic.arch=flashsac: FlashSAC critic as V(s), categorical CE, own optimizer + FlashSAC LR schedule ──
import math  # noqa: E402
from agents.modules.hand_ppo_nets import FlashSACValueCritic  # noqa: E402
from rsl_rl_flashsac.modules.flash_sac_layers import UnitLinear  # noqa: E402
FS = {"arch": "flashsac", "hidden_dim": 32, "lr_decay_updates": 40}
fs = make(critic=FS, seed=3)
c0 = fs.critics[0]
assert isinstance(c0, FlashSACValueCritic) and not hasattr(c0, "obs_norm")
y = torch.tensor([-7.0, -5.0, -1.234, 0.0, 0.05, 3.333, 5.0, 9.0])
t = c0.two_hot(y)
assert torch.allclose(t.sum(-1), torch.ones(len(y))) and torch.allclose((t * c0.bin_values).sum(-1), y.clamp(-5, 5), atol=1e-5)
for h in range(H):
    a_ids = {id(p) for g in fs.optimizers[h].param_groups for p in g["params"]}
    c_ids = {id(p) for g in fs.critic_optimizers[h].param_groups for p in g["params"]}
    assert a_ids == {id(p) for p in fs.actors[h].parameters()} and c_ids == {id(p) for p in fs.critics[h].parameters()}
bn_before = c0.embedder.norm.running_mean.clone()
for it in range(2):
    fs.begin_rollout()
    rollout(fs, rng, T)
    info = fs.update()
assert not torch.equal(bn_before, c0.embedder.norm.running_mean)          # batch statistics used in the update
for m in fs.critics.modules():
    if isinstance(m, UnitLinear):
        assert torch.allclose(m.w.weight.norm(dim=-1), torch.ones(m.w.weight.shape[0]), atol=1e-5)
steps = 2 * 5 * 4
want = 1.5e-4 + (3e-4 - 1.5e-4) * 0.5 * (1 + math.cos(math.pi * steps / 40))
assert abs(info["train/critic_lr_right"] - want) < 1e-12 and fs.critic_schedulers[1].last_epoch == steps, info["train/critic_lr_right"]
for k in ("loss/value_right", "train/value_mse_left", "train/critic_lr_left", "loss/surrogate_left"):
    assert k in info and np.isfinite(info[k]), k
x = torch.as_tensor(rng.standard_normal((5, D)).astype(np.float32))
with torch.no_grad():
    assert torch.equal(c0(x), c0(x))                                        # eval-mode BatchNorm: deterministic
with tempfile.TemporaryDirectory() as d:
    fs.save(d)
    fs2 = make(critic=FS, seed=9)
    fs2.load(d)
    with torch.no_grad():
        assert torch.allclose(fs2.critics[0](x), c0(x))
    assert fs2.critic_schedulers[0].last_epoch == steps
print(f"11. flashsac critic: two-hot exact, own Adam (LR {info['train/critic_lr_right']:.3e} = cosine), BN batch stats in the "
      f"update, unit-norm rows, metrics finite, save/load  OK")
# ── 12. actor.arch / critic.arch = simba: the alphanist (SimBa) residual trunk; joint optimizer + MSE like mlp ──
from agents.modules.hand_ppo_nets import SimbaBlock, SimbaTrunk  # noqa: E402
SB = {"arch": "simba", "simba": {"hidden_dim": 32}}
sb = make(critic=SB, actor=SB, seed=4)
for h in range(H):
    for net, out, gain in ((sb.actors[h], len(IDS[h]), 0.01), (sb.critics[h], 1, 1.0)):
        tr = net.mlp
        assert isinstance(tr, SimbaTrunk) and len(tr.blocks) == 2 and all(isinstance(b, SimbaBlock) for b in tr.blocks)
        assert tr.embed.in_features == FLAT + lat and tr.embed.out_features == 32 and tr.head.out_features == out
        for b in tr.blocks:
            assert b.fc1.out_features == 4 * 32 and b.fc2.out_features == 32 and isinstance(b.act, torch.nn.SiLU)
            assert b.ln.eps == 1e-6 and torch.count_nonzero(b.fc1.bias) == 0 and torch.count_nonzero(b.fc2.bias) == 0
            for fc in (b.fc1, b.fc2):                                   # flax he_normal: truncated at 2 std', var 2/fan_in
                sd = math.sqrt(2.0 / fc.in_features)
                assert fc.weight.abs().max() <= 2 * sd / 0.87962566103423978 + 1e-6
                assert abs(float(fc.weight.std()) / sd - 1.0) < 0.1, (float(fc.weight.std()), sd)
        w = tr.embed.weight                                             # orthogonal gain 1: orthonormal rows (32 < 37)
        assert torch.allclose(w @ w.T, torch.eye(32), atol=1e-5)
        hw = tr.head.weight                                             # orthogonal head, gain 0.01 (actor) / 1 (critic)
        assert torch.allclose(hw @ hw.T, gain ** 2 * torch.eye(out), atol=1e-6 * max(1.0, gain ** 2) + 1e-9)
        assert torch.count_nonzero(tr.embed.bias) == 0 and torch.count_nonzero(tr.head.bias) == 0
    b = sb.actors[h].mlp.blocks[0]                                      # residual: x + fc2(silu(fc1(ln(x))))
    x = torch.randn(7, 32)
    with torch.no_grad():
        ref = x + b.fc2(torch.nn.functional.silu(b.fc1(torch.nn.functional.layer_norm(x, (32,), b.ln.weight, b.ln.bias, 1e-6))))
        assert torch.allclose(b(x), ref, atol=1e-6)
    ids = {id(p) for g in sb.optimizers[h].param_groups for p in g["params"]}
    assert ids == {id(p) for p in sb.actors[h].parameters()} | {id(p) for p in sb.critics[h].parameters()}
assert sb.critic_optimizers == []
x = torch.as_tensor(rng.standard_normal((256, D)).astype(np.float32))
with torch.no_grad():
    m0 = sb.actors[0].mean(x)
assert float(m0.abs().max()) < 0.1, float(m0.abs().max())                 # initial means ~0 (gain 0.01 head)
for it in range(2):
    sb.begin_rollout()
    rollout(sb, rng, T)
    info = sb.update()
for k in ("loss/value_right", "loss/surrogate_left", "train/kl_left"):
    assert k in info and np.isfinite(info[k]), (k, info.get(k))
with tempfile.TemporaryDirectory() as d:
    sb.save(d)
    sb2 = make(critic=SB, actor=SB, seed=11)
    sb2.load(d)
    with torch.no_grad():
        assert torch.equal(sb2.actors[1].mean(x), sb.actors[1].mean(x)) and torch.equal(sb2.critics[0](x), sb.critics[0](x))
n_sb = sum(p.numel() for p in sb.actors[0].mlp.parameters())
print(f"12. simba trunk: embed {FLAT + lat}->32, 2 x [LN, 32->128 SiLU ->32, +skip], LN, head; SimBa init (orthogonal embed, "
      f"truncated He blocks, head gain 0.01 / 1), initial |mean| max {float(m0.abs().max()):.4f}, joint Adam, 2 updates "
      f"finite, save/load exact ({n_sb} trunk params)  OK")
print("ALL OK")
