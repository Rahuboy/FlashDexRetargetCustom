"""Gate: agents.hand_ppo with algorithm.impl=rl_games reproduces DexMachina's rl_games PPO epoch on identical data.

python tests/rlgames_equivalence.py <rl_games root (DexMachina's rl_games_dex)>   (from the repo root, FlashDexRetarget env)

For each case the agent is deep-copied just before its update; the copy is then updated by a reference built from
rl_games' own code: torch_ext.policy_kl / mean_list / apply_masks, common_losses.actor_loss / critic_loss,
schedulers.AdaptiveScheduler, datasets.PPODataset (+ update_mu_sigma), running_mean_std.RunningMeanStd,
models.ModelA2CContinuousLogStd's neglogp, and the literal source of A2CBase.discount_values, swap_and_flatten01 and
A2CAgent.bound_loss (compiled from the files: their modules import gym, absent here), following
ContinuousA2CBase.train_epoch / prepare_dataset and A2CAgent.calc_gradients (schedule_type standard, permute off,
truncate_grads). Prints the largest parameter difference per case, the LR trajectories and the value-RMS states;
exits non-zero on a mismatch."""
import ast
import copy
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
RLG = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(RLG))
from agents.hand_ppo import HandPPOAgent  # noqa: E402
from rl_games.algos_torch import torch_ext  # noqa: E402
from rl_games.algos_torch.models import ModelA2CContinuousLogStd  # noqa: E402
from rl_games.algos_torch.running_mean_std import RunningMeanStd  # noqa: E402
from rl_games.common import common_losses, datasets, schedulers  # noqa: E402


def rlg_source(rel: str, name: str, cls: str | None = None):
    """Compile one function / method of an rl_games file from its source (no module import)."""
    path = RLG / "rl_games" / rel
    src = path.read_text()
    tree = ast.parse(src)
    scope = tree.body
    if cls is not None:
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"torch": torch, "np": np}
    exec(compile(textwrap.dedent(ast.get_source_segment(src, node)), str(path), "exec"), ns)
    return ns[name]


discount_values = rlg_source("common/a2c_common.py", "discount_values", "A2CBase")
swap_and_flatten01 = rlg_source("common/a2c_common.py", "swap_and_flatten01")
bound_loss = rlg_source("algos_torch/a2c_continuous.py", "bound_loss", "A2CAgent")
neglogp = ModelA2CContinuousLogStd.Network.neglogp

N, T, K, FLAT, PER_STEP = 64, 6, 10, 21, 8
D = FLAT + K * PER_STEP
SIDES, IDS, H = ["right", "left"], [[0, 1, 2, 6], [3, 4, 5, 7]], 2
TRAJ = {"enable": True, "aggregation": "conv1d", "n_future": K, "n_sides": 1, "latent_dim": 16, "hidden_dim": 16,
        "num_layers": 2, "window_dims": [K * PER_STEP]}


def make(carry: int, clip_value: bool, epochs: int, seed: int) -> HandPPOAgent:
    cfg = OmegaConf.load(ROOT / "config/runner/hand_ppo.yaml").runner.agent
    rl = OmegaConf.load(ROOT / "config/runner/hand_ppo_rlgames.yaml").runner.agent
    cfg = OmegaConf.merge(cfg, rl, {"device": "cpu", "num_steps_per_env": T, "carry_steps": carry,
                                    "actor": {"hidden_dims": [32, 16]}, "critic": {"hidden_dims": [32, 16], "lr_decay_updates": 40},
                                    "algorithm": {"num_learning_epochs": epochs, "clip_value": clip_value},
                                    "empty_cache_after_update": False})
    torch.manual_seed(seed)
    return HandPPOAgent(cfg, D, SIDES, IDS, N, TRAJ, "cpu")


def rollout(ag: HandPPOAgent, rng: np.random.Generator, scale: float) -> None:
    n = ag.begin_rollout()
    obs = rng.standard_normal((N, D)).astype(np.float32)
    for _ in range(n):
        ag.act(obs)
        obs = rng.standard_normal((N, D)).astype(np.float32)
        ag.observe({"reward": (scale * rng.standard_normal((N, H))).astype(np.float32),
                    "terminated": rng.random(N) < 0.05, "truncated": rng.random(N) < 0.05, "next_observation": obs})


def reference_update(ref: HandPPOAgent, clip_value: bool) -> list[list[float]]:
    """rl_games' prepare_dataset + train_epoch on ``ref``'s window, one hand at a time; returns the LR per pass."""
    T_, N_ = ref.window, ref.num_envs
    with torch.no_grad():
        last_v = ref._values(ref._last_obs)                 # rl_games get_values (de-normalized), pre-update stats
    # rl_games' dones[t] = done before step t (ours: after step t); fdones = after the last step
    mb_fdones = torch.zeros(T_, N_)
    mb_fdones[1:] = ref.dones[:-1].float()
    fdones = ref.dones[-1].float()
    algo = SimpleNamespace(horizon_length=T_, gamma=ref.gamma, tau=ref.lam, carry_adv_horizon=0)
    lr_trace = []
    for h in range(ref.H):
        ids = ref.ids[h]
        mb_values = ref.values[..., h:h + 1]
        mb_rewards = ref.rewards[..., h:h + 1]                # shaped: x0.1 + gamma V time-out bootstrap
        mb_advs = discount_values(algo, fdones, last_v[:, h:h + 1], mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values
        values = swap_and_flatten01(mb_values)
        returns = swap_and_flatten01(mb_returns)
        advantages = returns - values                          # prepare_dataset
        rms = RunningMeanStd((1,))
        rms.running_mean.copy_(ref.value_rms[h].running_mean.reshape(1))
        rms.running_var.copy_(ref.value_rms[h].running_var.reshape(1))
        rms.count.copy_(ref.value_rms[h].count)
        rms.train()
        values = rms(values)
        returns = rms(returns)
        rms.eval()
        advantages = torch.sum(advantages, axis=1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ds = datasets.PPODataset(T_ * N_, T_ * N_ // ref.num_mb, False, False, "cpu", 1)
        ds.update_values_dict({
            "old_values": values, "old_logp_actions": swap_and_flatten01(-ref.logp[..., h]), "advantages": advantages,
            "returns": returns, "actions": swap_and_flatten01(ref.actions[..., ids]), "obs": swap_and_flatten01(ref.obs),
            "dones": None, "rnn_states": None, "rnn_masks": None,
            "mu": swap_and_flatten01(ref.mu[..., ids]).clone(), "sigma": swap_and_flatten01(ref.sigma[..., ids]).clone()})
        actor, critic, opt = ref.actors[h], ref.critics[h], ref.optimizers[h]
        params = list(actor.parameters()) + list(critic.parameters())
        sched = schedulers.AdaptiveScheduler(ref.desired_kl)
        lr = ref.lrs[h]
        trace = []
        for _ in range(ref.epochs):
            ep_kls = []
            for i in range(len(ds)):
                b = ds[i]
                mu = actor.mean(b["obs"])
                logstd = mu * 0.0 + actor.log_std_param          # rl_games continuous_a2c_logstd, fixed_sigma
                sigma = torch.exp(logstd)
                action_log_probs = neglogp(None, b["actions"], mu, sigma, logstd)
                v = critic(b["obs"]).unsqueeze(1)
                entropy = torch.distributions.Normal(mu, sigma, validate_args=False).entropy().sum(dim=-1)
                a_loss = common_losses.actor_loss(b["old_logp_actions"], action_log_probs, b["advantages"], True, ref.clip_param)
                c_loss = common_losses.critic_loss(None, b["old_values"], v, ref.clip_param, b["returns"], clip_value)
                b_loss = bound_loss(SimpleNamespace(bounds_loss_coef=ref.bounds_coef), mu)
                losses, _ = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], None)
                a_l, c_l, e_l, b_l = losses
                loss = a_l + 0.5 * c_l * 4.0 - e_l * ref.ent_coef + b_l * ref.bounds_coef
                for p in params:
                    p.grad = None
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, ref.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    kl = torch_ext.policy_kl(mu.detach(), sigma.detach(), b["mu"], b["sigma"], True)
                ep_kls.append(kl)
                ds.update_mu_sigma(mu.detach(), sigma.detach())
            av = torch_ext.mean_list(ep_kls)
            lr, _ = sched.update(lr, 0.0, 0, 0, av.item())
            for g in opt.param_groups:
                g["lr"] = lr
            trace.append(lr)
        ref.lrs[h] = lr
        ref.value_rms[h].running_mean.copy_(rms.running_mean.reshape(()))
        ref.value_rms[h].running_var.copy_(rms.running_var.reshape(()))
        ref.value_rms[h].count.copy_(rms.count)
        lr_trace.append(trace)
    # rsl_rl-style observation normalizers after the update, fresh rows only (as the agent; not part of rl_games)
    fresh = ref.obs[ref._carried_now:].flatten(0, 1)
    with torch.no_grad():
        for h in range(ref.H):
            ref.actors[h].update_normalization(fresh)
            ref.critics[h].update_normalization(fresh)
    return lr_trace


def max_diff(a: torch.nn.Module, b: torch.nn.Module) -> tuple[float, float]:
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    d = max(float((sa[k].double() - sb[k].double()).abs().max()) for k in sa)
    s = max(float(sa[k].double().abs().max()) for k in sa)
    return d, s


ok = True
for name, carry, clip_value, epochs, scale in (("ppo", 0, False, 6, 1.0), ("ppo clip_value", 0, True, 6, 1.0),
                                                ("ragger carry 3", 3, False, 6, 1.0), ("ppo big rewards", 0, False, 14, 30.0)):
    ag = make(carry, clip_value, epochs, seed=5)
    rng = np.random.default_rng(1)
    for it in range(3):                                         # the last update is compared (carry active from 2)
        rollout(ag, rng, scale)
        if it < 2:
            ag.update()
    ref = copy.deepcopy(ag)
    info = ag.update()
    lr_trace = reference_update(ref, clip_value)
    da, sa = max_diff(ag.actors, ref.actors)
    dc, sc = max_diff(ag.critics, ref.critics)
    dr, _ = max_diff(ag.value_rms, ref.value_rms)
    lr_ok = ag.lrs == ref.lrs
    good = da < 1e-5 * max(1.0, sa) and dc < 1e-5 * max(1.0, sc) and dr < 1e-9 and lr_ok
    ok &= good
    print(f"{name:16s} carried {int(info['train/carried_rows'])}: max |dparam| actor {da:.2e} (scale {sa:.2f}), critic "
          f"{dc:.2e}, value-RMS {dr:.1e}; LR per pass {[f'{x:.2e}' for x in lr_trace[0]]} / final ours {ag.lrs} "
          f"ref {ref.lrs}; kl {info['train/kl_right']:.2e} kl_rollout {info['train/kl_rollout_right']:.2e}  "
          f"{'MATCH' if good else 'MISMATCH'}")
print("ALL MATCH" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
