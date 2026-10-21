"""Multi-GPU agents.hand_ppo (common.distributed) on CPU with the gloo backend, 2 ranks, random data (no simulator):
python tests/test_hand_ppo_ddp.py  (from the repo root, FlashDexRetarget env).

Reference: one process with 2N envs and the same weights, updated on a window whose env columns are the two ranks' windows
side by side. With num_mini_batches 1 each rank's minibatch is its whole window, so the 2-rank update must equal the
reference update up to float summation order: the gradients entering the clip (averaged over ranks), the advantage, value
and observation normalization, the KL and learning rates, the adv/ panel and the weights. Both ranks must end bitwise
equal, rank 0's weights must win at construction, and impl=rl_games / critic.arch=flashsac must be refused."""
import os
import socket
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agents.hand_ppo import HandPPOAgent  # noqa: E402

N, T, K, FLAT, PER_STEP = 32, 6, 10, 21, 8   # envs per rank, window rows, future steps
D = FLAT + K * PER_STEP
SIDES, IDS, H = ["right", "left"], [[0, 1, 2, 6], [3, 4, 5, 7]], 2
TRAJ = {"enable": True, "aggregation": "conv1d", "n_future": K, "n_sides": 1, "latent_dim": 16, "hidden_dim": 16,
        "num_layers": 2, "window_dims": [K * PER_STEP]}
BUFFERS = ("obs", "actions", "mu", "sigma", "rewards", "values", "logp", "dones", "timeouts", "probs")
CASES = {"gaussian": {}, "categorical": {"actor": {"dist": "categorical"}}}

# the gradients that enter each clip (after the cross-rank average), in call order
GRADS: list[torch.Tensor] = []
_clip = torch.nn.utils.clip_grad_norm_


def _recording_clip(params, *args, **kwargs):
    params = list(params)
    GRADS.append(torch.cat([p.grad.reshape(-1) for p in params if p.grad is not None]).clone())
    return _clip(params, *args, **kwargs)


torch.nn.utils.clip_grad_norm_ = _recording_clip


def make(n_envs, extra=None, seed=0):
    cfg = OmegaConf.load(ROOT / "config/runner/hand_ppo.yaml").runner.agent
    cfg = OmegaConf.merge(cfg, {"device": "cpu", "num_steps_per_env": T, "carry_steps": 0,
                                "actor": {"hidden_dims": [32, 32]},
                                "critic": {"hidden_dims": [32, 32], "lr_decay_updates": 40},
                                "algorithm": {"num_mini_batches": 1, "num_learning_epochs": 2},
                                "empty_cache_after_update": False})
    if extra:
        cfg = OmegaConf.merge(cfg, extra)
    torch.manual_seed(seed)
    return HandPPOAgent(cfg, D, SIDES, IDS, n_envs, TRAJ, "cpu")


def state(ag):
    return {f"{m}.{k}": v.detach().clone() for m in ("actors", "critics", "value_rms")
            for k, v in getattr(ag, m).state_dict().items()}


def reference(path):
    """One process, 2N envs: a window from act / observe, then one update; saves everything the ranks compare to."""
    out = {}
    for name, extra in CASES.items():
        ag = make(2 * N, extra)
        init = state(ag)
        rng = np.random.default_rng(1)
        obs = rng.standard_normal((2 * N, D)).astype(np.float32)
        for _ in range(ag.begin_rollout()):
            ag.act(obs)
            obs = rng.standard_normal((2 * N, D)).astype(np.float32)
            ag.observe({"reward": rng.standard_normal((2 * N, H)).astype(np.float32), "terminated": rng.random(2 * N) < 0.1,
                        "truncated": rng.random(2 * N) < 0.05, "next_observation": obs})
        window = {k: getattr(ag, k).clone() for k in BUFFERS if getattr(ag, k) is not None}
        window["last_obs"] = ag._last_obs.clone()
        GRADS.clear()
        info = ag.update()
        out[name] = {"init": init, "window": window, "info": info, "state": state(ag), "lrs": list(ag.lrs),
                     "grads": list(GRADS)}
    torch.save(out, path)


def worker(rank, path, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=2)
    ref = torch.load(path, weights_only=False)
    cols = slice(rank * N, (rank + 1) * N)
    for name, extra in CASES.items():
        r = ref[name]
        ag = make(N, extra, seed=rank)                   # different init per rank: the broadcast must give rank 0's
        assert all(torch.equal(v, r["init"][k]) for k, v in state(ag).items()), (name, "construction broadcast")
        for k, v in r["window"].items():
            if k == "last_obs":
                ag._last_obs = v[cols].clone()
            else:
                getattr(ag, k).copy_(v[:, cols])
        ag.step = ag.window
        GRADS.clear()
        info = ag.update()
        # gradients: the first optimizer step of each hand (identical weights) tight, the second loose (it follows
        # Adam steps that already differ by summation order)
        assert len(GRADS) == len(r["grads"]) == 2 * 2 * H, len(GRADS)
        for i, (g, g_ref) in enumerate(zip(GRADS, r["grads"])):
            tol = dict(rtol=1e-5, atol=1e-7) if i < 2 * H else dict(rtol=1e-3, atol=1e-6)
            assert torch.allclose(g, g_ref, **tol), (name, i, float((g - g_ref).abs().max()))
        for k, v in state(ag).items():
            assert torch.allclose(v.double(), r["state"][k].double(), rtol=1e-4, atol=1e-5), (name, k)
        assert ag.lrs == r["lrs"], (name, ag.lrs, r["lrs"])
        for k in [k for k in r["info"] if k.startswith("adv/") or k.startswith("train/kl_")]:
            assert np.isclose(info[k], r["info"][k], rtol=1e-5, atol=1e-9), (name, k, info[k], r["info"][k])
        # both ranks hold the same networks, bit for bit
        mine = torch.cat([v.double().reshape(-1) for v in state(ag).values()])
        rank0 = mine.clone()
        dist.broadcast(rank0, src=0)
        assert torch.equal(mine, rank0), (name, "ranks differ")
    for extra in ({"algorithm": {"impl": "rl_games"}},
                  {"critic": {"arch": "flashsac", "hidden_dim": 32, "lr_decay_updates": 40}}):
        try:
            make(N, extra)
        except NotImplementedError:
            continue
        raise AssertionError(f"multi-GPU accepted {extra}")
    dist.destroy_process_group()


if __name__ == "__main__":
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "reference.pt")
        reference(path)
        mp.spawn(worker, args=(path, port), nprocs=2)
    print("2 ranks (gloo) == one process on the union (gaussian, categorical): construction broadcast, gradients, "
          "normalizers, KL / learning rates, adv/, weights; ranks bitwise equal; rl_games / flashsac refused  OK")
