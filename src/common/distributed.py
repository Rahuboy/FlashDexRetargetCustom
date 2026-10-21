"""Multi-GPU data parallelism for the on-policy agents: torchrun, one process per GPU.

Every rank runs its own Isaac Sim instance with its own ``num_envs`` envs (Isaac Lab's distributed mode binds rank r to GPU
``LOCAL_RANK``) and its own seed (seed + rank). The agent keeps one policy by averaging its gradients and update statistics
over every rank's rows (agents.hand_ppo); rank 0 alone logs, evaluates and checkpoints; env steps count every rank's envs.
Launch: ``NPROC=<gpus> bash scripts/train_flashdexretarget.sh ...``. Without torchrun no process group exists and every
helper below is a no-op (``world_size()`` 1, ``is_main()`` true)."""

from __future__ import annotations

import datetime

import torch
import torch.distributed as dist
import torch.nn as nn

_OPS = {"sum": dist.ReduceOp.SUM, "min": dist.ReduceOp.MIN, "max": dist.ReduceOp.MAX}


def init(device: str) -> None:
    """NCCL process group over torchrun's env:// rendezvous. The timeout is long because rank 0's evals pause it while
    the other ranks wait at their next collective."""
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))


def shutdown() -> None:
    if active():
        dist.destroy_process_group()


def active() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    return dist.get_world_size() if active() else 1


def is_main() -> bool:
    return not active() or dist.get_rank() == 0


def all_reduce_(t: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """``t`` reduced in place over every rank (``sum`` / ``mean`` / ``min`` / ``max``); returns ``t``."""
    if active():
        dist.all_reduce(t, op=_OPS["sum" if op == "mean" else op])
        if op == "mean":
            t /= dist.get_world_size()
    return t


def combine_moments(count: int, mean: torch.Tensor, var: torch.Tensor) -> tuple[float, torch.Tensor, torch.Tensor]:
    """(count, mean, unbiased variance) of every rank's rows from each rank's own (count, mean, unbiased variance);
    elementwise over ``mean`` / ``var``, accumulated in float64."""
    n = torch.tensor(float(count), dtype=torch.float64, device=mean.device)
    m, v = mean.double(), var.double()
    sums = all_reduce_(torch.cat([n.reshape(1), (n * m).reshape(-1)]))
    total, g_mean = sums[0], (sums[1:] / sums[0]).view_as(m)
    m2 = all_reduce_((n - 1) * v + n * (m - g_mean).square())
    return float(total), g_mean.to(mean.dtype), (m2 / (total - 1)).to(var.dtype)


def average_gradients_(*modules: nn.Module) -> None:
    """Average the gradients of ``modules`` over every rank in one flat all-reduce (rsl_rl's PPO.reduce_parameters)."""
    if not active():
        return
    grads = [p.grad for m in modules for p in m.parameters() if p.grad is not None]
    flat = all_reduce_(torch.cat([g.reshape(-1) for g in grads]), "mean")
    offset = 0
    for g in grads:
        g.copy_(flat[offset:offset + g.numel()].view_as(g))
        offset += g.numel()


def broadcast_(*modules: nn.Module) -> None:
    """Rank 0's parameters and buffers to every rank (in place)."""
    if active():
        for m in modules:
            for t in m.state_dict().values():
                dist.broadcast(t, src=0)


def any_(flag: bool) -> bool:
    """Whether ``flag`` is true on any rank (the same answer on every rank)."""
    if not active():
        return flag
    device = "cuda" if dist.get_backend() == "nccl" else "cpu"
    return bool(all_reduce_(torch.tensor(float(flag), device=device), "max").item())
