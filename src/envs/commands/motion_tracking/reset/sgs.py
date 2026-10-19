"""Success Guided Sampling (SGS; Zhang et al., arXiv 2610.12465, sgs-rl.pdf Sec. 3.2) of the training start state.

Configuration i = (clip c, start-frame bin b): the start frame is drawn uniformly inside bin b of [0, 0.99 T_c), b in
[0, num_bins), so N = num_clips * num_bins and equal weights reproduce the uniform sampler (uniform clip, uniform start
frame in [0, 0.99 T_c)). In the Isaac per-env-object scene an env may only draw its object's configurations: P is
restricted to them per env (the clip mix stays fixed by the scene; SGS re-weights the start-frame bins). Each configuration keeps its last ``history`` Boolean episode outcomes; p_i = their mean (0 with
no outcome yet). Weights (paper Eq. 1-2): w_i = (p_i + eps)^(kappa t) (1 - p_i + eps)^(kappa (1 - t)),
l_i = log(w_i + eps), P(i) = softmax(l / T). Every reset draws i.i.d. from P.

Outcome of a training episode (``success``):
  - "completion": the episode ended by a time-out-type termination (``motion_clip_ended`` / ``time_out``) with no other
    termination, i.e. it reached the end of the clip;
  - "tracking": completion AND the eval's SR MT-Obj criterion at k = 1 on the frames it covered: for every hand's
    object, the mean position error <= obj_trans (m) and the mean rotation error <= obj_rot_deg, means over the
    episode's steps after the first ``grace_steps`` (as evaluation.callbacks.tracking_performance). An episode with no
    step after the grace window counts by completion alone.
Only episodes that started from an SGS draw and ended inside a training env step are recorded; resets forced by the
eval (or any reset of a running episode) are not outcomes.
"""

from __future__ import annotations

import math
from typing import Any

import torch


class SuccessGuidedSampler:
    def __init__(self, num_frames: torch.Tensor, num_envs: int, num_sides: int, cfg: Any, device: str | torch.device):
        self.device = torch.device(device)
        self.num_frames = num_frames.to(self.device).long()            # (C,) frames per clip
        self.num_clips = int(self.num_frames.numel())
        self.num_bins = int(cfg.num_bins)
        self.n = self.num_clips * self.num_bins
        self.history = int(cfg.history)
        self.target = float(cfg.target)
        self.kappa = float(cfg.kappa)
        self.temperature = float(cfg.temperature)
        self.eps = float(cfg.eps)
        self.success_mode = str(cfg.success)
        if self.success_mode not in ("completion", "tracking"):
            raise ValueError(f"sampling.sgs.success must be 'completion' or 'tracking', got {self.success_mode!r}")
        self.grace_steps = int(cfg.grace_steps)
        self.obj_trans = float(cfg.obj_trans)
        self.obj_rot = math.radians(float(cfg.obj_rot_deg))
        if not 0.0 <= self.target <= 1.0 or self.kappa < 0 or self.temperature <= 0 or self.eps <= 0:
            raise ValueError(f"bad SGS parameters: t={self.target} kappa={self.kappa} T={self.temperature} eps={self.eps}")

        # outcome ring buffers
        self.outcomes = torch.zeros(self.n, self.history, dtype=torch.bool, device=self.device)
        self.counts = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.ptr = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.total_recorded = 0
        # running episode of each env: its configuration (-1 = not an SGS episode) and error sums
        self.env_cfg = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.ep_t = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.n_valid = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.sum_trans = torch.zeros(num_envs, num_sides, device=self.device)
        self.sum_rot = torch.zeros(num_envs, num_sides, device=self.device)

    # ── distribution ─────────────────────────────────────────────────────

    def success_rates(self) -> torch.Tensor:
        return self.outcomes.sum(dim=1).float() / self.counts.clamp_min(1).float()

    def probs(self) -> torch.Tensor:
        p = self.success_rates()
        w = (p + self.eps).pow(self.kappa * self.target) * (1.0 - p + self.eps).pow(self.kappa * (1.0 - self.target))
        return torch.softmax(torch.log(w + self.eps) / self.temperature, dim=0)

    def draw(self, env_ids: torch.Tensor, allowed: torch.Tensor | None = None,
             allowed_count: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """New configuration per env: returns (clip ids, start frames) and starts its episode bookkeeping.
        ``allowed`` (n, K) / ``allowed_count`` (n,): per-env configuration ids it may draw (padded; the Isaac per-env-object
        scene pins each env to its object's clips) -> P restricted to them and renormalized."""
        pr = self.probs()
        if allowed is None:
            idx = torch.multinomial(pr, len(env_ids), replacement=True)
        else:
            k = torch.arange(allowed.shape[1], device=self.device)[None, :]
            w = torch.where(k < allowed_count[:, None], pr[allowed], torch.zeros((), device=self.device))
            idx = allowed.gather(1, torch.multinomial(w, 1)).squeeze(1)
        clip = idx // self.num_bins
        b = (idx % self.num_bins).float()
        u = torch.rand(len(env_ids), device=self.device)
        start = (((b + u) / self.num_bins) * 0.99 * self.num_frames[clip].float()).long()
        self.env_cfg[env_ids] = idx
        self.ep_t[env_ids] = 0
        self.n_valid[env_ids] = 0
        self.sum_trans[env_ids] = 0.0
        self.sum_rot[env_ids] = 0.0
        return clip, start

    def invalidate(self, env_ids: torch.Tensor | None = None) -> None:
        """Running episodes stop counting (eval takeover)."""
        if env_ids is None:
            self.env_cfg.fill_(-1)
        else:
            self.env_cfg[env_ids] = -1

    # ── per-step accumulation and outcomes ───────────────────────────────

    def accumulate(self, err_trans: torch.Tensor, err_rot: torch.Tensor) -> None:
        """err_*: (B, S) per-side object position (m) / rotation (rad) errors of the current state."""
        if self.success_mode != "tracking":
            return
        live = self.env_cfg >= 0
        w = (live & (self.ep_t >= self.grace_steps)).float()
        self.sum_trans += err_trans * w[:, None]
        self.sum_rot += err_rot * w[:, None]
        self.n_valid += w.long()
        self.ep_t += live.long()

    def episode_success(self, env_ids: torch.Tensor, completed: torch.Tensor) -> torch.Tensor:
        if self.success_mode == "completion":
            return completed
        n = self.n_valid[env_ids].clamp_min(1).float()[:, None]
        ok = ((self.sum_trans[env_ids] / n <= self.obj_trans) & (self.sum_rot[env_ids] / n <= self.obj_rot)).all(dim=1)
        return completed & (ok | (self.n_valid[env_ids] == 0))

    def record(self, env_ids: torch.Tensor, completed: torch.Tensor) -> torch.Tensor:
        """Record the outcome of the episodes of env_ids that were SGS episodes; returns their successes."""
        cfg = self.env_cfg[env_ids]
        keep = cfg >= 0
        if not bool(keep.any()):
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        succ = self.episode_success(env_ids[keep], completed[keep])
        c = cfg[keep]
        order = torch.argsort(c, stable=True)
        c_s, s_s = c[order], succ[order]
        i = torch.arange(len(c_s), device=self.device)
        first = torch.ones_like(c_s, dtype=torch.bool)
        first[1:] = c_s[1:] != c_s[:-1]
        rank = i - torch.cummax(torch.where(first, i, torch.zeros_like(i)), dim=0).values
        self.outcomes[c_s, (self.ptr[c_s] + rank) % self.history] = s_s
        k = torch.bincount(c_s, minlength=self.n)
        self.ptr = (self.ptr + k) % self.history
        self.counts = torch.clamp(self.counts + k, max=self.history)
        self.total_recorded += int(len(c_s))
        self.env_cfg[env_ids[keep]] = -1
        return succ

    # ── logging / checkpoint ─────────────────────────────────────────────

    def stats(self) -> dict[str, float]:
        p, pr = self.success_rates(), self.probs()
        seen = self.counts > 0
        ent = float(-(pr * torch.log(pr.clamp_min(1e-30))).sum()) / math.log(self.n)
        return {
            "SGS/frac_seen": float(seen.float().mean()),
            "SGS/mean_success": float(p[seen].mean()) if bool(seen.any()) else 0.0,
            "SGS/frac_mid_0.2_0.8": float(((p > 0.2) & (p < 0.8) & seen).float().mean()),
            "SGS/entropy_norm": ent,
            "SGS/max_prob_x_N": float(pr.max()) * self.n,
            "SGS/episodes_recorded": float(self.total_recorded),
        }

    def state_dict(self) -> dict[str, Any]:
        return {"outcomes": self.outcomes.cpu(), "counts": self.counts.cpu(), "ptr": self.ptr.cpu(),
                "total_recorded": self.total_recorded, "num_bins": self.num_bins, "num_clips": self.num_clips,
                "history": self.history}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for k in ("num_bins", "num_clips", "history"):
            if int(state[k]) != int(getattr(self, k)):
                raise ValueError(f"SGS state mismatch on {k}: saved {state[k]} vs {getattr(self, k)}")
        self.outcomes.copy_(state["outcomes"].to(self.device))
        self.counts.copy_(state["counts"].to(self.device))
        self.ptr.copy_(state["ptr"].to(self.device))
        self.total_recorded = int(state["total_recorded"])
