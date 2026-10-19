"""MotionTrackingCommand mixin: per-env clip / frame state and the uniform start frame."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..motion_tracking_cfg import MotionTrackingCommandCfg


class SamplingMixin:
    def _init_sampling(self, cfg: MotionTrackingCommandCfg) -> None:
        self.motion_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.motion_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # Success Guided Sampling of (clip, start-frame bin) when sampling.mode == "sgs" (reset/sgs.py).
        self._sgs = None
        self._sgs_wrap = False
        if cfg.sampling.mode == "sgs":
            from .sgs import SuccessGuidedSampler

            self._sgs = SuccessGuidedSampler(
                self.motion_lib._motion_num_frames, self.num_envs, len(self._side_list), cfg.sampling.sgs, self.device
            )
            print(f"[sgs] {self._sgs.n} configurations = {self._sgs.num_clips} clips x {self._sgs.num_bins} bins; "
                  f"success={self._sgs.success_mode}", flush=True)

    def _sgs_record(self, env_ids: torch.Tensor) -> None:
        """Outcomes of the SGS episodes of env_ids that just ended: a training-step termination (time-out-type =
        completed) or, inside ``_update_command``, a clip end without a reset (completed). A reset of a running episode
        outside a step (forced by the eval / env.reset) records nothing."""
        env = self._env
        if self._sgs_wrap:
            ended = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
            completed = ended
        else:
            reset_buf = getattr(env, "reset_buf", None)
            tm = getattr(env, "termination_manager", None)
            if reset_buf is None or tm is None:
                ended = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
                completed = ended
            else:
                ended = reset_buf[env_ids] & (env.episode_length_buf[env_ids] > 0)
                completed = tm.time_outs[env_ids] & ~tm.terminated[env_ids]
        succ = self._sgs.record(env_ids[ended], completed[ended]) if bool(ended.any()) else None
        log = env.extras.setdefault("log", {})
        log.update(self._sgs.stats())
        if succ is not None and succ.numel() > 0:
            log["SGS/batch_success"] = float(succ.float().mean())

    def _sgs_accumulate(self) -> None:
        """Per-step object errors of the running SGS episodes (tracking success)."""
        if self._sgs.success_mode != "tracking" or not self.has_objects:
            return
        pre = {"right": "r", "left": "l"}
        trans = torch.stack([self.metrics[f"error_obj_trans_{pre[s]}"] for s in self._side_list], dim=-1)
        rot = torch.stack([self.metrics[f"error_obj_rot_{pre[s]}"] for s in self._side_list], dim=-1)
        self._sgs.accumulate(trans, rot)


    def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
        tr = self.motion_ids[env_ids]
        T_m = self.motion_lib._motion_num_frames[tr].float()
        self.motion_steps[env_ids] = (
            torch.rand(len(env_ids), device=self.device) * 0.99 * T_m
        ).long()
