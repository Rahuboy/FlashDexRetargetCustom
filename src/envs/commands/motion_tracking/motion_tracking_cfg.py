from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from mjlab.managers import CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from .motion_tracking import MotionTrackingCommand


@dataclass(kw_only=True)
class SgsCfg:
    """Success Guided Sampling (reset/sgs.py); used when ``MotionSamplingCfg.mode == "sgs"``."""

    num_bins: int = 32          # start-frame bins per clip: configurations = clips x bins
    history: int = 100          # outcomes kept per configuration (H)
    target: float = 0.5         # target success rate t
    kappa: float = 1.0          # concentration
    temperature: float = 2.0    # softmax temperature T
    eps: float = 1e-4
    success: Literal["completion", "tracking"] = "tracking"
    grace_steps: int = 15       # tracking: = the eval's TrackingPerformance grace_steps
    obj_trans: float = 0.03     # tracking: = the eval's SR@1 obj_trans (m)
    obj_rot_deg: float = 30.0   # tracking: = the eval's SR@1 obj_rot_deg


@dataclass(kw_only=True)
class MotionSamplingCfg:
    mode: Literal["uniform", "start", "sgs"] = "uniform"
    start_frame: int = 0
    sgs: SgsCfg = field(default_factory=SgsCfg)


@dataclass(kw_only=True)
class HandResetCfg:
    """Robot-side reset: warm-start noise around the reference hand state."""

    joint_position_range: tuple[float, float] = (0.0, 0.0)
    zero_init_vel: bool = False
    noise_to_initial_level: float = 1.0
    init_noise_scale: dict[str, float] = field(
        default_factory=lambda: {
            "wrist_trans": 0.01,  # m
            "wrist_rot_deg": 10.0,  # deg
            "finger_range_frac": 0.125,  # of joint range
            "wrist_trans_vel": 0.01,  # m/s
            "wrist_rot_vel": 0.01,  # rad/s
            "finger_vel": 0.1,  # rad/s, replaces the reference velocity
        }
    )


@dataclass(kw_only=True)
class ObjectSdfCfg:
    grid_extent: float = 0.30
    grid_n: int = 128


@dataclass(kw_only=True)
class XfrcGainsCfg:
    """PD gain pair (linear + rotational)."""
    
    pos: float = 0.0
    rot: float = 0.0


@dataclass(kw_only=True)
class ObjectXfrcCfg:
    """xfrc soft-PD object assist."""

    kp: XfrcGainsCfg = field(default_factory=XfrcGainsCfg)
    kd: XfrcGainsCfg = field(default_factory=XfrcGainsCfg)
    omega_rot: float = 0.0
    zeta_rot: float = 1.0
    vel_feedforward: bool = True
    rot_inertia_scale: bool = True
    implicit_damping: bool = False
    contact_scale: float = 1.0


@dataclass(kw_only=True)
class ObjectSettleCfg:
    """Post-reset assist settling window (video_to_data / CHORD: flash_chord ReferenceReset voc_decay_steps 20, step
    mode; IsaacLab v2d virtual_object_control_decay_steps 20). After a TRAINING reset to a start frame > 0, the xfrc
    assist runs at these full gains for ``steps`` env steps (in place of the pin_objects gains, or alone when
    pin_objects is off); with ``hold_reference`` the reference stays at the reset frame for those steps and advances
    after the last one. A reset to frame 0 and every eval reset get no window. Defaults = xfrc_curriculum_adaptive's
    level 0 (kp_init 80, kv_init 5, rot_stiffness_mult 96 -> rotation kp x96 / kv x sqrt(96), force_range 50)."""

    enable: bool = False
    steps: int = 20
    hold_reference: bool = True
    kp_pos: float = 80.0
    kv_pos: float = 5.0
    rot_mult: float = 96.0
    force_range: float = 50.0


@dataclass(kw_only=True)
class ObjectPerturbCfg:
    """Markov-burst object wrench for grasp robustness; never in eval."""

    enabled: bool = False
    force_scale: float = 0.0
    torque_scale: float = 0.0
    start_prob: float = 0.05
    continue_prob: float = 0.95
    gate_lag_s: float = 0.5
    height_thresh: float = 0.03
    require_assist_decayed: bool = True


@dataclass(kw_only=True)
class SupportDiskCfg:
    """CHORD kinematic table disks per motion."""

    enable: bool = False
    count: int = 4
    radii: list[float] = field(default_factory=lambda: [0.08, 0.15, 0.25, 0.4])
    height: float = 0.01
    radius_margin: float = 0.0
    park: list[float] = field(default_factory=lambda: [0.0, 2.0, 0.5])
    park_stride: float = 0.6
    friction: list[float] = field(default_factory=lambda: [1.0, 0.005, 0.0001])
    rgba: list[float] = field(default_factory=lambda: [0.55, 0.55, 0.6, 1.0])
    entity_names: list | None = None


@dataclass(kw_only=True)
class ObjectCfg:
    
    # Composition-derived (filled at scene build, not yaml).
    entity_names: dict[str, str | list[str]] | None = None
    mesh_paths: dict[str, str | list[str]] | None = None
    mesh_scales: dict[str, float | list[float]] | None = None
    shared_slots: list[bool] | None = None
    contact_sensor_mode: Literal["raw", "native"] = "raw"
    raw_sensor_specs: list | None = None
    swap_spec: dict | None = None

    # Yaml-tunable.
    sdf: ObjectSdfCfg = field(default_factory=ObjectSdfCfg)
    zero_init_vel: bool = False
    pin_objects: bool = False
    pin_mode: Literal["xfrc", "none"] = "xfrc"
    xfrc: ObjectXfrcCfg = field(default_factory=ObjectXfrcCfg)
    settle: ObjectSettleCfg = field(default_factory=ObjectSettleCfg)
    perturb: ObjectPerturbCfg = field(default_factory=ObjectPerturbCfg)
    static_offset: "StaticOffsetCfg" = field(default_factory=lambda: StaticOffsetCfg())
    support_disks: SupportDiskCfg = field(default_factory=SupportDiskCfg)


@dataclass(kw_only=True)
class StaticOffsetCfg:

    enable: bool = False
    prob: float = 0.5
    min_cm: float = 1.5
    max_cm: float = 3.5
    static_window_steps: int = 60
    static_eps_cm: float = 2.0
    away_from_other_hand: bool = True
    sides: tuple[str, ...] = ("left",)


@dataclass(kw_only=True)
class RefNoiseCfg:

    enable: bool = False
    obj_trans: float = 0.02  # m
    obj_rot_deg: float = 6.0
    obj_lin_vel: float = 0.05  # m/s
    obj_ang_vel_deg: float = 20.0
    wrist_trans: float = 0.02  # m
    wrist_rot_deg: float = 6.0
    wrist_lin_vel: float = 0.05  # m/s
    wrist_ang_vel_deg: float = 20.0
    depenetrate: bool = True
    depen_slack: float = 0.003  # m


@dataclass(kw_only=True)
class MotionTrackingCommandCfg(CommandTermCfg):

    motion_file: str
    entity_name: str
    finger_names: tuple[str, ...]
    joint_names: dict[str, list[str]]
    site_names: dict[str, dict]
    body_mapping: dict

    sampling: MotionSamplingCfg = field(default_factory=MotionSamplingCfg)
    hand: HandResetCfg = field(default_factory=HandResetCfg)
    object: ObjectCfg = field(default_factory=ObjectCfg)
    ref_noise: RefNoiseCfg = field(default_factory=RefNoiseCfg)

    # Debug-vis overlays (need debug_vis=True, except viz_object_collision).
    viz_robot_ghost: bool = False
    viz_object_ghost: bool = False
    viz_human_keypoint: bool = False
    viz_object_collision: bool = False
    viz_contact_wrench: bool = False

    def build(self, env: ManagerBasedRlEnv) -> MotionTrackingCommand:
        from .motion_tracking import MotionTrackingCommand

        return MotionTrackingCommand(self, env)
