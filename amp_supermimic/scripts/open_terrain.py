"""Open-arena 6-level curriculum terrain and omnidirectional command integration.

Provides an ``open_terrain`` profile without modifying the upstream
``third_party/trackerlab_supermimic`` submodule:
- Level 0: large flat square arena for learning omnidirectional/yaw locomotion.
- Levels 1..5: progressively harder isotropic rolling mountain hills, low
  terraced plateaus, and mild rough bumps, smoothly blended at tile seams.
- Full 6-mode omnidirectional + yaw commands (stop, forward, backward, strafe,
  in-place yaw, mixed twist) with ``CommandSim`` 6-level curriculum enabled via
  command-tracking survival success.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


OPEN_PROFILE = "open_terrain"
OPEN_FLAT_PROFILE = "open_flat"
OPEN_PROFILES = (OPEN_PROFILE, OPEN_FLAT_PROFILE)

# Adaptive velocity-command curriculum for ``open_flat`` when ``--command-range-mode full`` is used.
# All velocities are in real SI units (m/s and rad/s).
OPEN_FLAT_CURRICULUM = {
    "enabled": True,
    "update_interval_steps": 250,   # control steps (~100 iters at utd=0.4)
    "initial_alpha": 0.0,
    "step_up": 0.10,                # 10 stages from initial range to max range
    "step_down": 0.05,
    "upgrade_vel_track": 0.70,      # mean(body_velocity) / 0.20 >= 0.70
    "upgrade_yaw_track": 0.70,      # mean(yaw_rate) / 0.09 >= 0.70
    "upgrade_fall_rate": 0.10,      # falls per equivalent episode <= 10%
    "downgrade_vel_track": 0.45,    # mean(body_velocity) / 0.20 < 0.45
    "downgrade_fall_rate": 0.35,    # falls per equivalent episode >= 35%
    # Target command ranges [min, max] at alpha=0.0 -> alpha=1.0
    "fwd_vx_min": 0.20,
    "fwd_vx_start_max": 0.55,
    "fwd_vx_end_max": 2.00,
    "bwd_vx_min": 0.15,
    "bwd_vx_start_max": 0.45,
    "bwd_vx_end_max": 1.00,
    "strafe_vy_min": 0.10,
    "strafe_vy_start_max": 0.35,
    "strafe_vy_end_max": 0.75,
    "turn_wz_min": 0.30,
    "turn_wz_start_max": 0.90,
    "turn_wz_end_max": 1.60,
}

# Reward weights shared by ``open_terrain`` and ``open_flat`` (can be overridden by ``legacy`` configs).
OPEN_FLAT_REWARDS = {
    "velocity_weight": 0.20,
    "velocity_sigma": 0.12,
    "yaw_weight": 0.09,
    "yaw_sigma": 0.10,
    "alive_weight": 0.02,
    "upright_weight": -0.06,
    "clearance_weight": -0.04,
    "posture_weight": -0.0006,
    "action_rate_weight": -0.0008,
    "joint_vel_weight": -0.00002,
    "joint_limit_weight": -0.04,
    "progress_trans_weight": 0.04,
    "progress_yaw_weight": 0.02,
    "dont_wait_trans_weight": -0.05,
    "dont_wait_yaw_weight": -0.04,
}
BLOCK_HALF_EXTENT = 10.0  # in G1-normalized units (* scale); block width = 20 * scale
BLOCK_CENTERS_NORM = (
    (-20.0, -10.0),  # Level 0: flat
    (0.0, -10.0),    # Level 1: gentle rolling hills
    (20.0, -10.0),   # Level 2: moderate hills + mild roughness
    (-20.0, 10.0),   # Level 3: hills + low plateaus
    (0.0, 10.0),     # Level 4: steeper mixed hills + stepped plateaus
    (20.0, 10.0),    # Level 5: full open mountain/rough/stepped mix
)


def _smooth_envelope(dx: np.ndarray, dy: np.ndarray, half_size: float, margin: float) -> np.ndarray:
    """Raised-cosine taper to 0 along block boundaries so adjacent tiles meet smoothly."""
    wx = np.clip((half_size - np.abs(dx)) / margin, 0.0, 1.0)
    wy = np.clip((half_size - np.abs(dy)) / margin, 0.0, 1.0)
    sx = 0.5 - 0.5 * np.cos(math.pi * wx)
    sy = 0.5 - 0.5 * np.cos(math.pi * wy)
    return sx * sy


def generate_open_terrain(height: float, seed: int = 0):
    """Build a 3x2 grid of wide open blocks (Level 0 flat -> Level 5 mixed mountain)."""
    from trackerTask.supermimic.terrain import Terrain

    if not np.isfinite(height) or height <= 0:
        raise ValueError("Robot height must be finite and positive")
    scale = height / 0.76
    x_norm = np.linspace(-30.0, 30.0, 385)
    y_norm = np.linspace(-20.0, 20.0, 257)
    xx, yy = np.meshgrid(x_norm, y_norm)
    out = np.zeros_like(xx, dtype=np.float64)
    rng = np.random.default_rng(seed)

    # Low-amplitude seeded roughness grid interpolated bilinearly so heightfield has mild micro-relief.
    coarse = rng.uniform(-1.0, 1.0, size=(65, 97))
    ci = np.linspace(0, coarse.shape[0] - 1, yy.shape[0])
    cj = np.linspace(0, coarse.shape[1] - 1, xx.shape[1])
    i0 = np.clip(np.floor(ci).astype(int), 0, coarse.shape[0] - 2)
    j0 = np.clip(np.floor(cj).astype(int), 0, coarse.shape[1] - 2)
    fi = (ci - i0)[:, None]
    fj = (cj - j0)[None, :]
    rough = (
        (1 - fi) * (1 - fj) * coarse[i0[:, None], j0[None, :]]
        + fi * (1 - fj) * coarse[i0[:, None] + 1, j0[None, :]]
        + (1 - fi) * fj * coarse[i0[:, None], j0[None, :] + 1]
        + fi * fj * coarse[i0[:, None] + 1, j0[None, :] + 1]
    )

    for level, (cx, cy) in enumerate(BLOCK_CENTERS_NORM):
        strength = level / (len(BLOCK_CENTERS_NORM) - 1)
        if strength == 0.0:
            continue
        dx = xx - cx
        dy = yy - cy
        in_tile = (np.abs(dx) <= BLOCK_HALF_EXTENT) & (np.abs(dy) <= BLOCK_HALF_EXTENT)
        if not np.any(in_tile):
            continue
        env = _smooth_envelope(dx, dy, BLOCK_HALF_EXTENT, margin=1.8)
        phase = rng.uniform(-np.pi, np.pi, 4)
        theta = rng.uniform(0.0, np.pi)
        u = dx * math.cos(theta) + dy * math.sin(theta)
        v = -dx * math.sin(theta) + dy * math.cos(theta)

        # Isotropic rolling mountain ridges/valleys + secondary diagonal waves.
        hills = (
            0.19 * np.sin(0.75 * u + phase[0]) * np.cos(0.65 * v + phase[1])
            + 0.09 * np.sin(1.65 * dx + 1.25 * dy + phase[2])
            + 0.055 * np.cos(2.4 * u - 1.9 * v + phase[3])
        )
        # Smooth terraced low steps/plateaus starting from level >= 3.
        plateaus = 0.0
        if level >= 3:
            step_wave = 0.5 * (np.sin(0.85 * u + phase[1]) + np.cos(0.85 * v + phase[2]))
            # Smooth quantization into low climbable terraces (~5-11 cm for WBLWBL).
            terraced = np.tanh(3.5 * step_wave) * 0.09 + 0.05 * np.tanh(4.0 * np.sin(1.8 * step_wave))
            plateaus = ((level - 2) / 3.0) * terraced

        micro_rough = 0.035 * rough
        surface = strength * env * (hills + micro_rough) + env * plateaus
        out = np.where(in_tile, surface * scale, out)

    x_min, x_max = float(x_norm[0] * scale), float(x_norm[-1] * scale)
    y_min, y_max = float(y_norm[0] * scale), float(y_norm[-1] * scale)
    centers_y = tuple(float(cy * scale) for _, cy in BLOCK_CENTERS_NORM)
    terrain = Terrain(
        out.astype(np.float32),
        x_min,
        x_max,
        y_min,
        y_max,
        centers_y,
        float(24.0 * scale),
        seed,
        OPEN_PROFILE,
    )
    terrain.block_centers = tuple((float(cx * scale), float(cy * scale)) for cx, cy in BLOCK_CENTERS_NORM)
    return terrain


def generate_open_flat(height: float, seed: int = 0):
    """Build a pure flat open arena (locked to Level 0 flat ground)."""
    from trackerTask.supermimic.terrain import Terrain

    if not np.isfinite(height) or height <= 0:
        raise ValueError("Robot height must be finite and positive")
    scale = height / 0.76
    extent = 32.0 * scale
    return Terrain(
        np.zeros((129, 129), dtype=np.float32),
        -extent,
        extent,
        -extent,
        extent,
        (0.0,) * 6,
        24.0 * scale,
        seed,
        OPEN_FLAT_PROFILE,
    )


def open_flat_limits(c, alpha):
    a = max(0.0, min(1.0, float(alpha)))
    return (
        c["fwd_vx_start_max"] + a * (c["fwd_vx_end_max"] - c["fwd_vx_start_max"]),
        c["bwd_vx_start_max"] + a * (c["bwd_vx_end_max"] - c["bwd_vx_start_max"]),
        c["strafe_vy_start_max"] + a * (c["strafe_vy_end_max"] - c["strafe_vy_start_max"]),
        c["turn_wz_start_max"] + a * (c["turn_wz_end_max"] - c["turn_wz_start_max"]),
    )


def sample_open_flat(cfg, alpha, count, device, dtype, mode_set=None):
    import torch

    fwd_max, bwd_max, vy_max, wz_max = open_flat_limits(cfg, alpha)
    u = torch.rand(count, 4, device=device, dtype=dtype)
    if mode_set is None:
        mode = torch.randint(0, 6, (count,), device=device)
    else:
        mode_ids = {"stop": 0, "forward": 1, "backward": 2, "strafe": 3, "turn": 4, "mixed": 5, "forward_turn": 6}
        unknown = set(mode_set) - mode_ids.keys()
        if not mode_set or unknown:
            raise ValueError(f"Invalid open_flat command mode set: {sorted(unknown)}")
        allowed = torch.tensor([mode_ids[name] for name in mode_set], device=device, dtype=torch.long)
        mode = allowed[torch.randint(len(allowed), (count,), device=device)]
    commands = torch.zeros(count, 3, device=device, dtype=dtype)

    # mode 0: stop (0, 0, 0)
    forward = mode == 1
    reverse = mode == 2
    side = mode == 3
    turn = mode == 4
    mixed = mode == 5
    forward_turn = mode == 6

    commands[forward, 0] = cfg["fwd_vx_min"] + u[forward, 0] * (fwd_max - cfg["fwd_vx_min"])
    commands[reverse, 0] = -(cfg["bwd_vx_min"] + u[reverse, 0] * (bwd_max - cfg["bwd_vx_min"]))
    side_sign = torch.where(u[side, 1] < 0.5, -1.0, 1.0)
    commands[side, 1] = side_sign * (cfg["strafe_vy_min"] + u[side, 0] * (vy_max - cfg["strafe_vy_min"]))
    turn_sign = torch.where(u[turn, 2] < 0.5, -1.0, 1.0)
    commands[turn, 2] = turn_sign * (cfg["turn_wz_min"] + u[turn, 0] * (wz_max - cfg["turn_wz_min"]))
    commands[forward_turn, 0] = cfg["fwd_vx_min"] + u[forward_turn, 0] * (fwd_max - cfg["fwd_vx_min"])
    forward_turn_sign = torch.where(u[forward_turn, 1] < 0.5, -1.0, 1.0)
    commands[forward_turn, 2] = forward_turn_sign * (
        cfg["turn_wz_min"] + u[forward_turn, 2] * (wz_max - cfg["turn_wz_min"]))

    commands[mixed, 0] = -bwd_max + u[mixed, 0] * (0.85 * fwd_max + bwd_max)
    commands[mixed, 1] = (2.0 * u[mixed, 1] - 1.0) * (0.85 * vy_max)
    commands[mixed, 2] = (2.0 * u[mixed, 2] - 1.0) * (0.85 * wz_max)
    return commands, u


def update_open_flat_curriculum(self, transition, cfg, rewards):
    terms = transition["reward_terms"]
    self._vel_curriculum_steps += 1
    self._vel_curriculum_env_steps += self.num_envs
    self._vel_curriculum_vel_sum += float(terms["body_velocity"].sum().item())
    self._vel_curriculum_yaw_sum += float(terms["yaw_rate"].sum().item())
    self._vel_curriculum_dones += int(transition["done"].sum().item())
    self._vel_curriculum_falls += int(transition["episode_fall"].sum().item())

    if self._vel_curriculum_steps >= cfg["update_interval_steps"]:
        vel_max = max(float(rewards["velocity_weight"]), 1e-6)
        yaw_max = max(float(rewards["yaw_weight"]), 1e-6)
        vel_track = (self._vel_curriculum_vel_sum / max(self._vel_curriculum_env_steps, 1)) / vel_max
        yaw_track = (self._vel_curriculum_yaw_sum / max(self._vel_curriculum_env_steps, 1)) / yaw_max
        equiv_eps = max(
            float(self._vel_curriculum_dones),
            self._vel_curriculum_env_steps / float(self.max_episode_steps),
        )
        fall_rate = self._vel_curriculum_falls / max(equiv_eps, 1.0)

        old_alpha = self._vel_curriculum_alpha
        action_tag = "hold"
        if (
            vel_track >= cfg["upgrade_vel_track"]
            and yaw_track >= cfg["upgrade_yaw_track"]
            and fall_rate <= cfg["upgrade_fall_rate"]
            and old_alpha < 1.0
        ):
            self._vel_curriculum_alpha = min(1.0, round(old_alpha + cfg["step_up"], 4))
            action_tag = "UP"
        elif (
            (fall_rate >= cfg["downgrade_fall_rate"] or vel_track < cfg["downgrade_vel_track"])
            and old_alpha > 0.0
        ):
            self._vel_curriculum_alpha = max(0.0, round(old_alpha - cfg["step_down"], 4))
            action_tag = "DOWN"

        fwd_max, bwd_max, vy_max, wz_max = open_flat_limits(cfg, self._vel_curriculum_alpha)
        if action_tag != "hold":
            print(
                f"[open_flat curriculum] {action_tag}: alpha {old_alpha:.2f}->{self._vel_curriculum_alpha:.2f} "
                f"(vel_track={vel_track:.3f}, yaw_track={yaw_track:.3f}, fall_rate={fall_rate:.3f}) "
                f"fwd=[{cfg['fwd_vx_min']:.2f},{fwd_max:.2f}] bwd=[-{bwd_max:.2f},-{cfg['bwd_vx_min']:.2f}] "
                f"vy=±{vy_max:.2f} wz=±{wz_max:.2f}",
                flush=True,
            )
        self._vel_curriculum_steps = 0
        self._vel_curriculum_env_steps = 0
        self._vel_curriculum_vel_sum = 0.0
        self._vel_curriculum_yaw_sum = 0.0
        self._vel_curriculum_dones = 0
        self._vel_curriculum_falls = 0


def open_flat_reward(self, S, a, a_prev, c, r, terms=False):
    import torch
    from trackerLab.tracker_env.mdp import supermimic_command as mdp_cmd

    c = self.effective_command(c)
    p, R, _, _, q, qd = S
    scale = self.field.scale
    v_body, w_body = mdp_cmd.body_velocities(S)
    clearance = self._clearance(S)
    cmd_xy = c[..., 1:3] / scale
    vel_xy = v_body[..., :2] / scale
    ev = (vel_xy - cmd_xy).square().sum(-1)
    ew = (w_body[..., 2] - c[..., 3]).square()
    speed = torch.linalg.vector_norm(cmd_xy, dim=-1)
    translating = (speed > 0.10).to(ev.dtype)
    turning = (c[..., 3].abs() > 0.10).to(ev.dtype)
    projected = (vel_xy * cmd_xy).sum(-1) / speed.clamp_min(0.01)
    directed_yaw = w_body[..., 2] * torch.sign(c[..., 3])
    velocity_reward = r["velocity_weight"] * torch.exp(-ev / r["velocity_sigma"])
    yaw_reward = r["yaw_weight"] * torch.exp(-ew / r["yaw_sigma"])
    progress = (
        r["progress_trans_weight"] * translating * (projected / speed.clamp_min(0.01)).clamp(0.0, 1.0)
        + r["progress_yaw_weight"] * turning * (directed_yaw / c[..., 3].abs().clamp_min(0.05)).clamp(0.0, 1.0)
    )
    dont_wait = (
        r["dont_wait_trans_weight"] * translating * torch.sigmoid((0.5 * speed - projected) / 0.04)
        + r["dont_wait_yaw_weight"] * turning * torch.sigmoid((0.5 * c[..., 3].abs() - directed_yaw) / 0.08)
    )
    zero = torch.zeros_like(ev)
    values = (
        velocity_reward,
        yaw_reward,
        torch.full_like(ev, r["alive_weight"]),
        r["upright_weight"] * (1 - R[..., 2, 2]),
        r["clearance_weight"] * ((clearance - mdp_cmd.target_clearance(c)) / (0.2 * self.field.height)).square(),
        r["posture_weight"] * (q - self.default_q).square().mean(-1),
        r["action_rate_weight"] * (a - a_prev).square().mean(-1),
        r["joint_vel_weight"] * qd.square().mean(-1),
        r["joint_limit_weight"] * (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1),
        progress,
        zero,
        zero,
        dont_wait,
    )
    reward = sum(values)
    return (reward, dict(zip(mdp_cmd.TERMS, values))) if terms else reward

def patch_open_terrain(robot: str) -> None:
    """Register ``open_terrain`` and ``open_flat`` onto TrackerLab."""
    import torch
    import trackerTask.supermimic.terrain as task_terrain
    import trackerLab.tracker_env.mdp.supermimic_command as mdp_cmd
    from trackerLab.tracker_env.supermimic.terrain import TerrainSim
    from trackerLab.tracker_env.supermimic.command import CommandSim

    for profile in OPEN_PROFILES:
        if profile not in task_terrain.COMMAND_PROFILES:
            task_terrain.COMMAND_PROFILES = (*task_terrain.COMMAND_PROFILES, profile)
        if profile not in mdp_cmd.COMMAND_PROFILES:
            mdp_cmd.COMMAND_PROFILES = (*mdp_cmd.COMMAND_PROFILES, profile)
        task_id = f"locomotion/{robot}/{profile}"
        task_terrain.ALL_TASKS[task_id] = task_terrain.TerrainTask(task_id, robot, profile)

    if not getattr(task_terrain.generate, "_open_terrain_patched", False):
        orig_generate = task_terrain.generate

        def patched_generate(profile, height, seed=0):
            if profile == OPEN_PROFILE:
                return generate_open_terrain(height, seed)
            if profile == OPEN_FLAT_PROFILE:
                return generate_open_flat(height, seed)
            return orig_generate(profile, height, seed)

        patched_generate._open_terrain_patched = True
        task_terrain.generate = patched_generate

    if not getattr(CommandSim.__init__, "_open_terrain_patched", False):
        orig_cmd_init = CommandSim.__init__

        def patched_cmd_init(self, task, mdp, *args, **kwargs):
            orig_cmd_init(self, task, mdp, *args, **kwargs)
            if task.profile == OPEN_PROFILE:
                self.curriculum = True
                self.contract["terrain_curriculum_levels"] = len(BLOCK_CENTERS_NORM)
            elif task.profile == OPEN_FLAT_PROFILE:
                self.curriculum = False
                self.contract["terrain_curriculum_levels"] = 0
                self._vel_curriculum_alpha = float(OPEN_FLAT_CURRICULUM["initial_alpha"])
                self._vel_curriculum_steps = 0
                self._vel_curriculum_env_steps = 0
                self._vel_curriculum_vel_sum = 0.0
                self._vel_curriculum_yaw_sum = 0.0
                self._vel_curriculum_dones = 0
                self._vel_curriculum_falls = 0
                self.contract["velocity_command_curriculum"] = dict(OPEN_FLAT_CURRICULUM)

        patched_cmd_init._open_terrain_patched = True
        CommandSim.__init__ = patched_cmd_init

    if not getattr(CommandSim.sample_commands, "_open_flat_curriculum_patched", False):
        orig_sample_commands = CommandSim.sample_commands
        orig_cmd_step = CommandSim.step

        def _open_flat_limits(alpha: float):
            c = OPEN_FLAT_CURRICULUM
            return open_flat_limits(c, alpha)

        @torch.no_grad()
        def patched_sample_commands(self, ids=None, *, jump_height=None):
            if (
                self.mdp.profile != OPEN_FLAT_PROFILE
                or not OPEN_FLAT_CURRICULUM["enabled"]
                or self.stealth_keyboard_mode
                or hasattr(self, "_command_ranges")
                or self.mdp.terrain_navigation
            ):
                return orig_sample_commands(self, ids=ids, jump_height=jump_height)

            if ids is None:
                ids = torch.arange(self.num_envs, device=self.device)
            ids = ids.to(device=self.device, dtype=torch.long)
            height = self.jump_height if jump_height is None else jump_height
            if not math.isfinite(height) or height < 0:
                raise ValueError("Jump height must be finite and nonnegative")
            count = len(ids)
            if not count:
                return
            alpha = getattr(self, "_vel_curriculum_alpha", float(OPEN_FLAT_CURRICULUM["initial_alpha"]))
            commands, u = sample_open_flat(OPEN_FLAT_CURRICULUM, alpha, count, self.device, self.c.dtype)

            self.c[ids, 1:4] = commands
            self.c[ids, 4] = self.target_clearance_m
            self.c[ids, 5] = torch.where(u[:, 3] < self.jump_probability, height, 0.0)
            if not self.mdp.entry_duration_s:
                self.c[ids, 6] = 0
            self._command_age[ids] = 0
            lo, hi = self.command_resample_steps
            self._command_interval[ids] = torch.randint(lo, hi + 1, (count,), device=self.device)

        @torch.no_grad()
        def patched_cmd_step(self, action):
            transition = orig_cmd_step(self, action)
            if (
                self.mdp.profile == OPEN_FLAT_PROFILE
                and OPEN_FLAT_CURRICULUM["enabled"]
                and not self.stealth_keyboard_mode
                and not hasattr(self, "_command_ranges")
            ):
                update_open_flat_curriculum(self, transition, OPEN_FLAT_CURRICULUM, OPEN_FLAT_REWARDS)
            return transition

        patched_sample_commands._open_flat_curriculum_patched = True
        patched_cmd_step._open_flat_curriculum_patched = True
        CommandSim.sample_commands = patched_sample_commands
        CommandSim.step = patched_cmd_step

    if not getattr(TerrainSim._reset_state, "_open_terrain_patched", False):
        orig_terrain_reset = TerrainSim._reset_state

        def patched_terrain_reset(self, ids):
            orig_terrain_reset(self, ids)
            field_terrain = getattr(self.mdp.field, "terrain", None)
            profile = getattr(field_terrain, "profile", None)
            if profile == OPEN_FLAT_PROFILE:
                count = len(ids)
                xy = (2.0 * torch.rand((count, 2), device=self.device, dtype=self.qpos.dtype) - 1.0) * (4.0 * self.scale)
                ground = self.mdp.field.sample(xy)
                yaw = (2.0 * torch.rand(count, device=self.device, dtype=self.qpos.dtype) - 1.0) * math.pi
                quat = torch.stack(
                    (torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)),
                    dim=-1,
                )
                self.qpos[ids, self._root_qpos:self._root_qpos + 3] = torch.stack(
                    (xy[:, 0], xy[:, 1], ground + self.height + 0.02 * self.scale),
                    dim=-1,
                )
                self.qpos[ids, self._root_qpos + 3:self._root_qpos + 7] = quat
                return
            block_centers = getattr(field_terrain, "block_centers", None)
            if block_centers is None or profile != OPEN_PROFILE:
                return
            centers_tensor = getattr(self, "_open_block_centers", None)
            if centers_tensor is None or centers_tensor.device != self.device:
                centers_tensor = torch.tensor(block_centers, dtype=self.qpos.dtype, device=self.device)
                self._open_block_centers = centers_tensor
            count = len(ids)
            xy_center = centers_tensor[self.levels[ids]]
            jitter = (2.0 * torch.rand((count, 2), device=self.device, dtype=self.qpos.dtype) - 1.0) * (3.6 * self.scale)
            xy = xy_center + jitter
            ground = self.mdp.field.sample(xy)
            yaw = (2.0 * torch.rand(count, device=self.device, dtype=self.qpos.dtype) - 1.0) * math.pi
            quat = torch.stack(
                (torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)),
                dim=-1,
            )
            self.qpos[ids, self._root_qpos:self._root_qpos + 3] = torch.stack(
                (xy[:, 0], xy[:, 1], ground + self.height + 0.02 * self.scale),
                dim=-1,
            )
            self.qpos[ids, self._root_qpos + 3:self._root_qpos + 7] = quat

        patched_terrain_reset._open_terrain_patched = True
        TerrainSim._reset_state = patched_terrain_reset

    if not getattr(mdp_cmd.CommandMDP.reward, "_open_terrain_patched", False):
        orig_set_style = mdp_cmd.CommandMDP.set_style_weight
        orig_reward = mdp_cmd.CommandMDP.reward

        def patched_set_style_weight(self, weight):
            orig_set_style(self, weight)
            if self.profile in OPEN_PROFILES:
                r = OPEN_FLAT_REWARDS
                open_task_maximum = (
                    r["velocity_weight"]
                    + r["yaw_weight"]
                    + r["alive_weight"]
                    + r["progress_trans_weight"]
                    + r["progress_yaw_weight"]
                )
                self.max_step_reward = open_task_maximum + float(weight)

        def patched_reward(self, S, a, a_prev, c, terms=False):
            if self.profile not in OPEN_PROFILES:
                return orig_reward(self, S, a, a_prev, c, terms=terms)
            r = OPEN_FLAT_REWARDS
            return open_flat_reward(self, S, a, a_prev, c, r, terms)

        patched_set_style_weight._open_terrain_patched = True
        patched_reward._open_terrain_patched = True
        mdp_cmd.CommandMDP.set_style_weight = patched_set_style_weight
        mdp_cmd.CommandMDP.reward = patched_reward
