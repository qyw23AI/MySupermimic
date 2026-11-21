"""AMP-SuperMimic parkour and stepping-stones terrain + reward integration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np

try:
    from trackerTask.supermimic.terrain import Terrain
except ModuleNotFoundError:
    _tracker_task_src = Path(__file__).resolve().parents[2] / "third_party/trackerlab_supermimic/source/trackerTask"
    if str(_tracker_task_src) not in sys.path:
        sys.path.insert(0, str(_tracker_task_src))
    from trackerTask.supermimic.terrain import Terrain

PARKOUR_PROFILE = "amp_supermimic_parkour"
STEPPING_STONES_PROFILE = "amp_supermimic_stepping_stones"
PARKOUR_PROFILES = (PARKOUR_PROFILE, STEPPING_STONES_PROFILE)
PROFILE = PARKOUR_PROFILE  # Backward-compatible alias.

PARKOUR_TERRAIN_TYPES = ("flat_gap", "tire_array", "slope", "stair")
STEPPING_STONES_TERRAIN_TYPES = ("stepping_stones",)
TERRAIN_TYPES = PARKOUR_TERRAIN_TYPES

# Default parkour command sampling ranges (overridden from config.commands).
PARKOUR_COMMANDS = {
    "vx_range": (0.25, 0.60),
    "wz_range": (-0.30, 0.30),
    "wz_zero_probability": 0.30,
}

STEPPING_STONES_COMMANDS = {
    "vx_range": (0.20, 0.50),
    "wz_range": (-0.20, 0.20),
    "wz_zero_probability": 0.35,
}

# Default parkour reward weights (overridden from WblwblDogV1AmpSupermimicParkourCfg.rewards).
PARKOUR_REWARDS = {
    "velocity_weight": 0.04,
    "velocity_sigma": 0.25,
    "yaw_weight": 0.04,
    "yaw_sigma": 0.25,
    "alive_weight": 0.06,
    "upright_weight": -0.06,
    "clearance_weight": -0.004,
    "posture_weight": -0.0006,
    "action_rate_weight": -0.0008,
    "joint_vel_weight": -0.00002,
    "joint_limit_weight": -0.04,
    "progress_weight": 0.008,
    "lane_weight": -0.001,
    "heading_weight": -0.003,
    "dont_wait_weight": -0.01,
    "success_bonus": 0.5,
    "fall_penalty": 0.5,
}

# Default stepping-stones reward weights (overridden from WblwblDogV1AmpSupermimicSteppingStonesCfg.rewards).
STEPPING_STONES_REWARDS = {
    **PARKOUR_REWARDS,
    "velocity_weight": 0.03,
    "yaw_weight": 0.03,
    "alive_weight": 0.05,
    "clearance_weight": -0.002,
    "progress_weight": 0.008,
    "lane_weight": -0.002,
    "heading_weight": -0.005,
    "dont_wait_weight": -0.015,
    "success_bonus": 2.0,
    "fall_penalty": 1.0,
    "stone_foothold_weight": 0.050,
    "stone_edge_margin": 0.035,
    "stone_touch_progress_weight": 0.040,
    "stone_pit_contact_weight": -0.12,
    "stone_pit_fail_feet": 2,
    "stone_swing_clearance_weight": 0.020,
    "stone_swing_target_clearance": 0.08,
    "stone_slip_weight": -0.015,
    "stone_zone_margin": 0.02,
    "stone_visit_bonus": 1.5,
    "stone_all_visited_goal_bonus": 10.0,
    "stone_require_all_for_curriculum": True,
}


@dataclass
class WmpParkourTerrain(Terrain):
    lane_centers: np.ndarray
    lane_types: tuple[str, ...]
    lane_weights: tuple[float, ...]
    spawn_x: float
    max_init_terrain_level: int
    curriculum: bool
    stone_zone_half_x: tuple[float, ...] = ()
    stone_centers: tuple[tuple[tuple[float, float], ...], ...] = ()
    stone_half_size: float = 0.10
    stone_height: float = 0.10
    terrain_width: float = 2.4


def _at_level(values, level):
    return values[min(level, len(values) - 1)]


def _flat_gap(cfg, level, x, y, rng):
    gap = _at_level(cfg.flat_gap_widths, level)
    left, right = cfg.flat_gap_platform_edges
    half_width = rng.uniform(*cfg.flat_gap_corridor_width) / 2
    supported = (
        ((x >= x.min() + gap) & (x <= left - gap))
        | ((x >= left) & (x <= right))
        | (x >= right + gap)
    )
    return np.where(supported & (abs(y) <= half_width), 0., cfg.floor_height)


def _tire_array(cfg, level, x, y):
    height = _at_level(cfg.tire_heights, level)
    surface = np.broadcast_to(
        np.where(abs(y) <= cfg.tire_corridor_width / 2, 0., cfg.floor_height),
        (len(y), x.size),
    ).copy()
    mid_radius = (cfg.tire_outer_radius + cfg.tire_inner_radius) / 2
    half_section = (cfg.tire_outer_radius - cfg.tire_inner_radius) / 2
    for row in (-.5, .5):
        for column in (-1.5, -.5, .5, 1.5):
            dx = x - (cfg.tire_array_x + column * cfg.tire_pitch)
            dy = y - row * cfg.tire_pitch
            radius = np.hypot(dx, dy)
            ring = (radius >= cfg.tire_inner_radius) & (radius <= cfg.tire_outer_radius)
            angle = np.arctan2(*np.broadcast_arrays(dy, dx))[ring]
            section = np.arccos(np.clip((radius[ring] - mid_radius) / half_section, -1, 1))
            upper = np.sin(section)
            tread = (cfg.tire_tread_depth_fraction * (.5 + .5 * np.cos(12 * angle))
                     * (.5 + .5 * np.cos(4 * section)) * upper ** 2)
            surface[ring] = np.maximum(surface[ring], height * ((1 + upper) / 2 - tread))
    return surface


def _slope(cfg, level, x, y, sign):
    # Use one plane per lane with the lower end at ground level.
    grade = _at_level(cfg.slope_grades, level)
    low_end = -sign * cfg.terrain_length / 2
    height = sign * grade * (x - low_end)
    return np.broadcast_to(height, (len(y), x.size))


def _stair(cfg, level, x, y):
    step_h = _at_level(cfg.stair_step_heights, level)
    step_d = _at_level(cfg.stair_step_depths, level)
    n_steps = int(cfg.stair_num_steps)
    plat_half = float(cfg.stair_platform_length) / 2
    cx = float(cfg.stair_center_x)
    in_corridor = abs(y) <= cfg.stair_corridor_width / 2
    dx = np.abs(x - cx)
    dist_from_plat = np.maximum(dx - plat_half, 0.0)
    riser_from_top = np.ceil(dist_from_plat / step_d)
    height = np.maximum(n_steps - riser_from_top, 0.0) * step_h
    return np.where(in_corridor, height, cfg.floor_height)


def _stepping_stones(cfg, level, x, y):
    size, height = cfg.plum_stone_size, cfg.plum_stone_height
    pitch = 2 * (size + _at_level(cfg.plum_longitudinal_gaps, level))
    row_y = (size + _at_level(cfg.plum_row_gaps, level)) / 2
    stone_edge = 1.25 * pitch + size / 2
    platform_half = cfg.plum_platform_size / 2
    zone_half_x = stone_edge + _at_level(cfg.plum_platform_gaps, level)
    platform = zone_half_x + platform_half
    # Keep the flat competition floor at z=0 between the 10 cm stones/platforms;
    # ground contact inside the stone zone is penalized/terminated via rewards.
    surface = np.broadcast_to(
        np.where(abs(y) <= cfg.plum_road_width / 2, 0., cfg.floor_height),
        (len(y), x.size),
    ).copy()
    stone_centers = []
    for cy, centers in ((-row_y, (-1.25, -.25, .75)), (row_y, (-.75, .25, 1.25))):
        for cx in centers:
            sx = float(cx * pitch)
            sy = float(cy)
            stone_centers.append((sx, sy))
            surface[(abs(x - sx) <= size / 2) & (abs(y - sy) <= size / 2)] = height
    for cx in (-platform, platform):
        surface[(abs(x - cx) <= platform_half) & (abs(y) <= platform_half)] = height
    return surface, float(zone_half_x), tuple(stone_centers)


_plum_stones = _stepping_stones


def _generate_atlas(cfg, seed, *, preview_level=None, profile=None):
    """Build the training atlas or one identical row for the static viewer."""
    profile = profile or getattr(cfg, "profile", PARKOUR_PROFILE)
    if profile not in PARKOUR_PROFILES:
        raise ValueError(f"Unknown parkour profile {profile!r}; expected one of {PARKOUR_PROFILES}")

    scale = cfg.horizontal_scale
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("horizontal_scale must be finite and positive")
    dimensions = np.array([cfg.terrain_length, cfg.terrain_width]) / scale
    if (not np.isfinite(dimensions).all() or np.any(dimensions < 4)
            or not np.allclose(dimensions, np.round(dimensions))):
        raise ValueError("Terrain length and width must be positive multiples of horizontal_scale")
    if not 0 <= cfg.max_init_terrain_level < cfg.num_rows:
        raise ValueError("max_init_terrain_level must be between 0 and num_rows - 1")
    x_min, x_max = -cfg.terrain_length / 2, cfg.terrain_length / 2
    if not x_min < cfg.spawn_x < cfg.finish_x < x_max:
        raise ValueError("Set spawn_x and finish_x inside the terrain in travel order")
    if preview_level is not None and not 0 <= preview_level < cfg.num_rows:
        raise ValueError(f"{profile} level must be between 0 and {cfg.num_rows - 1}")

    weights = dict(cfg.terrain_proportions)
    if profile == STEPPING_STONES_PROFILE:
        allowed = set(STEPPING_STONES_TERRAIN_TYPES) | {"plum_stones"}
        if (set(weights) - allowed or not all(np.isfinite(w) and w >= 0 for w in weights.values())
                or sum(weights.values()) <= 0):
            raise ValueError(f"Stepping stones task expects positive weights for {STEPPING_STONES_TERRAIN_TYPES}")
        lane_specs = (("stepping_stones", (1,)),)
        normalized_weights = {"stepping_stones": float(weights.get("stepping_stones", weights.get("plum_stones", 1.0)))}
    else:
        allowed = set(PARKOUR_TERRAIN_TYPES)
        if (set(weights) - allowed or not all(np.isfinite(w) and w >= 0 for w in weights.values())
                or sum(weights.values()) <= 0):
            raise ValueError(f"Parkour task expects nonnegative weights for {PARKOUR_TERRAIN_TYPES}")
        lane_specs = (("flat_gap", (1,)), ("tire_array", (1,)), ("slope", (-1, 1)), ("stair", (1,)))
        normalized_weights = weights

    lanes = [(kind, sign, normalized_weights[kind] / len(signs))
             for kind, signs in lane_specs
             if normalized_weights.get(kind, 0) > 0 for sign in signs]
    if any(kind == "flat_gap" for kind, _, _ in lanes):
        left, right = cfg.flat_gap_platform_edges
        widest = max(cfg.flat_gap_widths)
        if not (x_min + widest < cfg.spawn_x < left - widest
                and left < right < cfg.finish_x - widest):
            raise ValueError("Keep the spawn and finish on supported road outside both flat gaps")

    levels = range(cfg.num_rows) if preview_level is None else (preview_level,)
    nx, ny = np.round(dimensions).astype(int)
    x = np.linspace(x_min, x_max, nx + 1)[None, :]
    y = (np.arange(ny) - ny / 2)[:, None] * scale
    total_lanes = len(levels) * len(lanes)
    y_min = -total_lanes * cfg.terrain_width / 2
    y_max = -y_min
    centers = (y_min + (np.arange(total_lanes) + .5) * cfg.terrain_width).reshape(-1, len(lanes))
    heights = np.empty((total_lanes * ny + 1, nx + 1), dtype=np.float32)
    stone_zone_half_x = []
    stone_centers_rows = []
    for row, level in enumerate(levels):
        for column, (kind, sign, _) in enumerate(lanes):
            if kind == "flat_gap":
                rng = np.random.default_rng(np.random.SeedSequence([seed, level, 0, 1]))
                surface = _flat_gap(cfg, level, x, y, rng)
            elif kind == "tire_array":
                surface = _tire_array(cfg, level, x, y)
            elif kind == "slope":
                surface = _slope(cfg, level, x, y, sign)
            elif kind == "stair":
                surface = _stair(cfg, level, x, y)
            else:
                surface, zone_half_x, centers_row = _stepping_stones(cfg, level, x, y)
                stone_zone_half_x.append(zone_half_x)
                stone_centers_rows.append(centers_row)
            start = (row * len(lanes) + column) * ny
            heights[start:start + ny] = surface
    heights[-1] = heights[-2]

    lane_types = tuple(f"slope_{'up' if sign > 0 else 'down'}" if kind == "slope" else kind
                       for kind, sign, _ in lanes)
    total_weight = sum(normalized_weights.values())
    lane_weights = tuple(weight / total_weight for _, _, weight in lanes)
    return WmpParkourTerrain(
        heights=heights, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        centers=tuple(centers[:, 0]), goal_x=cfg.finish_x, seed=seed, profile=profile,
        lane_centers=centers, lane_types=lane_types, lane_weights=lane_weights, spawn_x=cfg.spawn_x,
        max_init_terrain_level=cfg.max_init_terrain_level, curriculum=cfg.curriculum,
        stone_zone_half_x=tuple(stone_zone_half_x),
        stone_centers=tuple(stone_centers_rows),
        stone_half_size=float(getattr(cfg, "plum_stone_size", 0.20)) / 2.0,
        stone_height=float(getattr(cfg, "plum_stone_height", 0.10)),
        terrain_width=float(cfg.terrain_width),
    )


def generate_amp_supermimic_parkour(cfg, seed, *, preview_level=None):
    return _generate_atlas(cfg, seed, preview_level=preview_level, profile=PARKOUR_PROFILE)


def generate_amp_supermimic_stepping_stones(cfg, seed, *, preview_level=None):
    return _generate_atlas(cfg, seed, preview_level=preview_level, profile=STEPPING_STONES_PROFILE)


def generate_wmp_parkour(cfg, seed, *, preview_level=None, profile=None):
    return _generate_atlas(cfg, seed, preview_level=preview_level, profile=profile)


class _SteppingStonesHelper:
    """Pure-torch differentiable foot kinematics and stone-zone lookup for stepping stones."""

    def __init__(self, robot_name: str, device, terrain, foot_radii: tuple[float, ...] | None = None):
        import torch
        from trackerLab.kinematics.supermimic.robot import RobotKinematics
        from trackerTask.supermimic.gait import FOOT_BODIES, FOOT_OFFSETS
        from trackerTask.supermimic.robots import get_robot

        spec = get_robot(robot_name)
        feet = FOOT_BODIES[robot_name]
        self.kin = RobotKinematics(str(device), spec, tracked=feet)
        self.foot_offsets = torch.tensor(FOOT_OFFSETS[robot_name], dtype=torch.float32, device=device)
        radii = (0.02,) * len(feet) if foot_radii is None else foot_radii
        self.foot_radii = torch.tensor(radii, dtype=torch.float32, device=device)
        self.margin_offsets = torch.tensor(
            ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)),
            dtype=torch.float32,
            device=device,
        )
        self.ahead_offset = torch.tensor((0.06, 0.0), dtype=torch.float32, device=device)
        zone = getattr(terrain, "stone_zone_half_x", ()) or (0.0,)
        self.stone_zone_half_x = torch.tensor(zone, dtype=torch.float32, device=device)
        centers = getattr(terrain, "stone_centers", ()) or (((0.0, 0.0),) * 6,)
        self.stone_centers = torch.tensor(centers, dtype=torch.float32, device=device)
        self.stone_half_size = float(getattr(terrain, "stone_half_size", 0.10))
        self.stone_height = float(getattr(terrain, "stone_height", 0.10))
        self.y_min = float(terrain.y_min)
        self.terrain_width = float(getattr(terrain, "terrain_width", 2.4))
        self.num_rows = len(zone)

    def foot_sole_state(self, S):
        from trackerLab.kinematics.supermimic.rotations import cross3

        p, R, v, w = S[:4]
        p_body, R_body, v_body, w_body = self.kin.forward(p, R, v, w, S[4], S[5])
        arm = (R_body @ self.foot_offsets.to(R_body)[..., None]).squeeze(-1)
        foot_pos = p_body + arm
        foot_vel = v_body + cross3(w_body, arm)
        sole_z = foot_pos[..., 2] - self.foot_radii.to(foot_pos)
        return foot_pos, foot_vel, sole_z


def _foot_pit_state(mdp, S, c, r):
    import torch

    helper: _SteppingStonesHelper = mdp._stepping_stones_helper
    foot_pos, foot_vel, sole_z = helper.foot_sole_state(S)
    xy = torch.nan_to_num(foot_pos[..., :2])
    h_center = mdp.field.sample(xy)
    clearance_f = sole_z - h_center
    contact_w = (1.0 - clearance_f / 0.03).clamp(0.0, 1.0)
    level_idx = ((c[..., 0] - helper.y_min) / helper.terrain_width - 0.5).round().long().clamp(0, helper.num_rows - 1)
    zone_half_x = helper.stone_zone_half_x[level_idx]
    in_stone_zone = foot_pos[..., 0].abs() <= (zone_half_x[..., None] - float(r["stone_zone_margin"]))
    return helper, xy, foot_vel, sole_z, h_center, clearance_f, contact_w, in_stone_zone, level_idx


def _stepping_stones_terms(mdp, S, c, r):
    import torch

    helper, xy, foot_vel, sole_z, h_center, clearance_f, contact_w, in_stone_zone, level_idx = _foot_pit_state(mdp, S, c, r)

    # Reward full-sole support inside stone/platform tops (penalize overhanging edges).
    xy_margin = xy[..., None, :] + helper.margin_offsets.to(xy) * float(r["stone_edge_margin"])
    h_margin = mdp.field.sample(xy_margin)
    h_min_foot = torch.minimum(h_center, h_margin.amin(dim=-1))
    on_stone_interior = torch.sigmoid((h_min_foot - (helper.stone_height - 0.02)) / 0.01)
    stone_foothold = float(r["stone_foothold_weight"]) * (contact_w * on_stone_interior).mean(-1)

    # Reward stepping onto the 6 discrete stones while progressing forward (prevents camping on one stone).
    centers = helper.stone_centers[level_idx]
    dx = (xy[..., :, None, 0] - centers[..., None, :, 0]).abs()
    dy = (xy[..., :, None, 1] - (c[..., None, None, 0] + centers[..., None, :, 1])).abs()
    in_stone_box = (
        torch.sigmoid((helper.stone_half_size + 0.01 - dx) / 0.01)
        * torch.sigmoid((helper.stone_half_size + 0.01 - dy) / 0.01)
    )
    on_any_stone = (contact_w[..., None] * in_stone_box).amax(dim=-1)
    moving_fwd = torch.sigmoid((S[2][..., 0] / mdp.field.scale - 0.10) / 0.05)
    stone_touch_progress = float(r["stone_touch_progress_weight"]) * on_any_stone.amax(dim=-1) * moving_fwd

    # Penalize touching the z=0 pit floor inside the stepping-stones zone.
    is_pit_floor = torch.sigmoid((helper.stone_height - 0.03 - h_center) / 0.01)
    pit_contact_prob = contact_w * is_pit_floor * in_stone_zone.to(contact_w.dtype)
    stone_pit_contact = float(r["stone_pit_contact_weight"]) * pit_contact_prob.sum(-1)

    # Encourage adequate swing-foot clearance over 10 cm stone edges.
    swing_w = 1.0 - contact_w
    h_ahead = mdp.field.sample(xy + helper.ahead_offset.to(xy))
    h_ref = torch.maximum(h_center, h_ahead)
    swing_target = max(float(r["stone_swing_target_clearance"]), 1e-4)
    swing_score = ((sole_z - h_ref) / swing_target).clamp(0.0, 1.0)
    stone_swing_clearance = float(r["stone_swing_clearance_weight"]) * (swing_w * swing_score).mean(-1)

    # Suppress horizontal slip while supported on stones/platforms.
    foot_speed_xy = torch.linalg.vector_norm(foot_vel[..., :2], dim=-1) / mdp.field.scale
    stone_slip = float(r["stone_slip_weight"]) * (contact_w * foot_speed_xy).mean(-1)

    return {
        "stone_foothold": stone_foothold,
        "stone_touch_progress": stone_touch_progress,
        "stone_pit_contact": stone_pit_contact,
        "stone_swing_clearance": stone_swing_clearance,
        "stone_slip": stone_slip,
    }


def _parkour_reward(mdp, S, a, a_prev, c, r, terms=False):
    import torch
    from trackerLab.tracker_env.mdp import supermimic_command as mdp_cmd

    c_eff = mdp.effective_command(c)
    p, R, _, _, q, qd = S
    scale = mdp.field.scale
    v_body, w_body = mdp_cmd.body_velocities(S)
    clearance = mdp._clearance(S)
    ev = ((v_body[..., :2] - c_eff[..., 1:3]) / scale).square().sum(-1)
    ew = (w_body[..., 2] - c_eff[..., 3]).square()
    velocity_reward = float(r["velocity_weight"]) * torch.exp(-ev / float(r["velocity_sigma"]))
    yaw_reward = float(r["yaw_weight"]) * torch.exp(-ew / float(r["yaw_sigma"]))
    progress = float(r["progress_weight"]) * (S[2][..., 0] / scale).clamp(-1.0, 1.5)
    lane = float(r["lane_weight"]) * ((p[..., 1] - c_eff[..., 0]) / scale).square()
    heading = float(r["heading_weight"]) * (1.0 - R[..., 0, 0])
    dont_wait = (
        float(r["dont_wait_weight"])
        * (c_eff[..., 1] / scale > 0.15).to(ev.dtype)
        * torch.sigmoid((0.1 - v_body[..., 0] / scale) / 0.03)
    )
    values = (
        velocity_reward,
        yaw_reward,
        torch.full_like(ev, float(r["alive_weight"])),
        float(r["upright_weight"]) * (1.0 - R[..., 2, 2]),
        float(r["clearance_weight"]) * ((clearance - mdp_cmd.target_clearance(c_eff)) / (0.2 * mdp.field.height)).square(),
        float(r["posture_weight"]) * (q - mdp.default_q).square().mean(-1),
        float(r["action_rate_weight"]) * (a - a_prev).square().mean(-1),
        float(r["joint_vel_weight"]) * qd.square().mean(-1),
        float(r["joint_limit_weight"]) * (torch.relu(q - mdp.soft_hi) + torch.relu(mdp.soft_lo - q)).sum(-1),
        progress,
        lane,
        heading,
        dont_wait,
    )
    term_dict = dict(zip(mdp_cmd.TERMS, values))
    reward = sum(values)
    if mdp.profile == STEPPING_STONES_PROFILE:
        extra_terms = _stepping_stones_terms(mdp, S, c, r)
        term_dict.update(extra_terms)
        reward = reward + sum(extra_terms.values())
    return (reward, term_dict) if terms else reward


def patch_parkour(robot: str, foot_radii: tuple[float, ...] | None = None) -> None:
    """Register amp_supermimic_parkour and amp_supermimic_stepping_stones rewards and FK helpers."""
    import torch
    import trackerLab.tracker_env.mdp.supermimic_command as mdp_cmd
    from trackerLab.tracker_env.supermimic.command import CommandSim

    CommandMDP = mdp_cmd.CommandMDP
    if getattr(CommandMDP.reward, "_amp_parkour_patched", False):
        return

    orig_sample_commands = CommandSim.sample_commands

    @torch.no_grad()
    def patched_sample_commands(self, ids=None, *, jump_height=None):
        orig_sample_commands(self, ids=ids, jump_height=jump_height)
        if (
            self.mdp.profile not in PARKOUR_PROFILES
            or self.stealth_keyboard_mode
            or hasattr(self, "_command_ranges")
        ):
            return
        target_ids = torch.arange(self.num_envs, device=self.device) if ids is None else ids.to(device=self.device, dtype=torch.long)
        count = len(target_ids)
        if not count:
            return
        cmd_cfg = STEPPING_STONES_COMMANDS if self.mdp.profile == STEPPING_STONES_PROFILE else PARKOUR_COMMANDS
        vx_lo, vx_hi = map(float, cmd_cfg["vx_range"])
        wz_lo, wz_hi = map(float, cmd_cfg["wz_range"])
        zero_p = float(cmd_cfg["wz_zero_probability"])
        u = torch.rand(count, 3, device=self.device, dtype=self.c.dtype)
        self.c[target_ids, 1] = vx_lo + u[:, 0] * (vx_hi - vx_lo)
        self.c[target_ids, 2] = 0.0
        wz = wz_lo + u[:, 1] * (wz_hi - wz_lo)
        self.c[target_ids, 3] = torch.where(u[:, 2] < zero_p, torch.zeros_like(wz), wz)

    patched_sample_commands._amp_parkour_patched = True
    CommandSim.sample_commands = patched_sample_commands

    orig_reset_state = CommandSim._reset_state
    orig_cmd_step = CommandSim.step

    def patched_reset_state(self, ids):
        orig_reset_state(self, ids)
        if getattr(self.mdp, "profile", None) == STEPPING_STONES_PROFILE:
            if not hasattr(self, "_stones_visited"):
                self._stones_visited = torch.zeros((self.num_envs, 6), dtype=torch.bool, device=self.device)
                self._pre_reset_stones_visited = torch.zeros_like(self._stones_visited)
            self._pre_reset_stones_visited[ids] = self._stones_visited[ids]
            self._stones_visited[ids] = False

    @torch.no_grad()
    def patched_cmd_step(self, action):
        if getattr(self.mdp, "profile", None) != STEPPING_STONES_PROFILE:
            return orig_cmd_step(self, action)
        from trackerLab.tracker_env.mdp.supermimic import unflatten_state

        prev_levels = self.levels.clone()
        transition = orig_cmd_step(self, action)
        r = STEPPING_STONES_REWARDS
        helper: _SteppingStonesHelper = self.mdp._stepping_stones_helper
        S1 = unflatten_state(transition["s1"])
        c1 = transition["c1"]
        foot_pos, _, sole_z = helper.foot_sole_state(S1)
        level_idx = ((c1[..., 0] - helper.y_min) / helper.terrain_width - 0.5).round().long().clamp(0, helper.num_rows - 1)
        centers = helper.stone_centers[level_idx]
        dx = (foot_pos[..., :, None, 0] - centers[..., None, :, 0]).abs()
        dy = (foot_pos[..., :, None, 1] - (c1[..., None, None, 0] + centers[..., None, :, 1])).abs()
        on_stone = (
            (sole_z[..., None] <= helper.stone_height + 0.025)
            & (sole_z[..., None] >= helper.stone_height - 0.035)
            & (dx <= helper.stone_half_size + 0.015)
            & (dy <= helper.stone_half_size + 0.015)
        )
        stone_touched = on_stone.any(dim=1)  # (num_envs, 6)
        done = transition["done"]
        prior_visited = torch.where(done[:, None], self._pre_reset_stones_visited, self._stones_visited)
        newly_visited = stone_touched & ~prior_visited
        visited_now = prior_visited | stone_touched
        self._stones_visited = torch.where(done[:, None], self._stones_visited, visited_now)
        all_visited = visited_now.all(dim=-1)

        visit_rew = newly_visited.sum(dim=-1).to(transition["r"].dtype) * float(r["stone_visit_bonus"])
        all_goal_rew = (transition["episode_success"] & all_visited).to(transition["r"].dtype) * float(r["stone_all_visited_goal_bonus"])
        bonus = visit_rew + all_goal_rew
        transition["r"] = transition["r"] + bonus
        transition["reward_terms"]["stone_visit"] = visit_rew
        transition["reward_terms"]["stone_all_visited_goal"] = all_goal_rew
        transition["episode_stones_visited"] = torch.where(done, visited_now.sum(dim=-1), torch.zeros_like(prev_levels))
        transition["episode_all_stones_visited"] = torch.where(done, all_visited, torch.zeros_like(done))

        if r.get("stone_require_all_for_curriculum", True) and self.curriculum:
            incomplete = done & transition["episode_success"] & ~all_visited
            if incomplete.any():
                self.levels[incomplete] = prev_levels[incomplete]
                inc_ids = incomplete.nonzero(as_tuple=False).flatten()
                with self._warp_context():
                    self._reset_state(inc_ids)
                    self.mw.forward(self.model, self.data)
        return transition

    patched_reset_state._amp_parkour_patched = True
    patched_cmd_step._amp_parkour_patched = True
    CommandSim._reset_state = patched_reset_state
    CommandSim.step = patched_cmd_step

    orig_init = CommandMDP.__init__
    orig_set_style = CommandMDP.set_style_weight
    orig_reward = CommandMDP.reward
    orig_terminal_reward = CommandMDP.terminal_reward
    orig_termination = CommandMDP.termination
    orig_soft_outcomes = CommandMDP.soft_outcome_probabilities

    def _active_rewards(self):
        return STEPPING_STONES_REWARDS if self.profile == STEPPING_STONES_PROFILE else PARKOUR_REWARDS

    def patched_init(self, field, default_q, qlo, qhi, *args, **kwargs):
        orig_init(self, field, default_q, qlo, qhi, *args, **kwargs)
        if self.profile == STEPPING_STONES_PROFILE:
            self._stepping_stones_helper = _SteppingStonesHelper(
                robot, default_q.device, field.terrain, foot_radii=foot_radii,
            )

    def patched_set_style_weight(self, weight):
        orig_set_style(self, weight)
        if self.profile not in PARKOUR_PROFILES:
            return
        r = _active_rewards(self)
        task_max = (
            max(float(r["velocity_weight"]), 0.0)
            + max(float(r["yaw_weight"]), 0.0)
            + max(float(r["alive_weight"]), 0.0)
            + 1.5 * max(float(r["progress_weight"]), 0.0)
        )
        if self.profile == STEPPING_STONES_PROFILE:
            task_max += (
                max(float(r["stone_foothold_weight"]), 0.0)
                + max(float(r["stone_touch_progress_weight"]), 0.0)
                + max(float(r["stone_swing_clearance_weight"]), 0.0)
            )
        maximum = task_max + float(weight)
        success = float(r["success_bonus"]) + self.gamma * maximum / (1.0 - self.gamma)
        if not math.isfinite(success):
            raise ValueError("Style weight produces a nonfinite terminal payout")
        self.style_weight = float(weight)
        self.max_step_reward = maximum
        self.success_reward = success

    def patched_reward(self, S, a, a_prev, c, terms=False):
        if self.profile not in PARKOUR_PROFILES:
            return orig_reward(self, S, a, a_prev, c, terms=terms)
        return _parkour_reward(self, S, a, a_prev, c, _active_rewards(self), terms=terms)

    def patched_terminal_reward(self, success, fall):
        if self.profile not in PARKOUR_PROFILES:
            return orig_terminal_reward(self, success, fall)
        r = _active_rewards(self)
        return self.success_reward * success - float(r["fall_penalty"]) * fall

    def patched_termination(self, S, c):
        failure = orig_termination(self, S, c)
        if self.profile != STEPPING_STONES_PROFILE:
            return failure
        r = STEPPING_STONES_REWARDS
        fail_feet = int(r["stone_pit_fail_feet"])
        if fail_feet <= 0:
            return failure
        helper, _, _, _, h_center, clearance_f, _, in_stone_zone, _ = _foot_pit_state(self, S, c, r)
        pit_feet = (
            (clearance_f < 0.025)
            & (h_center < helper.stone_height - 0.04)
            & in_stone_zone
        ).sum(-1)
        return failure | (pit_feet >= fail_feet)

    def patched_soft_outcome_probabilities(self, S, c):
        success, fall = orig_soft_outcomes(self, S, c)
        if self.profile != STEPPING_STONES_PROFILE:
            return success, fall
        r = STEPPING_STONES_REWARDS
        fail_feet = int(r["stone_pit_fail_feet"])
        if fail_feet <= 0:
            return success, fall
        helper, _, _, _, h_center, clearance_f, _, in_stone_zone, _ = _foot_pit_state(self, S, c, r)
        p_pit = (
            torch.sigmoid((0.025 - clearance_f) / 0.01)
            * torch.sigmoid((helper.stone_height - 0.04 - h_center) / 0.01)
            * in_stone_zone.to(clearance_f.dtype)
        )
        pit_sum = p_pit.sum(-1)
        gate = (pit_sum / float(fail_feet)).clamp(0.0, 1.0)
        pit_fail = gate * torch.sigmoid((pit_sum - (float(fail_feet) - 0.5)) / 0.25)
        fall = 1.0 - (1.0 - fall) * (1.0 - pit_fail)
        success = torch.where(self._finite(S, c), success * (1.0 - pit_fail), torch.zeros_like(success))
        return success, fall

    patched_init._amp_parkour_patched = True
    patched_set_style_weight._amp_parkour_patched = True
    patched_reward._amp_parkour_patched = True
    patched_terminal_reward._amp_parkour_patched = True
    patched_termination._amp_parkour_patched = True
    patched_soft_outcome_probabilities._amp_parkour_patched = True

    CommandMDP.__init__ = patched_init
    CommandMDP.set_style_weight = patched_set_style_weight
    CommandMDP.reward = patched_reward
    CommandMDP.terminal_reward = patched_terminal_reward
    CommandMDP.termination = patched_termination
    CommandMDP.soft_outcome_probabilities = patched_soft_outcome_probabilities
