"""Interactive CPU MuJoCo 3D viewer for closed-loop WBLWBL policy gait inspection."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = ROOT / "amp_supermimic/outputs/trackerlab-amp/wblwbl_open_trot5_r2/last.pt"

CYCLE_STAGES = (
    ("forward_0.45", 0.45, 0.00, 0.00, 200),
    ("strafe_0.25", 0.00, 0.25, 0.00, 200),
    ("reverse_-0.30", -0.30, 0.00, 0.00, 200),
    ("turn_left_0.80", 0.00, 0.00, 0.80, 150),
    ("turn_right_-0.80", 0.00, 0.00, -0.80, 150),
    ("stop", 0.00, 0.00, 0.00, 100),
)


def _state_from_data(data, qpos_ids, qvel_ids):
    p = torch.tensor(data.qpos[:3], dtype=torch.float32)[None]
    qw, qx, qy, qz = data.qpos[3:7]
    R = torch.tensor(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=torch.float32,
    )[None]
    v = torch.tensor(data.qvel[:3], dtype=torch.float32)[None]
    w_body = torch.tensor(data.qvel[3:6], dtype=torch.float32)[None]
    w_world = (R @ w_body.unsqueeze(-1)).squeeze(-1)
    q = torch.tensor(data.qpos[qpos_ids], dtype=torch.float32)[None]
    qd = torch.tensor(data.qvel[qvel_ids], dtype=torch.float32)[None]
    return (p, R, v, w_world, q, qd)


def _load_policy_and_scene(
    ckpt_path: Path, profile_override: str | None, seed_override: int | None,
    *, ground_plane: bool = False, level: int = 0
):
    sys.path.insert(0, str(ROOT / "amp_supermimic/scripts"))
    import train as wblwbl_train

    spec = wblwbl_train.register(wblwbl_train.MJCF, wblwbl_train.DEFAULT_SOURCE_CONFIG)

    from trackerLab.algorithms.supermimic.terrain_networks import TerrainPolicy
    from trackerLab.tracker_env.mdp.supermimic_command import CommandMDP, body_velocities
    from trackerLab.tracker_env.mdp.supermimic_terrain import HeightField
    from trackerTask.supermimic.mujoco_terrain import build_model
    from trackerTask.supermimic.terrain import ROBOT_HEIGHTS, generate

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck_args = ck.get("arguments", {})
    profile = profile_override or ck_args.get("profile") or "open_terrain"
    if "/" in str(ck_args.get("task", "")) and profile_override is None:
        profile = str(ck_args["task"]).split("/")[-1]
    seed = int(ck_args.get("seed", 0) if seed_override is None else seed_override)
    action_limit = float(ck_args.get("action_limit", 20.0))
    hist_len = int(ck_args.get("hist", 10))

    from wmp_parkour import PARKOUR_PROFILES, generate_wmp_parkour

    if profile in PARKOUR_PROFILES:
        from amp_supermimic.legacy import apply_runtime_task_patches, load_task_config

        cfg, _ = load_task_config(profile)
        apply_runtime_task_patches(cfg)
        terrain = generate_wmp_parkour(cfg.terrain, seed, preview_level=level, profile=profile)
        ground_plane = False
    else:
        terrain = generate(profile, ROBOT_HEIGHTS[spec.name], seed=seed)
    model, spec, height, qpos_ids, qvel_ids, _, _, actuator_ids, _ = build_model(
        spec.name, terrain, ground_plane=ground_plane
    )
    data = mujoco.MjData(model)
    default_q = torch.tensor(spec.default_q, dtype=torch.float32)
    jids = [model.joint(n).id for n in spec.joint_names]
    qlo = torch.tensor(model.jnt_range[jids, 0], dtype=torch.float32)
    qhi = torch.tensor(model.jnt_range[jids, 1], dtype=torch.float32)
    field = HeightField(terrain, height, "cpu")
    mdp = CommandMDP(field, default_q, qlo, qhi, profile=profile, action_limit=action_limit)
    actor = TerrainPolicy(
        mdp.obs_dim, mdp.goal_dim, hist=hist_len, num_joints=spec.num_joints, action_limit=action_limit
    )
    actor.load_state_dict(ck["actor"])
    actor.eval()
    return ck, terrain, model, data, spec, height, qpos_ids, qvel_ids, actuator_ids, default_q, qlo, qhi, field, mdp, actor, body_velocities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT, help="Path to iteration_XXXXXX.pt or last.pt")
    parser.add_argument("--profile", type=str, default=None, help="Override terrain profile (default: from checkpoint)")
    parser.add_argument("--seed", type=int, default=None, help="Override terrain seed (default: from checkpoint)")
    parser.add_argument("--level", type=int, default=0, help="Terrain difficulty (default: 0)")
    parser.add_argument("--terrain-type", choices=("flat_gap", "tire_array", "slope_up", "slope_down", "stair", "stepping_stones"),
                        help="Select a parkour / stepping-stones lane (default: first enabled lane)")
    parser.add_argument("--vx", type=float, default=0.45, help="Command forward velocity m/s")
    parser.add_argument("--vy", type=float, default=0.0, help="Command lateral velocity m/s")
    parser.add_argument("--wz", type=float, default=0.0, help="Command yaw rate rad/s")
    parser.add_argument("--cycle", action="store_true", help="Cycle through forward/strafe/reverse/turn/stop demos")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (1.0 = real-time)")
    parser.add_argument("--no-follow", action="store_true", help="Do not track the robot with the camera")
    parser.add_argument("--no-scan", action="store_true", help="Hide 187 height-scan spheres and velocity arrows")
    parser.add_argument("--quadruped-scan", action="store_true",
                        help="Preview forward-biased quadruped scan box X=[-0.40, 0.90], Y=[-0.38, 0.38]")
    parser.add_argument("--apply-scan", action="store_true",
                        help="Also feed the overridden scan box into policy obs (default: visual overlay only)")
    parser.add_argument("--scan-x", type=float, nargs=2, metavar=("XMIN", "XMAX"),
                        help="Override visual height-scan X range in meters (17 rows)")
    parser.add_argument("--scan-y", type=float, nargs=2, metavar=("YMIN", "YMAX"),
                        help="Override visual height-scan Y range in meters (11 cols)")
    parser.add_argument("--hfield", action="store_true", help="Force heightfield even on Level 0")
    parser.add_argument("--headless-steps", type=int, default=0, help="Run N steps without GUI (for testing)")
    args = parser.parse_args()

    use_plane = (args.level == 0) and (not args.hfield)
    ckpt_path = args.checkpoint.resolve()
    (
        ck,
        terrain,
        model,
        data,
        spec,
        height,
        qpos_ids,
        qvel_ids,
        actuator_ids,
        default_q,
        qlo,
        qhi,
        field,
        mdp,
        actor,
        body_velocities,
    ) = _load_policy_and_scene(ckpt_path, args.profile, args.seed, ground_plane=use_plane, level=args.level)

    lvl = max(0, min(int(args.level), len(terrain.centers) - 1))
    from wmp_parkour import PARKOUR_PROFILES

    mixed_parkour = terrain.profile in PARKOUR_PROFILES
    if mixed_parkour:
        lane = args.terrain_type or terrain.lane_types[0]
        if lane not in terrain.lane_types:
            parser.error(f"{lane} is disabled in the WMP parkour terrain configuration")
        lvl = args.level
        cx, cy = terrain.spawn_x, float(terrain.lane_centers[0, terrain.lane_types.index(lane)])
    elif args.terrain_type is not None:
        parser.error("--terrain-type requires a WMP parkour profile")
    elif hasattr(terrain, "block_centers"):
        cx, cy = terrain.block_centers[lvl]
    else:
        cx, cy = 0.0, float(terrain.centers[lvl])

    xi = torch.zeros(1, spec.num_joints + 4, dtype=torch.float32)
    action_scale = torch.tensor(spec.action_scale, dtype=torch.float32)
    control_dt = 4.0 * float(model.opt.timestep)

    visual_offsets = field.offsets
    if args.quadruped_scan or args.scan_x is not None or args.scan_y is not None:
        sx = (-0.40, 0.90) if args.scan_x is None else tuple(args.scan_x)
        sy = (-0.38, 0.38) if args.scan_y is None else tuple(args.scan_y)
        gx, gy = torch.meshgrid(
            torch.linspace(sx[0], sx[1], 17),
            torch.linspace(sy[0], sy[1], 11),
            indexing="ij",
        )
        visual_offsets = torch.stack((gx.flatten(), gy.flatten()), -1)
        if args.apply_scan:
            field.offsets = visual_offsets
        print(
            f"[play_mujoco] scan grid ({'obs+visual' if args.apply_scan else 'visual-only'}): "
            f"X=[{sx[0]:+.2f}, {sx[1]:+.2f}]m, Y=[{sy[0]:+.2f}, {sy[1]:+.2f}]m (17x11=187 pts)",
            flush=True,
        )

    def reset_robot():
        gz = 0.0 if use_plane else float(field.sample(torch.tensor([[cx, cy]], dtype=torch.float32))[0])
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = [cx, cy, gz + height + 0.01]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[qpos_ids] = spec.default_q
        data.ctrl[actuator_ids] = spec.default_q
        mujoco.mj_forward(model, data)
        a_prev = torch.zeros(1, spec.num_joints)
        c = torch.tensor([[cy, args.vx, args.vy, args.wz, height, 0.0, 0.0]], dtype=torch.float32)
        S0 = _state_from_data(data, qpos_ids, qvel_ids)
        obs0 = mdp.obs(S0, a_prev, xi, c)
        hist = obs0[:, None].expand(-1, actor.hist, -1).clone()
        return a_prev, hist

    def current_cmd(step_idx: int):
        if not args.cycle:
            return "custom", args.vx, args.vy, args.wz
        total = sum(s[4] for s in CYCLE_STAGES)
        pos = step_idx % total
        acc = 0
        for label, vx, vy, wz, dur in CYCLE_STAGES:
            acc += dur
            if pos < acc:
                return label, vx, vy, wz
        return CYCLE_STAGES[0][:4]

    print(
        f"[play_mujoco] loaded {ckpt_path} (iter={ck.get('iteration')}) | level={lvl} | "
        + ("mode=cycle" if args.cycle else f"cmd=({args.vx:+.2f}, {args.vy:+.2f}, {args.wz:+.2f})"),
        flush=True,
    )

    if args.headless_steps > 0:
        a_prev, hist = reset_robot()
        falls = 0
        with torch.no_grad():
            for step_idx in range(args.headless_steps):
                label, vx, vy, wz = current_cmd(step_idx)
                c = torch.tensor([[cy, vx, vy, wz, height, 0.0, 0.0]], dtype=torch.float32)
                S = _state_from_data(data, qpos_ids, qvel_ids)
                obs = mdp.obs(S, a_prev, xi, c)
                hist = torch.cat((hist[:, 1:], obs[:, None]), 1)
                act = actor(hist, mdp.goal(S, c)).clamp(-mdp.action_limit, mdp.action_limit)
                data.ctrl[actuator_ids] = (default_q + act[0] * action_scale).clamp(qlo, qhi).numpy()
                for _ in range(4):
                    mujoco.mj_step(model, data)
                a_prev = act
                S1 = _state_from_data(data, qpos_ids, qvel_ids)
                if bool(mdp.termination(S1, c)[0]):
                    falls += 1
                    a_prev, hist = reset_robot()
                elif mixed_parkour and data.qpos[0] >= terrain.goal_x:
                    a_prev, hist = reset_robot()
        S1 = _state_from_data(data, qpos_ids, qvel_ids)
        vb, wb = body_velocities(S1)
        print(
            f"[headless {args.headless_steps} steps] falls={falls} pos=({data.qpos[0]:.2f},{data.qpos[1]:.2f},{data.qpos[2]:.2f}) "
            f"vel=({float(vb[0,0]):+.2f},{float(vb[0,1]):+.2f},{float(wb[0,2]):+.2f})",
            flush=True,
        )
        return

    from mujoco import viewer as mj_viewer
    from live_viewer import _add_arrow, _add_sphere

    def _draw_overlays(viewer, S, c_vec):
        scn = viewer.user_scn
        scn.ngeom = 0
        if args.no_scan:
            return
        pos = S[0][0].detach().cpu().numpy().astype(np.float64)
        R = S[1][0].detach().cpu().numpy().astype(np.float64)
        vel = S[2][0].detach().cpu().numpy().astype(np.float64)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        cy_v, sy_v = np.cos(yaw), np.sin(yaw)
        origin = pos + np.array([0.0, 0.0, 0.28], dtype=np.float64)
        cmd_vx = float(cy_v * c_vec[0, 1] - sy_v * c_vec[0, 2])
        cmd_vy = float(sy_v * c_vec[0, 1] + cy_v * c_vec[0, 2])
        _add_arrow(scn, origin, origin + np.array([cmd_vx, cmd_vy, 0.0]), 0.015, [0.1, 0.9, 0.2, 0.95])
        _add_arrow(
            scn,
            origin + np.array([0.0, 0.0, 0.035]),
            origin + np.array([float(vel[0]), float(vel[1]), 0.035]),
            0.013,
            [1.0, 0.55, 0.1, 0.9],
        )
        wz_cmd = float(c_vec[0, 3])
        if abs(wz_cmd) > 0.05:
            fwd = np.array([cy_v, sy_v, 0.0], dtype=np.float64)
            left = np.array([-sy_v, cy_v, 0.0], dtype=np.float64)
            head = origin + 0.26 * fwd + np.array([0.0, 0.0, 0.07], dtype=np.float64)
            _add_arrow(scn, head, head + 0.32 * wz_cmd * left, 0.013, [0.2, 0.65, 1.0, 0.95])
        xy = field.scan_xy((S[0], S[1]), offsets=visual_offsets)
        z = (torch.zeros_like(xy[..., 0]) if use_plane else field.sample(xy))
        pts = torch.cat((xy, z.unsqueeze(-1)), dim=-1)[0].detach().cpu().numpy().astype(np.float64)
        for pt in pts:
            _add_sphere(scn, pt + np.array([0.0, 0.0, 0.01]), 0.012, [0.1, 0.9, 1.0, 0.6])

    a_prev, hist = reset_robot()
    step_idx = 0
    last_label = ""
    with mj_viewer.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[:] = 1
        viewer.cam.lookat[:] = [float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2])]
        viewer.cam.distance = 2.2
        viewer.cam.elevation = -20.0
        viewer.cam.azimuth = 135.0
        _draw_overlays(viewer, _state_from_data(data, qpos_ids, qvel_ids),
                       torch.tensor([[cy, args.vx, args.vy, args.wz, height, 0.0, 0.0]], dtype=torch.float32))
        viewer.sync()

        with torch.no_grad():
            while viewer.is_running():
                t0 = time.monotonic()
                label, vx, vy, wz = current_cmd(step_idx)
                if label != last_label:
                    print(f"[stage] {label} cmd=({vx:+.2f}, {vy:+.2f}, {wz:+.2f})", flush=True)
                    last_label = label

                c = torch.tensor([[cy, vx, vy, wz, height, 0.0, 0.0]], dtype=torch.float32)
                S = _state_from_data(data, qpos_ids, qvel_ids)
                obs = mdp.obs(S, a_prev, xi, c)
                hist = torch.cat((hist[:, 1:], obs[:, None]), 1)
                act = actor(hist, mdp.goal(S, c)).clamp(-mdp.action_limit, mdp.action_limit)
                data.ctrl[actuator_ids] = (default_q + act[0] * action_scale).clamp(qlo, qhi).numpy()
                for _ in range(4):
                    mujoco.mj_step(model, data)
                a_prev = act
                step_idx += 1

                S1 = _state_from_data(data, qpos_ids, qvel_ids)
                if bool(mdp.termination(S1, c)[0]):
                    print(f"[reset] fell at step {step_idx}, resetting to level {lvl}", flush=True)
                    a_prev, hist = reset_robot()
                elif mixed_parkour and data.qpos[0] >= terrain.goal_x:
                    print(f"[reset] reached the finish on {lane} level {lvl}", flush=True)
                    a_prev, hist = reset_robot()
                elif step_idx % 50 == 0:
                    vb, wb = body_velocities(S1)
                    m = mdp.command_metrics(S1, c)
                    print(
                        f"[t={step_idx * control_dt:5.1f}s] {label:16s} | cmd=({vx:+.2f},{vy:+.2f},{wz:+.2f}) "
                        f"actual=({float(vb[0,0]):+.2f},{float(vb[0,1]):+.2f},{float(wb[0,2]):+.2f}) "
                        f"clr={float(m['physical_clearance_m'][0]):.3f}m",
                        flush=True,
                    )

                if not args.no_follow:
                    viewer.cam.lookat[:] = [float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2])]
                _draw_overlays(viewer, _state_from_data(data, qpos_ids, qvel_ids), c)
                viewer.sync()
                if args.speed > 0:
                    dt_sleep = (control_dt / args.speed) - (time.monotonic() - t0)
                    if dt_sleep > 0:
                        time.sleep(dt_sleep)


if __name__ == "__main__":
    main()
