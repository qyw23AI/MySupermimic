"""Generate four-beat WBLWBL Dog V1 references for a 3 x 0.4 x 0.1 m bridge."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj
import numpy as np
from scipy.optimize import LinearConstraint, least_squares, minimize
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2]
FPS = 50
BRIDGE_LENGTH, BRIDGE_WIDTH, BRIDGE_HEIGHT = 3.0, 0.40, 0.10
LEGS = ("RR", "RL", "FR", "FL")
SIDES = np.array([-1.0, 1.0, -1.0, 1.0])
DUTY = 0.80
SWING_HEIGHT = 0.035
SOLE_CLEARANCE = 0.0005
PROFILES = (
    ("bridge_walk_w28_v22", 0.28, 0.22, 1.20),
    ("bridge_walk_w30_v28", 0.30, 0.28, 1.20),
    ("bridge_walk_w32_v32", 0.32, 0.32, 1.12),
)


def footprint_bounds(points):
    # ConvexHull returns unit outward normals and edge offsets.
    return ConvexHull(points).equations


def design_plan(time, period, speed, track, neutral_feet, radius, *, gait):
    offsets, duty = (([0.0, 0.50, 0.50, 0.0], .62) if gait == "trot"
                     else ([0.0, 0.50, 0.75, 0.25], DUTY))
    phase = time[:, None] / period + offsets
    cycle = np.floor(phase)
    fraction = phase - cycle
    stance = fraction <= duty + 1e-10
    u = np.clip((fraction - duty) / (1 - duty), 0, 1)
    transfer = u**3 * (10 - 15 * u + 6 * u**2)
    touchdown = (cycle - offsets) * period
    feet = np.empty((len(time), 4, 3))
    feet[:, :, 0] = neutral_feet[:, 0] + speed * (touchdown + duty * period / 2 + period * transfer)
    feet[:, :, 1] = SIDES * track / 2
    feet[:, :, 2] = radius + SOLE_CLEARANCE + SWING_HEIGHT * 64 * (u * (1 - u))**3
    return feet, stance, duty


def balance_offsets(time, period, feet, stance, body_xy, neutral_com_xy):
    """Fit a smooth periodic trunk shift subject to nominal static support margins."""
    phase = 2 * np.pi * time / period
    basis = np.column_stack([np.ones(len(time)), np.sin(phase), np.cos(phase),
                             np.sin(2 * phase), np.cos(2 * phase),
                             np.sin(3 * phase), np.cos(3 * phase)])
    constraints, upper = [], []
    for i in range(len(time)):
        planes = footprint_bounds(feet[i, stance[i], :2])
        normal = planes[:, :2]
        constraints.extend(np.einsum("ij,k->ijk", normal, basis[i]).reshape(-1, 14))
        upper.extend(-0.010 - planes[:, 2] - normal @ (body_xy[i] + neutral_com_xy))
    weights = np.tile([1.0, 1.0, 1.0, 3.0, 3.0, 8.0, 8.0], 2)
    result = minimize(
        lambda coef: 0.5 * np.dot(weights * coef, coef), np.zeros(14),
        jac=lambda coef: weights * coef, method="SLSQP",
        constraints=[LinearConstraint(constraints, -np.inf, upper)],
        options={"ftol": 1e-11, "maxiter": 150},
    )
    if not result.success:
        raise RuntimeError(f"Support trajectory fit failed: {result.message}")
    return basis @ result.x.reshape(2, -1).T


def solve_cycle(robot, neutral, track, speed, period, *, gait="walk"):
    from retarget_lvr_reference import metrics

    time = np.arange(round(period * FPS)) / FPS
    neutral_feet = robot.fk(neutral)
    mj.mj_comPos(robot.model, robot.data)
    root_body = robot.model.jnt_bodyid[0]
    neutral_com = robot.data.subtree_com[root_body, :2].copy()
    targets, stance, duty = design_plan(time, period, speed, track, neutral_feet, robot.radius, gait=gait)
    root = np.zeros((len(time), 7))
    root[:, 0] = speed * time
    if gait == "walk":
        root[:, :2] += balance_offsets(time, period, targets, stance, root[:, :2], neutral_com)
    # A level trunk with slightly more knee flexion than the nominal WBL stance.
    root[:, 2] = 0.365
    root[:, 6] = 1.0
    lo, hi = robot.limits.copy()
    hi[2::3] = np.minimum(hi[2::3], -1.10)
    rows, feet, support_margins = [], [], []
    for center, target, support in zip(root, targets, stance):
        def residual(q):
            return np.r_[(robot.fk(q, root=center[:3]) - target).ravel(), 0.0002 * (q - neutral)]

        # Start every solve on the same bent-knee branch so cycles join cleanly.
        fit = least_squares(residual, neutral, bounds=(lo + 1e-5, hi - 1e-5),
                            max_nfev=50, gtol=1e-11)
        if not fit.success:
            raise RuntimeError(f"Swing/stance IK failed: {fit.message}")
        true_feet = robot.fk(fit.x, root=center[:3])
        mj.mj_comPos(robot.model, robot.data)
        if gait == "walk":
            planes = footprint_bounds(true_feet[support, :2])
            support_margins.append(np.min(-(planes[:, :2] @ robot.data.subtree_com[root_body, :2] + planes[:, 2])))
        feet.append(true_feet)
        rows.append(np.r_[center, fit.x])

    rows, feet = np.array(rows), np.array(feet)
    error = np.linalg.norm(feet - targets, axis=2)
    sole = feet[:, :, 2] - robot.radius
    edge_margin = BRIDGE_WIDTH / 2 - (np.abs(feet[:, :, 1]) + robot.radius)
    geom_speed = np.linalg.norm(np.diff(np.r_[feet, feet[:1] + [speed * period, 0, 0]], axis=0), axis=2) * FPS
    stays = stance & np.roll(stance, -1, axis=0)
    static_margin = float(min(support_margins)) if support_margins else None
    if gait == "trot" and (not np.array_equal(stance[:, 0], stance[:, 3])
                           or not np.array_equal(stance[:, 1], stance[:, 2])
                           or not np.any(~stance[:, 0]) or not np.any(~stance[:, 1])
                           or stance.sum(axis=1).min() != 2):
        raise RuntimeError("Expected two alternating diagonal pairs with stance overlap")
    if (error.max() > .001 or sole.min() < 0 or edge_margin.min() < .01
            or (static_margin is not None and static_margin < .005)):
        raise RuntimeError(f"Geometry check failed: {track=}, {speed=}, {error.max()=}, "
                           f"{sole.min()=}, {edge_margin.min()=}, {static_margin=}")
    stats = metrics(robot, rows, neutral_feet)
    wrap = np.r_[rows, rows[:1] + np.r_[speed * period, np.zeros(18)]]
    stats.update(
        nominal_speed_m_s=speed, period_s=period, duty_factor=duty, feet_center_track_m=track,
        normal_stance_track_m=neutral_feet[[1, 3], 1].mean() - neutral_feet[[0, 2], 1].mean(),
        sphere_radius_m=robot.radius.tolist(), min_lateral_edge_margin_m=float(edge_margin.min()),
        max_stance_slip_m_s=float(geom_speed[stays].max()),
        min_projected_com_support_margin_m=static_margin,
        min_support_feet=int(stance.sum(axis=1).min()),
        maximum_ik_residual_m=float(error.max()),
        max_joint_step_rad=float(np.abs(np.diff(wrap[:, 7:], axis=0)).max()),
        peak_swing_clearance_m=float(sole.max()),
    )
    return rows, feet, stance, stats


def extend_clip(rows, feet, stance, stride):
    # Tile the cycle up to the last frame where the complete foot spheres fit on the deck.
    minimum_foot_x = feet[:, :, 0].min()
    cycles = int(np.ceil(BRIDGE_LENGTH / stride)) + 1
    poses = np.tile(rows, (cycles, 1))
    points = np.tile(feet, (cycles, 1, 1))
    travel = np.repeat(np.arange(cycles) * stride, len(rows))
    poses[:, 0] += travel
    points[:, :, 0] += travel[:, None]
    margin = 0.04
    deck_start = minimum_foot_x - margin
    fits = points[:, :, 0].max(axis=1) <= deck_start + BRIDGE_LENGTH - margin
    stop = int(np.flatnonzero(~fits)[0])
    return poses[:stop], np.tile(stance, (cycles, 1))[:stop], deck_start


def bridge_model(path, deck_start):
    spec = mj.MjSpec.from_file(str(path))
    for geom in spec.worldbody.geoms:
        if geom.name != "floor":
            geom.contype = 0
            geom.conaffinity = 0
            geom.rgba = [0, 0, 0, 0]
    spec.worldbody.add_geom(
        name="review_bridge", type=mj.mjtGeom.mjGEOM_BOX,
        pos=[deck_start + BRIDGE_LENGTH / 2, 0, BRIDGE_HEIGHT / 2],
        size=[BRIDGE_LENGTH / 2, BRIDGE_WIDTH / 2, BRIDGE_HEIGHT / 2],
        rgba=[0.24, 0.54, 0.36, 1.0],
    )
    return spec.compile()


def review_clip(model, names, rows, video):
    data = mj.MjData(model)
    joints = [model.joint(name).qposadr[0] for name in names]
    foot_ids = {
        i for i in range(model.ngeom)
        if model.body(model.geom_bodyid[i]).name in {f"{leg}_calf" for leg in LEGS}
        and model.geom_type[i] == mj.mjtGeom.mjGEOM_SPHERE
    }
    ground_ids = {model.geom("floor").id, model.geom("review_bridge").id}
    collisions = set()
    proc = None
    renderer = None
    try:
        if video:
            renderer = mj.Renderer(model, height=360, width=640)
            proc = subprocess.Popen([
                "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                "-video_size", "1280x360", "-framerate", "25", "-i", "-", "-an",
                "-vf", "pad=iw:ih+30:0:30:color=0x151c26,"
                "drawtext=text='WBL bridge reference | oblique view':x=14:y=7:fontsize=15:fontcolor=white,"
                "drawtext=text='Top view | deck width 40 cm':x=654:y=7:fontsize=15:fontcolor=white",
                "-c:v", "libx264", "-threads", "2", "-crf", "20", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(video),
            ], stdin=subprocess.PIPE)
            cameras = []
            for azimuth, elevation in ((135, -24), (90, -85)):
                cam = mj.MjvCamera()
                cam.type = mj.mjtCamera.mjCAMERA_FREE
                cam.distance, cam.azimuth, cam.elevation = 1.36, azimuth, elevation
                cameras.append(cam)
        for i, row in enumerate(rows):
            data.qpos[:3] = row[:3] + [0, 0, BRIDGE_HEIGHT]
            data.qpos[3:7] = row[[6, 3, 4, 5]]
            data.qpos[joints] = row[7:]
            mj.mj_forward(model, data)
            for contact in data.contact:
                a, b = contact.geom
                if contact.dist < -1e-5 and not ((a in foot_ids and b in ground_ids)
                                                or (b in foot_ids and a in ground_ids)):
                    collisions.add((model.geom(a).name or str(a), model.geom(b).name or str(b)))
            if proc and i % 2 == 0:
                panels = []
                for cam in cameras:
                    cam.lookat[:] = [row[0], 0, .26]
                    renderer.update_scene(data, camera=cam)
                    panels.append(renderer.render().copy())
                proc.stdin.write(np.concatenate(panels, axis=1).tobytes())
    finally:
        if renderer:
            renderer.close()
        if proc:
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError("ffmpeg failed")
    if collisions:
        raise RuntimeError(f"Unexpected MJCF contacts on the bridge: {sorted(collisions)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", type=Path, default=ROOT.parent /
                        "robocon2027_sim_control/deploy_cpp/robot/wblwbl_dog_v1/xml/wblwbl_dog_v1.xml")
    parser.add_argument("--gait", choices=("walk", "trot"), default="walk")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--render", action="store_true", help="Render synchronized oblique and top views")
    args = parser.parse_args()
    if args.output is None:
        folder = "narrow_bridge_trot" if args.gait == "trot" else "narrow_bridge_candidates"
        args.output = ROOT / "amp_supermimic/data/wblwbl" / folder
    profiles = PROFILES if args.gait == "walk" else (
        ("bridge_trot_w26_v26", .26, .26, .84),
        ("bridge_trot_w28_v32", .28, .32, .76),
        ("bridge_trot_w30_v38", .30, .38, .72),
    )

    # Reuse the existing WBL reference contract and geometric FK implementation.
    from retarget_go2 import target_init
    from retarget_lvr_reference import Kinematics
    names = json.loads((ROOT / "amp_supermimic/configs/wblwbl_dog_v1.plan.json").read_text())[
        "contract"]["policy_joint_names"]
    if [f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "calf")] != names:
        raise ValueError("Unexpected WBLWBL joint order")
    robot = Kinematics(args.mjcf, names)
    neutral = np.array([target_init()["default_joint_angles"][name] for name in names])
    args.output.mkdir(parents=True, exist_ok=True)
    files, clips = [], {}
    for name, track, speed, period in profiles:
        rows, feet, stance, stats = solve_cycle(robot, neutral, track, speed, period, gait=args.gait)
        rows, stance, start = extend_clip(rows, feet, stance, stride=speed * period)
        # AMP exports use a zero-height world floor; the elevated deck is only in the review scene.
        rows[:, 0] -= start
        bridge = bridge_model(args.mjcf, 0.0)
        csv = args.output / f"{name}.csv"
        video = args.output / f"{name}.mp4" if args.render else None
        review_clip(bridge, names, rows, video)
        np.savetxt(csv, rows, delimiter=",", fmt="%.10g")
        np.savetxt(args.output / f"{name}_stance.csv", stance, delimiter=",", fmt="%d",
                   header=",".join(LEGS), comments="")
        stats.update(frames=len(rows), duration_s=(len(rows)-1)/FPS, torso_travel_m=float(np.ptp(rows[:, 0])),
                     csv_sha256=hashlib.sha256(csv.read_bytes()).hexdigest(),
                     unexpected_penetrating_contact_pairs=0)
        files.append(csv.name)
        clips[name] = stats
        print(f"{name}: {stats['frames']} frames, "
              f"edge {stats['min_lateral_edge_margin_m']*1000:.1f} mm, "
              f"minimum planned supports {stats['min_support_feet']}", flush=True)
    manifest = {"version": 1, "root": ".", "source_fps": FPS, "quaternion_order": "xyzw",
                "joint_names": names, "files": files}
    (args.output / "train.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = {"source_kind": ("procedural_four_beat_footsteps_and_wbl_ik" if args.gait == "walk"
                              else "procedural_diagonal_footsteps_and_wbl_ik"),
              "status": "kinematic_candidate",
              "bridge_length_width_height_m": [BRIDGE_LENGTH, BRIDGE_WIDTH, BRIDGE_HEIGHT],
              "target_mjcf_sha256": hashlib.sha256(args.mjcf.read_bytes()).hexdigest(),
              "stance_labels_leg_order": LEGS, "fps": FPS, "clips": clips}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
