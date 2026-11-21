"""Generate two-phase WBL high-wall jump references on the 30 cm competition wall."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from generate_obstacle_gaits import DEFAULT_MODEL, FPS, LEGS, ROOT, SOLE_GAP, ObstacleIK, audit, ease, render

import mujoco as mj
import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

OUTPUT = ROOT / "amp_supermimic/data/wblwbl/competition_gaits"
DT = 1 / FPS
GRAVITY = 9.81
NAMES = [f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "calf")]

# Each pose specifies (base x, base z, pitch), (rear-pair x and sole z),
# and (front-pair x and sole z). All marks are in wall-local world coordinates.
START = ((-.490, .365, 0.), (-.725, 0.), (-.260, 0.))
LOAD = ((-.470, .324, -.08), (-.725, 0.), (-.260, 0.))
LAUNCH_FRONT = ((-.402, .419, -.31), (-.725, 0.), (-.187, .149))
FRONT_ON_WALL = ((-.256, .441, -.42), (-.519, 0.), (.001, .30))
RESET_REAR = ((-.245, .411, -.44), (-.464, 0.), (.001, .30))
LOAD_AGAIN = ((-.235, .392, -.46), (-.464, 0.), (.001, .30))
LAUNCH_REAR = ((-.153, .484, -.33), (-.464, 0.), (.030, .365))
FRONT_DOWN_DIRECT = ((.430, .475, .19), (.220, .322), (.655, 0.))
FRONT_DOWN_REAR_ON_WALL = ((.354, .499, .25), (.010, .30), (.550, 0.))
END_AFTER_REAR_TOUCH = ((.532, .362, 0.), (.315, 0.), (.758, 0.))


def blend(start, end, portion):
    start = np.asarray(start)
    return start + ease(portion)[..., None] * (np.asarray(end) - start)


def foot_path(keyframes, u):
    times, values = zip(*keyframes)
    return PchipInterpolator(times, values, axis=0)(u)


def make_plan(robot, route):
    poses, target_feet, contacts, labels = [], [], [], []
    current = START

    def append(label, body, rear, front, support, *, final=False):
        # Nearby segments share a boundary. The next segment supplies that row.
        keep = slice(None) if final else slice(None, -1)
        poses.extend(np.column_stack((body[:, 0], np.zeros(len(body)), body[:, 1:]))[keep])
        for r, f in zip(rear[keep], front[keep]):
            target_feet.append(((r[0], -.183, r[1]), (r[0], .183, r[1]),
                                (f[0], -.180, f[1]), (f[0], .180, f[1])))
        contacts.extend(support[keep])
        labels.extend([label] * len(body[keep]))

    def shift(label, frames, next_pose, *, lift_rear=0., lift_front=0.):
        nonlocal current
        u = np.arange(frames + 1) / frames
        body, rear, front = (blend(a, b, u) for a, b in zip(current, next_pose))
        support = np.ones((len(u), 4), dtype=bool)
        bump = 64 * (u * (1 - u))**3
        if lift_rear:
            rear[:, 1] += lift_rear * bump
            support[1:-1, :2] = False
        if lift_front:
            front[:, 1] += lift_front * bump
            support[1:-1, 2:] = False
        append(label, body, rear, front, support)
        current = next_pose

    def push(label, frames, launch, landing, flight_frames, front_midpoint):
        nonlocal current
        u = np.arange(frames + 1) / frames
        duration = flight_frames * DT
        vx = (landing[0][0] - launch[0][0]) / duration
        vz = (landing[0][1] - launch[0][1]) / duration + .5 * GRAVITY * duration
        body = CubicHermiteSpline(
            [0., 1.], [current[0], launch[0]],
            [[0., 0., 0.], [vx * frames * DT, vz * frames * DT, 0.]],
        )(u)
        front = foot_path(((0., current[2]), front_midpoint, (1., launch[2])), u)
        support = np.tile([True, True, False, False], (len(u), 1))
        support[0, 2:] = True
        append(label, body, blend(current[1], launch[1], u), front, support)
        current = launch

    def jump(label, frames, landing, rear_knots, front_knots):
        nonlocal current
        u = np.arange(frames + 1) / frames
        duration = frames * DT
        a, b = current[0], landing[0]
        body = blend(a, b, u)
        body[:, 0] = a[0] + (b[0] - a[0]) * u
        body[:, 1] = a[1] + (b[1] - a[1]) * u + .5 * GRAVITY * duration**2 * u * (1 - u)
        rear = foot_path(((0., current[1]), *rear_knots, (1., landing[1])), u)
        front = foot_path(((0., current[2]), *front_knots, (1., landing[2])), u)
        append(label, body, rear, front, np.zeros((len(u), 4), dtype=bool))
        current = landing

    def land_rear(frames, landing, *, via_top):
        nonlocal current
        u = np.arange(frames + 1) / frames
        body = blend(current[0], landing[0], u)
        if via_top:
            body[:, 1] = current[0][1] + ease((u - .21) / .79) * (landing[0][1] - current[0][1])
        apex = .325 if via_top else current[1][1]
        x = current[1][0] + ease((u - .10) / .66) * (landing[1][0] - current[1][0])
        z = np.where(u < .3, current[1][1] + (apex - current[1][1]) * ease(u / .27),
                     apex + (landing[1][1] - apex) * ease((u - .39) / .61))
        support = np.tile([False, False, True, True], (len(u), 1))
        support[0, :2] = via_top
        append("Rear pair lands on far floor", body, np.column_stack((x, z)),
               blend(current[2], landing[2], u), support)
        current = landing

    shift("Crouch and load rear legs", 12, LOAD)
    push("First rear-pair push", 7, LAUNCH_FRONT, FRONT_ON_WALL, 10, (.48, (-.218, .055)))
    jump("Front pair jumps onto wall", 10, FRONT_ON_WALL, ((.53, (-.629, .074)),),
         ((.28, (-.131, .261)), (.65, (-.050, .330))))
    shift("Brace front pair on crest", 14, ((-.252, .428, -.42), FRONT_ON_WALL[1], FRONT_ON_WALL[2]))
    shift("Reset rear pair for second push", 14, RESET_REAR, lift_rear=.044)
    shift("Compress for second jump", 8, LOAD_AGAIN)
    direct = route == "direct"
    front_down = FRONT_DOWN_DIRECT if direct else FRONT_DOWN_REAR_ON_WALL
    push("Second rear-pair push", 9, LAUNCH_REAR, front_down, 16, (.5, (.001, .312)))
    jump("Second jump - both pairs clear the wall", 16, front_down,
         ((.27, (-.355, .215)), (.55, (-.145, .389)), (.77, (.099, .380) if direct else (-.043, .390))),
         ((.36, (.226, .346)), (.73, (.487, .182))))
    if direct:
        land_rear(16, ((.544, .381, .07), (.345, 0.), FRONT_DOWN_DIRECT[2]), via_top=False)
        shift("Settle on far side", 12, ((.546, .365, 0.), (.345, 0.), FRONT_DOWN_DIRECT[2]))
    else:
        shift("Rear pair braces on wall", 8,
              ((.416, .483, .19), FRONT_DOWN_REAR_ON_WALL[1], FRONT_DOWN_REAR_ON_WALL[2]))
        shift("Front pair moves forward on far floor", 15,
              ((.451, .474, .17), FRONT_DOWN_REAR_ON_WALL[1], END_AFTER_REAR_TOUCH[2]), lift_front=.046)
        land_rear(18, END_AFTER_REAR_TOUCH, via_top=True)
        shift("Settle on far side", 8, END_AFTER_REAR_TOUCH)
    append("Complete", np.asarray([current[0]]), [current[1]], [current[2]], [[True] * 4], final=True)
    feet = np.asarray(target_feet)
    feet[..., 2] += robot.kin.radius + SOLE_GAP
    return np.asarray(poses), feet, np.asarray(contacts, dtype=bool), labels


def check_physics(robot, rows, kp, kd):
    """Free-base tracking diagnostic. Apply only clipped torque to the 12 leg actuators."""
    model, data = robot.model, robot.data
    mj.mj_resetData(model, data)
    robot.load(rows[0])
    joint_ids = [model.joint(name).id for name in NAMES]
    dof_ids = model.jnt_dofadr[joint_ids]
    actuators = [next(a for a in range(model.nu) if model.actuator_trnid[a, 0] == j) for j in joint_ids]
    wall_id = model.geom("obstacle").id
    feet = [
        next(g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body(leg + "_calf").id
             and model.geom_type[g] == mj.mjtGeom.mjGEOM_SPHERE)
        for leg in LEGS
    ]
    to_step = np.arange(int((len(rows) - 1) * DT / model.opt.timestep) + 1) * model.opt.timestep
    time = np.arange(len(rows)) * DT
    targets = np.column_stack([np.interp(to_step, time, column) for column in rows[:, 7:].T])
    limit = model.actuator_ctrlrange[actuators]
    max_x, max_front_sole = float(data.qpos[0]), 0.
    pair_frames = 0
    for target in targets:
        torque = kp * (target - data.qpos[robot.joint_slots]) - kd * data.qvel[dof_ids]
        data.ctrl[actuators] = np.clip(torque, limit[:, 0], limit[:, 1])
        mj.mj_step(model, data)
        max_x = max(max_x, float(data.qpos[0]))
        max_front_sole = max(max_front_sole, float((data.geom_xpos[feet[2:], 2] - robot.kin.radius[2:]).min()))
        front_support = set()
        for c in data.contact:
            if wall_id in c.geom and c.pos[2] >= robot.height - .01 and abs(c.frame[2]) > .7:
                front_support.update(set(c.geom).intersection(feet[2:]))
        pair_frames += len(front_support) == 2
    final_feet_x = data.geom_xpos[feet, 0].tolist()
    cleared = min(final_feet_x) > robot.length / 2 + max(robot.kin.radius)
    return {"kp": kp, "kd": kd, "body_max_x_m": max_x, "max_lower_front_sole_height_m": max_front_sole,
            "front_pair_simultaneous_crest_contact_s": pair_frames * model.opt.timestep,
            "final_body_xyz_m": data.qpos[:3].tolist(), "final_feet_x_m": final_feet_x,
            "completed_two_phase_crossing": bool(cleared and pair_frames > 0 and data.qpos[2] > .22)}


def write_clip(folder, robot, rows, toes, stance, phases, route, args):
    direct = route == "direct"
    report, top = audit(robot, rows, toes, stance, "wall",
                        wall_top_feet=(2, 3) if direct else (0, 1, 2, 3), static_support=False)
    if direct and top[:, :2].any():
        raise RuntimeError("Direct route unexpectedly uses a rear crest support")
    paired = top[:, 2:].all(axis=1)
    airborne = ~stance.any(axis=1)
    flight_starts = np.flatnonzero(airborne & ~np.r_[False, airborne[:-1]])
    flight_ends = np.flatnonzero(airborne & ~np.r_[airborne[1:], False]) + 1
    if len(flight_starts) != 2 or not paired[flight_ends[0]:flight_starts[1]].any():
        raise RuntimeError("Expected first flight, front crest support, and second flight")
    report.update(
        route=route, max_body_root_height_m=float(rows[:, 2].max()),
        planned_front_pair_crest_support_s=float(paired.sum() * DT),
        planned_flight_intervals_s=(np.column_stack((flight_starts, flight_ends)) * DT).tolist(),
        phase_labels={"0": "swing_or_flight", "1": "floor_support", "2": "crest_support"},
        source_mjcf_sha256=hashlib.sha256(args.mjcf.read_bytes()).hexdigest(),
    )
    if args.physics_check:
        report["free_base_pd_tracking"] = [check_physics(robot, rows, *gains)
                                           for gains in ((40., 1.), (60., 2.), (90., 2.8), (120., 3.5))]
    folder.mkdir(parents=True, exist_ok=True)
    csv = folder / f"{folder.name}.csv"
    np.savetxt(csv, rows, delimiter=",", fmt="%.10g")
    np.savetxt(folder / "support.csv", np.where(top, 2, stance.astype(int)), fmt="%d",
               delimiter=",", header=",".join(LEGS), comments="")
    (folder / "train.json").write_text(json.dumps({"version": 1, "root": ".", "source_fps": FPS,
        "quaternion_order": "xyzw", "joint_names": NAMES, "files": [csv.name]}, indent=2) + "\n")
    boundaries = [0, *(i for i in range(1, len(phases)) if phases[i] != phases[i-1]), len(phases)]
    report["phases"] = [{"name": phases[a], "start_frame": a, "stop_frame_exclusive": b}
                        for a, b in zip(boundaries[:-1], boundaries[1:])]
    report["csv_sha256"] = hashlib.sha256(csv.read_bytes()).hexdigest()
    (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{folder.name}: {len(rows)} frames, {report['duration_s']:.2f} s", flush=True)
    if args.render:
        title = "Rear pair clears crest directly" if direct else "Rear pair briefly braces on crest"
        render(robot, rows, folder / "reference.mp4", "wall", titles=("WBL - two-stage high-wall jump", title))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("all", "direct", "rear_touch"), default="all")
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--physics-check", action="store_true")
    args = parser.parse_args()
    for route in ("direct", "rear_touch") if args.route == "all" else (args.route,):
        robot = ObstacleIK(args.mjcf, NAMES, "wall")
        roots, toes, stance, phases = make_plan(robot, route)
        rows = np.array([robot.solve(root, target) for root, target in zip(roots, toes)])
        write_clip(args.output / f"high_wall_jump_{route}", robot, rows, toes, stance, phases, route, args)


if __name__ == "__main__":
    main()
