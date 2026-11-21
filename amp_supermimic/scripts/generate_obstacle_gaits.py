"""Generate WBL bridge trot and wall climb references with MuJoCo geometry checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from generate_bridge_walk import BRIDGE_HEIGHT, BRIDGE_LENGTH, BRIDGE_WIDTH, FPS, LEGS, ROOT, SIDES

import mujoco as mj
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp

from retarget_go2 import target_init
from retarget_lvr_reference import Kinematics

WALL_SIZE = (.08, .80, .30)  # Longitudinal thickness, lateral width, height.
SOLE_GAP = .001
AUDIT_RATE = 4 * FPS
DEFAULT_MODEL = (
    ROOT.parent / "robocon2027_sim_control/deploy_cpp/robot/wblwbl_dog_v1/xml/wblwbl_dog_v1.xml"
)
DEFAULT_OUTPUT = ROOT / "amp_supermimic/data/wblwbl/competition_gaits"
CLIPS = {"bridge": "single_bridge_trot", "wall": "high_wall_climb"}


def ease(u):
    """Minimum-jerk interpolation with zero endpoint velocity and acceleration."""
    u = np.clip(u, 0., 1.)
    return u**3 * (10 - 15 * u + 6 * u**2)


class ObstacleIK:
    def __init__(self, asset, names, kind):
        self.kin = Kinematics(asset, names)
        self.neutral = np.array([target_init()["default_joint_angles"][name] for name in names])
        self.previous = self.neutral.copy()
        self.lo, self.hi = self.kin.limits.copy()
        self.hi[2::3] = np.minimum(self.hi[2::3], -.50)
        spec = mj.MjSpec.from_file(str(asset))
        for geom in list(spec.worldbody.geoms):
            if geom.name != "floor":
                geom.contype = geom.conaffinity = 0
                geom.rgba = [0, 0, 0, 0]
        dimensions = (BRIDGE_LENGTH, BRIDGE_WIDTH, BRIDGE_HEIGHT) if kind == "bridge" else WALL_SIZE
        self.length, self.width, self.height = dimensions
        self.center_x = self.length / 2 if kind == "bridge" else 0.
        spec.worldbody.add_geom(
            name="obstacle", type=mj.mjtGeom.mjGEOM_BOX, size=np.array(dimensions) / 2,
            pos=[self.center_x, 0, self.height / 2], rgba=[.19, .54, .34, 1],
        )
        self.model = spec.compile()
        self.data = mj.MjData(self.model)
        self.joint_slots = [self.model.joint(name).qposadr[0] for name in names]

    def load(self, row):
        self.data.qpos[:3] = row[:3]
        self.data.qpos[3:7] = row[[6, 3, 4, 5]]
        self.data.qpos[self.joint_slots] = row[7:]
        mj.mj_forward(self.model, self.data)

    def solve(self, root, toes):
        orientation = Rotation.from_euler("y", root[3])
        local = orientation.inv().apply(toes - root[:3])

        def residual(joints):
            return (self.kin.fk(joints) - local).ravel()

        result = least_squares(residual, self.previous, bounds=(self.lo + 1e-7, self.hi - 1e-7),
                               max_nfev=60, gtol=1e-10)
        if not result.success:
            raise RuntimeError(f"IK solve failed at root {root}: {result.message}")
        self.previous = result.x
        return np.r_[root[:3], orientation.as_quat(), result.x]


def bridge_plan(robot):
    speed, period, duty = .32, .84, .63
    stride = speed * period
    time = np.arange(round(4.32 / speed * FPS)) / FPS
    body_x = -.57 + speed * time
    # RR–FL and RL–FR remain half a cycle apart through both height transitions.
    phase = (time[:, None] / period + [0., .5, .5, 0.]) % 1.
    swing = np.clip((phase - duty) / (1 - duty), 0., 1.)
    stance = swing == 0.
    neutral_x = robot.kin.fk(robot.neutral)[:, 0]
    last_step = neutral_x + body_x[:, None] - stride * (phase - duty / 2)

    def placement(x):
        for edge in (0., BRIDGE_LENGTH):
            offset = x - edge
            x = np.where(np.abs(offset) < .044, edge + np.where(offset < 0., -.044, .044), x)
        return x, BRIDGE_HEIGHT * ((x > 0.) & (x < BRIDGE_LENGTH))

    start_x, start_z = placement(last_step)
    end_x, end_z = placement(last_step + stride)
    foot_x = start_x + (end_x - start_x) * ease(swing)
    foot_z = start_z + .036 * 64 * (swing * (1 - swing))**3
    change_height = start_z != end_z
    apex = np.maximum(start_z, end_z) + .040
    edge_z = np.where(swing < .22, start_z + (apex - start_z) * ease(swing / .22),
                      apex + (end_z - apex) * ease((swing - .76) / .24))
    edge_x = start_x + (end_x - start_x) * ease((swing - .20) / .58)
    foot_x = np.where(change_height, edge_x, foot_x)
    foot_z = np.where(change_height, edge_z, foot_z)
    feet = np.stack((foot_x, np.broadcast_to(.14 * SIDES, foot_x.shape),
                     foot_z + robot.kin.radius + SOLE_GAP), axis=-1)
    up = ease((body_x + .29) / .58)
    down = ease((body_x - (BRIDGE_LENGTH - .29)) / .58)
    root = np.column_stack((body_x, np.zeros_like(body_x), .365 + BRIDGE_HEIGHT * (up - down),
                            -.14 * (4 * up * (1 - up) - 4 * down * (1 - down))))
    phases = np.full(len(root), "Bridge deck - diagonal trot", dtype=object)
    phases[(start_x < 0.).any(axis=1)] = "Approach and step onto deck"
    phases[(end_x > BRIDGE_LENGTH).any(axis=1)] = "Trot off the far end"
    return root, feet, stance, phases


def wall_plan(robot):
    roots = [np.array([-.43, 0., .365, 0.])]
    feet = [np.array([[-.65, -.184, 0.], [-.65, .184, 0.],
                      [-.19, -.184, 0.], [-.19, .184, 0.]])]
    support, labels = [np.ones(4, dtype=bool)], ["Ready"]

    def phase(label, seconds, root_target, leg=None, landing=None, lift=.06):
        start_root = roots[-1].copy()
        start_feet = feet[-1].copy()
        for u in np.linspace(0., 1., round(seconds * FPS) + 1)[1:]:
            roots.append(start_root + ease(u) * (np.asarray(root_target) - start_root))
            toes = start_feet.copy()
            standing = np.ones(4, dtype=bool)
            if leg is not None:
                start, end = start_feet[leg], np.asarray(landing)
                # Advance to a point short of the wall while lifting, then traverse its upper edge.
                if start[2] < end[2]:
                    progress = .32 * ease(u / .33) + .68 * ease((u - .35) / .52)
                else:
                    delay = .04 if start[2] > end[2] else .16
                    progress = ease((u - delay) / (.78 - delay))
                toes[leg, :2] = start[:2] + progress * (end[:2] - start[:2])
                apex = max(start[2], end[2]) + lift
                toes[leg, 2] = (start[2] + (apex - start[2]) * ease(u / .36) if u < .46
                                else apex + (end[2] - apex) * ease((u - .57) / .43))
                standing[leg] = u >= 1.
            feet.append(toes)
            support.append(standing)
            labels.append(label)

    rr, rl, fr, fl = range(4)
    top = WALL_SIZE[2]
    phase("Raise front trunk", .50, [-.37, 0., .420, -.47])
    phase("Right front onto wall", .66, [-.33, .027, .433, -.44], fr, [.01, -.18, top], .046)
    phase("Left front onto wall", .66, [-.30, -.016, .456, -.51], fl, [.01, .18, top], .025)
    phase("Hold both front feet on top", .40, [-.28, 0., .452, -.42])
    phase("Right rear steps under trunk", .60, [-.24, .022, .444, -.29], rr, [-.275, -.184, 0.], .052)
    phase("Left rear steps under trunk", .60, [-.205, -.018, .449, -.25], rl, [-.275, .184, 0.], .052)
    phase("Lift body over near edge", .50, [-.096, 0., .490, -.13])
    # These two placements have a narrow reach/clearance window. Parameters were
    # fitted against the shin and wall collision geometry at sub-frame times.
    phase("Right front onto far floor", .72, [-.048785, .016, .492536, -.065038],
          fr, [.299012, -.184, 0.], .015247)
    phase("Left front onto far floor", .72, [-.011242, -.016, .482669, -.064770],
          fl, [.295179, .184, 0.], .008101)
    phase("Move weight to far side", .36, [.036, 0., .440, .12])
    phase("Right rear onto wall", .70, [.098, .022, .437, .29], rr, [.01, -.18, top], .029)
    phase("Left rear onto wall", .70, [.140, -.016, .443, .34], rl, [.01, .18, top], .029)
    phase("Hold both rear feet on top", .30, [.22, 0., .461, .25])
    phase("Advance right front on floor", .65, [.29, .019, .481, .20], fr, [.715, -.184, 0.], .05)
    phase("Advance left front on floor", .65, [.37, -.019, .484, .16], fl, [.715, .184, 0.], .05)
    phase("Move hind hips beyond crest", .50, [.421, 0., .490, .18])
    phase("Right rear onto far floor", .70, [.478, .022, .473, .15], rr, [.324, -.184, 0.], .020)
    phase("Left rear onto far floor", .70, [.519, -.016, .443, .11], rl, [.324, .184, 0.], .020)
    phase("Settle on far side", .42, [.555, 0., .369, 0.])
    toes = np.asarray(feet)
    toes[:, :, 2] += robot.kin.radius + SOLE_GAP
    return np.asarray(roots), toes, np.asarray(support), labels


def audit(robot, rows, toes, stance, kind, *, wall_top_feet=(0, 1, 2, 3), static_support=True):
    feet = np.array([robot.kin.fk(row[7:], row[3:7], row[:3]) for row in rows])
    error = np.linalg.norm(feet - toes, axis=-1)
    inner_edge = robot.length / 2 - np.abs(feet[..., 0] - robot.center_x)
    side_edge = robot.width / 2 - np.abs(feet[..., 1])
    foot_on_top = stance & (np.abs(feet[..., 2] - robot.kin.radius - robot.height) < .002)
    edge_margin = np.minimum(inner_edge, side_edge) - robot.kin.radius
    top_margin = float(edge_margin[foot_on_top].min())
    near_side, far_side = robot.center_x + np.asarray([-1., 1.]) * robot.length / 2
    if (not np.isfinite(rows).all() or error.max() > .0005 or top_margin < .004
            or np.any(rows[:, 7:] < robot.lo) or np.any(rows[:, 7:] > robot.hi)
            or np.any(feet[0, :, 0] + robot.kin.radius >= near_side)
            or np.any(feet[-1, :, 0] - robot.kin.radius <= far_side)):
        raise RuntimeError(f"{kind} pose audit failed: IK {error.max():.6f} m, margin {top_margin:.6f} m")
    if kind == "bridge":
        if (not np.array_equal(stance[:, 0], stance[:, 3])
                or not np.array_equal(stance[:, 1], stance[:, 2])
                or stance.sum(axis=1).min() != 2):
            raise RuntimeError("Bridge plan lost its diagonal support pairing")
    elif not foot_on_top[:, wall_top_feet].any(axis=0).all():
        raise RuntimeError("A required foot never reached a planned stance on the wall")

    time = np.arange(len(rows)) / FPS
    dense_time = np.arange((len(rows) - 1) * AUDIT_RATE // FPS + 1) / AUDIT_RATE
    dense = np.column_stack([np.interp(dense_time, time, column) for column in rows.T])
    dense[:, 3:7] = Slerp(time, Rotation.from_quat(rows[:, 3:7]))(dense_time).as_quat()
    worst = (0., None)
    for i, row in enumerate(dense):
        robot.load(row)
        for contact in robot.data.contact:
            if contact.dist < worst[0]:
                worst = (float(contact.dist), i / AUDIT_RATE)
    if worst[0] < -.00010:
        raise RuntimeError(f"{kind}: geometry penetration {-worst[0]:.6f} m at {worst[1]:.3f}s")

    # Projected COM is an informative check for the stepped climb. The actual
    # support wrench and dynamic feasibility require contact simulation.
    minimum_com_margin = None
    if kind == "wall" and static_support:
        margins = []
        for row, centers, support in zip(rows, feet, stance):
            robot.load(row)
            planes = ConvexHull(centers[support, :2]).equations
            com = robot.data.subtree_com[robot.model.jnt_bodyid[0], :2]
            margins.append(float(np.min(-(planes[:, :2] @ com + planes[:, 2]))))
        minimum_com_margin = min(margins)

    stay = stance[:-1] & stance[1:]
    return {
        "frames": len(rows), "fps": FPS, "duration_s": float(time[-1]),
        "body_forward_travel_m": float(rows[-1, 0] - rows[0, 0]),
        "obstacle_length_width_height_m": [robot.length, robot.width, robot.height],
        "nominal_step_width_m": .28 if kind == "bridge" else .368,
        "planned_support_frames_on_top_by_leg": foot_on_top.sum(axis=0).tolist(),
        "maximum_foot_fit_error_m": float(error.max()),
        "max_planned_stance_foot_speed_m_s": float(np.linalg.norm(np.diff(feet, axis=0), axis=-1)[stay].max() * FPS),
        "minimum_planted_foot_edge_margin_m": top_margin,
        "minimum_top_lateral_margin_m": float((side_edge - robot.kin.radius)[foot_on_top].min()),
        "worst_geometry_penetration_m": -worst[0],
        "collision_audit_rate_hz": AUDIT_RATE,
        "min_planned_support_count": int(stance.sum(axis=1).min()),
        "min_static_projected_com_margin_m": minimum_com_margin,
        "peak_joint_speed_rad_s": float(np.abs(np.diff(rows[:, 7:], axis=0)).max() * FPS),
        "knee_most_extended_rad": float(rows[:, 9::3].max()),
        "status": "kinematic_candidate",
    }, foot_on_top


def render(robot, rows, video, kind, *, titles=None):
    width, height = 576, 320
    camera_views = ((138, -25), (90, -69)) if kind == "bridge" else ((126, -20), (90, -10))
    cameras = []
    for azimuth, elevation in camera_views:
        view = mj.MjvCamera()
        view.type = mj.mjtCamera.mjCAMERA_FREE
        view.azimuth, view.elevation, view.distance = azimuth, elevation, 1.60
        cameras.append(view)
    if titles is None:
        titles = ("WBL - full bridge trot", "Bridge width 40 cm") if kind == "bridge" else (
            "WBL - step over the 30 cm wall", "Side view | wall thickness 8 cm")
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{2 * width}x{height}", "-framerate", str(FPS // 2), "-i", "-", "-an",
        "-vf", "pad=iw:ih+28:0:28:color=0x151c26,"
        f"drawtext=text='{titles[0]}':x=12:y=6:fontsize=15:fontcolor=white,"
        f"drawtext=text='{titles[1]}':x={width + 12}:y=6:fontsize=15:fontcolor=white",
        "-c:v", "libx264", "-threads", "2", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(video),
    ]
    with mj.Renderer(robot.model, height=height, width=width) as renderer:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
            for row in rows[::2]:
                robot.load(row)
                panels = []
                for camera in cameras:
                    camera.lookat[:] = [row[0], 0., .25]
                    renderer.update_scene(robot.data, camera=camera)
                    panels.append(renderer.render().copy())
                process.stdin.write(np.concatenate(panels, axis=1).tobytes())
            process.stdin.close()
            if process.wait():
                raise RuntimeError(f"ffmpeg failed: {video}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("bridge", "wall", "all"), default="all")
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    contract = json.loads((ROOT / "amp_supermimic/configs/wblwbl_dog_v1.plan.json").read_text())
    names = contract["contract"]["policy_joint_names"]
    if names != [f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "calf")]:
        raise ValueError("Unexpected WBL foot/joint order")
    kinds = tuple(CLIPS) if args.kind == "all" else (args.kind,)
    for kind in kinds:
        key = CLIPS[kind]
        robot = ObstacleIK(args.mjcf, names, kind)
        plan = bridge_plan if kind == "bridge" else wall_plan
        roots, toes, stance, phases = plan(robot)
        rows = np.array([robot.solve(root, target) for root, target in zip(roots, toes)])
        report, on_top = audit(robot, rows, toes, stance, kind)
        folder = args.output / key
        folder.mkdir(parents=True, exist_ok=True)
        csv = folder / f"{key}.csv"
        np.savetxt(csv, rows, delimiter=",", fmt="%.10g")
        support = np.where(on_top, 2, stance.astype(int))
        np.savetxt(folder / "support.csv", support, delimiter=",", fmt="%d",
                   header=",".join(LEGS), comments="")
        changes = [0, *(i for i in range(1, len(phases)) if phases[i] != phases[i-1]), len(phases)]
        report["phases"] = [{"name": str(phases[a]), "start_frame": a, "stop_frame_exclusive": b}
                            for a, b in zip(changes[:-1], changes[1:])]
        report["support_labels"] = {"0": "swing", "1": "floor", "2": "obstacle_top"}
        report["csv_sha256"] = hashlib.sha256(csv.read_bytes()).hexdigest()
        report["mjcf_sha256"] = hashlib.sha256(args.mjcf.read_bytes()).hexdigest()
        (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        manifest = {"version": 1, "root": ".", "source_fps": FPS,
                    "quaternion_order": "xyzw", "joint_names": names, "files": [csv.name]}
        (folder / "train.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{key}: {len(rows)} frames, span {report['body_forward_travel_m']:.2f} m, "
              f"max joint speed {report['peak_joint_speed_rad_s']:.2f} rad/s", flush=True)
        if args.render:
            render(robot, rows, folder / "reference.mp4", kind)


if __name__ == "__main__":
    main()
