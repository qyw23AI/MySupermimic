#!/usr/bin/env python3
"""Interactive CPU MuJoCo 3D viewer for inspecting AMP reference motion pools or CSV clips."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parents[1]
DEFAULT_POOL = ROOT / "amp_supermimic/data/flat_trainning/pool/train.json"
DEFAULT_MJCF = ROOT.parent / "robocon2027_sim_control/deploy_cpp/robot/wblwbl_dog_v1/xml/wblwbl_dog_v1.xml"
DEFAULT_JOINTS = [
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
]


def _add_arrow(scene, p0, p1, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
        np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        g, mujoco.mjtGeom.mjGEOM_ARROW, float(radius),
        np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64),
    )
    scene.ngeom += 1


def _load_clips(motion_config: Path | None, csv_path: Path | None, clip_filter: str | None):
    if csv_path is not None:
        arr = np.loadtxt(csv_path, delimiter=",", ndmin=2)
        return [(csv_path.name, arr, 50.0, DEFAULT_JOINTS)]
    cfg = json.loads(motion_config.read_text(encoding="utf-8"))
    if cfg.get("quaternion_order", "xyzw") != "xyzw":
        raise ValueError(f"Unsupported quaternion_order in {motion_config}")
    fps = float(cfg.get("source_fps", 50.0))
    jnames = list(cfg["joint_names"])
    root_dir = motion_config.parent / cfg.get("root", ".")
    clips = []
    for name in cfg["files"]:
        if clip_filter and clip_filter not in name:
            continue
        arr = np.loadtxt(root_dir / name, delimiter=",", ndmin=2)
        clips.append((name, arr, fps, jnames))
    if not clips:
        raise ValueError(f"No clips matched in {motion_config}")
    return clips


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion-config", type=Path, default=DEFAULT_POOL,
                    help="Path to train.json / test.json motion pool")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Optional single 19-col CSV clip (overrides --motion-config)")
    ap.add_argument("--clip", type=str, default=None,
                    help="Optional substring filter for clip names in --motion-config")
    ap.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    ap.add_argument("--pause", type=float, default=0.5, help="Pause seconds between clips")
    ap.add_argument("--once", action="store_true", help="Play all clips once and exit")
    ap.add_argument("--no-follow", action="store_true", help="Do not follow robot with camera")
    ap.add_argument("--headless-steps", type=int, default=0,
                    help="Run N frames without GUI (for testing)")
    args = ap.parse_args()

    clips = _load_clips(args.motion_config, args.csv, args.clip)
    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    for i in range(model.ngeom):
        if model.geom_bodyid[i] == 0 and model.geom(i).name != "floor":
            model.geom_contype[i] = 0
            model.geom_conaffinity[i] = 0
            model.geom_rgba[i, 3] = 0.0
    data = mujoco.MjData(model)

    def set_frame(row, qids):
        data.qpos[:3] = row[:3]
        data.qpos[3:7] = row[[6, 3, 4, 5]]  # xyzw -> wxyz
        data.qpos[qids] = row[7:]
        mujoco.mj_forward(model, data)

    if args.headless_steps > 0:
        steps = 0
        for name, arr, fps, jnames in clips:
            qids = np.array([model.joint(n).qposadr[0] for n in jnames])
            for row in arr:
                set_frame(row, qids)
                steps += 1
                if steps >= args.headless_steps:
                    print(f"[headless {steps} frames] last_clip={name} z={data.qpos[2]:.3f}")
                    return
        print(f"[headless {steps} frames] clips={len(clips)}")
        return

    print(f"[play_motion] loaded {len(clips)} clip(s); cycling in MuJoCo viewer...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.distance = 1.65
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -22.0
        while viewer.is_running():
            for name, arr, fps, jnames in clips:
                if not viewer.is_running():
                    break
                qids = np.array([model.joint(n).qposadr[0] for n in jnames])
                dt = 1.0 / (fps * max(args.speed, 1e-3))
                v_body = Rotation.from_quat(arr[:, 3:7]).inv().apply(
                    np.gradient(arr[:, :3], 1.0 / fps, axis=0)
                )
                rpy = Rotation.from_quat(arr[:, 3:7]).as_euler("xyz", degrees=True)
                print(
                    f"[clip] {name} | frames={len(arr)} ({len(arr)/fps:.2f}s) "
                    f"| v_med=({np.median(v_body[:,0]):+.2f},{np.median(v_body[:,1]):+.2f}) m/s "
                    f"| roll_med={np.median(rpy[:,0]):+.2f} deg",
                    flush=True,
                )
                for k, row in enumerate(arr):
                    if not viewer.is_running():
                        break
                    t0 = time.perf_counter()
                    set_frame(row, qids)
                    if not args.no_follow:
                        viewer.cam.lookat[:] = [float(data.qpos[0]), float(data.qpos[1]), 0.25]
                    with viewer.lock():
                        scn = viewer.user_scn
                        scn.ngeom = 0
                        p = data.qpos[:3].copy() + np.array([0.0, 0.0, 0.25])
                        v_world = Rotation.from_quat(row[3:7]).apply(v_body[k])
                        if np.linalg.norm(v_world[:2]) > 0.03:
                            _add_arrow(scn, p, p + np.array([v_world[0], v_world[1], 0.0]) * 0.6,
                                       0.015, [0.1, 0.9, 0.2, 0.95])
                    viewer.sync()
                    rem = dt - (time.perf_counter() - t0)
                    if rem > 0:
                        time.sleep(rem)
                if args.pause > 0 and viewer.is_running():
                    time.sleep(args.pause)
            if args.once:
                break


if __name__ == "__main__":
    main()
