#!/usr/bin/env python3
"""Render selected Kine2Go Go2 reference motions for visual review."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco

ROOT = Path(__file__).resolve().parents[2]
CLIPS = (
    "ai4_dog_walk_00",
    "ai4_dog_walk_01",
    "ai4_dog_pace",
    "ai4_dog_left_turn",
    "ai4_dog_right_turn",
    "ai4_dog_synth_circle_walk",
    "ai4_dog_synth_eight_walk",
)

ACTION_RANGES = {
    "ai4_dog_walk_00": {"source_file": "dog_walk00_joint_pos.txt", "start": 101, "end": 580},
    "ai4_dog_walk_01": {"source_file": "dog_walk01_joint_pos.txt", "start": 377, "end": 1050},
    "ai4_dog_pace": {"source_file": "dog_walk00_joint_pos.txt", "start": 162, "end": 201},
    "ai4_dog_left_turn": {"source_file": "dog_walk09_joint_pos.txt", "start": 1085, "end": 1124},
    "ai4_dog_right_turn": {"source_file": "dog_walk09_joint_pos.txt", "start": 2404, "end": 2450},
    "ai4_dog_synth_circle_walk": {"source_file": "circle_walk.txt", "start": 446, "end": 806},
    "ai4_dog_synth_eight_walk": {"source_file": "eight_walk.txt", "start": 0, "end": None},
}


def base_yaw_xyzw(q):
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "amp_supermimic/data/go2/inward_review")
    parser.add_argument("--output", type=Path, default=ROOT / "amp_supermimic/outputs/go2_inward_review")
    parser.add_argument("--video-fps", type=int, default=30)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "third_party/trackerlab_supermimic/scripts/rls/supermimic"))
    from bootstrap import configure_paths
    configure_paths()
    from trackerTask.supermimic.robots import get_robot

    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    if tuple(manifest["clips"]) != CLIPS:
        raise ValueError("Downloaded clip selection differs from this preview script")

    spec = get_robot("go2")
    model = mujoco.MjModel.from_xml_path(str(Path(spec.mjcf_path).resolve()))
    data = mujoco.MjData(model)
    joint_ids = np.array([model.joint(name).id for name in spec.joint_names])
    joint_qpos = model.jnt_qposadr[joint_ids]
    feet = ("FL", "FR", "RL", "RR")
    foot_body_ids = np.array([model.body(f"{leg}_calf").id for leg in feet])
    foot_offset = np.array([0.0, 0.0, -0.213])
    foot_side = np.array([1.0, -1.0, 1.0, -1.0])

    motions = []
    source_fps = int(manifest["source_fps"])
    if source_fps % args.video_fps:
        raise ValueError("Video FPS must evenly divide the source FPS")
    stride = source_fps // args.video_fps
    full_frame_counts = {}
    selected_dir = args.data_dir / "selected_clips"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected_records = []
    for name in CLIPS:
        full_rows = np.loadtxt(args.data_dir / f"{name}.csv", delimiter=",", ndmin=2)
        if full_rows.shape[1] != 19 or not np.isfinite(full_rows).all():
            raise ValueError(f"Malformed reference trajectory: {name}")
        full_frame_counts[name] = len(full_rows)
        action_range = ACTION_RANGES[name]
        start = int(action_range["start"])
        end = action_range["end"]
        stop = len(full_rows) if end is None else int(end) + 1
        if start < 0 or stop > len(full_rows) or stop - start < 2:
            raise ValueError(f"Invalid source range for {name}: {start}:{stop} in {len(full_rows)} frames")
        rows = full_rows[start:stop]
        selected_path = selected_dir / f"{name}.csv"
        np.savetxt(selected_path, rows, delimiter=",", fmt="%.10f")
        indices = np.arange(0, len(rows), stride, dtype=int)
        repeats = 4 if len(indices) <= 30 else 1
        motions.append((name, rows, indices, repeats))
        selected_records.append({
            "file": selected_path.name,
            "source_file": action_range["source_file"],
            "source_start_frame": start,
            "source_end_frame_inclusive": None if end is None else int(end),
            "frames": int(len(rows)),
            "duration_seconds": round((len(rows) - 1) / source_fps, 3),
            "downloaded_reference_frames": int(len(full_rows)),
            "sha256": __import__("hashlib").sha256(selected_path.read_bytes()).hexdigest(),
        })
    selected_manifest = {
        "dataset": manifest["dataset"],
        "revision": manifest["revision"],
        "pipeline_repository": "https://github.com/nomagiclab/kine2go-pipeline",
        "pipeline_revision": json.loads((ROOT / "third_party/trackerlab_supermimic/scripts/rls/supermimic/resources.lock.json").read_text())["go2_motions"]["pipeline_revision"],
        "source_fps": source_fps,
        "joint_names": list(spec.joint_names),
        "columns": ["root_x", "root_y", "root_z", "root_qx", "root_qy", "root_qz", "root_qw"] + list(spec.joint_names),
        "frame_range_convention": "source ranges from motion_ranges.txt; start and end are inclusive; end null means through final frame",
        "license": "CC BY-NC 4.0 for all selected ai4_dog clips; retain attribution in each clip.json",
        "clips": selected_records,
    }
    (selected_dir / "manifest.json").write_text(json.dumps(selected_manifest, indent=2) + "\n")

    width, height = 640, 360
    args.output.mkdir(parents=True, exist_ok=True)
    video_path = args.output / "go2_expert_inward_candidates.mp4"
    metrics_path = args.output / "inward_candidate_metrics.json"

    segment_frames = []
    current_frame = 0
    for name, rows, indices, repeats in motions:
        motion_frames = len(indices) * repeats
        segment_frames.append({
            "clip": name,
            "start_frame": current_frame,
            "motion_end_frame": current_frame + motion_frames,
            "end_frame": current_frame + motion_frames + args.video_fps // 2,
            "source_frames": len(rows),
            "duration_seconds": round((len(rows) - 1) / source_fps, 3),
            "preview_repeats": repeats,
        })
        current_frame += motion_frames + args.video_fps // 2

    draw_filters = [
        "drawbox=x=0:y=0:w=iw:h=62:color=black@0.72:t=fill",
        "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
        "text='Go2 reference poses | fixed world azimuth views':"
        "x=16:y=5:fontsize=18:fontcolor=white",
    ]
    for index, segment in enumerate(segment_frames, 1):
        start = segment["start_frame"] / args.video_fps
        end = max(start, (segment["motion_end_frame"] - 1) / args.video_fps)
        label = f"{index:02d} - {segment['clip']}"
        draw_filters.append(
            "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
            f"text='{label}':x=16:y=31:fontsize=21:fontcolor=0xFFD166:"
            f"enable='between(t,{start:.3f},{end:.3f})'"
        )
    ffmpeg = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{width * 2}x{height}",
        "-framerate", str(args.video_fps), "-i", "-",
        "-vf", ",".join(draw_filters),
        "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(video_path),
    ]
    proc = subprocess.Popen(ffmpeg, stdin=subprocess.PIPE)
    main_renderer = mujoco.Renderer(model, height=height, width=width)
    front_renderer = mujoco.Renderer(model, height=height, width=width)
    main_camera = mujoco.MjvCamera()
    front_camera = mujoco.MjvCamera()
    for camera in (main_camera, front_camera):
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.elevation = -12
    main_camera.distance = 1.25
    front_camera.distance = 0.95

    metrics = {
        "dataset": manifest["dataset"],
        "revision": manifest["revision"],
        "source_fps": source_fps,
        "video_fps": args.video_fps,
        "playback": "direct kinematic reference playback in the Go2 MJCF; no policy or PD simulation",
        "lateral_measurement": "foot lateral position in base frame, signed outward-positive for each side",
        "clips": [],
    }

    try:
        for name, rows, indices, repeats in motions:
            lateral_values = []
            hip_values = []
            last_frame = None
            for _ in range(repeats):
                for frame_index in indices:
                    row = rows[frame_index]
                    data.qpos[:3] = row[:3]
                    data.qpos[3:7] = row[[6, 3, 4, 5]]
                    data.qpos[joint_qpos] = row[7:]
                    mujoco.mj_forward(model, data)

                    quat_xyzw = row[3:7]
                    yaw = base_yaw_xyzw(quat_xyzw)
                    target = row[:3].copy()
                    target[2] += 0.06
                    main_camera.lookat[:] = target
                    main_camera.azimuth = 135.0  # fixed in world frame so root yaw remains visible
                    front_camera.lookat[:] = target
                    front_camera.azimuth = 0.0  # fixed in world frame so root yaw remains visible
                    main_renderer.update_scene(data, camera=main_camera)
                    front_renderer.update_scene(data, camera=front_camera)
                    left = main_renderer.render()
                    right = front_renderer.render()
                    last_frame = np.concatenate((left, right), axis=1)
                    proc.stdin.write(last_frame.tobytes())

                    root_rotation = np.array(data.xmat[model.body("base").id]).reshape(3, 3)
                    foot_world = data.xpos[foot_body_ids] + np.einsum(
                        "nij,j->ni",
                        data.xmat[foot_body_ids].reshape(-1, 3, 3),
                        foot_offset,
                    )
                    foot_local = np.einsum(
                        "ij,nj->ni", root_rotation.T, foot_world - row[:3]
                    )
                    lateral_values.extend((foot_local[:, 1] * foot_side).tolist())
                    hip_values.extend(row[[7, 10, 13, 16]].tolist())
            if last_frame is not None:
                for _ in range(args.video_fps // 2):
                    proc.stdin.write(last_frame.tobytes())
            lateral = np.asarray(lateral_values)
            hip = np.asarray(hip_values)
            metrics["clips"].append({
                "name": name,
                "frames": int(len(rows)),
                "downloaded_frames": int(full_frame_counts[name]),
                "duration_seconds": round((len(rows) - 1) / source_fps, 3),
                "preview_repeats": repeats,
                "signed_foot_lateral_m": {
                    "minimum": float(lateral.min()),
                    "p05": float(np.quantile(lateral, 0.05)),
                    "median": float(np.median(lateral)),
                    "fraction_crossing_body_midline": float(np.mean(lateral < 0.0)),
                },
                "hip_abduction_joint_rad": {
                    "minimum": float(hip.min()),
                    "maximum": float(hip.max()),
                },
            })
    finally:
        if proc.stdin:
            proc.stdin.close()
        main_renderer.close()
        front_renderer.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed while encoding the review video")

    metrics["video"] = str(video_path)
    metrics["video_bytes"] = video_path.stat().st_size
    metrics["segments"] = segment_frames
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({
        "video": str(video_path),
        "bytes": video_path.stat().st_size,
        "metrics": str(metrics_path),
        "clips": metrics["clips"],
    }, indent=2))


if __name__ == "__main__":
    main()
