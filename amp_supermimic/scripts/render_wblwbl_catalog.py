"""Render every categorized WBLWBL reference motion as a full MP4 video."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "amp_supermimic" / "data" / "wblwbl"
MJCF = ROOT.parent / "robocon2027_sim_control" / "deploy_cpp" / "robot" / "wblwbl_dog_v1" / "xml" / "wblwbl_dog_v1.xml"
WIDTH = 640
HEIGHT = 360
OUTPUT_FPS = 25


def load_model():
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == 0 and model.geom(geom_id).name != "floor":
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
            model.geom_rgba[geom_id, 3] = 0.0
    model.geom("floor").rgba[:] = [0.55, 0.64, 0.59, 1.0]
    return model


def render_motion(model, entry, source_fps, output_fps):
    csv_path = DATA_DIR / entry["motion"]
    output_path = csv_path.parent / "full_gait.mp4"
    partial_path = output_path.with_name("full_gait.partial.mp4")
    report = json.loads((csv_path.parent / "report.json").read_text())
    rows = np.loadtxt(csv_path, delimiter=",")
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.shape[1] != 19:
        raise ValueError(f"{csv_path} has {rows.shape[1]} columns; expected 19")
    if len(rows) != entry["frames"]:
        raise ValueError(f"{csv_path} has {len(rows)} frames; index says {entry['frames']}")
    if not np.isfinite(rows).all():
        raise ValueError(f"{csv_path} contains non-finite values")

    joint_ids = np.array([model.joint(name).id for name in report["joint_names"]])
    qpos_addrs = np.array([model.jnt_qposadr[joint_id] for joint_id in joint_ids])
    if len(qpos_addrs) != 12:
        raise ValueError(f"{csv_path} joint list has {len(qpos_addrs)} entries; expected 12")

    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 1.6
    camera.azimuth = 145
    camera.elevation = -24
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    sample_step = source_fps // output_fps
    frame_indices = list(range(0, len(rows), sample_step))

    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(output_fps), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame_idx in frame_indices:
            row = rows[frame_idx]
            data.qpos[:3] = row[:3]
            # CSV stores xyzw; MuJoCo's free joint stores wxyz.
            data.qpos[3:7] = row[[6, 3, 4, 5]]
            data.qpos[qpos_addrs] = row[7:]
            mujoco.mj_forward(model, data)

            camera.lookat[:] = data.qpos[:3]
            camera.lookat[2] = max(0.22, float(data.qpos[2]))
            renderer.update_scene(data, camera=camera)
            process.stdin.write(renderer.render().tobytes())
        process.stdin.close()
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")
        partial_path.replace(output_path)
    except BaseException:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        process.wait()
        partial_path.unlink(missing_ok=True)
        raise
    finally:
        renderer.close()

    return {
        "category": entry["category"],
        "motion": entry["motion"],
        "video": output_path.relative_to(DATA_DIR).as_posix(),
        "frames": len(rows),
        "source_duration_seconds": len(rows) / source_fps,
        "fps": output_fps,
        "duration_seconds": len(frame_indices) / output_fps,
        "video_frames": len(frame_indices),
        "size_bytes": output_path.stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", action="append", help="Render only this category; may be repeated")
    parser.add_argument("--output-fps", type=int, default=OUTPUT_FPS)
    args = parser.parse_args()
    if args.output_fps < 1:
        parser.error("--output-fps must be positive")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required but was not found on PATH")
    if not MJCF.is_file():
        parser.error(f"WBLWBL MJCF was not found: {MJCF}")

    index = json.loads((DATA_DIR / "action_index.json").read_text())
    source_fps = int(index["fps"])
    if args.output_fps > source_fps or source_fps % args.output_fps:
        parser.error(f"--output-fps must evenly divide the source rate ({source_fps} fps)")
    entries = index["motions"]
    if args.category:
        categories = set(args.category)
        known = {entry["category"] for entry in entries}
        unknown = categories - known
        if unknown:
            parser.error(f"unknown categories: {', '.join(sorted(unknown))}")
        entries = [entry for entry in entries if entry["category"] in categories]

    model = load_model()
    rendered = []
    for number, entry in enumerate(entries, 1):
        result = render_motion(model, entry, source_fps, args.output_fps)
        rendered.append(result)
        print(
            f"[{number}/{len(entries)}] {result['category']}/{Path(result['motion']).parent.name}: "
            f"{result['duration_seconds']:.2f}s -> {result['video']} ({result['size_bytes']:,} bytes)",
            flush=True,
        )

    existing_index_path = DATA_DIR / "video_index.json"
    if args.category and existing_index_path.is_file():
        existing = json.loads(existing_index_path.read_text())
        by_video = {item["video"]: item for item in existing.get("videos", [])}
        by_video.update({item["video"]: item for item in rendered})
        rendered = sorted(by_video.values(), key=lambda item: item["video"])
    video_index = {"source_fps": source_fps, "output_fps": args.output_fps, "videos": rendered}
    existing_index_path.write_text(json.dumps(video_index, indent=2) + "\n")
    print(f"Rendered {len(entries)} videos; index: {existing_index_path}")


if __name__ == "__main__":
    main()
