"""Open an interactive MuJoCo 3D viewer (or save screenshots) for SuperMimic terrains."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import struct
import sys
import zlib

ROOT = Path(__file__).resolve().parents[2]


def _save_png(path: Path, rgb) -> None:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack("!IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def _build_scene(profile: str, seed: int, level: int, mjcf: Path, source_config: Path, render_size=(1280, 720)):
    sys.path.insert(0, str(ROOT / "amp_supermimic/scripts"))
    import train as wblwbl_train

    spec = wblwbl_train.register(mjcf, source_config)

    import mujoco
    import torch
    from trackerTask.supermimic.terrain import ROBOT_HEIGHTS, generate
    from trackerTask.supermimic.mujoco_terrain import build_model
    from trackerLab.tracker_env.mdp.supermimic_terrain import HeightField

    terrain = generate(profile, ROBOT_HEIGHTS[spec.name], seed=seed)
    model, _, height, qpos_ids, _, _, _, _, _ = build_model(
        spec.name, terrain, ground_plane=False, render_size=render_size
    )
    data = mujoco.MjData(model)
    hf = HeightField(terrain, height, "cpu")
    level = max(0, min(level, len(terrain.centers) - 1))
    if hasattr(terrain, "block_centers"):
        x, y = terrain.block_centers[level]
    else:
        x, y = 0.0, float(terrain.centers[level])
    gz = float(hf.sample(torch.tensor([[x, y]]))[0])
    data.qpos[:3] = [x, y, gz + height]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[qpos_ids] = spec.default_q
    mujoco.mj_forward(model, data)
    return model, data, terrain


def main():
    import train as wblwbl_train

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=wblwbl_train.PROFILES, default="mountain")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--mjcf", type=Path, default=wblwbl_train.MJCF)
    parser.add_argument("--source-config", type=Path, default=wblwbl_train.DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--snapshot-dir", type=Path, default=None,
                        help="Save multi-angle PNG screenshots instead of opening GUI viewer")
    args = parser.parse_args()

    if args.snapshot_dir is not None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        if Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json").exists():
            os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")

    import mujoco

    model, data, terrain = _build_scene(
        args.profile, args.seed, args.level, args.mjcf.resolve(), args.source_config.resolve()
    )

    if args.snapshot_dir is not None:
        args.snapshot_dir.mkdir(parents=True, exist_ok=True)
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 1
        renderer = mujoco.Renderer(model, height=720, width=1280)
        if args.profile == "open_terrain":
            cx5, cy5 = terrain.block_centers[-1]
            views = [
                (f"{args.profile}_overview.png", [0.0, 0.0, 0.0], 34.0, -34, 135),
                (f"{args.profile}_level5_side.png", [cx5, cy5, 0.1], 9.0, -18, 125),
                (f"{args.profile}_topdown.png", [0.0, 0.0, 0.0], 34.0, -89, 90),
            ]
        else:
            views = [
                (f"{args.profile}_overview.png", [3.0, 0.0, 0.15], 12.5, -32, 135),
                (f"{args.profile}_level5_side.png", [3.2, float(terrain.centers[-1]), 0.15], 5.5, -15, 90),
                (f"{args.profile}_topdown.png", [3.2, 0.0, 0.0], 14.0, -89, 90),
            ]
        try:
            for filename, lookat, dist, elev, azim in views:
                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.lookat[:] = lookat
                cam.distance = dist
                cam.elevation = elev
                cam.azimuth = azim
                renderer.update_scene(data, camera=cam, scene_option=opt)
                _save_png(args.snapshot_dir / filename, renderer.render())
        finally:
            renderer.close()
        return

    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[:] = 1
        if args.profile == "open_terrain":
            viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
            viewer.cam.distance = 32.0
        else:
            viewer.cam.lookat[:] = [3.0, 0.0, 0.15]
            viewer.cam.distance = 12.0
        viewer.cam.elevation = -32.0
        viewer.cam.azimuth = 135.0
        viewer.sync()
        while viewer.is_running():
            viewer.sync()
            import time
            time.sleep(0.05)


if __name__ == "__main__":
    main()
