"""Render a saved one-hop MuJoCo rollout to MP4 for visual review."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .brace_experiment import CONTROL_DT, make_model
from .contact_route import add_sensors


def render(asset: Path, replay: Path, output: Path, *, width: int = 1280, height: int = 720,
           fps: int = 50, runup_frames: int = 16):
    with np.load(replay, allow_pickle=False) as data:
        states = np.asarray(data["states"])[0]
        air = np.asarray(data["air"])[0].astype(bool)
        success = bool(np.asarray(data["success"]).reshape(-1)[0])
        score = float(np.asarray(data["scores"]).reshape(-1)[0])
    valid_flight = np.zeros_like(air)
    edges = np.flatnonzero(np.diff(np.r_[False, air, False]))
    for start, stop in zip(edges[::2], edges[1::2]):
        if stop - start < 6 or start == 0:
            continue
        rise = states[start:stop, 2].max() - states[start - 1, 2]
        if states[start, 21] >= .15 and rise >= .06:
            valid_flight[start:stop] = True
    model = make_model(asset, configure=add_sensors)
    sim = mj.MjData(model)
    camera = mj.MjvCamera()
    mj.mjv_defaultCamera(camera)
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [-.18, 0., .30]
    camera.distance = 2.25
    camera.azimuth = 90.
    camera.elevation = -12.
    output.parent.mkdir(parents=True, exist_ok=True)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 21) if font_path.exists() else ImageFont.load_default()
    with mj.Renderer(model, height=height, width=width) as renderer, imageio.get_writer(
        output, fps=fps, codec="libx264", quality=8, macro_block_size=1
    ) as writer:
        for frame, state in enumerate(states):
            sim.qpos[:] = state[:model.nq]
            sim.qvel[:] = state[model.nq:model.nq + model.nv]
            mj.mj_forward(model, sim)
            renderer.update_scene(sim, camera=camera)
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image)
            phase = ("RUN-UP" if frame < runup_frames else "FLIGHT" if valid_flight[frame]
                     else "UNSUPPORTED / FALL" if air[frame] else "NO TAKEOFF / RECOVERY")
            draw.text((24, 24), f"WBL one-hop candidate | {phase} | t={frame * CONTROL_DT:.2f}s",
                      font=font, fill="white", stroke_width=2, stroke_fill="black")
            draw.text((24, 64), f"root x={state[0]:+.2f} m  z={state[2]:.2f} m  score={score:.2f}",
                      font=font, fill="white", stroke_width=2, stroke_fill="black")
            status = "PASS" if success else "NOT CLEARED: wall remains ahead"
            color = "#50ff50" if success else "#ff5555"
            draw.text((24, height - 48), status, font=font, fill=color, stroke_width=2, stroke_fill="black")
            writer.append_data(np.asarray(image))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--runup-frames", type=int, default=16)
    args = parser.parse_args()
    path = render(args.asset, args.replay, args.output,
                  width=args.width, height=args.height, fps=args.fps,
                  runup_frames=args.runup_frames)
    print(path.resolve())


if __name__ == "__main__":
    main()
