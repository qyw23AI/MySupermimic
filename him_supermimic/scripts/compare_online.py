#!/usr/bin/env python3
"""Record paired Isaac Lab viewport playback of HIM checkpoints."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import traceback

from train_online import HIM_ROOT


COMMANDS = ((0.5, 0.0, 0.0), (0.0, 0.25, 0.0), (0.2, 0.0, 0.6), (0.8, 0.0, 0.0))
STEPS_PER_COMMAND = 150
CAPTURE_EVERY = 2


def run(args):
    import numpy as np
    import torch

    from himloco_lab.adapter import LegacyRunnerAdapter
    from himloco_lab.config import class_to_dict, load_config
    from himloco_lab.visualization import ViewportCapture, configure_viewport
    from him_supermimic.him_policy import DifferentiableHIM
    from him_supermimic.himlab import RecordingHIMLabEnv, make_flat_env_cfg
    from him_supermimic.replay import EpisodeReplay
    from warmup_critic import load_policy

    import himloco_lab.config as him_config

    if not Path(him_config.__file__).resolve().is_relative_to(HIM_ROOT):
        raise RuntimeError("Expected the pinned MyHIMLoco submodule")
    saved_config = args.run / "config.json"
    amp_run = saved_config.is_file() and json.loads(saved_config.read_text()).get("algorithm") == "amp_hie_ppo"
    cold_start = saved_config.is_file() and (args.run / "initial_policy.pt").is_file()
    global COMMANDS
    if amp_run:
        COMMANDS = ((0.5, 0., 0.), (0.2, 0., 0.6), (0.2, 0., -0.6), (0., 0., 0.))
    selection = {"commands": COMMANDS, "steps_per_command": STEPS_PER_COMMAND}
    if cold_start or amp_run:
        config = json.loads(saved_config.read_text())
        settings = config["legacy"]
        available = sorted(int(path.stem.split("_", 1)[1]) for path in args.run.glob("checkpoint_*.pt"))
        if not available:
            raise ValueError("Comparison requires at least one trained checkpoint")
        iterations = args.iterations or [0, available[-1]]
        selection["mode"] = "amp_hie_ppo" if amp_run else "cold_start"
    else:
        report = json.loads((args.run / "report.json").read_text())
        evaluated = report["evaluations"]
        if not evaluated:
            raise ValueError("Comparison requires at least one trained checkpoint")
        best = max(evaluated, key=lambda row: (-row["falls"], row["reward_per_step"]))
        iterations = args.iterations or [0, best["iteration"]]
        settings = report["legacy_config"]
        selection.update(baseline=report["baseline"], best_trained=best,
                         best_overall_iteration=report["best_iteration"], mode="a5_fine_tuning")

    labels = ["initial" if i == 0 else f"iter_{i:04d}" for i in iterations]
    checkpoints = [(args.run / ("initial_policy.pt" if cold_start and i == 0 else f"checkpoint_{i:04d}.pt"))
                   .resolve(strict=True) for i in iterations]
    selection["iterations"] = iterations
    (args.output / "comparison.json").write_text(json.dumps(selection, indent=2) + "\n")
    legacy, training = load_config(task="wblwbl_dog_v1", stage=3, num_envs=1)

    def playback_policy(checkpoint, device):
        if amp_run:
            return load_policy(config, checkpoint, device)[1]
        if not cold_start:
            return load_policy({"legacy": class_to_dict(legacy), "training": training}, checkpoint, device)[1]

        from rsl_rl.modules import HIMActorCritic

        layout = settings["env"]
        policy = DifferentiableHIM(HIMActorCritic(
            layout["num_observations"], layout["num_privileged_obs"],
            layout["num_one_step_observations"], layout["num_actions"],
            **config["training"]["policy"])).to(device)
        saved = torch.load(checkpoint, weights_only=True, map_location=device)
        policy.load_state_dict(saved if checkpoint.name == "initial_policy.pt" else saved["model_state_dict"])
        return policy.eval()

    for name, scale in settings["rewards"]["scales"].items():
        setattr(legacy.rewards.scales, name, scale)
    for name, value in settings["domain_rand"].items():
        if isinstance(value, bool) and hasattr(legacy.domain_rand, name):
            setattr(legacy.domain_rand, name, value)
    legacy.commands.curriculum = False
    legacy.commands.heading_command = False
    legacy.noise.add_noise = False
    legacy.noise.noise_level = 0.0
    cfg = make_flat_env_cfg(legacy, args.device, args.output / "asset_cache")
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.eye = (2.4, 2.2, 1.5)
    cfg.viewer.lookat = (0.0, 0.0, 0.12)
    cfg.sim.render_interval = cfg.decimation * CAPTURE_EVERY
    env = RecordingHIMLabEnv(cfg)
    try:
        configure_viewport((768, 432))
        adapter = LegacyRunnerAdapter(env)
        command = torch.zeros(3, device=env.device)
        original_resample = env.core._resample_commands

        def scheduled_command(ids):
            if ids is not None:
                env.core.commands[ids, :3] = command

        env.core._resample_commands = scheduled_command
        try:
            for label, checkpoint in zip(labels, checkpoints, strict=True):
                policy = playback_policy(checkpoint, env.device)
                capture = ViewportCapture(args.output / label, CAPTURE_EVERY)
                torch.manual_seed(701)
                np.random.seed(701)
                with torch.inference_mode():
                    for buffer in (env.core.obs_buf, env.core.privileged_obs_buf, env.core.actions,
                                   env.core.last_actions, env.core.last_last_actions, env.core.commands):
                        buffer.zero_()
                    command.copy_(command.new_tensor(COMMANDS[0]))
                    env.replay = None
                    adapter.reset()
                    env.replay = EpisodeReplay(1, env.num_envs, fields=("next_state_pre_reset", "reward", "terminated"))
                    with (args.output / f"{label}.csv").open("w", newline="") as file:
                        rows = csv.writer(file)
                        rows.writerow(("time_s", "target_vx", "target_vy", "target_yaw",
                                       "actual_vx", "actual_vy", "actual_yaw", "reward", "fall"))
                        for step in range(STEPS_PER_COMMAND * len(COMMANDS)):
                            target = COMMANDS[step // STEPS_PER_COMMAND]
                            command.copy_(command.new_tensor(target))
                            env.core.commands[:, :3] = command
                            history = adapter.get_observations().clone()
                            history[:, :3] = command * env.core.commands_scale
                            adapter.step(policy(history))
                            capture.on_step()
                            data = env.replay.data
                            state = data["next_state_pre_reset"][0, 0]
                            rows.writerow(((step + 1) * env.step_dt, *target, *state[7:9].tolist(),
                                           state[12].item(), data["reward"][0, 0].item(), int(data["terminated"][0, 0])))
                    capture.capture_final(env.sim, "last.png")
        finally:
            env.core._resample_commands = original_resample
        videos = []
        for label, iteration in zip(labels, iterations, strict=True):
            frames = list((args.output / label).glob("frame_*.png"))
            expected = STEPS_PER_COMMAND * len(COMMANDS) // CAPTURE_EVERY
            if len(frames) != expected:
                raise RuntimeError(f"{label}: expected {expected} viewport frames, got {len(frames)}")
            output = args.output / f"{label}.mp4"
            title = "Initial HIM" if label == "initial" else f"Iteration {iteration}"
            filters = [f"drawtext=text='{title}':x=16:y=14:fontsize=25:fontcolor=white:box=1:boxcolor=black@0.65"]
            for index, (vx, vy, yaw) in enumerate(COMMANDS):
                start = index * STEPS_PER_COMMAND / 50
                end = (index + 1) * STEPS_PER_COMMAND / 50
                filters.append(
                    f"drawtext=text='vx {vx:+.2f}  vy {vy:+.2f}  yaw {yaw:+.2f}':"
                    "x=16:y=h-th-16:fontsize=21:fontcolor=white:box=1:boxcolor=black@0.65:"
                    f"enable='gte(t,{start})*lt(t,{end})'"
                )
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "25",
                            "-pattern_type", "glob", "-i", str(args.output / label / "frame_*.png"),
                            "-vf", ",".join(filters), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            str(output)], check=True)
            videos.append(output)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *[arg for video in videos for arg in ("-i", str(video))],
                        "-filter_complex", f"hstack=inputs={len(videos)}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        str(args.output / "comparison.mp4")], check=True)
    finally:
        env.close()


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, nargs="+", help="Checkpoint iterations in display order")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.iterations is not None and (len(set(args.iterations)) < 2 or len(set(args.iterations)) != len(args.iterations) or min(args.iterations) < 0):
        parser.error("Specify at least two distinct nonnegative checkpoint iterations")
    if args.headless:
        parser.error("Recording requires the Isaac Lab GUI viewport")
    args.run = args.run.resolve(strict=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    app = AppLauncher(args).app
    try:
        run(args)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    sys.stdout.flush()
    app.close()


if __name__ == "__main__":
    main()
