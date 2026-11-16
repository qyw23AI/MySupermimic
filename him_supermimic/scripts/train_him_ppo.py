#!/usr/bin/env python3
"""Train and smoke-check the Stage 3 WBLWBL HIM-PPO baseline."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "him_supermimic/configs/wblwbl_dog_v1.plan.json"


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def evaluate(env, adapter, policy, steps):
    from him_supermimic.evaluation import evaluation

    result = evaluation(env, adapter, policy, steps)
    # Keep the original report aliases for existing consumers.
    return {**result, "mean_reward_per_step": result["reward_per_step"],
            "tracking_rows": result["transitions"],
            "linear_velocity_rmse_m_s": result["linear_rmse_m_s"],
            "yaw_rate_rmse_rad_s": result["yaw_rmse_rad_s"],
            "terminations": result["falls"],
            "terminations_per_robot_minute": result["falls_per_robot_minute"]}


def save_replay(env, replay, path):
    import torch

    rows = replay.ordered()
    torch.save({"transitions": {k: v.cpu() for k, v in rows.items()}, "schema_version": 1,
                "reward_term_names": env.reward_term_names}, path)


def collect(args, env, adapter, runner, output):
    import torch

    checkpoint = args.collect_checkpoint.resolve(strict=True)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    runner.alg.actor_critic.load_state_dict(saved["model_state_dict"])
    policy = runner.get_inference_policy()
    noise = torch.Generator(device=env.device).manual_seed(args.seed + 10000)
    from him_supermimic.replay import EpisodeReplay

    env.replay = EpisodeReplay(args.collect_steps, env.num_envs)
    start = time.monotonic()
    with torch.inference_mode():
        for _ in range(args.collect_steps):
            action = policy(adapter.get_observations())
            exploration = torch.randn(action.shape, generator=noise, device=env.device)
            adapter.step(action + args.action_noise * exploration)
    save_replay(env, env.replay, output / "replay.pt")
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "action_noise_std": args.action_noise,
        "transitions": args.collect_steps * env.num_envs,
        "elapsed_seconds": time.monotonic() - start,
    }
    write_json(output / "collection.json", report)
    print(json.dumps(report, indent=2), flush=True)


def run(args, output):
    import numpy as np
    import torch
    from himloco_lab.adapter import LegacyRunnerAdapter
    from himloco_lab.config import class_to_dict, load_config
    from rsl_rl.runners import HIMOnPolicyRunner
    import himloco_lab.config as him_config
    import rsl_rl.runners.him_on_policy_runner as him_runner

    for module in (him_config, him_runner):
        if not Path(module.__file__).resolve().is_relative_to(args.himloco_root):
            raise RuntimeError(f"{module.__name__} was imported outside {args.himloco_root}: {module.__file__}")

    from him_supermimic.himlab import RecordingHIMLabEnv, make_flat_env_cfg
    from him_supermimic.replay import EpisodeReplay

    plan = json.loads(PLAN.read_text())
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    legacy, train_cfg = load_config(task="wblwbl_dog_v1", num_envs=args.num_envs, stage=3)
    if args.profile == "core_reward_v1":
        overrides = plan["controlled_flat"]
        for term in overrides["omitted_terms"]:
            setattr(legacy.rewards.scales, term, 0.0)
        for option, value in overrides["domain_rand_overrides"].items():
            setattr(legacy.domain_rand, option, value)
        legacy.commands.curriculum = overrides["command_curriculum"]
    if args.no_observation_noise:
        # Legacy height measurements use noise_level even when add_noise is false.
        legacy.noise.add_noise = False
        legacy.noise.noise_level = 0.0
    if args.rollout_steps is not None:
        train_cfg["runner"]["num_steps_per_env"] = args.rollout_steps

    snapshot = subprocess.check_output([
        sys.executable, str(ROOT / "him_supermimic/scripts/snapshot_himloco.py"),
        "--source", str(args.himloco_root), "--output", str(output / "source.json"),
    ], text=True)
    print(snapshot, end="", flush=True)
    cfg = make_flat_env_cfg(legacy, args.device, output / "asset_cache")
    cfg.seed = args.seed
    write_json(output / "config.json", {
        "task": "wblwbl_dog_v1", "stage": 3, "profile": args.profile,
        "seed": args.seed, "device": args.device, "num_envs": args.num_envs,
        "iterations": args.iterations, "eval_steps": args.eval_steps,
        "replay_steps": args.replay_steps, "smoke_replay": args.verify_replay,
        "no_observation_noise": args.no_observation_noise,
        "legacy": class_to_dict(legacy), "training": train_cfg,
        "torch": torch.__version__,
        "source": str(args.himloco_root),
        "upstreams": json.loads((ROOT / "manifests/upstreams.json").read_text()),
    })
    env = RecordingHIMLabEnv(cfg)
    try:
        contract = plan["contract"]
        if (
            env.core.num_obs != contract["policy_input"]
            or env.core.num_actions != contract["num_actions"]
            or env.core.dof_names != contract["policy_joint_names"]
            or abs(env.step_dt - contract["control_dt"]) > 1e-8
        ):
            raise RuntimeError("WBLWBL observation, action, joint order or timing contract differs")
        if args.probe_world_model is not None:
            from him_supermimic.action_probe import probe

            probe(args, env, class_to_dict(legacy), output)
            return
        adapter = LegacyRunnerAdapter(env)
        start = time.monotonic()
        runner = HIMOnPolicyRunner(adapter, train_cfg, log_dir=str(output), device=env.device)
        if args.collect_checkpoint is not None:
            collect(args, env, adapter, runner, output)
            return
        if args.replay_steps:
            env.replay = EpisodeReplay(args.replay_steps, env.num_envs)
        runner.learn(args.iterations, init_at_random_ep_len=True)
        train_seconds = time.monotonic() - start

        checkpoint = output / f"model_{runner.current_learning_iteration}.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if any(not torch.isfinite(value).all() for value in saved["model_state_dict"].values()):
            raise RuntimeError("Checkpoint contains nonfinite model weights")
        policy = runner.get_inference_policy()
        training_replay = env.replay
        env.replay = None
        metrics = evaluate(env, adapter, policy, args.eval_steps)
        report = {
            "training_seconds": train_seconds,
            "training_transitions": runner.tot_timesteps,
            "checkpoint": checkpoint.name,
            "evaluation": metrics,
        }
        if training_replay is not None:
            save_replay(env, training_replay, output / "replay.pt")
            report["replay_steps_retained"] = training_replay.size
        write_json(output / "metrics.json", report)
        if args.verify_replay:
            from him_supermimic.verify import verify_replay

            write_json(output / "replay_validation.json", verify_replay(env, adapter, policy, replay_path=output / "verification_replay.pt"))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--himloco-root", type=Path, default=ROOT / "third_party/MyHIMLoco")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("untouched", "core_reward_v1"), default="untouched")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--eval-steps", type=int, default=48)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--replay-steps", type=int, default=0)
    parser.add_argument("--verify-replay", action="store_true")
    parser.add_argument("--probe-world-model", type=Path)
    parser.add_argument("--probe-replay", type=Path)
    parser.add_argument("--collect-checkpoint", type=Path, help="Collect frozen HIM policy rollouts")
    parser.add_argument("--collect-steps", type=int, default=512)
    parser.add_argument("--action-noise", type=float, default=0.10, help="Collection action noise std")
    parser.add_argument("--no-observation-noise", action="store_true",
                        help="Disable proprioceptive and height noise for numerical validation")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.num_envs, args.iterations, args.eval_steps) < 1 or args.replay_steps < 0:
        parser.error("num-envs, iterations and eval-steps must be positive; replay-steps must be nonnegative")
    if args.rollout_steps is not None and args.rollout_steps < 2:
        parser.error("PPO requires at least two rollout steps")
    if args.verify_replay and args.num_envs < 3:
        parser.error("Replay validation requires at least three environments")
    if args.collect_steps < 1 or not 0 <= args.action_noise < float("inf"):
        parser.error("collect-steps must be positive and action-noise must be finite and nonnegative")
    if args.collect_checkpoint is not None and (args.replay_steps or args.verify_replay):
        parser.error("collection writes its own replay; omit --replay-steps and --verify-replay")
    if (args.probe_world_model is None) != (args.probe_replay is None):
        parser.error("Specify both --probe-world-model and --probe-replay")
    if args.probe_world_model and (args.num_envs < 2 or args.num_envs % 2
                                  or args.profile != "core_reward_v1" or args.collect_checkpoint):
        parser.error("Probe requires an even number of controlled-flat environments")
    args.himloco_root = args.himloco_root.resolve()
    if not (args.himloco_root / "isaaclab_migration/himloco_lab/config.py").is_file():
        parser.error(f"MyHIMLoco source missing at {args.himloco_root}. "
                     "Initialize the third_party/MyHIMLoco submodule first.")
    sys.path[:0] = [str(ROOT), str(args.himloco_root / "isaaclab_migration"), str(args.himloco_root / "rsl_rl")]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    app = AppLauncher(args).app
    try:
        run(args, output)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Isaac Sim shutdown may exit the process with status 0.
        # Close on the successful path and preserve Python failures here.
        raise
    sys.stdout.flush()
    sys.stderr.flush()
    app.close()


if __name__ == "__main__":
    main()
