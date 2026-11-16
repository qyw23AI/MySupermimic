"""Command line for offline checks, PhysX training, and fixed-command evaluation."""

import argparse
import json
from pathlib import PosixPath

from amp_supermimic.legacy import (
    build_upstream_train_args, list_tasks, resolve_repo_path,
)
from .config import motion_receipt, read_snapshot, resolve, robot_from, snapshot, write_json

FLAT = "wblwbl_dog_v1_physx_flat"


def task_options(parser):
    parser.add_argument("--task", choices=list_tasks(backend="physx"), default=FLAT)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--motion-config")
    parser.add_argument("--amp-weight", type=float)
    parser.add_argument("--amp-batch", type=int)
    parser.add_argument("--amp-updates", type=int)
    parser.add_argument("--amp-history", type=int)
    parser.add_argument("--source", "--wmp-source")
    parser.add_argument("--competition-stage", choices=("eval", "learn"))
    parser.add_argument("--level", type=int)
    for flag in ("iterations", "wm-only-iters", "warmup-steps", "horizon", "hist", "wm-window",
                 "wm-batches", "wm-batch", "pi-batch", "buffer-steps"):
        parser.add_argument(f"--{flag}", type=int)


def preflight(args):
    env, train = resolve(args)
    robot = robot_from(env)
    _, motions = motion_receipt(env, robot)
    report = snapshot(args.task, env, train) | {
        "robot_sha256": robot.sha256, "joint_names": list(robot.joints),
        "foot_centers_m": robot.foot_centers, "foot_radii_m": robot.foot_radii,
        "asset_path": str(robot.path), "control_dt": env.control.physics_dt * env.control.decimation,
        "motion_audit": motions,
    }
    if env.terrain.profile != "plane":
        from .wmp_terrain import CourseAtlas
        atlas = CourseAtlas(env.terrain, "cpu")
        vertices, faces = atlas.mesh()
        report["terrain"] = {"rows": atlas.rows, "types": atlas.column_types,
                             "vertices": len(vertices), "triangles": len(faces),
                             "l0": {kind: atlas.cells[0, col].parameters
                                    for col, kind in enumerate(atlas.column_types)}}
    return report


def train(args):
    from .runtime import Runtime, configure_trackerlab

    env, training = resolve(args)
    if args.cpu_check:
        print(json.dumps(preflight(args), indent=2, ensure_ascii=False))
        return
    if not env.env.device.startswith("cuda"):
        raise ValueError("TrackerLab training requires a CUDA device. Use --check for offline validation.")
    robot = robot_from(env)
    run = resolve_repo_path(args.run_dir)
    plan = snapshot(args.task, env, training)
    if args.resume:
        if args.warm_start:
            raise ValueError("Choose either --resume or --warm-start")
        if getattr(env.terrain, "competition_stage", None) == "learn":
            raise ValueError("Resume of a WMP learning curriculum requires persisted course statistics")
        saved = json.loads((run / "legacy_task_config.json").read_text())
        saved["train_cfg"]["runner"]["iterations"] = training.runner.iterations
        if saved != plan:
            raise ValueError("Resume requires the original task parameters; only iterations can change")
        if not (run / "last.pt").is_file():
            raise FileNotFoundError(run / "last.pt")
    else:
        run.mkdir(parents=True, exist_ok=False)

    motion, receipt = motion_receipt(env, robot, None if args.resume else run / "references")
    identity = {"robot_sha256": robot.sha256, "motion_audit": receipt}
    identity_path = run / "input_identity.json"
    if args.resume:
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Resume input asset or expert files differ from the saved run")
        if motion is None:
            from .references import audit_manifest

            motion = run / "references" / "train.json"
            original = [record["sha256"] for record in receipt["train"].values()]
            restored = audit_manifest(motion, robot.joints, env.amp.amp_history)
            if [record["sha256"] for record in restored.values()] != original:
                raise ValueError("The run-local AMP training manifest has changed")
    else:
        write_json(identity_path, identity)
        write_json(run / "legacy_task_config.json", plan)
    flags = ["--task", args.task, "--run-dir", str(run), "--amp-motion-config", str(motion)]
    if args.resume:
        flags += ["--resume", str(run / "last.pt")]
    if args.warm_start:
        flags += ["--warm-start", str(resolve_repo_path(args.warm_start)), "--warm-start-reset-world"]
    upstream_args = build_upstream_train_args(env, training, flags)
    configure_trackerlab()
    from train_terrain import parse_args
    parsed = parse_args(upstream_args)
    write_json(run / "physx_config.json", plan | identity | {
        "arguments": vars(parsed), "control_dt": env.control.physics_dt * env.control.decimation,
        "resolved_urdf": str(robot.path),
        "resolved_wmp_source": str(resolve_repo_path(env.terrain.source))
        if env.terrain.profile != "plane" else None,
    })
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=not args.gui, device=env.env.device)
    runtime = None
    try:
        runtime = Runtime(args.task, env, robot, run)
        runtime.train(upstream_args, receipt)
        run_config = run / "run_config.json"
        effective = json.loads(run_config.read_text())
        effective["physx"] = json.loads((run / "physx_config.json").read_text())
        effective["perception"].update(
            source="plane" if runtime.atlas is None else "WMP analytic mesh top-surface query",
            physical_ray_validation=False,
        )
        write_json(run_config, effective)
    finally:
        try:
            if runtime is not None and runtime.simulator is not None and runtime.simulator.events is not None:
                write_json(run / "wmp_course_metrics.json", {
                    key: value.tolist() if hasattr(value, "tolist") else value
                    for key, value in runtime.simulator.events.course.state_dict().items()
                })
        finally:
            launcher.app.close()


def evaluate(args):
    import torch
    from .runtime import Runtime, configure_trackerlab

    checkpoint = resolve_repo_path(args.checkpoint)
    task, env, train_cfg = read_snapshot(checkpoint.parent / "physx_config.json")
    robot = robot_from(env)
    with torch.serialization.safe_globals([PosixPath]):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    contract = saved["simulator_contract"]
    if saved["task"] != task or contract["backend"] != "isaaclab-physx" or robot.sha256 != contract["robot_sha256"]:
        raise ValueError("Checkpoint task, backend or URDF hash does not match the run")
    if args.level is not None and env.terrain.profile == "plane":
        raise ValueError("--level applies to WMP tasks")
    if env.terrain.profile != "plane":
        env.terrain.competition_stage = "eval"
        if args.level is not None:
            if not 0 <= args.level < 10:
                raise ValueError("WMP level must be 0–9")
            env.terrain.level = args.level
    if min(args.num_envs, args.steps) < 1:
        raise ValueError("Expected positive environment and evaluation step counts")
    env.env.num_envs, env.env.seed = args.num_envs, args.seed
    if args.device:
        env.env.device = args.device
    configure_trackerlab()
    from isaaclab.app import AppLauncher
    from trackerLab.algorithms.supermimic.terrain_networks import TerrainPolicy
    from trackerLab.tracker_env.mdp.supermimic import unflatten_state
    from trackerLab.tracker_env.mdp.supermimic_command import body_velocities

    output = resolve_repo_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    launcher = AppLauncher(headless=not args.gui, device=env.env.device)
    try:
        torch.manual_seed(args.seed)
        runtime = Runtime(task, env, robot, output)
        sim = runtime.create(train_cfg.algorithm.gamma)
        hist = train_cfg.world_model.hist
        actor = TerrainPolicy(sim.mdp.obs_dim, sim.mdp.goal_dim, hist=hist,
                              num_joints=len(robot.joints), action_limit=env.env.action_limit)
        actor.load_state_dict(saved["actor"])
        actor.to(sim.device).eval()
        results = []
        with torch.inference_mode():
            for command in args.command or env.commands.samples:
                sim.reset()
                errors = torch.zeros(5, device=sim.device)
                sim.twists = sim.c.new_tensor([command])
                sim.sample_commands(torch.arange(sim.num_envs, device=sim.device))
                previous = sim.c.new_zeros((sim.num_envs, hist, sim.mdp.obs_dim))
                for _ in range(args.steps):
                    state, observation = sim.state(), sim.observe()
                    previous = torch.where(sim.ep_step[:, None, None] == 0, observation[:, None],
                                           torch.cat((previous[:, 1:], observation[:, None]), 1))
                    transition = sim.step(actor(previous, sim.mdp.goal(state, sim.c)), fixed_command=True)
                    v, w = body_velocities(unflatten_state(transition["s1"]))
                    sampled = transition["c1"]
                    errors[0] += (v[:, :2] - sampled[:, 1:3]).square().sum()
                    errors[1] += (w[:, 2] - sampled[:, 3]).square().sum()
                    errors[2] += transition["episode_fall"].sum()
                    errors[3] += transition["episode_success"].sum()
                    errors[4] += (transition["episode_strict_success"] & transition["done"]).sum()
                samples = args.steps * sim.num_envs
                velocity, yaw, falls, success, strict = errors.tolist()
                results.append({"command": list(command), "velocity_rmse_mps": (velocity / samples) ** .5,
                                "yaw_rmse_rad_s": (yaw / samples) ** .5, "falls": int(falls),
                                "falls_per_robot_minute": falls / (samples * sim.dt / 60),
                                "successes": int(success), "strict_successes": int(strict)})
        write_json(output / "evaluation.json", {"task": task, "seed": args.seed, "num_envs": args.num_envs,
                                                 "steps": args.steps, "results": results})
    finally:
        launcher.app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-tasks", action="store_true")
    commands = parser.add_subparsers(dest="mode")
    check = commands.add_parser("preflight")
    task_options(check)
    training = commands.add_parser("train")
    task_options(training)
    training.add_argument("--run-dir", required=True)
    training.add_argument("--check", dest="cpu_check", action="store_true")
    training.add_argument("--resume", action="store_true")
    training.add_argument("--warm-start")
    training.add_argument("--gui", action="store_true")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--seed", type=int, default=801)
    evaluation.add_argument("--num-envs", type=int, default=16)
    evaluation.add_argument("--steps", type=int, default=1000)
    evaluation.add_argument("--level", type=int)
    evaluation.add_argument("--device")
    evaluation.add_argument("--command", nargs=3, type=float, action="append")
    evaluation.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    if args.list_tasks:
        print("\n".join(list_tasks(backend="physx")))
    elif args.mode == "preflight":
        print(json.dumps(preflight(args), indent=2, ensure_ascii=False))
    elif args.mode == "train":
        train(args)
    elif args.mode == "evaluate":
        evaluate(args)
    else:
        parser.error("select preflight, train or evaluate")
