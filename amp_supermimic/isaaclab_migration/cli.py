"""Command line for offline checks, PhysX training, and fixed-command evaluation."""

import argparse
import json
from pathlib import PosixPath

from amp_supermimic.legacy import (
    build_upstream_train_args, list_tasks, resolve_repo_path,
)
from .config import DEFAULT_FRICTION_RANGE, motion_receipt, read_snapshot, resolve, robot_from, snapshot, write_json

FLAT = "wblwbl_dog_v1_physx_flat"


def task_options(parser):
    parser.add_argument("--task", choices=list_tasks(backend="physx"), default=FLAT)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--action-limit", type=float)
    parser.add_argument("--project-joint-targets", action=argparse.BooleanOptionalAction, default=None,
                        help="Feed physically bounded joint targets to the world model")
    parser.add_argument("--friction-range", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="Sample the effective ground contact coefficient in this range at training resets")
    parser.add_argument("--motion-config")
    parser.add_argument("--amp-weight", type=float)
    parser.add_argument("--amp-batch", type=int)
    parser.add_argument("--amp-updates", type=int)
    parser.add_argument("--amp-history", type=int)
    parser.add_argument("--command-training-modes", dest="training_modes", nargs="+",
                        choices=("stop", "forward", "backward", "strafe", "turn", "mixed", "forward_turn"),
                        help="Restrict open_flat command sampling to the selected motion modes")
    parser.add_argument("--source", "--wmp-source")
    parser.add_argument("--competition-stage", choices=("eval", "learn"))
    parser.add_argument("--level", type=int)
    for flag in ("compile", "compile-critic", "fused-optimizers"):
        parser.add_argument(f"--{flag}", action=argparse.BooleanOptionalAction, default=None)
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
    if env.terrain.profile not in ("plane", "open_flat", "parkour"):
        from .wmp_terrain import CourseAtlas
        atlas = CourseAtlas(env.terrain, "cpu")
        vertices, faces = atlas.mesh()
        report["terrain"] = {"rows": atlas.rows, "types": atlas.column_types,
                             "vertices": len(vertices), "triangles": len(faces),
                             "l0": {kind: atlas.cells[0, col].parameters
                                    for col, kind in enumerate(atlas.column_types)}}
    return report


def load_checkpoint(path):
    import torch

    with torch.serialization.safe_globals([PosixPath]):
        return torch.load(path, map_location="cpu", weights_only=True)


def action_diagnostic_tensors(policy_action, xi, default_q, qlo, qhi, action_scale,
                               action_limit, joint_pos, joint_vel):
    """Return device-side sums for policy/action/limit diagnostics without per-step syncs."""
    import torch

    clipped_action = policy_action.clamp(-action_limit, action_limit)
    raw_target = default_q + xi[:, 1:-3] + clipped_action * action_scale
    target = raw_target.clamp(qlo, qhi)
    saturated = policy_action.abs() >= action_limit - 1e-6
    target_clipped = (raw_target < qlo) | (raw_target > qhi)
    at_joint_limit = (joint_pos <= qlo + 1e-3) | (joint_pos >= qhi - 1e-3)
    sums = torch.stack((
        policy_action.abs().sum(), saturated.sum().to(policy_action.dtype),
        target_clipped.sum().to(policy_action.dtype), (raw_target - target).abs().sum(),
        at_joint_limit.sum().to(policy_action.dtype),
        joint_pos.abs().sum(), joint_vel.abs().sum(),
    ))
    return sums, policy_action.abs().max()


def validate_warm_start(path, env, training, robot):
    saved = load_checkpoint(path)
    args = saved["arguments"]
    contract = saved["simulator_contract"]
    scale = contract["action_scale"]
    scales = (scale,) * len(robot.joints) if isinstance(scale, (int, float)) else tuple(scale)
    if (saved["robot"] != env.robot.name or tuple(contract["joint_names"]) != robot.joints
            or tuple(contract["default_q"]) != robot.default_q
            or scales != (env.control.action_scale,) * len(robot.joints)):
        raise ValueError("Warm-start joint order, nominal pose or action scale differs from PhysX")
    expected = {"hist": training.world_model.hist, "action_limit": env.env.action_limit}
    for key, value in expected.items():
        if args.get(key) != value:
            raise ValueError(f"Warm-start requires matching {key}: checkpoint={args.get(key)}, requested={value}")


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
    warm_start = resolve_repo_path(args.warm_start)
    if args.resume:
        if args.warm_start:
            raise ValueError("Choose either --resume or --warm-start")
        if getattr(env.terrain, "competition_stage", None) == "learn":
            raise ValueError("Resume of a WMP learning curriculum requires persisted course statistics")
        saved = json.loads((run / "legacy_task_config.json").read_text())
        saved["env_cfg"]["control"].setdefault("project_joint_targets", False)
        saved["env_cfg"]["terrain"].setdefault("friction_range", list(DEFAULT_FRICTION_RANGE))
        saved["train_cfg"]["runner"]["iterations"] = training.runner.iterations
        if saved != plan:
            raise ValueError("Resume requires the original task parameters; only iterations can change")
        if not (run / "last.pt").is_file():
            raise FileNotFoundError(run / "last.pt")
    else:
        if warm_start is not None:
            validate_warm_start(warm_start, env, training, robot)
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
    if warm_start is not None:
        flags += ["--warm-start", str(warm_start), "--warm-start-reset-world"]
    upstream_args = build_upstream_train_args(env, training, flags)
    configure_trackerlab()
    from train_terrain import parse_args
    parsed = parse_args(upstream_args)
    write_json(run / "physx_config.json", plan | identity | {
        "arguments": vars(parsed), "control_dt": env.control.physics_dt * env.control.decimation,
        "resolved_urdf": str(robot.path),
        "resolved_wmp_source": str(resolve_repo_path(env.terrain.source))
        if hasattr(env.terrain, "source") else None,
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
            source=("TrackerLab triangulated heightfield" if env.terrain.profile == "parkour"
                    else "plane" if runtime.atlas is None else "WMP analytic mesh top-surface query"),
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
            launcher.app.close(wait_for_replicator=False, skip_cleanup=not args.gui)


def evaluate(args):
    import torch
    from .runtime import Runtime, configure_trackerlab

    checkpoint = resolve_repo_path(args.checkpoint)
    task, env, train_cfg = read_snapshot(checkpoint.parent / "physx_config.json")
    hold_on_event = args.no_auto_reset or args.stop_on_instability
    if hold_on_event:
        # A diagnostic rollout must remain one continuous episode; do not let
        # the training-time 20 s timeout silently reset the robot mid-view.
        env.env.max_episode_steps = max(env.env.max_episode_steps, args.steps + 1)
    robot = robot_from(env)
    saved = load_checkpoint(checkpoint)
    contract = saved["simulator_contract"]
    if saved["task"] != task or contract["backend"] != "isaaclab-physx" or robot.sha256 != contract["robot_sha256"]:
        raise ValueError("Checkpoint task, backend or URDF hash does not match the run")
    if args.level is not None and env.terrain.profile in ("plane", "open_flat"):
        raise ValueError("--level applies to terrain tasks")
    if env.terrain.profile == "parkour":
        env.terrain.level = 0 if args.level is None else args.level
        if not 0 <= env.terrain.level < 6:
            raise ValueError("TrackerLab parkour levels are 0–5")
    elif env.terrain.profile not in ("plane", "open_flat"):
        env.terrain.competition_stage = "eval"
        if args.level is not None:
            from .wmp_terrain import OBSTACLE_LEVELS

            maximum = OBSTACLE_LEVELS.get(env.terrain.profile, 9)
            if not 0 <= args.level <= maximum:
                raise ValueError(f"{env.terrain.profile} level must be 0–{maximum}")
            env.terrain.level = args.level
    if min(args.num_envs, args.steps) < 1:
        raise ValueError("Expected positive environment and evaluation step counts")
    if not 0. < args.instability_upright <= 1.:
        raise ValueError("--instability-upright must lie in (0, 1]")
    if not 0. < args.instability_clearance_ratio <= 1.:
        raise ValueError("--instability-clearance-ratio must lie in (0, 1]")
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
    print(f"Starting PhysX evaluation ({args.num_envs} envs, {args.steps} steps per command)"
          f" -> {output}", flush=True)
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
            commands = args.command or env.commands.samples
            for index, command in enumerate(commands, 1):
                label = f"[{index}/{len(commands)}] command {tuple(command)}"
                print(f"{label}: starting rollout", flush=True)
                errors = torch.zeros(7, device=sim.device)
                diag_sum = torch.zeros(7, device=sim.device)
                max_action = torch.zeros((), device=sim.device)
                motion_sum = torch.zeros(5, device=sim.device)
                min_upright = torch.ones((), device=sim.device)
                min_clearance = torch.full((), float("inf"), device=sim.device)
                min_root_height = torch.full((), float("inf"), device=sim.device)
                max_root_height = torch.zeros((), device=sim.device)
                root_height_sq_sum = torch.zeros((), device=sim.device)
                root_vz_sq_sum = torch.zeros((), device=sim.device)
                max_abs_root_vz = torch.zeros((), device=sim.device)
                upright_low_steps = torch.zeros((), device=sim.device)
                clearance_low_steps = torch.zeros((), device=sim.device)
                actual_steps = 0
                stop_reason = None
                sim.twists = sim.c.new_tensor([command])
                sim.reset(fixed_command=True)
                previous = sim.c.new_zeros((sim.num_envs, hist, sim.mdp.obs_dim))
                for step in range(1, args.steps + 1):
                    state, observation = sim.state(), sim.observe()
                    previous = torch.where(sim.ep_step[:, None, None] == 0, observation[:, None],
                                           torch.cat((previous[:, 1:], observation[:, None]), 1))
                    action = actor(previous, sim.mdp.goal(state, sim.c))
                    xi = sim.xi.clone()
                    transition = sim.step(action, fixed_command=True, auto_reset=not hold_on_event)
                    next_state = unflatten_state(transition["s1"])
                    v, w = body_velocities(next_state)
                    actual_steps = step
                    upright = next_state[1][:, 2, 2]
                    clearance = next_state[0][:, 2] - sim.mdp.field.sample(next_state[0][:, :2])
                    root_height = next_state[0][:, 2]
                    min_upright = torch.minimum(min_upright, upright.min())
                    min_clearance = torch.minimum(min_clearance, clearance.min())
                    min_root_height = torch.minimum(min_root_height, root_height.min())
                    max_root_height = torch.maximum(max_root_height, root_height.max())
                    root_height_sq_sum += root_height.square().sum()
                    root_vz_sq_sum += v[:, 2].square().sum()
                    max_abs_root_vz = torch.maximum(max_abs_root_vz, v[:, 2].abs().max())
                    upright_low_steps += (upright < args.instability_upright).sum()
                    clearance_low_steps += (
                        clearance < args.instability_clearance_ratio * env.robot.standing_height
                    ).sum()
                    sampled = transition["c1"]
                    sums, action_max = action_diagnostic_tensors(
                        action, xi, sim.default_q, sim.qlo, sim.qhi,
                        env.control.action_scale, env.env.action_limit,
                        next_state[4], next_state[5])
                    diag_sum += sums
                    max_action = torch.maximum(max_action, action_max)
                    body_delta = (state[1].transpose(-1, -2) @
                                  (next_state[0] - state[0]).unsqueeze(-1)).squeeze(-1)
                    motion_sum += torch.stack((v[:, 0].sum(), v[:, 1].sum(), w[:, 2].sum(),
                                               body_delta[:, 0].sum(), next_state[0][:, 2].sum()))
                    errors[0] += (v[:, :2] - sampled[:, 1:3]).square().sum()
                    errors[1] += (w[:, 2] - sampled[:, 3]).square().sum()
                    errors[2] += transition["episode_fall"].sum()
                    errors[3] += transition["episode_success"].sum()
                    errors[4] += (transition["episode_strict_success"] & transition["done"]).sum()
                    errors[5] += transition["done"].sum()
                    errors[6] += transition["episode_timeout"].sum()
                    if args.stop_on_instability:
                        unstable = ((upright < args.instability_upright)
                                    | (clearance < args.instability_clearance_ratio
                                       * env.robot.standing_height))
                        if bool(unstable.any()):
                            stop_reason = "visual_instability_threshold"
                            print(f"{label}: instability threshold crossed at step {step}; "
                                  f"min upright={min_upright.item():.3f}, "
                                  f"min clearance={min_clearance.item():.3f} m", flush=True)
                            break
                    if hold_on_event and bool(transition["done"].any()):
                        stop_reason = ("fall" if bool(transition["episode_fall"].any())
                                       else "terminal")
                        print(f"{label}: stopped on {stop_reason} at step {step}", flush=True)
                        break
                    if step % 250 == 0 or step == args.steps:
                        print(f"{label}: {step}/{args.steps} steps", flush=True)
                samples = actual_steps * sim.num_envs
                joint_samples = samples * len(robot.joints)
                velocity, yaw, falls, success, strict, episodes, timeouts = errors.tolist()
                action_abs, saturated, target_clipped, target_clip_abs, at_limit, joint_abs, joint_speed = diag_sum.tolist()
                mean_vx, mean_vy, mean_yaw, body_dx, root_height = (motion_sum / samples).tolist()
                results.append({
                    "command": list(command), "evaluated_control_steps": actual_steps,
                    "stop_reason": stop_reason,
                    "velocity_rmse_mps": (velocity / samples) ** .5,
                    "yaw_rmse_rad_s": (yaw / samples) ** .5, "falls": int(falls),
                    "falls_per_robot_minute": falls / (samples * sim.dt / 60),
                    "successes": int(success), "strict_successes": int(strict),
                    "completed_episodes": int(episodes), "timeouts": int(timeouts),
                    "strict_success_rate": strict / episodes if episodes else None,
                    "mean_body_vx_mps": mean_vx, "mean_body_vy_mps": mean_vy,
                    "mean_body_yaw_rate_rad_s": mean_yaw,
                    "mean_body_dx_per_step_m": body_dx, "mean_root_height_m": root_height,
                    "min_root_height_m": float(min_root_height),
                    "max_root_height_m": float(max_root_height),
                    "root_height_std_m": float((root_height_sq_sum / samples - root_height ** 2).clamp_min(0).sqrt()),
                    "root_vertical_speed_rms_mps": float((root_vz_sq_sum / samples).sqrt()),
                    "min_root_clearance_m": float(min_clearance),
                    "min_upright_projection": float(min_upright),
                    "max_abs_root_vertical_velocity_mps": float(max_abs_root_vz),
                    "upright_below_threshold_fraction": float(upright_low_steps / samples),
                    "clearance_below_threshold_fraction": float(clearance_low_steps / samples),
                    "instability_upright_threshold": args.instability_upright,
                    "instability_clearance_ratio": args.instability_clearance_ratio,
                    "mean_abs_action": action_abs / joint_samples, "max_abs_action": float(max_action),
                    "action_saturation_fraction": saturated / joint_samples,
                    "joint_target_clip_fraction": target_clipped / joint_samples,
                    "joint_target_clip_mean_rad": target_clip_abs / joint_samples,
                    "joint_limit_occupancy_fraction": at_limit / joint_samples,
                    "mean_abs_joint_position_rad": joint_abs / joint_samples,
                    "mean_abs_joint_velocity_rad_s": joint_speed / joint_samples,
                })
                if stop_reason is not None:
                    break
        write_json(output / "evaluation.json", {"task": task, "seed": args.seed, "num_envs": args.num_envs,
                                                 "steps": args.steps, "simulator_contract": sim.contract,
                                                 "results": results})
        print(f"Evaluation complete. Results: {output / 'evaluation.json'}", flush=True)
        if args.gui and args.hold_viewer:
            print("Isaac Sim is paused at the final robot pose. Close the Isaac Sim window to exit.",
                  flush=True)
            while launcher.app.is_running():
                launcher.app.update()
    finally:
        launcher.app.close(wait_for_replicator=False, skip_cleanup=not args.gui)


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
    evaluation.add_argument("--no-auto-reset", action="store_true",
                             help="Keep one continuous episode and stop on the first terminal event")
    evaluation.add_argument("--hold-viewer", action="store_true",
                             help="Keep Isaac Sim open and paused after the rollout finishes or stops")
    evaluation.add_argument("--stop-on-instability", action="store_true",
                             help="Stop the rollout when upright/clearance diagnostics cross thresholds")
    evaluation.add_argument("--instability-upright", type=float, default=0.70,
                             help="Diagnostic minimum R[2,2] threshold (default: 0.70)")
    evaluation.add_argument("--instability-clearance-ratio", type=float, default=0.65,
                             help="Diagnostic minimum root clearance / standing height (default: 0.65)")
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
