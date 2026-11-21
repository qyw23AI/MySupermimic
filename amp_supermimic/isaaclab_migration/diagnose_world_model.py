"""Compare a trained terrain world model against real PhysX rollouts.

Example:
  python -m amp_supermimic.isaaclab_migration.diagnose_world_model \
    --checkpoint amp_supermimic/outputs/physx/RUN/last.pt \
    --output amp_supermimic/outputs/physx/RUN/world-model-diagnostic \
    --num-envs 4 --steps 64 --command 0.5 0 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _group_rmse(error, start: int, end: int) -> float:
    return float(error[..., start:end].square().mean().sqrt().detach().cpu())


def _error_groups(error):
    return {
        "root_position_m": _group_rmse(error, 0, 3),
        "root_rotation_rad": _group_rmse(error, 3, 6),
        "root_linear_velocity_mps": _group_rmse(error, 6, 9),
        "root_angular_velocity_rad_s": _group_rmse(error, 9, 12),
        "joint_position_rad": _group_rmse(error, 12, 24),
        "joint_velocity_rad_s": _group_rmse(error, 24, 36),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=801)
    parser.add_argument("--device", default=None)
    parser.add_argument("--command", nargs=3, type=float, default=(0.5, 0.0, 0.0))
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--domain-randomization", action="store_true",
                        help="apply the PhysX reset-time friction and joint-offset randomization")
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1:
        raise ValueError("num-envs and steps must be positive")

    import torch
    from .cli import load_checkpoint, read_snapshot, resolve_repo_path, robot_from
    from .runtime import Runtime, configure_trackerlab

    checkpoint = resolve_repo_path(args.checkpoint)
    output = resolve_repo_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    task, env, train_cfg = read_snapshot(checkpoint.parent / "physx_config.json")
    robot = robot_from(env)
    saved = load_checkpoint(checkpoint)
    contract = saved["simulator_contract"]
    if saved["task"] != task or contract["backend"] != "isaaclab-physx":
        raise ValueError("Expected a matching Isaac Lab PhysX checkpoint")
    if robot.sha256 != contract["robot_sha256"]:
        raise ValueError("Checkpoint URDF hash does not match the configured robot")
    env.env.num_envs, env.env.seed = args.num_envs, args.seed
    if args.device:
        env.env.device = args.device

    configure_trackerlab()
    from isaaclab.app import AppLauncher
    from trackerLab.algorithms.supermimic.terrain_networks import TerrainPolicy
    from trackerLab.tracker_env.mdp.supermimic import normalize_xi, unflatten_state
    from trackerLab.tracker_env.mdp.supermimic_command import body_velocities

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
        hidden = tuple(getattr(train_cfg.world_model, "hidden", (2048, 2048, 2048)))
        world = runtime.make_world_model(sim.mdp.field, len(robot.joints), hidden=hidden)
        world.load_state_dict(saved["world"])
        world.to(sim.device).eval()

        sim.twists = sim.c.new_tensor([args.command])
        sim.reset(fixed_command=not args.domain_randomization)
        if args.domain_randomization:
            # Keep the randomized physical context, but evaluate one fixed command.
            sim.c[:, 1:4] = sim.c.new_tensor(args.command)
        history = sim.c.new_zeros((sim.num_envs, hist, sim.mdp.obs_dim))
        rollout_state = tuple(value.clone() for value in sim.state())
        one_step_sq = torch.zeros(36, device=sim.device)
        one_step_count = 0
        horizons = sorted(set((1, 8, 32, args.steps)))
        rollout_snapshots = {}
        actual_velocity_sum = torch.zeros(3, device=sim.device)
        actual_height_sum = torch.zeros((), device=sim.device)
        fall_count = torch.zeros((), device=sim.device)
        done_count = torch.zeros((), device=sim.device)

        with torch.inference_mode():
            for step_index in range(args.steps):
                state, observation = sim.state(), sim.observe()
                history = torch.where(sim.ep_step[:, None, None] == 0, observation[:, None],
                                      torch.cat((history[:, 1:], observation[:, None]), 1))
                action = actor(history, sim.mdp.goal(state, sim.c))
                xi_n = normalize_xi(sim.xi.clone())
                one_step_prediction = world.step(state, action, xi_n)
                rollout_prediction = world.step(rollout_state, action, xi_n)
                transition = sim.step(action, fixed_command=True)
                real_next = unflatten_state(transition["s1"])

                one_error = world.state_error(one_step_prediction, real_next)
                rollout_error = world.state_error(rollout_prediction, real_next)
                one_step_sq += one_error.square().sum(0)
                one_step_count += sim.num_envs
                if step_index + 1 in horizons:
                    rollout_snapshots[str(step_index + 1)] = _error_groups(rollout_error)

                body_v, body_w = body_velocities(real_next)
                actual_velocity_sum += torch.stack((body_v[:, 0].sum(), body_v[:, 1].sum(),
                                                    body_w[:, 2].sum()))
                actual_height_sum += real_next[0][:, 2].sum()
                fall_count += transition["episode_fall"].sum()
                done_count += transition["done"].sum()

                rollout_state = rollout_prediction
                done = transition["done"]
                if bool(done.any()):
                    post_reset = sim.state()
                    rollout_state = tuple(
                        torch.where(done.view(-1, *([1] * (value.ndim - 1))), post_reset[index], value)
                        for index, value in enumerate(rollout_state)
                    )

        per_component_rmse = (one_step_sq / max(one_step_count, 1)).sqrt()
        one_step_metrics = {
            "root_position_m": float(per_component_rmse[0:3].square().mean().sqrt().cpu()),
            "root_rotation_rad": float(per_component_rmse[3:6].square().mean().sqrt().cpu()),
            "root_linear_velocity_mps": float(per_component_rmse[6:9].square().mean().sqrt().cpu()),
            "root_angular_velocity_rad_s": float(per_component_rmse[9:12].square().mean().sqrt().cpu()),
            "joint_position_rad": float(per_component_rmse[12:24].square().mean().sqrt().cpu()),
            "joint_velocity_rad_s": float(per_component_rmse[24:36].square().mean().sqrt().cpu()),
        }
        samples = args.steps * sim.num_envs
        result = {
            "task": task,
            "checkpoint": str(checkpoint),
            "checkpoint_iteration": saved.get("iteration"),
            "command": list(args.command),
            "domain_randomization": args.domain_randomization,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "simulator_contract": sim.contract,
            "one_step_prediction_rmse": one_step_metrics,
            "open_loop_rollout_rmse_by_horizon": rollout_snapshots,
            "real_physx_mean_body_vx_mps": float((actual_velocity_sum[0] / samples).cpu()),
            "real_physx_mean_body_vy_mps": float((actual_velocity_sum[1] / samples).cpu()),
            "real_physx_mean_yaw_rate_rad_s": float((actual_velocity_sum[2] / samples).cpu()),
            "real_physx_mean_root_height_m": float((actual_height_sum / samples).cpu()),
            "falls": int(fall_count.item()),
            "completed_episodes": int(done_count.item()),
        }
        (output / "world_model_diagnostic.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        launcher.app.close(wait_for_replicator=False, skip_cleanup=not args.gui)


if __name__ == "__main__":
    main()
