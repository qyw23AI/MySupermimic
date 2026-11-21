"""Replay a retargeted reference gait through the real PhysX PD actuators.

This bypasses the learned actor and world model. It answers whether the
imported robot, joint order, PD drives, effort limits and contact geometry can
physically execute the reference trajectory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="amp_supermimic/outputs/physx/open-flat-mergefixed-pilot-20261117/physx_config.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--clip", default="forward__procedural_forward")
    parser.add_argument("--start-frame", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    if args.start_frame < 0 or args.steps < 1:
        raise ValueError("start-frame must be nonnegative and steps must be positive")

    import torch
    from .cli import read_snapshot, resolve_repo_path, robot_from
    from .runtime import Runtime, configure_trackerlab

    config_path = resolve_repo_path(args.config)
    output = resolve_repo_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    task, env, train_cfg = read_snapshot(config_path)
    robot = robot_from(env)
    env.env.num_envs, env.env.seed = 1, 1701
    if args.device:
        env.env.device = args.device

    configure_trackerlab()
    from isaaclab.app import AppLauncher
    from trackerLab.motion_buffer.motion_lib.supermimic import MotionLib
    from trackerLab.kinematics.supermimic.rotations import mat_to_quat, quat_to_mat, so3_log, yaw_mat
    from trackerLab.tracker_env.mdp.supermimic_command import body_velocities

    launcher = AppLauncher(headless=not args.gui, device=env.env.device)
    try:
        runtime = Runtime(task, env, robot, output)
        kin = robot.kinematics(runtime.device, runtime.spec,
                               tracked=runtime.spec.tracked_bodies, base=runtime.kinematics_class)
        motion_path = resolve_repo_path(env.amp.motion_config)
        motion = MotionLib.from_config(str(motion_path), kin, runtime.device)
        if args.clip not in motion.names:
            raise ValueError(f"Clip {args.clip!r} not found; available: {motion.names}")
        clip_index = motion.names.index(args.clip)
        clip_start = int(motion.clip_start[clip_index].item())
        clip_end = int(motion.clip_end[clip_index].item())
        start = clip_start + args.start_frame
        if start + args.steps > clip_end:
            raise ValueError(f"Requested {args.steps} steps exceed clip end; available {clip_end - start}")

        sim = runtime.create(train_cfg.algorithm.gamma)
        sim.reset(fixed_command=True)
        ids = torch.tensor([0], device=sim.device, dtype=torch.long)
        start_state = sim.state()
        root_pose = sim.robot.data.root_link_pose_w[ids].clone()
        current_R = quat_to_mat(root_pose[:, 3:7])
        reference_R = motion.root_R[start].unsqueeze(0)
        alignment = yaw_mat(current_R) @ yaw_mat(reference_R).transpose(-1, -2)
        aligned_R = alignment @ reference_R
        root_xy = root_pose[:, :2].clone()
        floor = sim.mdp.field.sample(root_xy)
        root_pose[:, 2] = floor + motion.root_p[start, 2]
        root_pose[:, 3:7] = mat_to_quat(aligned_R)
        root_vel = torch.cat((
            (alignment @ motion.root_v[start].view(1, 3, 1)).squeeze(-1),
            (alignment @ motion.root_w[start].view(1, 3, 1)).squeeze(-1),
        ), dim=-1)
        offsets = sim.xi[ids, 1:-3].clone()
        q0 = motion.q[start].unsqueeze(0) + offsets
        qd0 = motion.qd[start].unsqueeze(0)
        sim.robot.write_root_link_pose_to_sim(root_pose, env_ids=ids)
        sim.robot.write_root_link_velocity_to_sim(root_vel, env_ids=ids)
        sim.robot.write_joint_state_to_sim(q0, qd0, env_ids=ids, joint_ids=sim.joint_ids)
        sim.robot.set_joint_position_target(q0, joint_ids=sim.joint_ids)
        sim.scene.write_data_to_sim()
        sim.scene.update(sim.cfg.control.physics_dt)

        origin = sim.state()[0][0].clone()
        root_pos_errors, root_rot_errors = [], []
        body_v_errors, body_w_errors = [], []
        joint_pos_errors, joint_vel_errors = [], []
        actual_vx, reference_vx = [], []
        target_clip_fractions, limit_occupancies = [], []
        torque_abs_means, torque_abs_maxes = [], []
        joint_tracking_abs_means = []
        for offset in range(1, args.steps + 1):
            frame = start + offset
            target = motion.q[frame].unsqueeze(0) + offsets
            target_clip_fractions.append(((target < sim.qlo) | (target > sim.qhi)).float().mean())
            sim.robot.set_joint_position_target(target, joint_ids=sim.joint_ids)
            for _ in range(sim.cfg.control.decimation):
                sim.scene.write_data_to_sim()
                sim.sim.step()
                sim.scene.update(sim.cfg.control.physics_dt)

            state = sim.state()
            expected_p = origin + (alignment @ (motion.root_p[frame] - motion.root_p[start]).view(3, 1)).squeeze(-1)
            expected_R = alignment[0] @ motion.root_R[frame]
            expected_v = (alignment[0] @ motion.root_v[frame])
            expected_w = (alignment[0] @ motion.root_w[frame])
            actual_v, actual_w = body_velocities(state)
            expected_body_v = expected_R.transpose(-1, -2) @ expected_v
            expected_body_w = expected_R.transpose(-1, -2) @ expected_w
            root_pos_errors.append(torch.linalg.vector_norm(state[0][0] - expected_p))
            root_rot_errors.append(torch.linalg.vector_norm(so3_log(state[1][0] @ expected_R.transpose(-1, -2))))
            body_v_errors.append(torch.linalg.vector_norm(actual_v[0] - expected_body_v))
            body_w_errors.append(torch.linalg.vector_norm(actual_w[0] - expected_body_w))
            joint_pos_errors.append(torch.linalg.vector_norm(state[4][0] - target[0]) / len(robot.joints) ** 0.5)
            joint_vel_errors.append(torch.linalg.vector_norm(state[5][0] - motion.qd[frame]) / len(robot.joints) ** 0.5)
            actual_q = sim.robot.data.joint_pos[:, sim.joint_ids]
            limit_occupancies.append(((actual_q - sim.qlo < 0.05) | (sim.qhi - actual_q < 0.05)).float().mean())
            joint_tracking_abs_means.append((actual_q - target).abs().mean())
            applied = getattr(sim.robot.data, "applied_torque", None)
            if applied is not None:
                tau = applied[:, sim.joint_ids].abs()
                torque_abs_means.append(tau.mean())
                torque_abs_maxes.append(tau.max())
            actual_vx.append(actual_v[0, 0])
            reference_vx.append(expected_body_v[0])

        def rmse(values):
            tensor = torch.stack(values)
            return float(tensor.square().mean().sqrt().detach().cpu())

        result = {
            "task": task,
            "clip": args.clip,
            "start_frame": args.start_frame,
            "steps": args.steps,
            "control_dt_s": sim.dt,
            "physics_dt_s": sim.cfg.control.physics_dt,
            "decimation": sim.cfg.control.decimation,
            "merge_fixed_joints": sim.merge_fixed_joints,
            "solver_position_iterations": sim.cfg.env.solver_iterations,
            "solver_velocity_iterations": getattr(sim.cfg.env, "solver_velocity_iterations", 1),
            "root_position_rmse_m": rmse(root_pos_errors),
            "root_orientation_rmse_rad": rmse(root_rot_errors),
            "body_linear_velocity_rmse_mps": rmse(body_v_errors),
            "body_angular_velocity_rmse_rad_s": rmse(body_w_errors),
            "joint_position_rmse_rad": rmse(joint_pos_errors),
            "joint_velocity_rmse_rad_s": rmse(joint_vel_errors),
            "actual_mean_body_vx_mps": float(torch.stack(actual_vx).mean().cpu()),
            "reference_mean_body_vx_mps": float(torch.stack(reference_vx).mean().cpu()),
            "joint_target_clip_fraction": float(torch.stack(target_clip_fractions).mean().cpu()),
            "joint_limit_occupancy_fraction_0p05rad": float(torch.stack(limit_occupancies).mean().cpu()),
            "joint_tracking_mean_abs_error_rad": float(torch.stack(joint_tracking_abs_means).mean().cpu()),
            "applied_torque_mean_abs_nm": float(torch.stack(torque_abs_means).mean().cpu()) if torque_abs_means else None,
            "applied_torque_max_abs_nm": float(torch.stack(torque_abs_maxes).max().cpu()) if torque_abs_maxes else None,
            "final_root_height_m": float(sim.state()[0][0, 2].cpu()),
        }
        (output / "reference_tracking_diagnostic.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        launcher.app.close(wait_for_replicator=False, skip_cleanup=not args.gui)


if __name__ == "__main__":
    main()
