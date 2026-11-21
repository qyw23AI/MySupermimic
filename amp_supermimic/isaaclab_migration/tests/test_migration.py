"""Offline regression tests for PhysX migration contracts. No simulator startup."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from amp_supermimic.isaaclab_migration.cli import FLAT, action_diagnostic_tensors, validate_warm_start
from amp_supermimic.isaaclab_migration.config import motion_receipt, resolve, robot_from
from amp_supermimic.isaaclab_migration.runtime import Runtime, configure_trackerlab
from amp_supermimic.isaaclab_migration.wmp_terrain import CourseAtlas, CourseEvents
from amp_supermimic.legacy import class_to_dict, list_tasks, load_task_config
from amp_supermimic.scripts.open_terrain import sample_open_flat


class MigrationContracts(unittest.TestCase):
    def test_physx_open_flat_inherits_warp_training_contract(self):
        warp, warp_train = load_task_config("wblwbl_dog_v1_open_flat")
        physx, physx_train = resolve(SimpleNamespace(task="wblwbl_dog_v1_physx_open_flat"))
        warp_env = class_to_dict(warp.env)
        physx_env = class_to_dict(physx.env)
        physx_env.pop("solver_velocity_iterations")
        self.assertEqual(warp_env, physx_env)
        for field in ("amp", "rewards", "domain_rand"):
            self.assertEqual(class_to_dict(getattr(warp, field)), class_to_dict(getattr(physx, field)))
        self.assertEqual(class_to_dict(warp_train.world_model), class_to_dict(physx_train.world_model))
        self.assertEqual(class_to_dict(warp_train.algorithm), class_to_dict(physx_train.algorithm))
        self.assertEqual(class_to_dict(warp.commands.curriculum), class_to_dict(physx.commands.curriculum))

    def test_open_flat_command_mode_filter_supports_walk_turn_and_moving_turn(self):
        cfg = {
            "fwd_vx_min": 0.20, "fwd_vx_start_max": 0.55, "fwd_vx_end_max": 2.0,
            "bwd_vx_min": 0.15, "bwd_vx_start_max": 0.45, "bwd_vx_end_max": 1.0,
            "strafe_vy_min": 0.10, "strafe_vy_start_max": 0.35, "strafe_vy_end_max": 0.75,
            "turn_wz_min": 0.30, "turn_wz_start_max": 0.90, "turn_wz_end_max": 1.60,
        }
        commands, _ = sample_open_flat(cfg, 0.0, 768, "cpu", torch.float32,
                                        mode_set=("forward", "turn", "forward_turn"))
        self.assertTrue(torch.all(commands[:, 0] >= 0))
        self.assertTrue(torch.all(commands[:, 1] == 0))
        straight = (commands[:, 0] > 0) & (commands[:, 2] == 0)
        in_place_turn = (commands[:, 0] == 0) & (commands[:, 2] != 0)
        moving_turn = (commands[:, 0] > 0) & (commands[:, 2] != 0)
        self.assertTrue(torch.any(straight))
        self.assertTrue(torch.any(in_place_turn))
        self.assertTrue(torch.any(moving_turn))
        self.assertTrue(torch.any(commands[:, 2] > 0))
        self.assertTrue(torch.any(commands[:, 2] < 0))

    def test_physx_flat_keeps_explicit_feet_and_native_contact_solver(self):
        flat, _ = resolve(SimpleNamespace(task="wblwbl_dog_v1_physx_open_flat"))
        parkour, _ = resolve(SimpleNamespace(task="wblwbl_dog_v1_physx_parkour"))
        self.assertFalse(flat.robot.merge_fixed_joints)
        self.assertEqual(flat.env.solver_velocity_iterations, 1)
        self.assertFalse(parkour.robot.merge_fixed_joints)

    def test_trackerlab_parkour_uses_heightfield_diagonal(self):
        from amp_supermimic.isaaclab_migration.parkour import collision_mesh

        env, _ = resolve(SimpleNamespace(task="wblwbl_dog_v1_physx_parkour", level=2))
        runtime = Runtime("parkour", env, robot_from(env), None, "cpu")
        field = runtime.field(runtime.terrain, env.robot.standing_height, "cpu")
        vertices, faces = collision_mesh(runtime.terrain)
        points = torch.tensor(vertices[faces[::739]].mean(1), dtype=torch.float32)
        torch.testing.assert_close(field.sample(points[:, :2]), points[:, 2], atol=1e-5, rtol=1e-5)

    def test_task_overrides_are_isolated(self):
        env, train = resolve(SimpleNamespace(task=FLAT, num_envs=2, iterations=3))
        self.assertEqual((env.env.num_envs, train.runner.iterations), (2, 3))
        fresh, fresh_train = resolve(SimpleNamespace(task=FLAT))
        self.assertEqual((fresh.env.num_envs, fresh_train.runner.iterations), (256, 500))
        self.assertIsNone(fresh.robot.source_config)
        self.assertNotIn(FLAT, list_tasks(backend="warp"))
        with self.assertRaises(ValueError):
            resolve(SimpleNamespace(task=FLAT, level=2))

    def test_action_diagnostics_detect_saturation_and_target_clipping(self):
        action = torch.tensor([[0.0, 20.0, -30.0]])
        xi = torch.zeros(1, 7)
        default_q = torch.zeros(3)
        qlo, qhi = torch.full((3,), -1.0), torch.full((3,), 1.0)
        joint_pos = torch.tensor([[-1.0, 0.0, 1.0]])
        joint_vel = torch.tensor([[1.0, -2.0, 3.0]])
        sums, maximum = action_diagnostic_tensors(
            action, xi, default_q, qlo, qhi, .25, 20.0, joint_pos, joint_vel)
        self.assertEqual(sums.tolist(), [50.0, 2.0, 2.0, 8.0, 2.0, 2.0, 6.0])
        self.assertEqual(maximum.item(), 30.0)

    def test_warm_start_preserves_action_limit(self):

        env, train = resolve(SimpleNamespace(task=FLAT, action_limit=20.))
        robot = robot_from(env)
        saved = {
            "robot": env.robot.name,
            "arguments": {"hist": train.world_model.hist, "action_limit": 20.},
            "simulator_contract": {"joint_names": list(robot.joints), "default_q": list(robot.default_q),
                                   "action_scale": [env.control.action_scale] * len(robot.joints)},
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "actor.pt"
            torch.save(saved, path)
            validate_warm_start(path, env, train, robot)
            env.env.action_limit = 4.
            with self.assertRaisesRegex(ValueError, "action_limit"):
                validate_warm_start(path, env, train, robot)

    def test_real_references_feed_trackerlab_amp(self):
        env, train = resolve(SimpleNamespace(task=FLAT))
        robot = robot_from(env)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, audit = motion_receipt(env, robot, root)
            self.assertEqual((len(audit["train"]), len(audit["test"])), (4, 2))
            runtime = Runtime(FLAT, env, robot, root, "cpu")
            from trackerLab.algorithms.supermimic.amp_style import AMPStyle
            from trackerLab.motion_buffer.motion_lib.supermimic import MotionLib

            kin = robot.kinematics("cpu", runtime.spec)
            motion = MotionLib.from_config(str(manifest), kin, "cpu")
            field = runtime.field(runtime.terrain, env.robot.standing_height, "cpu")
            style = AMPStyle(kin, motion, env.robot.standing_height, ground_sampler=field.sample,
                             history_steps=env.amp.amp_history, hidden=(32, 16), replay_capacity=64)
            self.assertTrue(torch.isfinite(style.sample_expert(8)).all())
            self.assertEqual(len(field.offsets), 187)
            self.assertEqual(len(field.model_offsets), 54)
            lower, upper = torch.tensor(robot.limits).T
            mdp = runtime.make_mdp(field, torch.tensor(robot.default_q), lower, upper)
            self.assertEqual(mdp.obs_dim, 232)
            self.assertEqual(train.world_model.hist, 10)

    def test_reference_reset_adds_terrain_height_in_the_correct_frame(self):
        from amp_supermimic.isaaclab_migration.backend import PhysxCommandSim

        scene = PhysxCommandSim.__new__(PhysxCommandSim)
        scene.atlas, scene.parkour = None, False
        scene.mdp = SimpleNamespace(field=SimpleNamespace(
            sample=lambda xy: .1 * xy[..., 0] + .05 * xy[..., 1]))
        world_xy = torch.tensor([[10.2, 20.4]])
        origins = torch.tensor([[10.0, 20.0]])
        reference_z = torch.tensor([.4])
        # Local HeightField: subtract the per-environment world origin first.
        torch.testing.assert_close(scene._reference_root_height(world_xy, origins, reference_z),
                                   torch.tensor([.44]))
        # WMP atlas coordinates are global and must not subtract the environment origin.
        scene.atlas = object()
        torch.testing.assert_close(scene._reference_root_height(world_xy, origins, reference_z),
                                   torch.tensor([2.44]))

    def test_transition_keeps_pre_reset_state(self):
        configure_trackerlab()
        from amp_supermimic.isaaclab_migration.backend import PhysxCommandSim
        from trackerLab.algorithms.supermimic.buffer import ReplayBuffer

        env, _ = resolve(SimpleNamespace(task=FLAT))
        robot = robot_from(env)
        runtime = Runtime(FLAT, env, robot, None, "cpu")
        field = runtime.field(runtime.terrain, env.robot.standing_height, "cpu")
        lower, upper = torch.tensor(robot.limits).T
        scene = PhysxCommandSim.__new__(PhysxCommandSim)
        scene.cfg, scene.events, scene.dt = env, None, .02
        scene.parkour = scene.open_flat = False
        scene.default_q, scene.qlo, scene.qhi = torch.tensor(robot.default_q), lower, upper
        scene.mdp = runtime.make_mdp(field, scene.default_q, lower, upper)
        scene.c = torch.tensor([[0., .5, 0., 0., .4, 0., 0.]])
        scene.xi = torch.zeros(1, 16)
        scene.a_prev = torch.zeros(1, 12)
        scene.ep_return = torch.zeros(1)
        scene.ep_tracking = torch.zeros(1)
        scene.ep_step = torch.zeros(1, dtype=torch.long)
        scene.command_interval = torch.tensor([150])
        scene.command_age = torch.tensor([12])
        scene.levels = torch.zeros(1, dtype=torch.long)
        scene.joint_ids = list(range(12))
        position = torch.tensor([[0., 0., .4]])

        def integrate():
            position[0, 0] += .0025

        def reset(ids):
            position[ids, 0] = -1.
            scene.a_prev[ids] = 0.
            scene.c[ids] = 0.
            scene.ep_step[ids] = 0
            scene.ep_return[ids] = 0.
            scene.ep_tracking[ids] = 0.

        scene.state = lambda: (position.clone(), torch.eye(3)[None],
                               torch.tensor([[.5, 0., 0.]]), torch.zeros(1, 3),
                               scene.default_q[None], torch.zeros(1, 12))
        scene.sim = SimpleNamespace(step=lambda **kwargs: integrate(), has_gui=lambda: False)
        scene.scene = SimpleNamespace(write_data_to_sim=lambda: None, update=lambda dt: None)
        scene.robot = SimpleNamespace(set_joint_position_target=lambda *a, **k: None)
        scene.reset = lambda ids, **kwargs: reset(ids)
        scene.sample_commands = lambda ids, **kwargs: None

        first_action = torch.full((1, 12), .25)
        first = scene.step(first_action, fixed_command=True)
        self.assertFalse(first["done"].item())
        torch.testing.assert_close(first["a_prev"], torch.zeros_like(first_action))
        torch.testing.assert_close(scene.a_prev, first_action)

        # End the next transition deliberately: its replay row must retain the
        # pre-reset state and previous action, while reset clears live history.
        scene.ep_step.fill_(env.env.max_episode_steps - 1)
        next_action = torch.full((1, 12), .5)
        transition = scene.step(next_action, fixed_command=True)
        self.assertTrue(transition["truncated"].item())
        self.assertFalse(transition["terminated"].item())
        self.assertAlmostEqual(transition["s1"][0, 0].item(), .02)
        self.assertEqual(position[0, 0].item(), -1.)
        torch.testing.assert_close(transition["a_prev"], first_action)
        torch.testing.assert_close(transition["a"], next_action)
        torch.testing.assert_close(scene.a_prev, torch.zeros_like(next_action))
        replay = ReplayBuffer(4, 1, "cpu", 12, scenario_dim=7, command_timing=True)
        replay.add(transition)
        self.assertEqual(replay.size, 1)

        # Viewer diagnostics can preserve the terminal pose instead of
        # immediately teleporting the robot to its reset state.
        scene.ep_step.fill_(env.env.max_episode_steps - 1)
        position[0, 0] = 0.
        scene.ep_return.zero_()
        scene.ep_tracking.zero_()
        scene.a_prev.zero_()
        scene.c[0] = torch.tensor([0., .5, 0., 0., .4, 0., 0.])
        held = scene.step(next_action, fixed_command=True, auto_reset=False)
        self.assertTrue(held["truncated"].item())
        self.assertAlmostEqual(position[0, 0].item(), .01)
        self.assertEqual(scene.ep_step.item(), env.env.max_episode_steps)


    def test_wmp_l0_mesh_query_and_strict_finish(self):
        env, _ = resolve(SimpleNamespace(task="wblwbl_dog_v1_physx_wmp_stair_gap"))
        atlas = CourseAtlas(env.terrain, "cpu")
        vertices, faces = atlas.mesh()
        self.assertGreater(len(vertices), 0)
        self.assertLess(faces.max(), len(vertices))
        course = atlas.cells[0, 0]
        center = torch.tensor(atlas.center(0, 0))
        x, y, _, _, z = course.boxes[4]
        target = center + torch.tensor([x, y, z])
        launch = course.boxes[3]
        gap_x = (launch[0] + launch[2] / 2 + x - course.boxes[4][2] / 2) / 2
        sample = torch.stack([target[:2], center[:2] + torch.tensor([gap_x, 0.])])
        torch.testing.assert_close(atlas.sample(sample), torch.tensor([z, course.floor_height]))

        events = CourseEvents(atlas, 1, "cpu", .02)
        command = torch.tensor([[4., .25, 0., 0., .4, 0., 0.]])
        forces = torch.tensor([[[0., 0., 10.]]]).expand(1, 4, 3)
        fallen = torch.zeros(1, dtype=torch.bool)
        total = 0.
        episode_steps = torch.ones(1, dtype=torch.long)
        for index in [1, 2, 3] + [4] * 17:
            x, y, _, _, z = course.boxes[index]
            sole = (center + torch.tensor([x, y, z])).expand(1, 4, 3)
            pos = sole[:, 0, :] + torch.tensor([0., 0., .4])
            state = pos, torch.eye(3)[None], torch.zeros(1, 3), None, None, None
            success, failure, rewards = events.update(state, sole, forces, command, fallen, fallen,
                                                       episode_steps, env.rewards)
            total += float(sum(rewards.values()).item())
            self.assertFalse(failure.item())
        self.assertTrue(success.item())
        self.assertTrue(events.course.strict.item())
        self.assertAlmostEqual(total, 3.5)
        events.finish(torch.tensor([0]))
        self.assertEqual(events.course.strict_wins.sum().item(), 1)
        self.assertFalse(events.course.success.item())

    def test_wmp_curriculum_selects_next_geometry(self):
        env, _ = resolve(SimpleNamespace(task="wblwbl_dog_v1_physx_wmp_gap_combos"))
        atlas = CourseAtlas(env.terrain, "cpu")
        events = CourseEvents(atlas, 2, "cpu", .02)
        kind = int(events.course.kind[0])
        events.course.window_n[kind, 0] = 49
        events.course.window_wins[kind, 0] = 39
        events.course.success[0] = True
        events.core.episode_length_buf = torch.ones(2, dtype=torch.long)
        events.core.time_out_buf = torch.zeros(2, dtype=torch.bool)
        events.finish(torch.tensor([0]))
        self.assertEqual(events.core.terrain_levels.tolist(), [1, 0])
        torch.testing.assert_close(events.core.env_origins[0], atlas.spawns[1, 0])


if __name__ == "__main__":
    configure_trackerlab()
    unittest.main()
