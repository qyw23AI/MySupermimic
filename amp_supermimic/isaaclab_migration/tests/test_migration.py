"""Offline regression tests for PhysX migration contracts. No simulator startup."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from amp_supermimic.isaaclab_migration.cli import FLAT
from amp_supermimic.isaaclab_migration.config import motion_receipt, resolve, robot_from
from amp_supermimic.isaaclab_migration.runtime import Runtime, configure_trackerlab
from amp_supermimic.isaaclab_migration.wmp_terrain import CourseAtlas, CourseEvents
from amp_supermimic.legacy import list_tasks


class MigrationContracts(unittest.TestCase):
    def test_task_overrides_are_isolated(self):
        env, train = resolve(SimpleNamespace(task=FLAT, num_envs=2, iterations=3))
        self.assertEqual((env.env.num_envs, train.runner.iterations), (2, 3))
        fresh, fresh_train = resolve(SimpleNamespace(task=FLAT))
        self.assertEqual((fresh.env.num_envs, fresh_train.runner.iterations), (256, 500))
        self.assertIsNone(fresh.robot.source_config)
        self.assertNotIn(FLAT, list_tasks(backend="warp"))
        with self.assertRaises(ValueError):
            resolve(SimpleNamespace(task=FLAT, level=2))

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
        scene.default_q, scene.qlo, scene.qhi = torch.tensor(robot.default_q), lower, upper
        scene.mdp = runtime.make_mdp(field, scene.default_q, lower, upper)
        scene.c = torch.tensor([[0., .5, 0., 0., .4, 0., 0.]])
        scene.xi = torch.zeros(1, 16)
        scene.a_prev = torch.zeros(1, 12)
        scene.ep_return = torch.zeros(1)
        scene.ep_tracking = torch.zeros(1)
        scene.ep_step = torch.tensor([env.env.max_episode_steps - 1])
        scene.command_interval = torch.tensor([150])
        scene.command_age = torch.tensor([12])
        scene.levels = torch.zeros(1, dtype=torch.long)
        scene.joint_ids = list(range(12))
        position = torch.tensor([[0., 0., .4]])

        def integrate():
            position[0, 0] += .0025

        def reset(ids):
            position[ids, 0] = -1.

        scene.state = lambda: (position.clone(), torch.eye(3)[None],
                               torch.tensor([[.5, 0., 0.]]), torch.zeros(1, 3),
                               scene.default_q[None], torch.zeros(1, 12))
        scene.sim = SimpleNamespace(step=integrate)
        scene.scene = SimpleNamespace(write_data_to_sim=lambda: None, update=lambda dt: None)
        scene.robot = SimpleNamespace(set_joint_position_target=lambda *a, **k: None)
        scene.reset = reset
        scene.sample_commands = lambda ids: None

        transition = scene.step(torch.zeros(1, 12))
        self.assertTrue(transition["truncated"].item())
        self.assertFalse(transition["terminated"].item())
        self.assertAlmostEqual(transition["s1"][0, 0].item(), .01)
        self.assertEqual(position[0, 0].item(), -1.)
        replay = ReplayBuffer(4, 1, "cpu", 12, scenario_dim=7, command_timing=True)
        replay.add(transition)
        self.assertEqual(replay.size, 1)

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
