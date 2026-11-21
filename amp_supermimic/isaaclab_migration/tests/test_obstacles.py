"""Validate competition geometry and obstacle-specific contact outcomes on CPU."""

from types import SimpleNamespace
import unittest

import torch

from amp_supermimic.isaaclab_migration.config import resolve
from amp_supermimic.isaaclab_migration.wmp_terrain import CourseAtlas, CourseEvents


class ObstacleContracts(unittest.TestCase):
    def setup_course(self, kind, level):
        cfg, _ = resolve(SimpleNamespace(task=f"wblwbl_dog_v1_physx_{kind}",
                                         competition_stage="eval", level=level))
        atlas = CourseAtlas(cfg.terrain, "cpu")
        events = CourseEvents(atlas, 1, "cpu", .02)
        return cfg, atlas, events

    def test_competition_geometry_matches_height_query(self):
        for kind, level, size in (("bridge_trot", 9, (3., .40, .10)),
                                  ("high_wall", 6, (.08, .80, .30))):
            with self.subTest(kind=kind):
                _, atlas, _ = self.setup_course(kind, level)
                center = atlas.center(level, 0)
                torch.testing.assert_close(torch.tensor(atlas.cells[level, 0].boxes[0][2:]),
                                           torch.tensor(size))
                x, y, _ = center
                points = torch.tensor([[x, y], [x + size[0] / 2 + .01, y],
                                       [x, y + size[1] / 2 + .01]])
                torch.testing.assert_close(atlas.sample(points), torch.tensor([size[2], 0., 0.]))

    def test_bridge_requires_both_pairs_and_rejects_off_bridge_floor_support(self):
        cfg, atlas, events = self.setup_course("bridge_trot", 9)
        x, y, _ = atlas.center(9, 0)
        root = torch.tensor([[x, y, .46]])
        sole = torch.tensor([[[x-.19, y-.14, .10], [x-.19, y+.14, .10],
                              [x+.19, y-.14, .10], [x+.19, y+.14, .10]]])
        forces = torch.zeros_like(sole)
        events.course.entered[:] = True
        events.course.success[:] = True
        for step in range(36):
            events.course.strict[:] = True  # Candidate success from WMP's crossing check.
            forces.zero_()
            pair = [0, 3] if (step // 4) % 2 == 0 else [1, 2]
            forces[:, pair, 2] = 20.
            events.obstacle_rewards(root, sole, forces, cfg.rewards)
            if step == 12:
                self.assertFalse(events.course.strict.item())
        self.assertTrue(events.course.strict.item())
        sole[:, :2, 2] = 0.
        forces[:, :2, 2] = 20.
        events.obstacle_rewards(root, sole, forces, cfg.rewards)
        self.assertTrue(events.course.failed.item())
        self.assertFalse(events.course.success.item())

    def test_wall_requires_sustained_front_foot_wall_top_contact(self):
        cfg, atlas, events = self.setup_course("high_wall", 6)
        x, y, _ = atlas.center(6, 0)
        root = torch.tensor([[x-.14, y, .46]])
        sole = torch.tensor([[[x-.3, y-.14, 0.], [x-.3, y+.14, 0.],
                              [x-.05, y-.14, .30], [x+.05, y+.14, .30]]])
        forces = torch.zeros_like(sole)
        forces[:, 2, 2] = 20.
        events.course.strict[:] = True
        events.obstacle_rewards(root, sole, forces, cfg.rewards)
        self.assertFalse(events.course.strict.item())  # Upward force outside the top.
        sole[:, 2, 0] = x
        for frame in range(4):
            events.course.strict[:] = True
            reward = events.obstacle_rewards(root, sole, forces, cfg.rewards)
            self.assertEqual(reward["wall_front_support"].item(), float(frame == 3))
            self.assertEqual(events.course.strict.item(), frame == 3)
