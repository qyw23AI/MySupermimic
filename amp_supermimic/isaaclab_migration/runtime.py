"""Connect TrackerLab command training to the PhysX implementation."""

from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

from amp_supermimic.legacy import ROOT


def configure_trackerlab():
    path = str(ROOT / "third_party/trackerlab_supermimic/scripts/rls/supermimic")
    if path not in sys.path:
        sys.path.insert(0, path)
    from bootstrap import configure_paths
    configure_paths()


class Runtime:
    def __init__(self, task, env, robot, run, device=None):
        configure_trackerlab()
        from trackerLab.tracker_env.mdp.supermimic_command import CommandMDP
        from trackerTask.supermimic.robots import RobotSpec
        from trackerTask.supermimic import terrain
        from .wmp_terrain import CourseAtlas

        self.env, self.robot, self.run = env, robot, run
        self.command_mdp = CommandMDP
        from trackerLab.kinematics.supermimic.robot import RobotKinematics
        self.kinematics_class = RobotKinematics
        self.device = device or env.env.device
        self.atlas = (CourseAtlas(env.terrain, self.device)
                      if env.terrain.profile != "plane" else None)
        name = env.robot.name
        # A distinct profile keeps TrackerLab's velocity-only forward filter
        # from removing both turning directions from the AMP expert windows.
        self.task = terrain.TerrainTask(task, name, "physx_command")
        self.spec = RobotSpec(name, robot.joints, (robot.root, *robot.feet), robot.root,
                              robot.root, "", robot.default_q,
                              (env.control.action_scale,) * len(robot.joints))
        self.simulator = None
        self.terrain = terrain.generate("velocity", env.robot.standing_height, env.env.seed)
        if self.atlas is not None:
            self.terrain = replace(self.terrain, x_min=0., y_min=0.,
                                   x_max=self.atlas.rows * self.atlas.tile_size,
                                   y_max=self.atlas.cols * self.atlas.tile_size)

    def field(self, data, height, device):
        from trackerLab.tracker_env.mdp.supermimic_terrain import HeightField

        field = HeightField(data, height, device)
        if self.atlas is not None:
            field.sample = self.atlas.sample
        return field

    def make_mdp(self, field, default_q, qlo, qhi, **kwargs):
        scales = {"body_velocity": self.env.rewards.body_velocity,
                  "yaw_rate": self.env.rewards.yaw_rate}

        class PhysxCommandMDP(self.command_mdp):
            def reward(self, state, action, previous, command, terms=False):
                _, values = super().reward(state, action, previous, command, terms=True)
                for key, scale in scales.items():
                    values[key] = values[key] * scale
                total = sum(values.values())
                return (total, values) if terms else total

        # The task uses command tracking. WMP owns hard course completion.
        kwargs["profile"] = "velocity"
        return PhysxCommandMDP(field, default_q, qlo, qhi, **kwargs)

    def make_simulator(self, task, mdp, *args, **kwargs):
        from .backend import PhysxCommandSim

        self.simulator = PhysxCommandSim(self.env, self.robot, self.spec, mdp, self.atlas, self.run)
        return self.simulator

    def create(self, gamma):
        import torch

        field = self.field(self.terrain, self.env.robot.standing_height, self.device)
        lower, upper = torch.tensor(self.robot.limits, device=self.device).T
        mdp = self.make_mdp(field, torch.tensor(self.spec.default_q, device=self.device), lower, upper,
                            gamma=gamma, action_limit=self.env.env.action_limit,
                            clearance_ratio=self.env.env.clearance_ratio)
        return self.make_simulator(self.task, mdp)

    def _joint_limit_adapter(self, *args, **kwargs):
        """Provide the only model fields read by TrainerLab's model setup."""
        from numpy import asarray

        ids = {name: SimpleNamespace(id=index) for index, name in enumerate(self.robot.joints)}
        limits = SimpleNamespace(joint=ids.__getitem__, jnt_range=asarray(self.robot.limits))
        contract = {"motors_effort_velocity_kp_kd_armature": self.robot.motors}
        return (limits, None, None, None, None, None, None, None, contract)

    def train(self, arguments, receipt):
        import train_terrain
        from trackerTask.supermimic import gait, terrain
        from trackerLab.kinematics.supermimic import robot as kinematics
        from trackerLab.tracker_env.mdp import supermimic_command
        from trackerLab.tracker_env.supermimic import command

        name = self.spec.name
        height = self.env.robot.standing_height
        provenance = {"kind": "wblwbl-physx-local-csv-audit", "splits": receipt}
        with ExitStack() as stack:
            # Keep the original optimization and replay code. The scoped hooks
            # cover only task/asset initialization and the real rollout boundary.
            for module, key, value in (
                (train_terrain, "get_task", lambda name: self.task),
                (train_terrain, "robot_parameters", lambda name: (self.spec, height, self.robot.motors)),
                (train_terrain, "HeightField", self.field),
                (train_terrain, "build_model", self._joint_limit_adapter),
                (train_terrain, "validate_style_references", lambda *a, **kw: provenance),
                (terrain, "generate", lambda *a, **kw: self.terrain),
                (supermimic_command, "CommandMDP", self.make_mdp),
                (command, "CommandSim", self.make_simulator),
                (kinematics, "RobotKinematics",
                 lambda device, spec: self.robot.kinematics(device, spec, base=self.kinematics_class)),
                (sys, "argv", [str(Path(train_terrain.__file__)), *arguments]),
            ):
                stack.enter_context(patch.object(module, key, value))
            stack.enter_context(patch.dict(terrain.ROBOT_HEIGHTS, {name: height}))
            stack.enter_context(patch.dict(gait.FOOT_BODIES, {name: self.robot.feet}))
            stack.enter_context(patch.dict(gait.FOOT_OFFSETS, {name: self.robot.foot_centers}))
            train_terrain.main()
