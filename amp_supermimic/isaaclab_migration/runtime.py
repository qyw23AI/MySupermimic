"""Connect TrackerLab command training to the PhysX implementation."""

from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

from amp_supermimic.legacy import ROOT, class_to_dict
from amp_supermimic.scripts.open_terrain import open_flat_reward


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
        from trackerLab.tracker_env.supermimic.command import CommandSim
        from trackerTask.supermimic.robots import RobotSpec
        from trackerTask.supermimic import terrain
        from .wmp_terrain import CourseAtlas

        self.env, self.robot, self.run = env, robot, run
        self.command_mdp = CommandMDP
        self.reference_filter = CommandSim.set_reference_resets
        from trackerLab.kinematics.supermimic.robot import RobotKinematics
        self.kinematics_class = RobotKinematics
        self.device = device or env.env.device
        self.atlas = (CourseAtlas(env.terrain, self.device)
                      if env.terrain.profile not in ("plane", "open_flat", "parkour") else None)
        name = env.robot.name
        # A distinct profile keeps TrackerLab's velocity-only forward filter
        # from removing both turning directions from the AMP expert windows.
        profile = env.terrain.profile
        self.task = terrain.TerrainTask(task, name, profile if profile in ("open_flat", "parkour") else "physx_command")
        self.spec = RobotSpec(name, robot.joints, (robot.root, *robot.feet), robot.root,
                              robot.root, "", robot.default_q,
                              (env.control.action_scale,) * len(robot.joints))
        self.simulator = None
        self.terrain = terrain.generate("parkour" if profile == "parkour" else "velocity",
                                        env.robot.standing_height, env.env.seed)
        if self.atlas is not None:
            self.terrain = replace(self.terrain, x_min=0., y_min=0.,
                                   x_max=self.atlas.rows * self.atlas.tile_size,
                                   y_max=self.atlas.cols * self.atlas.tile_size)

    def field(self, data, height, device):
        from trackerLab.tracker_env.mdp.supermimic_terrain import HeightField

        field = HeightField(data, height, device)
        if self.env.terrain.profile == "open_flat":
            # Open-flat scan points outside the sampled patch still measure the
            # same z=0 ground plane; do not feed the OOB sentinel to the policy.
            field.flat_outside = True
        if self.atlas is not None:
            field.sample = self.atlas.sample
        return field

    def make_mdp(self, field, default_q, qlo, qhi, **kwargs):
        import torch

        rewards = class_to_dict(self.env.rewards)
        profile = self.env.terrain.profile
        open_flat = profile == "open_flat"
        obstacle_atlas = self.atlas
        obstacle_task = profile in ("narrow_bridge", "high_wall")
        if obstacle_task:
            tile_size = obstacle_atlas.tile_size
            boxes = default_q.new_tensor([obstacle_atlas.cells[row, 0].boxes[0]
                                         for row in range(obstacle_atlas.rows)])
            foot_kin = self.robot.kinematics(self.device, self.spec, tracked=self.robot.feet,
                                            base=self.kinematics_class)
            ball_offset = default_q.new_tensor(self.robot.foot_centers)
            radius = default_q.new_tensor(self.robot.foot_radii)

        class PhysxCommandMDP(self.command_mdp):
            def set_style_weight(self, weight):
                super().set_style_weight(weight)
                if open_flat:
                    self.max_step_reward = weight + sum(rewards[key] for key in (
                        "velocity_weight", "yaw_weight", "alive_weight",
                        "progress_trans_weight", "progress_yaw_weight"))

            def reward(self, state, action, previous, command, terms=False):
                if open_flat:
                    return open_flat_reward(self, state, action, previous, command, rewards, terms)
                _, values = super().reward(state, action, previous, command, terms=True)
                for key in ("body_velocity", "yaw_rate"):
                    values[key] = values[key] * rewards.get(key, 1.)
                if obstacle_task:
                    values.update(self.obstacle_terms(state, command))
                total = sum(values.values())
                return (total, values) if terms else total

            def termination(self, state, command):
                if not open_flat:
                    return super().termination(state, command)
                # PhysX open-flat uses an infinite GroundPlane. The finite
                # HeightField bounds only limit sampled scan data, so leaving
                # that rectangle is not a physical fall.
                upright = state[1][..., 2, 2]
                clearance = self._clearance(state)
                return ((clearance < self._minimum_clearance(command))
                        | (upright < .35) | ~self._finite(state, command))

            def obstacle_terms(self, state, command):
                pos, rot, vel, _, _, _ = state
                lane_err = (pos[..., 1] - command[..., 0]) / .18
                moving = (vel[..., 0] / .25).clamp(0., 1.)
                terms = {"obstacle_lane": rewards["lane_centering"] * moving
                         * torch.exp(-lane_err.square()) * rot[..., 0, 0].clamp(0., 1.)}
                row = (pos[..., 0] / tile_size).floor().long().clamp(0, len(boxes) - 1)
                distance = pos[..., 0] - (row + .5) * tile_size
                half_length, half_width, top = (boxes[row, 2] * .5, boxes[row, 3] * .5, boxes[row, 4])
                feet, orientation, foot_v, foot_w = foot_kin.forward(*state)
                offset = (orientation @ ball_offset[..., None]).squeeze(-1)
                feet = feet + offset
                side = (feet[..., 1] - command[..., 0, None]).abs()
                if profile == "narrow_bridge":
                    near_deck = torch.sigmoid((half_length + .20 - distance.abs()) / .08)
                    overhang = (side + radius + .015 - half_width[..., None]).clamp(min=0.)
                    terms["bridge_foot_edges"] = -rewards["foot_edges"] * near_deck * (
                        (overhang / .10).square().mean(-1).clamp(max=4.))
                else:
                    near = torch.exp(-((distance + .15) / .28).square())
                    rise = ((pos[..., 2] - field.height) / top).clamp(0., 1.)
                    terms["wall_clearance"] = rewards["wall_clearance"] * near * rise * moving
                    fx = feet[..., 2:, 0] - (row[..., None] + .5) * tile_size
                    fz = feet[..., 2:, 2] - radius[2:] - top[..., None]
                    # Smooth proxy for a front foot settled on the top face.
                    # The simulator measures sustained support separately.
                    toe_v = foot_v + torch.cross(foot_w, offset, dim=-1)
                    front_v = toe_v[..., 2:, :].square().sum(-1)
                    alignment = (fx / .060).square() + (fz / .070).square()
                    inside = torch.sigmoid((half_width[..., None] - side[..., 2:]) / .04)
                    support = (torch.exp(-alignment - front_v / .35) * inside).amax(-1)
                    approach = torch.exp(-((distance + .22) / .23).square())
                    terms["wall_foot_placement"] = rewards["wall_foot_placement"] * approach * support
                return terms

        # The task uses command tracking. WMP owns hard course completion.
        kwargs["profile"] = self.terrain.profile
        return PhysxCommandMDP(field, default_q, qlo, qhi, **kwargs)

    def make_simulator(self, task, mdp, *args, **kwargs):
        from .backend import PhysxCommandSim

        self.simulator = PhysxCommandSim(self.env, self.robot, self.spec, mdp, self.atlas, self.run)
        self.simulator.reference_filter = self.reference_filter
        return self.simulator

    def create(self, gamma):
        import torch

        field = self.field(self.terrain, self.env.robot.standing_height, self.device)
        lower, upper = torch.tensor(self.robot.limits, device=self.device).T
        mdp = self.make_mdp(field, torch.tensor(self.spec.default_q, device=self.device), lower, upper,
                            gamma=gamma, action_limit=self.env.env.action_limit,
                            clearance_ratio=self.env.env.clearance_ratio)
        return self.make_simulator(self.task, mdp)

    def make_world_model(self, field, num_joints, **kwargs):
        if not getattr(self.env.control, "project_joint_targets", False):
            from trackerLab.algorithms.supermimic.terrain_networks import TerrainWorldModel
            return TerrainWorldModel(field, num_joints, **kwargs)
        from .world_model import JointTargetWorldModel
        return JointTargetWorldModel(field, self.robot, self.env.control,
                                     self.env.env.action_limit, **kwargs)

    def _joint_limit_adapter(self, *args, **kwargs):
        """Provide the only model fields read by TrainerLab's model setup."""
        from numpy import asarray

        ids = {name: SimpleNamespace(id=index) for index, name in enumerate(self.robot.joints)}
        limits = SimpleNamespace(joint=ids.__getitem__, jnt_range=asarray(self.robot.limits))
        contract = {"motors_effort_velocity_kp_kd_armature": self.robot.motors}
        return (limits, None, None, None, None, None, None, None, contract)

    def reference_floor_distances(self, values, name):
        """Measure the minimum distance from the imported URDF spheres to flat ground."""
        import torch
        from trackerLab.kinematics.supermimic.rotations import quat_to_mat

        state = torch.as_tensor(values, device=self.device, dtype=torch.float32)
        p, q = state[:, :3], state[:, 7:]
        rotation = quat_to_mat(state[:, [6, 3, 4, 5]])
        zero = torch.zeros_like(p)
        kin = self.robot.kinematics(self.device, self.spec, tracked=self.robot.feet, base=self.kinematics_class)
        feet, frames, *_ = kin.forward(p, rotation, zero, zero, q, torch.zeros_like(q))
        centers = state.new_tensor(self.robot.foot_centers)
        z = (feet + (frames @ centers[..., None]).squeeze(-1))[..., 2]
        return (z - state.new_tensor(self.robot.foot_radii)).amin(-1).cpu().numpy()

    def train(self, arguments, receipt):
        import train_terrain
        import prepare_style_references
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
                (train_terrain, "TerrainWorldModel", self.make_world_model),
                (train_terrain, "build_model", self._joint_limit_adapter),
                (train_terrain, "validate_style_references", lambda *a, **kw: provenance),
                (prepare_style_references, "physical_floor_distances", self.reference_floor_distances),
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
