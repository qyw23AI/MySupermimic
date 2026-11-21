"""Condition the learned dynamics on the joint targets backed by PhysX."""

import torch

from trackerLab.algorithms.supermimic.terrain_networks import TerrainWorldModel


def clamp_joint_targets(action, nominal, scale, lower, upper):
    return (nominal + action * scale).clamp(lower, upper)


class JointTargetWorldModel(TerrainWorldModel):
    """Keep actor history in policy coordinates and project only dynamics inputs."""

    def __init__(self, field, robot, control, action_limit, **kwargs):
        super().__init__(field, len(robot.joints), **kwargs)
        self.action_limit = action_limit
        self.action_scale = control.action_scale
        self.register_buffer("default_angles", torch.tensor(robot.default_q), persistent=False)
        lower, upper = torch.tensor(robot.limits).T.contiguous()
        self.register_buffer("lower", lower, persistent=False)
        self.register_buffer("upper", upper, persistent=False)

    def inputs(self, state, action, context):
        # Invert the joint offset slice of TrackerLab's normalize_xi.
        nominal = self.default_angles + .01 * context[..., 1:1 + self.num_joints]
        target = clamp_joint_targets(action.clamp(-self.action_limit, self.action_limit),
                                     nominal, self.action_scale, self.lower, self.upper)
        issued_action = (target - nominal) / self.action_scale
        return super().inputs(state, issued_action, context)
