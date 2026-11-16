"""Differentiable residual dynamics for the controlled WBLWBL flat-ground task."""

import torch
from torch import nn


CONTEXT_FIELDS = ("friction", "restitution", "kp_factor", "kd_factor")


def quaternion_product(left, right):
    a, b = left[..., :3], right[..., :3]
    w, z = left[..., 3:], right[..., 3:]
    return torch.cat((w * b + z * a + torch.cross(a, b, dim=-1), w * z - (a * b).sum(-1, keepdim=True)), -1)


def canonical_quaternion(q):
    q = torch.nn.functional.normalize(q, dim=-1)
    return torch.where(q[..., 3:] < 0, -q, q)


def rotation_difference(q0, q1):
    inverse = torch.cat((-q0[..., :3], q0[..., 3:]), -1)
    relative = canonical_quaternion(quaternion_product(inverse, q1))
    axis = relative[..., :3]
    length = axis.norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(length, relative[..., 3:])
    scale = torch.where(length < 1e-6, 2.0, angle / length.clamp_min(1e-6))
    return axis * scale


def state_difference(state, target):
    """Euclidean deltas and shortest-path local rotation, 36 channels."""
    return torch.cat((
        target[..., :3] - state[..., :3],
        rotation_difference(state[..., 3:7], target[..., 3:7]),
        target[..., 7:] - state[..., 7:],
    ), -1)


def compose_state(state, delta):
    rotation = delta[..., 3:6]
    half = rotation.norm(dim=-1, keepdim=True) * 0.5
    increment = torch.cat((0.5 * torch.sinc(half / torch.pi) * rotation, half.cos()), -1)
    quat = canonical_quaternion(quaternion_product(state[..., 3:7], increment))
    return torch.cat((state[..., :3] + delta[..., :3], quat, state[..., 7:] + delta[..., 6:]), -1)


def state_features(state, action, prev_action, context):
    # On an infinite plane, x/y translation cannot change the next increment.
    return torch.cat((state[..., 2:3], canonical_quaternion(state[..., 3:7]), state[..., 7:],
                      action, prev_action, context), -1)


class WorldModel(nn.Module):
    def __init__(self, hidden=(512, 512, 512)):
        super().__init__()
        dims = (35 + 24 + len(CONTEXT_FIELDS), *hidden, 36)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        for name, size in (("input", dims[0]), ("delta", dims[-1])):
            self.register_buffer(f"{name}_mean", torch.zeros(size))
            self.register_buffer(f"{name}_std", torch.ones(size))

    @torch.no_grad()
    def fit_normalization(self, state, next_state, action, prev_action, context):
        features = state_features(state, action, prev_action, context)
        delta = state_difference(state, next_state)
        for prefix, values in (("input", features), ("delta", delta)):
            values = values.flatten(end_dim=-2)
            getattr(self, f"{prefix}_mean").copy_(values.mean(0))
            floor = 1e-3 if prefix == "input" else 1e-4
            getattr(self, f"{prefix}_std").copy_(values.std(0, unbiased=False).clamp_min(floor))

    def forward(self, state, action, prev_action, context):
        features = state_features(state, action, prev_action, context)
        delta = self.net((features - self.input_mean) / self.input_std)
        return compose_state(state, delta * self.delta_std + self.delta_mean)

    def rollout(self, initial, actions, previous_actions, context):
        """Unroll from one observed seed using the logged actions, without teacher forcing."""
        state = initial
        predictions = []
        for action, previous, parameters in zip(actions.unbind(1), previous_actions.unbind(1),
                                                 context.unbind(1), strict=True):
            state = self(state, action, previous, parameters)
            predictions.append(state)
        return torch.stack(predictions, dim=1)
