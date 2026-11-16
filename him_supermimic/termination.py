"""Soft continuation models over the A1 state layout."""
import torch
from torch import nn

from .flat_task import projected_gravity


def heuristic_survival(states):
    return torch.sigmoid((states[..., 2] - 0.16) / 0.015) * torch.sigmoid(
        (-projected_gravity(states)[..., 2] - 0.5) / 0.08)


class ContactSurvival(nn.Module):
    """Predict a contact terminal from pose and velocities without simulator-only inputs."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(34, 64), nn.SiLU(), nn.Linear(64, 1))
        self.register_buffer("mean", torch.zeros(34))
        self.register_buffer("std", torch.ones(34))
        self.register_buffer("logit_offset", torch.zeros(()))

    @staticmethod
    def features(states):
        return torch.cat((states[..., 2:3], projected_gravity(states), states[..., 7:]), dim=-1)

    def logits(self, states):
        return self.net((self.features(states) - self.mean) / self.std).squeeze(-1) + self.logit_offset

    def forward(self, states):
        return torch.sigmoid(-self.logits(states))
