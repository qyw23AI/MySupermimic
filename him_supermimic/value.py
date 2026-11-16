"""History-conditioned Critic and fixed TD(lambda) targets."""

import torch
from torch import nn

from .dynamics import canonical_quaternion


class Critic(nn.Module):
    def __init__(self, width=315):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 256), nn.ELU(), nn.Linear(256, 128),
                                 nn.ELU(), nn.Linear(128, 1))
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("std", torch.ones(width))

    @staticmethod
    def features(state, history, command, context):
        # Keep 37 state channels, but remove absolute x/y and quaternion sign ambiguity.
        state = torch.cat((torch.zeros_like(state[..., :2]), state[..., 2:3],
                           canonical_quaternion(state[..., 3:7]), state[..., 7:]), dim=-1)
        return torch.cat((state, history, command, context), dim=-1)

    @torch.no_grad()
    def fit_input_stats(self, features):
        self.mean.copy_(features.mean(dim=0))
        self.std.copy_(features.std(dim=0, unbiased=False).clamp_min(1e-3))

    def forward(self, features):
        return self.net((features - self.mean) / self.std).squeeze(-1)


class HorizonCritic(nn.Module):
    """Critic conditioned on the fraction of the return horizon remaining."""

    def __init__(self):
        super().__init__()
        self.net = Critic(316)
        self.register_buffer("rmean", torch.tensor(0.5))
        self.register_buffer("rstd", torch.tensor(0.3))

    def forward(self, features, remaining_fraction):
        remaining = (remaining_fraction - self.rmean) / self.rstd
        return self.net(torch.cat((features, remaining.unsqueeze(-1)), dim=-1))


@torch.no_grad()
def td_lambda_targets(rewards, values, terminated, gamma=0.99, lam=0.95, *, survival=None):
    """Values has one more time step than rewards. Only physical termination cuts bootstrap."""
    target = values[:, -1]
    result = []
    for t in reversed(range(rewards.shape[1])):
        continuation = (~terminated[:, t]).to(rewards.dtype)
        if survival is not None:
            continuation = continuation * survival[:, t]
        target = rewards[:, t] + gamma * continuation * ((1 - lam) * values[:, t + 1] + lam * target)
        result.append(target)
    return torch.stack(result[::-1], dim=1)
