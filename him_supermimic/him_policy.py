"""HIM policy with the legacy weights and differentiable history input."""

import torch
from torch import nn


class DifferentiableHIM(nn.Module):
    def __init__(self, legacy):
        super().__init__()
        self.actor = legacy.actor
        self.estimator = legacy.estimator
        self.frame_width = legacy.num_one_step_obs

    def forward(self, history):
        encoded = self.estimator.encoder(history)
        velocity = encoded[..., :3]
        latent = torch.nn.functional.normalize(encoded[..., 3:], dim=-1, p=2)
        return self.actor(torch.cat((history[..., :self.frame_width], velocity, latent), dim=-1))
