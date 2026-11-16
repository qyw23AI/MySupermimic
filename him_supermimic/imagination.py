"""Roll out the frozen or trainable HIM Actor through learned flat-ground dynamics."""

from typing import NamedTuple

import torch

from .value import Critic


class ImaginedTrajectory(NamedTuple):
    features: torch.Tensor
    rewards: torch.Tensor
    tracking_rewards: torch.Tensor
    actions: torch.Tensor
    states: torch.Tensor


def imagine(policy, model, task, seed, horizon, *, observation_noise=None):
    """Start at recorded history. Caller supplies windows before command resampling."""
    state, history, command, context, previous, older, episode_step = seed
    features, rewards, tracking_rewards, actions, states = [], [], [], [], [state]
    for t in range(horizon):
        features.append(Critic.features(state, history, command, context))
        action = task.clip_action(policy(history))
        next_state = model(state, action, previous, context)
        command = task.update_command(command, next_state, episode_step + t, command)
        terms = task.reward_terms(state, next_state, command, action, previous, older)
        rewards.append(sum(terms.values()))
        tracking_rewards.append(terms["tracking_lin_vel"] + terms["tracking_ang_vel"])
        actions.append(action)
        noise = None if observation_noise is None else observation_noise[:, t]
        history = task.update_history(history, task.actor_frame(next_state, command, action, noise))
        state, older, previous = next_state, previous, action
        states.append(state)
    features.append(Critic.features(state, history, command, context))
    return ImaginedTrajectory(*(torch.stack(items, dim=1)
                                for items in (features, rewards, tracking_rewards, actions, states)))
