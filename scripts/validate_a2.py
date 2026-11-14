#!/usr/bin/env python3
"""Compare the Torch bridge to original HIMLoco labels in a flat A1 replay."""

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from him_supermimic.flat_task import CORE_TERMS, FlatTask


OBS_ATOL = 1e-5
REWARD_ATOL = 1e-4
REWARD_RTOL = 1e-3


def compare(actual, expected, atol, *, rtol=0.0):
    if actual.numel() == 0:
        return None
    error = (actual - expected).abs()
    if not bool((error <= atol + rtol * expected.abs()).all()):
        raise ValueError(f"Mismatch, max absolute error = {error.max().item():.8g}")
    return error.max().item()


def observation_errors(task, actor, critic, rows, usable):
    running = usable & ~(rows["terminated"] | rows["truncated"])
    actor_label = rows["actor_history_after_step"][..., :45]
    critic_label = rows["next_critic_obs_pre_reset"]
    obs_limit = task.normalization["clip_observations"]
    actor = actor.clamp(-obs_limit, obs_limit)
    # Legacy terminal critic observations are captured without _get_observations' clipping.
    critic = torch.where(running.unsqueeze(-1), critic.clamp(-obs_limit, obs_limit), critic)
    frame_noise, height_noise = task.observation_noise_bounds(actor)
    bounds = torch.cat((frame_noise, frame_noise.new_zeros(6), height_noise))
    blocks = {
        "command": slice(0, 3), "angular_velocity": slice(3, 6), "gravity": slice(6, 9),
        "joint_position": slice(9, 21), "joint_velocity": slice(21, 33), "action": slice(33, 45),
        "linear_velocity": slice(45, 48), "disturbance": slice(48, 51), "height_scan": slice(51, None),
    }
    report = {}
    for name, block in blocks.items():
        report[name] = {
            "critic_max_abs": compare(critic[usable][..., block], critic_label[usable][..., block],
                                      bounds[block] + OBS_ATOL),
            "noise_bound": bounds[block].max().item(),
        }
        if name not in ("linear_velocity", "disturbance", "height_scan"):
            report[name]["actor_max_abs"] = compare(
                actor[running][..., block], actor_label[running][..., block], bounds[block] + OBS_ATOL)
    return report


def gradient_check(task, rows, usable):
    mask = usable & ~(rows["command_resampled"] | rows["terminated"] | rows["truncated"])
    state = rows["state"][mask][:16].double()
    step = rows["episode_step"][mask][:16]
    command = rows["command"][mask][:16].double()
    action = rows["raw_action"][mask][:16].double().requires_grad_()
    next_state = rows["next_state_pre_reset"][mask][:16].double().requires_grad_()
    previous = rows["prev_action"][mask][:16].double()
    older = rows["prev_prev_action"][mask][:16].double()
    history = rows["actor_history"][mask][:16].double()
    if not len(step):
        raise ValueError("Replay needs an ordinary row to check gradients")

    def objective(prediction, raw_action):
        executed = task.clip_action(raw_action)
        goal = task.update_command(command, prediction, step, command)
        obs = task.update_history(history, task.actor_frame(prediction, goal, executed))
        reward = sum(task.reward_terms(state, prediction, goal, executed, previous, older).values())
        # Deliberately includes observations so all three paths exercise autograd.
        return reward.mean() + 0.1 * obs[..., :45].square().mean()

    gradients = torch.autograd.grad(objective(next_state, action), (next_state, action))
    state_grad, action_grad = gradients
    for grad in gradients:
        if not torch.isfinite(grad).all() or grad.norm() == 0:
            raise ValueError("Expected finite, nonzero task gradients")
    # Deterministic directions, avoiding discontinuous sampled-command changes.
    state_direction = state_grad / state_grad.norm()
    action_direction = action_grad / action_grad.norm()
    step_size = 1e-6
    center = (next_state.detach(), action.detach())
    plus = objective(center[0] + step_size * state_direction, center[1] + step_size * action_direction)
    minus = objective(center[0] - step_size * state_direction, center[1] - step_size * action_direction)
    difference = (plus - minus) / (2 * step_size)
    derivative = (state_grad * state_direction).sum() + (action_grad * action_direction).sum()
    return {
        "state_norm": state_grad.norm().item(),
        "action_norm": action_grad.norm().item(),
        "directional_derivative_error": compare(derivative, difference, 1e-7, rtol=1e-6),
    }


def validate(replay_path, config_path, joint_names):
    replay = torch.load(replay_path, map_location="cpu", weights_only=True)
    if replay["schema_version"] != 1:
        raise ValueError(f"Unsupported A1 schema {replay['schema_version']}")
    config = json.loads(config_path.read_text())
    if (config["task"], config["stage"]) != ("wblwbl_dog_v1", 3):
        raise ValueError("A2 expects WBLWBL Dog V1 Stage 3 replay")
    rows = replay["transitions"]
    task = FlatTask(config["legacy"], joint_names)
    # A1 logs the post-push velocity. Legacy reward/observation uses pre-push cached body velocity.
    usable = ~rows["push_applied"]
    if not usable.any():
        raise ValueError("No rows with matching legacy reward-state timing")
    sampled = rows["command_pre_reset"]
    next_state = rows["next_state_pre_reset"]
    command = task.update_command(rows["command"], next_state, rows["episode_step"], sampled)
    torch.testing.assert_close(task.command_resampled(rows["episode_step"]), rows["command_resampled"])
    execution = task.clip_action(rows["raw_action"])
    frame = task.actor_frame(next_state, command, execution)
    critic = task.critic_frame(next_state, frame, rows["pending_force_next_body"])
    # Actor history always prepends a frame. For reset rows, the new frame belongs to the new episode.
    history = task.update_history(rows["actor_history"], frame)
    terms = task.reward_terms(rows["state"], next_state, command, execution,
                              rows["prev_action"], rows["prev_prev_action"],
                              terminal_torque=rows["terminal_torque"])
    labels = dict(zip(replay["reward_term_names"], rows["reward_terms"].unbind(dim=-1), strict=True))
    if missing := set(CORE_TERMS) - labels.keys():
        raise ValueError(f"Missing core reward labels: {sorted(missing)}")
    errors = {}
    for name in labels.keys() & terms.keys():
        try:
            errors[name] = compare(terms[name][usable], labels[name][usable], REWARD_ATOL, rtol=REWARD_RTOL)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc
    total = sum(terms[name] for name in labels) if labels.keys() <= terms.keys() else None
    return {
        "status": "passed", "profile": config["profile"], "replay": str(replay_path),
        "rows": usable.numel(), "push_rows_excluded": int((~usable).sum()),
        "terminations": int(rows["terminated"].sum()), "timeouts": int(rows["truncated"].sum()),
        "command_resamples": int(rows["command_resampled"].sum()),
        "command_max_abs": compare(command, rows["command_pre_reset"], OBS_ATOL),
        "action_max_abs": compare(execution, rows["executed_action"], 0),
        "history_shift_max_abs": compare(history[..., 45:], rows["actor_history_after_step"][..., 45:], 0),
        "observation": observation_errors(task, frame, critic, rows, usable),
        "weighted_reward_term_max_abs": dict(sorted(errors.items())),
        "total_reward_max_abs": None if total is None else compare(total[usable], rows["reward"][usable],
                                                                  REWARD_ATOL, rtol=REWARD_RTOL),
        "reward_terms_requiring_additional_inputs": sorted(labels.keys() - terms.keys()),
        "gradient": gradient_check(task, rows, usable),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.replay.resolve().parent
    source = json.loads((directory / "source.json").read_text())
    report = validate(args.replay.resolve(), directory / "config.json", source["asset"]["joint_names"])
    output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
