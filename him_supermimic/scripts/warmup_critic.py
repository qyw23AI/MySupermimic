#!/usr/bin/env python3
"""Verify the HIM gradient path and warm a Critic on replay-seeded imagination."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party/MyHIMLoco/rsl_rl")]
from him_supermimic.dynamics import CONTEXT_FIELDS, WorldModel
from him_supermimic.flat_task import FlatTask
from him_supermimic.him_policy import DifferentiableHIM
from him_supermimic.imagination import imagine
from him_supermimic.replay import EpisodeReplay
from him_supermimic.value import Critic, td_lambda_targets


HORIZON = 32
TD_HORIZON = 8
SEED_FIELDS = ("state", "actor_history", "command", "context", "prev_action", "prev_prev_action", "episode_step")
GAMMA = 0.99
LAMBDA = 0.95


class SeedReplay:
    def __init__(self, path, device):
        self.path = path.resolve(strict=True)
        self.config = json.loads((self.path.parent / "config.json").read_text())
        self.source = json.loads((self.path.parent / "source.json").read_text())
        if (self.config["task"], self.config["stage"], self.config["profile"]) != (
            "wblwbl_dog_v1", 3, "core_reward_v1"
        ):
            raise ValueError(f"{path}: expected A3 controlled flat-ground replay")
        data = torch.load(self.path, map_location="cpu", weights_only=True)
        if data["schema_version"] != 1:
            raise ValueError("Expected A1 schema 1")
        rows = data["transitions"]
        fields = set(SEED_FIELDS) - {"context"} | {
            "episode_id", "command_resampled", "terminated", "truncated", "reward",
        }
        self.rows = {k: rows[k].to(device) for k in fields}
        self.rows["context"] = torch.cat([rows[k] for k in CONTEXT_FIELDS], dim=-1).to(device)
        replay = EpisodeReplay(*rows["episode_id"].shape)
        replay.data = self.rows
        replay.total_steps = replay.capacity
        self.starts = replay.valid_starts(HORIZON)
        if not len(self.starts):
            raise ValueError(f"{path}: no valid {HORIZON}-step windows")
        if not all(torch.isfinite(x).all() for x in self.rows.values()):
            raise ValueError(f"{path}: nonfinite replay")

    def seeds(self, indices):
        t, env = self.starts[indices].unbind(-1)
        return tuple(self.rows[k][t, env] for k in SEED_FIELDS)

    def physical_returns(self, indices):
        start, env = self.starts[indices].unbind(-1)
        time = start[:, None] + torch.arange(HORIZON, device=start.device)
        return discounted(self.rows["reward"][time, env[:, None]])


def discounted(rewards):
    return (rewards * GAMMA ** torch.arange(rewards.shape[1], device=rewards.device)).sum(dim=1)


def load_policy(config, checkpoint, device):
    from rsl_rl.modules.him_actor_critic import HIMActorCritic

    import rsl_rl.modules.him_actor_critic as source
    if not Path(source.__file__).resolve().is_relative_to(ROOT / "third_party/MyHIMLoco"):
        raise RuntimeError("Expected HIM from the pinned MyHIMLoco submodule")
    env = config["legacy"]["env"]
    legacy = HIMActorCritic(env["num_observations"], env["num_privileged_obs"],
                            env["num_one_step_observations"], env["num_actions"],
                            **config["training"]["policy"]).to(device)
    saved = torch.load(checkpoint, weights_only=True, map_location=device)
    legacy.load_state_dict(saved["model_state_dict"])
    return legacy.eval(), DifferentiableHIM(legacy).eval()


def positive_norm(gradient, name):
    norm = gradient.norm().item()
    if not torch.isfinite(gradient).all() or norm <= 0:
        raise ValueError(f"Invalid gradient for {name}")
    return norm


def verify(legacy, policy, model, task, data):
    policy.estimator.requires_grad_(False)
    model.requires_grad_(False)
    history = data.rows["actor_history"].reshape(-1, 270)[::13]
    with torch.no_grad():
        error = (legacy.act_inference(history) - policy(history)).abs().max().item()
    if error > 1e-6:
        raise ValueError(f"Legacy inference mismatch: {error}")
    # Isolate the older frames, which reach the Actor only via the encoder.
    leaf = history[:128].clone().requires_grad_()
    new_grad, = torch.autograd.grad(policy(leaf).square().mean(), leaf)
    old_grad, = torch.autograd.grad(legacy.act_inference(leaf).square().mean(), leaf)
    old_older = old_grad[:, policy.frame_width:].abs().max().item()
    if old_older != 0:
        raise ValueError("Expected the legacy estimator path to detach history")

    # Use a temporary Actor copy so the verified PPO checkpoint stays the rollout policy.
    probe = deepcopy(policy)
    seed = data.seeds(torch.arange(128, device=history.device) % len(data.starts))
    # Stack outputs are diagnostic views. Probe the first action by hooking the first model input.
    first = []

    def record_first_action(_module, inputs):
        if not first:
            first.append(inputs[1])

    with model.register_forward_pre_hook(record_first_action):
        trajectory = imagine(probe, model, task, seed, TD_HORIZON)
        objective = trajectory.tracking_rewards[:, -1].mean()
    first_norm = positive_norm(torch.autograd.grad(objective, first[0], retain_graph=True)[0],
                               "final tracking reward -> first action")
    frozen_before = [p.detach().clone() for p in (*probe.estimator.parameters(), *model.parameters())]
    actor_before = [p.detach().clone() for p in probe.actor.parameters()]
    optimizer = torch.optim.Adam(probe.actor.parameters(), lr=5e-5)
    optimizer.zero_grad(set_to_none=True)
    (-objective).backward()
    actor_grad = positive_norm(torch.cat([p.grad.flatten() for p in probe.actor.parameters()]), "Actor")
    optimizer.step()
    frozen = (*probe.estimator.parameters(), *model.parameters())
    if any(p.grad is not None or not torch.equal(p, before) for p, before in zip(frozen, frozen_before, strict=True)):
        raise ValueError("Frozen estimator or dynamics changed")
    delta = torch.cat([(p.detach() - before).flatten()
                       for p, before in zip(probe.actor.parameters(), actor_before, strict=True)])
    return {
        "parity_rows": len(history), "action_max_abs": error, "legacy_older_history_gradient_max_abs": old_older,
        "differentiable_older_history_gradient_norm": positive_norm(
            new_grad[:, policy.frame_width:], "older history"),
        "first_action_gradient_norm": first_norm, "actor_gradient_norm": actor_grad,
        "actor_update_norm": positive_norm(delta, "Actor update"),
        "frozen_estimator_and_world_model_unchanged": True,
    }


@torch.no_grad()
def collect_imagination(policy, model, task, dataset, count, rng, *, include_physics=False):
    indices = torch.randperm(len(dataset.starts), device=model.input_mean.device, generator=rng)[:count]
    features, rewards, physical = [], [], []
    for chunk in indices.split(256):
        seed = dataset.seeds(chunk)
        noise_bound, _ = task.observation_noise_bounds(seed[0])
        noise = (2 * torch.rand((len(chunk), HORIZON, 45), device=indices.device, generator=rng) - 1) * noise_bound
        trajectory = imagine(policy, model, task, seed, HORIZON, observation_noise=noise)
        features.append(trajectory.features)
        rewards.append(trajectory.rewards)
        if include_physics:
            physical.append(dataset.physical_returns(chunk))
    return torch.cat(features), torch.cat(rewards), torch.cat(physical) if physical else None


def fit(critic, optimizer, features, target, updates, rng):
    if not torch.isfinite(target).all():
        raise ValueError("Nonfinite Critic targets")
    for _ in range(updates):
        index = torch.randint(len(target), (512,), device=target.device, generator=rng)
        loss = (critic(features[index]) - target[index]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()


@torch.no_grad()
def metrics(prediction, target, constant):
    error = prediction - target
    if not torch.isfinite(error).all():
        raise ValueError("Nonfinite Critic predictions")
    variance = target.var(unbiased=False).item()
    return {"count": len(target), "rmse": error.square().mean().sqrt().item(),
            "constant_rmse": (constant - target).square().mean().sqrt().item(),
            "mean_error": error.mean().item(),
            "explained_variance": 1 - error.var(unbiased=False).item() / variance if variance else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("policy", "world-model", "train", "validation", "test", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--train-seeds", type=int, default=8192)
    parser.add_argument("--eval-seeds", type=int, default=3072)
    args = parser.parse_args()
    if min(args.train_seeds, args.eval_seeds) < 1:
        parser.error("seed counts must be positive")
    torch.manual_seed(args.seed)
    rng = torch.Generator(device=args.device).manual_seed(args.seed)
    train, val, test = [SeedReplay(p, args.device) for p in (args.train, args.validation, args.test)]
    if len({data.config["seed"] for data in (train, val, test)}) != 3:
        parser.error("Expected three replay collection seeds")
    if any(data.source["files"] != train.source["files"] or data.source["asset"] != train.source["asset"]
           for data in (val, test)):
        parser.error("Replay source contracts differ")
    joint_names = train.source["asset"]["joint_names"]
    task = FlatTask(train.config["legacy"], joint_names)
    wm_checkpoint = torch.load(args.world_model, weights_only=True, map_location=args.device)
    if (wm_checkpoint["joint_names"] != joint_names or tuple(wm_checkpoint["context_fields"]) != CONTEXT_FIELDS
        or wm_checkpoint["state_schema"] != 1):
        raise ValueError("World model metadata differs from the replay contract")
    model = WorldModel(wm_checkpoint["hidden"]).to(args.device).eval()
    model.load_state_dict(wm_checkpoint["model_state_dict"])
    legacy, policy = load_policy(train.config, args.policy, args.device)
    gradient_report = verify(legacy, policy, model, task, test)
    print(json.dumps(gradient_report), flush=True)
    policy.requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    train_f, train_r, _ = collect_imagination(policy, model, task, train, args.train_seeds, rng)
    val_f, val_r, _ = collect_imagination(policy, model, task, val, args.eval_seeds, rng)
    baseline = discounted(train_r).mean()
    critic = Critic().to(args.device)
    critic.fit_input_stats(train_f.flatten(0, 1))
    optimizer = torch.optim.Adam(critic.parameters(), lr=2e-4)
    # Predict a 32-step discounted return at each seed. Rolling TD uses
    # a shorter remaining-value estimate to avoid adding an extra 32-step tail.
    train_return, val_return = discounted(train_r), discounted(val_r)
    fit(critic, optimizer, train_f[:, 0], train_return, 1200, rng)
    frozen = deepcopy(critic).eval().requires_grad_(False)
    remaining = HORIZON - torch.arange(TD_HORIZON + 1, device=args.device)
    tail_scale = (1 - GAMMA ** remaining) / (1 - GAMMA ** HORIZON)

    @torch.no_grad()
    def targets(features, rewards):
        values = frozen(features[:, :TD_HORIZON + 1]) * tail_scale
        return td_lambda_targets(rewards[:, :TD_HORIZON], values,
                                 torch.zeros_like(rewards[:, :TD_HORIZON], dtype=torch.bool),
                                 GAMMA, LAMBDA)[:, 0]

    train_td, val_td = targets(train_f, train_r), targets(val_f, val_r)
    curve = []
    best = float("inf")

    def save_if_best(step):
        nonlocal best
        with torch.no_grad():
            value = critic(val_f[:, 0])
            evaluation = metrics(value, val_return, baseline)
            td_error = metrics(value, val_td, train_td.mean())
        curve.append({"td_updates": step, "return": evaluation, "td_lambda": td_error})
        print(json.dumps(curve[-1]), flush=True)
        if evaluation["rmse"] < best:
            best = evaluation["rmse"]
            torch.save({"model_state_dict": critic.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "input_width": 315, "td_updates": step, "gamma": GAMMA, "lambda": LAMBDA,
                        "return_horizon": HORIZON, "context_fields": CONTEXT_FIELDS,
                        "joint_names": joint_names, "state_schema": 1}, args.output / "critic.pt")

    save_if_best(0)
    # Keep the direct rollout target in the objective to bound bootstrap drift.
    for step in range(200, 1001, 200):
        fit(critic, optimizer, train_f[:, 0], 0.75 * train_return + 0.25 * train_td, 200, rng)
        save_if_best(step)
    selected = torch.load(args.output / "critic.pt", weights_only=True, map_location=args.device)
    critic.load_state_dict(selected["model_state_dict"])
    test_f, test_r, physical = collect_imagination(policy, model, task, test, args.eval_seeds, rng,
                                                   include_physics=True)
    with torch.no_grad():
        prediction = critic(test_f[:, 0])
        evaluation = {
            "imagined_td_lambda": metrics(prediction, targets(test_f, test_r), train_td.mean()),
            "imagined_32_step_return": metrics(prediction, discounted(test_r), baseline),
            "logged_physics_32_step_return": metrics(prediction, physical, baseline),
        }
    report = {
        "gradient": gradient_report, "validation": curve, "test": evaluation,
        "best_td_updates": selected["td_updates"], "horizon": HORIZON, "td_horizon": TD_HORIZON,
        "gamma": GAMMA, "lambda": LAMBDA, "seed": args.seed, "elapsed_seconds": time.monotonic() - start,
        "seed_counts": {"train": len(train_f), "validation": len(val_f), "test": len(test_f)},
        "replay": {name: {"path": str(split.path), "seed": split.config["seed"]}
                   for name, split in zip(("train", "validation", "test"), (train, val, test))},
        "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        "world_model_sha256": hashlib.sha256(args.world_model.read_bytes()).hexdigest(),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(evaluation, indent=2), flush=True)


if __name__ == "__main__":
    main()
