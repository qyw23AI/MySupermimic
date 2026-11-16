#!/usr/bin/env python3
"""Fit an 8-step autoregressive WBLWBL world model and evaluate independent trajectories."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from him_supermimic.dynamics import CONTEXT_FIELDS, WorldModel, rotation_difference, state_difference
from him_supermimic.flat_task import projected_gravity
from him_supermimic.replay import EpisodeReplay


HORIZONS = (1, 8, 16, 32)
FIELD_NAMES = ("position_m", "orientation_rad", "tilt_rad", "linear_velocity_m_s",
               "angular_velocity_rad_s", "joint_position_rad", "joint_velocity_rad_s")


class Dataset:
    def __init__(self, path, device):
        config = json.loads((path.parent / "config.json").read_text())
        if (config["task"], config["stage"], config["profile"]) != ("wblwbl_dog_v1", 3, "core_reward_v1"):
            raise ValueError(f"{path}: expected the controlled WBLWBL flat-ground profile")
        self.config = config
        self.source = json.loads((path.parent / "source.json").read_text())
        self.collection = json.loads((path.parent / "collection.json").read_text())
        bundle = torch.load(path, weights_only=True, map_location="cpu")
        if bundle["schema_version"] != 1:
            raise ValueError("Expected A1 replay schema 1")
        rows = bundle["transitions"]
        self.path = path
        self.steps, self.envs = rows["episode_id"].shape
        for name in ("push_applied", "applied_force_body", "pending_force_next_body"):
            if rows[name].any():
                raise ValueError(f"{path}: unexpected disturbance in controlled data ({name})")
        if not torch.allclose(rows["substep_actions"], rows["executed_action"].unsqueeze(2).expand_as(rows["substep_actions"])):
            raise ValueError(f"{path}: unexpected actuator delay")
        fields = ("state", "next_state_pre_reset", "executed_action", "prev_action",
                  "episode_id", "terminated", "truncated", "command_resampled")
        self.rows = {name: rows[name].to(device) for name in fields}
        self.rows["context"] = torch.cat([rows[name] for name in CONTEXT_FIELDS], -1).to(device)
        if any(not torch.isfinite(field).all() for field in self.rows.values()):
            raise ValueError("Nonfinite replay data")
        continuous = self.rows["episode_id"][1:] == self.rows["episode_id"][:-1]
        torch.testing.assert_close(self.rows["state"][1:][continuous],
                                   self.rows["next_state_pre_reset"][:-1][continuous], atol=1e-5, rtol=0)
        self.replay = EpisodeReplay(self.steps, self.envs)
        self.replay.data = self.rows
        self.replay.total_steps = self.steps

    def windows(self, starts, horizon):
        time_indices = starts[:, :1] + torch.arange(horizon, device=starts.device)
        def select(key):
            return self.rows[key][time_indices, starts[:, 1:2]]

        return (select("state")[:, 0], select("next_state_pre_reset"), select("executed_action"),
                select("prev_action"), select("context"))

    def description(self):
        return {"path": str(self.path), "seed": self.config["seed"],
                "control_steps": self.steps, "num_envs": self.envs, "transitions": self.steps * self.envs,
                "terminations": int(self.rows["terminated"].sum()), "timeouts": int(self.rows["truncated"].sum()),
                "source_head": self.source["head"], "collection": self.collection}


def error_channels(prediction, target):
    rotation = rotation_difference(prediction[..., 3:7], target[..., 3:7]).norm(dim=-1)
    ga, gb = projected_gravity(prediction), projected_gravity(target)
    tilt = torch.atan2(torch.cross(ga, gb, dim=-1).norm(dim=-1), (ga * gb).sum(dim=-1))
    error = prediction - target
    return torch.stack((
        error[..., :3].square().mean(-1), rotation.square(), tilt.square(),
        error[..., 7:10].square().mean(-1), error[..., 10:13].square().mean(-1),
        error[..., 13:25].square().mean(-1), error[..., 25:37].square().mean(-1),
    ), dim=-1)


@torch.no_grad()
def evaluate(model, dataset, horizons, batch_size):
    horizon = max(horizons)
    starts = dataset.replay.valid_starts(horizon)
    if not len(starts):
        raise ValueError(f"{dataset.path}: no {horizon}-step windows")
    device = starts.device
    indices = torch.tensor(horizons, device=device) - 1
    totals = torch.zeros(3, 2, len(horizons), len(FIELD_NAMES), dtype=torch.float64, device=device)
    counts = torch.zeros(3, len(horizons), dtype=torch.long, device=device)
    for chunk in starts.split(batch_size):
        initial, target, actions, previous, context = dataset.windows(chunk, horizon)
        prediction = model.rollout(initial, actions, previous, context)[:, indices]
        moving = ((target[..., 7:9].norm(dim=-1).mean(1) > 0.15)
                  | (target[..., 12].abs().mean(1) > 0.25))
        target = target[:, indices]
        end_time = chunk[:, :1] + indices
        falls = dataset.rows["terminated"][end_time, chunk[:, 1:2]]
        masks = torch.stack((torch.ones_like(falls), moving[:, None].expand_as(falls), falls))
        squared = torch.stack((error_channels(prediction, target),
                               error_channels(initial[:, None].expand_as(target), target)))
        if not torch.isfinite(squared).all():
            raise ValueError("Nonfinite rollout errors")
        totals += (squared.unsqueeze(0) * masks[:, None, :, :, None]).sum(dim=2)
        counts += masks.sum(dim=1)
    result = {}
    for i, group in enumerate(("all", "motion", "terminal")):
        result[group] = {}
        for j, h in enumerate(horizons):
            count = int(counts[i, j])
            model_rmse, persistence = (totals[i, :, j] / max(count, 1)).sqrt().tolist()
            result[group][str(h)] = {
                "windows": count,
                "model": dict(zip(FIELD_NAMES, model_rmse)) if count else None,
                "persistence": dict(zip(FIELD_NAMES, persistence)) if count else None,
            }
    return result


def motion_gate(values):
    entry = values["motion"]["8"]
    caps = {"joint_position_rad": 0.05, "linear_velocity_m_s": 0.2, "tilt_rad": 0.05}
    if entry["windows"] == 0:
        return {"passed": False, "reason": "No motion windows"}
    checks = {}
    for name, cap in caps.items():
        rmse, persistence = entry["model"][name], entry["persistence"][name]
        improvement = 1 - rmse / persistence if persistence > 0 else None
        checks[name] = {"rmse": rmse, "max_rmse": cap, "improvement": improvement,
                        "passed": rmse <= cap and improvement is not None and improvement >= 0.2}
    return {"passed": all(item["passed"] for item in checks.values()), "channels": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "validation", "test", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    if min(args.updates, args.batch_size, args.validate_every) < 1:
        parser.error("Updates, batch size and validation interval must be positive")
    paths = [p.resolve(strict=True) for p in (args.train, args.validation, args.test)]
    if len(set(paths)) != 3:
        parser.error("Train, validation and test must be independent replay files")
    torch.manual_seed(args.seed)
    train, validation, test = [Dataset(path, args.device) for path in paths]
    datasets = (train, validation, test)
    if len({d.config["seed"] for d in datasets}) != 3:
        parser.error("Collection seeds must differ across splits")
    if any(d.source["files"] != train.source["files"] or d.source["asset"] != train.source["asset"]
           or d.config["legacy"]["sim"] != train.config["legacy"]["sim"] for d in datasets[1:]):
        parser.error("Replay source and state contract must match across splits")
    args.output.mkdir(parents=True, exist_ok=False)
    model = WorldModel().to(args.device)
    d = train.rows
    model.fit_normalization(d["state"], d["next_state_pre_reset"], d["executed_action"],
                            d["prev_action"], d["context"])
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    horizon = 8
    starts = train.replay.valid_starts(horizon)
    if not len(starts):
        raise ValueError("Training replay has no legal windows")
    best_score = float("inf")
    scores = []
    start = time.monotonic()
    for step in range(1, args.updates + 1):
        selected = starts[torch.randint(len(starts), (args.batch_size,), device=starts.device)]
        initial, target, actions, previous, context = train.windows(selected, horizon)
        prediction = model.rollout(initial, actions, previous, context)
        loss = (state_difference(prediction, target) / model.delta_std).square().mean()
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite loss at update {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % args.validate_every and step != args.updates:
            continue
        metrics = evaluate(model, validation, (8,), args.batch_size)["all"]["8"]["model"]
        score = sum(metrics[name] / cap for name, cap in {
            "joint_position_rad": 0.05, "linear_velocity_m_s": 0.2, "tilt_rad": 0.05}.items())
        scores.append({"update": step, "training_loss": loss.item(), "validation_score": score})
        print(json.dumps(scores[-1]), flush=True)
        if score < best_score:
            best_score = score
            torch.save({"model_state_dict": model.state_dict(), "update": step, "hidden": [512] * 3,
                        "context_fields": CONTEXT_FIELDS, "train_horizon": horizon,
                        "joint_names": train.source["asset"]["joint_names"], "state_schema": 1},
                       args.output / "world_model.pt")
    best = torch.load(args.output / "world_model.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    validation_metrics = evaluate(model, validation, HORIZONS, args.batch_size)
    test_metrics = evaluate(model, test, HORIZONS, args.batch_size)
    report = {
        "status": "evaluated", "training_horizon": horizon, "hidden": [512] * 3,
        "batch_size": args.batch_size, "seed": args.seed, "learning_rate": 2e-4,
        "best_update": best["update"], "training_seconds_including_evaluation": time.monotonic() - start,
        "data": {name: dataset.description() for name, dataset in zip(("train", "validation", "test"), datasets)},
        "learning_curve": scores, "validation": validation_metrics, "test": test_metrics,
        "validation_motion_gate": motion_gate(validation_metrics), "test_motion_gate": motion_gate(test_metrics),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["test_motion_gate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
