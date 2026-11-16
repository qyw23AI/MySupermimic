#!/usr/bin/env python3
"""Fit the contact proxy and compare world model budgets on fixed replay."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from train_world_model import Dataset, evaluate
from him_supermimic.dynamics import WorldModel, state_difference
from him_supermimic.termination import ContactSurvival, heuristic_survival


def calibrate(args):
    def load(paths):
        return [torch.load(p, weights_only=True, map_location=args.device)["transitions"] for p in paths]

    def non_timeout(rows):
        mask = ~rows["truncated"]
        return rows["next_state_pre_reset"][mask], rows["terminated"][mask].float()

    groups = [load(paths) for paths in (args.train, args.validation, args.test)]
    model = ContactSurvival().to(args.device)
    state, label = (torch.cat(items) for items in zip(*(non_timeout(d) for d in groups[0]), strict=True))
    pos, neg = (label == 1).nonzero().flatten(), (label == 0).nonzero().flatten()
    if not len(pos) or not len(neg):
        raise ValueError("Need both contact terminals and healthy steps to fit the proxy")
    x = model.features(state)
    model.mean.copy_(x.mean(0))
    model.std.copy_(x.std(0, unbiased=False).clamp_min(1e-3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(400):
        idx = torch.cat((pos[torch.randint(len(pos), (64,), device=args.device)],
                         neg[torch.randint(len(neg), (192,), device=args.device)]))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model.logits(state[idx]), label[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Choose only the scalar calibration offset on independent validation runs.
    with torch.no_grad():
        vstate, vlabel = (torch.cat(items) for items in zip(*(non_timeout(d) for d in groups[1]), strict=True))
        logits = model.logits(vstate)
        offsets = torch.linspace(-16, 4, 201, device=args.device)
        scores = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[None, :] + offsets[:, None], vlabel.expand(len(offsets), -1), reduction="none").mean(1)
        model.logit_offset.copy_(offsets[scores.argmin()])

        def measure(rows, continuation):
            p = 1 - continuation(rows["next_state_pre_reset"])
            keep = ~rows["truncated"]
            target = rows["terminated"]
            t = target & keep
            n = ~target & keep
            points = {}
            for cutoff in (0.01, 0.1, 0.5, 0.9):
                predicted = p >= cutoff
                # Lead time of the first alarm within the final eight recorded steps.
                lead = torch.full_like(target, -1, dtype=torch.long)
                for offset in range(8):
                    same = rows["episode_id"][offset:] == rows["episode_id"][:len(target)-offset]
                    lead[offset:] = torch.where(
                        same & predicted[:len(target)-offset] & keep[:len(target)-offset],
                        offset, lead[offset:])
                hits = lead[t]
                points[str(cutoff)] = {"tp": int((predicted & t).sum()), "fp": int((predicted & n).sum()),
                                      "fn": int((~predicted & t).sum()),
                                      "terminal_alarm_lead_histogram": torch.bincount(hits + 1, minlength=9).tolist()}
            return {"terminals": int(t.sum()), "nonterminals": int(n.sum()),
                    "brier": (p[keep] - target[keep].float()).square().mean().item(),
                    "thresholds": points}

        report = {split: {str(path): { "heuristic": measure(rows, heuristic_survival),
                                        "learned": measure(rows, model)}
                         for path, rows in zip(paths, group, strict=True)}
                  for split, paths, group in zip(("validation", "test"),
                        (args.validation, args.test), groups[1:], strict=True)}
    source = json.loads((args.train[0].parent / "source.json").read_text())
    args.output.mkdir(parents=True)
    torch.save({"state_schema": 1, "joint_names": source["asset"]["joint_names"],
                "model_state_dict": model.state_dict()}, args.output / "termination.pt")
    return report


def budget(args):
    warm = torch.load(args.world_model, weights_only=True, map_location=args.device)
    model = WorldModel(warm["hidden"]).to(args.device)
    model.load_state_dict(warm["model_state_dict"])
    training = [Dataset(path, args.device) for path in args.train]
    splits = {"validation": [Dataset(path, args.device) for path in args.validation],
              "test": [Dataset(path, args.device) for path in args.test]}
    starts = [d.replay.valid_starts(8) for d in training]
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    report = {}
    schedule = (0, 100, 200, 400)
    for step in range(schedule[-1] + 1):
        if step in schedule:
            report[str(step)] = {name: {str(d.path): evaluate(model, d, (1, 8), 512) for d in datasets}
                                 for name, datasets in splits.items()}
        if step == schedule[-1]:
            break
        recent = step % (len(training) - 1) + 1
        batches = [training[i].windows(starts[i][torch.randint(len(starts[i]), (128,), device=args.device)], 8)
                   for i in (0, recent)]
        seed, target, actions, previous, context = (torch.cat(b) for b in zip(*batches, strict=True))
        loss = (state_difference(model.rollout(seed, actions, previous, context), target)
                / model.delta_std).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    args.output.mkdir(parents=True)
    return report


def evaluate_checkpoint(args):
    checkpoint = torch.load(args.world_model, weights_only=True, map_location=args.device)
    model = WorldModel(checkpoint["world_model_hidden"]).to(args.device).eval()
    model.load_state_dict(checkpoint["world_model_state_dict"])
    args.output.mkdir(parents=True)
    return {str(path): evaluate(model, Dataset(path, args.device), (1, 8), 512)
            for path in args.test}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("calibrate", "budget", "evaluate"))
    for name in ("train", "validation", "test"):
        parser.add_argument(f"--{name}", type=Path, nargs="+", default=[])
    parser.add_argument("--world-model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.test or (args.mode != "evaluate" and (not args.train or not args.validation)):
        parser.error("Expected independent replay paths for the requested mode")
    if args.mode != "calibrate" and args.world_model is None:
        parser.error("--world-model is required for model diagnostics")
    paths = [p.resolve(strict=True) for p in args.train + args.validation + args.test]
    if len(set(paths)) != len(paths):
        parser.error("Replay splits must use different files")
    torch.manual_seed(704)
    report = {"calibrate": calibrate, "budget": budget, "evaluate": evaluate_checkpoint}[args.mode](args)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
