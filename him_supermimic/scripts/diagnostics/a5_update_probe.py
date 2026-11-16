#!/usr/bin/env python3
"""Evaluate frozen HIM checkpoints and probe one actual A5 Adam update."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_online import ROOT, HORIZON, GAMMA, ONLINE_FIELDS, choose_windows, update_policy
from him_supermimic.evaluation import schedules

PLAN = ROOT / "him_supermimic/configs/wblwbl_dog_v1.plan.json"
DEV_SEEDS = (801, 802, 803)
TEST_SEEDS = (901, 902, 903)
SCALES = (0.0, 0.25, 0.5, 1.0, -0.5)


def evaluate_checkpoints(args, env, adapter, config):
    import torch
    from him_supermimic.evaluation import evaluation
    from warmup_critic import load_policy

    legacy, policy = load_policy(config, args.checkpoint, env.device)
    seeds = TEST_SEEDS if args.mode == "blind" else DEV_SEEDS
    result = {"checkpoint": str(args.checkpoint), "native": evaluation(
        env, adapter, legacy.act_inference, 16 if args.smoke else 160)}
    if args.parity:
        reference = result["native"]
        wrapped = evaluation(env, adapter, policy, 16 if args.smoke else 160)
        histories = adapter.get_observations()
        with torch.inference_mode():
            action_error = (legacy.act_inference(histories) - policy(histories)).abs().max().item()
        result["parity"] = {"action_max_abs": action_error,
                            "wrapped_native": wrapped,
                            "deltas": {key: wrapped[key] - reference[key] for key in
                                       ("linear_rmse_m_s", "yaw_rmse_rad_s", "reward_per_step", "falls")}}
        if action_error > 1e-6:
            raise RuntimeError(f"HIM inference mismatch: {action_error}")
    steps = 16 if args.smoke else max(1500, int(env.core.max_episode_length) + 1)
    result["steps_per_suite"] = steps
    result["suites"] = {}
    result["command_plans"] = {}
    for seed in seeds:
        suite, commands, random = schedules(env, seed)
        result["command_plans"][seed] = {"fixed": commands, "random": random, "hold_steps": 200}
        result["suites"][seed] = {
            name: evaluation(env, adapter, legacy.act_inference, steps, seeds=(seed,), schedule=schedule)
            for name, schedule in suite.items()}
    return result


def update_probe(args, env, adapter, config):
    import torch

    from him_supermimic.dynamics import CONTEXT_FIELDS, WorldModel
    from him_supermimic.evaluation import evaluation, reset_episode_batch, set_command
    from him_supermimic.flat_task import FlatTask
    from him_supermimic.imagination import imagine
    from him_supermimic.replay import EpisodeReplay
    from him_supermimic.termination import heuristic_survival
    from him_supermimic.value import Critic
    from warmup_critic import load_policy

    _, baseline = load_policy(config, args.checkpoint, env.device)
    _, anchor = load_policy(config, args.initial, env.device)
    proposed = deepcopy(baseline)
    saved = torch.load(args.checkpoint, map_location=env.device, weights_only=True)
    if saved["joint_names"] != env.core.dof_names or tuple(saved["context_fields"]) != CONTEXT_FIELDS:
        raise ValueError("Checkpoint state or action layout differs")
    if saved["critic_value_mode"] != "scaled_fixed_horizon":
        raise ValueError("Expected original combined A5 checkpoints with tail disabled")
    model = WorldModel(saved["world_model_hidden"]).to(env.device)
    model.load_state_dict(saved["world_model_state_dict"])
    model.eval().requires_grad_(False)
    value = Critic().to(env.device)
    value.load_state_dict(saved["critic_state_dict"])
    optimizer = torch.optim.Adam(proposed.actor.parameters())
    optimizer.load_state_dict(saved["actor_optimizer_state_dict"])
    critic_optimizer = torch.optim.Adam(value.parameters())
    critic_optimizer.load_state_dict(saved["critic_optimizer_state_dict"])
    task = FlatTask(config["legacy"], env.core.dof_names)
    rng = torch.Generator(device=env.device).manual_seed(args.seed + 43)

    def start(seed):
        env.replay = None
        reset_episode_batch(env, adapter, seed)

    def seed_from(row):
        return (row["state"], row["actor_history"], row["command"],
                torch.cat([row[key] for key in CONTEXT_FIELDS], dim=-1),
                row["prev_action"], row["prev_prev_action"], row["episode_step"])

    start(args.seed)
    env.replay = EpisodeReplay(32, env.num_envs, fields=ONLINE_FIELDS)
    with torch.no_grad():
        for _ in range(32):
            act = baseline(adapter.get_observations())
            adapter.step(act + 0.10 * torch.randn(act.shape, device=env.device, generator=rng))
    step, ids = choose_windows(env.replay, 256, rng).unbind(-1)
    batch = {key: tensor[step % env.replay.capacity, ids] for key, tensor in env.replay.data.items()}
    update = update_policy(proposed, model, value, task, seed_from(batch),
                           (optimizer, critic_optimizer), rng, saved["critic_return_horizon"], 0.0,
                           reference_policy=anchor, anchor_weight=0.01)
    print("Candidate Adam update generated", flush=True)

    candidates = {}
    with torch.no_grad():
        origin = list(baseline.actor.parameters())
        direction = [after - before for before, after in zip(origin, proposed.actor.parameters(), strict=True)]
        for scale in SCALES:
            candidate = deepcopy(baseline).eval().requires_grad_(False)
            for dst, src, delta in zip(candidate.actor.parameters(), origin, direction, strict=True):
                dst.copy_(src + scale * delta)
            candidates[scale] = candidate

    # Freeze the rate command at the fork in both simulator and model.
    task.heading_command = False
    horizons = (8, 32, 160)
    fields = ("state", "actor_history", "command", "episode_step", "prev_action", "prev_prev_action",
              *CONTEXT_FIELDS, "next_state_pre_reset", "command_pre_reset", "reward", "terminated", "truncated")
    prefix_sources = {"checkpoint": baseline}
    if args.failure_source:
        _, prefix_sources["historical_failure"] = load_policy(config, args.failure_source, env.device)
    output = []
    action_output = []

    @torch.no_grad()
    def branch(prefix_policy, seed, offset, candidate, perturb=0.0, direction=None, frozen_command=None):
        start(seed)
        for _ in range(offset):
            adapter.step(prefix_policy(adapter.get_observations()))
        env.replay = EpisodeReplay(1, env.num_envs, fields=fields)
        if frozen_command is None:
            frozen_command = env.core.commands[:, :3].clone()
        schedule = lambda _: frozen_command
        active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        reward = torch.zeros_like(active, dtype=torch.float32)
        linear = reward.clone()
        yaw = reward.clone()
        duration = reward.clone()
        falls = active & False
        results = {}
        for t in range(horizons[-1] if direction is None else 1):
            set_command(env, schedule, t)
            action = candidate(adapter.get_observations())
            adapter.step(action if direction is None else action + perturb * direction)
            row = {key: tensor[0] for key, tensor in env.replay.data.items()}
            if t == 0:
                initial = {key: tensor.clone() for key, tensor in row.items()}
            state = row["next_state_pre_reset"]
            cmd = row["command_pre_reset"]
            reward += (GAMMA ** t) * row["reward"] * active
            linear += (state[:, 7:9] - cmd[:, :2]).square().sum(-1) * active
            yaw += (state[:, 12] - cmd[:, 2]).square() * active
            duration += active
            falls |= active & row["terminated"]
            active &= ~(row["terminated"] | row["truncated"])
            if t + 1 in horizons:
                results[t + 1] = torch.stack((reward, linear / duration, yaw / duration,
                                              falls.float(), duration), dim=-1)
        if direction is not None:
            return initial, torch.cat((state[:, 7:9], state[:, 12:13], row["reward"][:, None]), -1)
        return initial, results

    def mismatch(reference, other):
        return torch.maximum((reference["state"] - other["state"]).abs().amax(-1),
                             (reference["actor_history"] - other["actor_history"]).abs().amax(-1))

    action_rng = torch.Generator(device=env.device).manual_seed(args.seed + 44)
    offsets = (48,) if args.smoke else (48, 112, 176, 240)
    for source, prefix_policy in prefix_sources.items():
        for offset in offsets:
            print(f"Probing {source}, prefix {offset} steps", flush=True)
            reference, base = branch(prefix_policy, args.seed, offset, baseline)
            frozen = reference["command"][:, :3]
            duplicate, null = branch(prefix_policy, args.seed, offset, baseline, frozen_command=frozen)
            noise_start = mismatch(reference, duplicate)
            near = base[8][:, 3].bool()
            turning = (frozen[:, 2].abs() >= 0.2) & ~near
            if source == "historical_failure":
                # Label with the actual behavior policy if the candidate recovers after the fork.
                _, behavior = branch(prefix_policy, args.seed, offset, prefix_policy, frozen_command=frozen)
                near |= behavior[8][:, 3].bool()
                turning &= ~near
            seed = seed_from(reference)
            with torch.no_grad():
                predictions = {}
                for scale, policy in candidates.items():
                    trajectory = imagine(policy, model, task, seed, HORIZON)
                    surv = heuristic_survival(trajectory.states[:, 1:])
                    prefix = torch.cat((torch.ones_like(surv[:, :1]), surv.cumprod(1)[:, :-1]), 1)
                    discounts = trajectory.rewards.new_tensor([GAMMA ** t for t in range(HORIZON)])
                    ret = (trajectory.rewards * prefix * discounts).sum(-1)
                    penalty = (trajectory.actions[:, 0] - task.clip_action(anchor(seed[1]))).square().mean(-1)
                    predictions[scale] = (ret, ret - 0.01 * penalty)
            for scale in SCALES[1:]:
                initial, observed = branch(prefix_policy, args.seed, offset, candidates[scale],
                                           frozen_command=frozen)
                output.append({"source": source, "offset": offset, "scale": scale,
                               "near_fall": near.cpu(), "turning": turning.cpu(),
                               "start_error": mismatch(reference, initial).cpu(),
                               "duplicate_start_error": noise_start.cpu(),
                               "predicted_return_delta": (predictions[scale][0] - predictions[0.0][0]).cpu(),
                               "predicted_objective_delta": (predictions[scale][1] - predictions[0.0][1]).cpu(),
                               "baseline": {h: x.cpu() for h, x in base.items()},
                               "delta": {h: (x - base[h]).cpu() for h, x in observed.items()},
                               "duplicate_delta": {h: (null[h] - x).cpu() for h, x in base.items()}})
            action = baseline(seed[1]).detach()
            random_direction = torch.nn.functional.normalize(torch.randn(
                action.shape, device=env.device, generator=action_rng), dim=-1)
            _, zero = branch(prefix_policy, args.seed, offset, baseline, direction=random_direction,
                             frozen_command=frozen)
            _, zero_repeat = branch(prefix_policy, args.seed, offset, baseline, direction=random_direction,
                                    frozen_command=frozen)
            def response(a):
                executed = task.clip_action(a)
                state = model(seed[0], executed, seed[4], seed[3])
                reward = sum(task.reward_terms(seed[0], state, seed[2], executed, seed[4], seed[5]).values())
                return torch.cat((state[:, 7:9], state[:, 12:13], reward[:, None]), -1)

            _, model_jvp = torch.autograd.functional.jvp(response, action, random_direction)
            for epsilon in (0.02, 0.04):
                initial_plus, plus = branch(prefix_policy, args.seed, offset, baseline, epsilon,
                                            random_direction, frozen)
                initial_minus, minus = branch(prefix_policy, args.seed, offset, baseline, -epsilon,
                                              random_direction, frozen)
                with torch.no_grad():
                    predicted = response(action + epsilon * random_direction) - response(action - epsilon * random_direction)
                action_output.append({"source": source, "offset": offset, "epsilon": epsilon,
                                      "near_fall": near.cpu(), "turning": turning.cpu(),
                                      "physical_difference": (plus - minus).cpu(),
                                      "predicted_difference": predicted.cpu(), "model_jvp": model_jvp.cpu(),
                                      "zero_difference": (zero - zero_repeat).cpu(),
                                      "start_error": mismatch(initial_plus, initial_minus).cpu(),
                                      "termination_disagreement": (initial_plus["terminated"] != initial_minus["terminated"]).cpu(),
                                      "clipped": ((action.abs() + epsilon * random_direction.abs()) >
                                                  env.core.cfg.normalization.clip_actions).any(-1).cpu()})
    direction_summary = {}
    for low, high in zip(action_output[::2], action_output[1::2], strict=True):
        key = low["source"]
        summaries = direction_summary.setdefault(key, {})
        for group, mask in (("near_fall", low["near_fall"]),
                            ("turning", low["turning"]),
                            ("healthy", ~(low["near_fall"] | low["turning"]))):
            totals = summaries.setdefault(group, {"rows": 0, "reliable": [0] * 4, "agree": [0] * 4})
            totals["rows"] += int(mask.sum())
            reliable = mask[:, None]
            for row in (low, high):
                clean = ~(row["clipped"] | row["termination_disagreement"])
                reliable = (reliable & clean[:, None] & (row["start_error"] <= 1e-3)[:, None]
                            & (row["physical_difference"].abs() > 3 * row["zero_difference"].abs().clamp_min(1e-6)))
            sign = high["physical_difference"].sign()
            reliable &= sign == low["physical_difference"].sign()
            for name, count in (("reliable", reliable.sum(0)),
                                ("agree", (reliable & (sign == low["model_jvp"].sign())).sum(0))):
                totals[name] = [old + int(new) for old, new in zip(totals[name], count, strict=True)]

    env.replay = None
    env.command_override = None
    reset_metrics = {
        str(scale): evaluation(env, adapter, policy, 16 if args.smoke else 512,
                               seeds=(args.seed,))
        for scale, policy in candidates.items()
    }
    torch.save({"candidate_rows": output, "action_rows": action_output,
                "candidate_channels": ["discounted_reward", "linear_mse", "yaw_mse", "fall", "alive_steps"],
                "action_channels": ["vx", "vy", "yaw_rate", "reward"]}, args.output / "probe.pt")
    return {"checkpoint": str(args.checkpoint), "iteration": saved["iteration"], "seed": args.seed,
            "failure_source": str(args.failure_source) if args.failure_source else None,
            "scales": SCALES, "horizons": horizons, "offsets": offsets, "training_batch": 256,
            "update": update, "reset_metrics": reset_metrics, "parameter_step_l2": sum(d.square().sum().item() for d in direction) ** 0.5,
            "near_fall_rows": {source: sum(int(row["near_fall"].sum()) for row in output
                                          if row["source"] == source and row["scale"] == 1.0)
                               for source in prefix_sources},
            "action_direction_counts": direction_summary,
            "action_direction_channels": ["vx", "vy", "yaw_rate", "reward"],
            "artifact": "probe.pt"}


def run(args):
    import torch
    from himloco_lab.adapter import LegacyRunnerAdapter
    from himloco_lab.config import class_to_dict, load_config
    from him_supermimic.himlab import RecordingHIMLabEnv, make_flat_env_cfg

    torch.manual_seed(args.seed)
    legacy, train = load_config(task="wblwbl_dog_v1", num_envs=args.num_envs, stage=3)
    plan = json.loads(PLAN.read_text())
    profile = plan["controlled_flat"]
    for term in profile["omitted_terms"]:
        setattr(legacy.rewards.scales, term, 0.0)
    for name, enabled in profile["domain_rand_overrides"].items():
        setattr(legacy.domain_rand, name, enabled)
    legacy.commands.curriculum = profile["command_curriculum"]
    cfg = make_flat_env_cfg(legacy, args.device, args.output / "asset_cache")
    cfg.seed = args.seed
    env = RecordingHIMLabEnv(cfg)
    try:
        contract = plan["contract"]
        if env.core.dof_names != contract["policy_joint_names"] or abs(env.step_dt - contract["control_dt"]) > 1e-8:
            raise RuntimeError("Unexpected WBLWBL joint order or control timing")
        adapter = LegacyRunnerAdapter(env)
        config = {"legacy": class_to_dict(legacy), "training": train}
        report = (update_probe if args.mode == "probe" else evaluate_checkpoints)(args, env, adapter, config)
        report["mode"] = args.mode
        report["num_envs"] = args.num_envs
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Saved {args.output / 'report.json'}", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        env.close()


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("development", "blind", "probe"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--failure-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=801)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or (args.mode == "probe" and args.initial is None):
        parser.error("Positive num-envs and an initial checkpoint for probe mode are required")
    args.output.mkdir(parents=True, exist_ok=False)
    app = AppLauncher(args).app
    try:
        run(args)
    except Exception:
        traceback.print_exc()
        raise
    app.close()


if __name__ == "__main__":
    main()
