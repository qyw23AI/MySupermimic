#!/usr/bin/env python3
"""Alternate Isaac Lab collection, model fitting and 8-step HIM BPTT."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
HIM_ROOT = ROOT / "third_party/MyHIMLoco"
sys.path[:0] = [str(ROOT), str(HIM_ROOT / "isaaclab_migration"), str(HIM_ROOT / "rsl_rl")]

HORIZON = 8
GAMMA = 0.99
LAMBDA = 0.95
ONLINE_FIELDS = (
    "state", "next_state_pre_reset", "executed_action", "prev_action", "prev_prev_action",
    "actor_history", "next_critic_obs_pre_reset", "command", "episode_step", "episode_id",
    "terminated", "truncated", "command_resampled", "friction", "restitution", "kp_factor", "kd_factor",
)
from him_supermimic.evaluation import EVAL_SEEDS, evaluation



def choose_windows(replay, count, rng):
    import torch

    starts = replay.valid_starts(HORIZON)
    if len(starts) == 0:
        raise ValueError("No legal online replay windows")
    return starts[torch.randint(len(starts), (count,), device=starts.device, generator=rng)]


def model_update(model, optimizer, replay, real_starts, offline, old_starts):
    import torch

    from him_supermimic.dynamics import CONTEXT_FIELDS, state_difference

    step, env_id = real_starts.unbind(-1)
    slot = (step[:, None] + torch.arange(HORIZON, device=step.device)) % replay.capacity

    def online(name):
        return replay.data[name][slot, env_id[:, None]]

    seed, target, action, previous, context = (
        online("state")[:, 0], online("next_state_pre_reset"), online("executed_action"), online("prev_action"),
        torch.cat([online(key) for key in CONTEXT_FIELDS], -1),
    )
    batch = [torch.cat((a, b), dim=0) for a, b in zip(
        (seed, target, action, previous, context), offline.windows(old_starts, HORIZON), strict=True)]
    model.requires_grad_(True)
    prediction = model.rollout(batch[0], *batch[2:])
    loss = (state_difference(prediction, batch[1]) / model.delta_std).square().mean()
    optimize(loss, optimizer, model.parameters())
    model.requires_grad_(False)
    return loss.item()


def optimize(loss, optimizer, parameters):
    import torch

    if not torch.isfinite(loss):
        raise ValueError("Nonfinite objective")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    return norm.item()


def fin_horizon_scale(remaining, horizon):
    return (1 - GAMMA ** remaining) / (1 - GAMMA ** horizon)


def update_policy(policy, model, value, task, seed, optimizers, rng, value_horizon, tail_weight,
                  critic_all_steps=False, critic_horizon_conditioned=False, tail_cap=None,
                  target_value=None, survival_model=None, reference_policy=None, anchor_weight=0.0):
    import torch

    from him_supermimic.termination import heuristic_survival
    from him_supermimic.imagination import imagine
    from him_supermimic.value import td_lambda_targets

    actor_opt, critic_opt = optimizers
    base_state = seed[0]
    bounds, _ = task.observation_noise_bounds(base_state)
    noise = (2 * torch.rand((len(base_state), HORIZON, 45), device=base_state.device, generator=rng) - 1) * bounds
    policy.estimator.requires_grad_(False)
    value.requires_grad_(False)
    traj = imagine(policy, model, task, seed, HORIZON, observation_noise=noise)
    states = traj.states[:, 1:]
    survival = heuristic_survival(states) if survival_model is None else survival_model(states)
    prefix = torch.cat((torch.ones_like(survival[:, :1]), survival.cumprod(dim=1)), dim=1)
    discount = GAMMA ** torch.arange(HORIZON, device=base_state.device)
    imagined = (traj.rewards * prefix[:, :-1] * discount).sum(dim=1)
    if critic_horizon_conditioned:
        remaining = torch.full((len(base_state),), (value_horizon - HORIZON) / value_horizon,
                               device=base_state.device)
        tail = value(traj.features[:, -1], remaining)
    else:
        tail = value(traj.features[:, -1]) * fin_horizon_scale(value_horizon - HORIZON, value_horizon)
    reward_objective = imagined.mean()
    weighted_tail = tail_weight * GAMMA ** HORIZON * prefix[:, -1] * tail
    tail_objective = weighted_tail.mean()
    objective = (imagined + weighted_tail).mean()
    stats = {}
    if tail_cap is not None:
        params = tuple(policy.actor.parameters())
        reward_grad = torch.cat([g.flatten() for g in torch.autograd.grad(
            reward_objective, params, retain_graph=True)])
        tail_grad = torch.cat([g.flatten() for g in torch.autograd.grad(
            tail_objective, params, retain_graph=True)])
        reward_norm, tail_norm = reward_grad.norm(), tail_grad.norm()
        if not torch.isfinite(reward_norm + tail_norm):
            raise ValueError("Nonfinite Actor gradient component")
        multiplier = (torch.ones_like(tail_norm) if tail_cap == float("inf") else
                      (tail_cap * reward_norm / tail_norm.clamp_min(1e-12)).clamp(max=1))
        stats.update(reward_grad_norm=reward_norm.item(), tail_grad_norm=tail_norm.item(),
                     tail_grad_cosine=torch.nn.functional.cosine_similarity(reward_grad, tail_grad, dim=0).item(),
                     tail_grad_multiplier=multiplier.item())
        # Keep the reported objective intact while rescaling only the tail gradient.
        objective = objective + (multiplier - 1) * (tail_objective - tail_objective.detach())
    if reference_policy is not None:
        with torch.no_grad():
            reference_action = task.clip_action(reference_policy(seed[1]))
        anchor = (traj.actions[:, 0] - reference_action).square().mean()
        objective = objective - anchor_weight * anchor
        stats["actor_anchor_rmse"] = anchor.sqrt().item()
    actor_norm = optimize(-objective, actor_opt, policy.actor.parameters())

    bootstrap = value if target_value is None else target_value
    with torch.no_grad():
        if critic_horizon_conditioned:
            remaining = (value_horizon - torch.arange(HORIZON + 1, device=base_state.device)) / value_horizon
            values = bootstrap(traj.features.detach(), remaining.expand(len(base_state), -1))
            value_targets = values
        else:
            values = bootstrap(traj.features.detach())
            scale = fin_horizon_scale(value_horizon - torch.arange(HORIZON + 1, device=base_state.device), value_horizon)
            value_targets = values * scale
        targets = td_lambda_targets(traj.rewards.detach(), value_targets,
                                     torch.zeros_like(survival, dtype=torch.bool), GAMMA, LAMBDA,
                                     survival=survival.detach())
    value.requires_grad_(True)
    critic_features = traj.features.detach()
    if critic_horizon_conditioned:
        remaining = (value_horizon - torch.arange(HORIZON + 1, device=base_state.device)) / value_horizon
        critic_predictions = value(critic_features, remaining.expand(len(base_state), -1))
        critic_all_steps = True
    else:
        critic_predictions = value(critic_features)
    critic_loss = (critic_predictions[:, :HORIZON] - targets).square().mean() if critic_all_steps else (
        critic_predictions[:, 0] - targets[:, 0]
    ).square().mean()
    optimize(critic_loss, critic_opt, value.parameters())
    return {**stats,"imagined_8_step_return": imagined.mean().item(), "actor_objective": objective.item(),
            "actor_grad_norm": actor_norm, "critic_loss": critic_loss.item(),
            "imagined_survival_at_8": prefix[:, -1].mean().item()}


def run(args, output):
    from copy import deepcopy

    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from him_supermimic.tensorboard_metrics import EVAL_TAGS, UPDATE_TAGS, log_metrics
    from himloco_lab.adapter import LegacyRunnerAdapter
    from himloco_lab.config import class_to_dict, load_config
    import himloco_lab.config as him_config

    from him_supermimic.dynamics import CONTEXT_FIELDS, WorldModel
    from him_supermimic.flat_task import FlatTask
    from him_supermimic.himlab import RecordingHIMLabEnv, make_flat_env_cfg
    from him_supermimic.replay import EpisodeReplay
    from him_supermimic.value import Critic, HorizonCritic
    from train_world_model import Dataset
    from warmup_critic import load_policy

    if not Path(him_config.__file__).resolve().is_relative_to(HIM_ROOT):
        raise RuntimeError("Expected environment from the pinned MyHIMLoco submodule")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = torch.Generator(device=args.device).manual_seed(args.seed + 1)
    extra_model_rng = torch.Generator(device=args.device).manual_seed(args.seed + 2)
    legacy, training = load_config(task="wblwbl_dog_v1", stage=3, num_envs=args.num_envs)
    plan = json.loads((ROOT / "him_supermimic/configs/wblwbl_dog_v1.plan.json").read_text())
    profile = plan["controlled_flat"]
    for term in profile["omitted_terms"]:
        setattr(legacy.rewards.scales, term, 0.0)
    for name, enabled in profile["domain_rand_overrides"].items():
        setattr(legacy.domain_rand, name, enabled)
    legacy.commands.curriculum = profile["command_curriculum"]
    config = {"legacy": class_to_dict(legacy), "training": training}
    wm = torch.load(args.world_model, map_location=args.device, weights_only=True)
    cv = torch.load(args.critic, map_location=args.device, weights_only=True)
    joints = plan["contract"]["policy_joint_names"]
    for checkpoint in (wm, cv):
        if (checkpoint["joint_names"] != joints or tuple(checkpoint["context_fields"]) != CONTEXT_FIELDS
            or checkpoint["state_schema"] != 1):
            raise ValueError("Checkpoint metadata differs from WBLWBL state/action layout")
    horizon = cv["return_horizon"]
    if horizon <= HORIZON or (cv["gamma"], cv["lambda"]) != (GAMMA, LAMBDA):
        raise ValueError("Expected the compatible A4 finite-horizon Critic")
    model = WorldModel(wm["hidden"]).to(args.device)
    model.load_state_dict(wm["model_state_dict"])
    critic_horizon_conditioned = cv.get("value_mode") == "remaining_horizon"
    if critic_horizon_conditioned != (args.critic_target == "horizon_all_steps"):
        raise ValueError("Critic checkpoint value mode must match --critic-target")
    value = (HorizonCritic() if critic_horizon_conditioned else Critic(cv["input_width"])).to(args.device)
    value.load_state_dict(cv["model_state_dict"])
    original_policy, policy = load_policy(config, args.policy, args.device)
    reference_policy = deepcopy(policy).eval().requires_grad_(False) if args.actor_anchor else None
    target_value = deepcopy(value).eval().requires_grad_(False) if args.critic_target_tau else None
    survival_model = None
    if args.termination_model:
        from him_supermimic.termination import ContactSurvival

        ckpt = torch.load(args.termination_model, map_location=args.device, weights_only=True)
        if ckpt["joint_names"] != joints or ckpt["state_schema"] != 1:
            raise ValueError("Incompatible termination checkpoint")
        survival_model = ContactSurvival().to(args.device).eval().requires_grad_(False)
        survival_model.load_state_dict(ckpt["model_state_dict"])
    # Fixed recorded histories track action drift independently of the current online replay.
    histories = torch.load(args.replay, map_location="cpu", weights_only=True)["transitions"]["actor_history"]
    flat_history = histories.flatten(0, 1)
    fixed_history = flat_history[::max(1, len(flat_history) // 1024)][:1024].to(args.device)
    with torch.inference_mode():
        initial_actions = policy(fixed_history)
    del histories, flat_history
    offline = Dataset(args.replay, args.device)
    old_starts = offline.replay.valid_starts(HORIZON)
    if offline.source["asset"]["joint_names"] != joints or not len(old_starts):
        raise ValueError("Expected compatible A3 training replay windows")
    task = FlatTask(config["legacy"], joints)
    cfg = make_flat_env_cfg(legacy, args.device, output / "asset_cache")
    cfg.seed = args.seed
    env = RecordingHIMLabEnv(cfg)
    writer = None
    try:
        writer = SummaryWriter(str(output / "tensorboard"))
        if env.core.dof_names != joints or abs(env.step_dt - task.dt) > 1e-8:
            raise RuntimeError("Simulator layout differs from trained weights")
        adapter = LegacyRunnerAdapter(env)
        model_opt = torch.optim.Adam(model.parameters(), lr=2e-5)
        actor_opt = torch.optim.Adam(policy.actor.parameters(), lr=args.actor_lr)
        critic_opt = torch.optim.Adam(value.parameters(), lr=2e-4)
        if str(env.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(env.device)
        baseline = evaluation(env, adapter, original_policy.act_inference, args.eval_steps)
        report = {"baseline": baseline, "updates": [], "evaluations": [],
                  "settings": {k: str(v) if isinstance(v, Path) or v == float("inf") else v for k, v in vars(args).items()},
                  "legacy_config": config["legacy"], "eval_seeds": EVAL_SEEDS, "horizon": HORIZON,
                  "checkpoint_sha256": {k: hashlib.sha256(getattr(args, k).read_bytes()).hexdigest()
                                        for k in ("policy", "world_model", "critic", "termination_model")
                                        if getattr(args, k)}}
        log_metrics(writer, baseline, EVAL_TAGS)
        print(json.dumps({"baseline": baseline}), flush=True)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        with torch.inference_mode():
            adapter.reset()
        env.replay = EpisodeReplay(args.replay_capacity, env.num_envs, fields=ONLINE_FIELDS)
        def save_checkpoint(path, iteration):
            temporary = path.with_suffix(".tmp")
            torch.save({"model_state_dict": original_policy.state_dict(), "world_model_state_dict": model.state_dict(),
                        "critic_state_dict": value.state_dict(),
                        "target_critic_state_dict": target_value.state_dict() if target_value is not None else None,
                        "actor_optimizer_state_dict": actor_opt.state_dict(),
                        "model_optimizer_state_dict": model_opt.state_dict(), "critic_optimizer_state_dict": critic_opt.state_dict(),
                        "estimator_optimizer_state_dict": policy.estimator.optimizer.state_dict(), "iteration": iteration,
                        "joint_names": joints, "world_model_hidden": wm["hidden"], "critic_return_horizon": horizon,
                        "context_fields": CONTEXT_FIELDS, "critic_value_mode": cv.get("value_mode", "scaled_fixed_horizon")}, temporary)
            temporary.replace(path)

        def save_report():
            temporary = output / "report.tmp"
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temporary.replace(output / "report.json")

        def score(metrics):
            return (-metrics["falls"], metrics["reward_per_step"])

        best = score(baseline)
        report["best_iteration"] = 0
        report["best_checkpoint"] = "checkpoint_0000.pt"
        report["best_evaluation"] = baseline
        save_checkpoint(output / report["best_checkpoint"], 0)
        save_report()
        def sync():
            if str(env.device).startswith("cuda"):
                torch.cuda.synchronize(env.device)
        training_start = time.monotonic()
        worse_falls = 0
        guard_breaches = 0
        for iteration in range(1, args.iterations + 1):
            replay = env.replay
            sync()
            collection_start = time.monotonic()
            with torch.inference_mode():
                reward_sum = torch.zeros((), device=env.device)
                for _ in range(args.collect_steps):
                    action = policy(adapter.get_observations())
                    noise = torch.randn(action.shape, device=action.device, generator=rng) * 0.10
                    reward_sum += adapter.step(action + noise)[2].sum()
            sync()
            collection_seconds = time.monotonic() - collection_start
            losses = []
            for index in range(args.model_updates):
                model_rng = rng if index < 2 else extra_model_rng
                recent = choose_windows(replay, 128, model_rng)
                historical = old_starts[torch.randint(len(old_starts), (128,), device=args.device, generator=model_rng)]
                losses.append(model_update(model, model_opt, replay, recent, offline, historical))
            step, env_id = choose_windows(replay, 256, rng).unbind(-1)
            slot = step % replay.capacity
            row = {key: values[slot, env_id] for key, values in replay.data.items()}
            him_metrics = {}
            if not args.freeze_him:
                policy.estimator.requires_grad_(True)
                estimator_loss, swap_loss = policy.estimator.update(
                    row["actor_history"], row["next_critic_obs_pre_reset"], lr=1e-5)
                if not np.isfinite([estimator_loss, swap_loss]).all():
                    raise ValueError("Nonfinite HIM auxiliary loss")
                him_metrics = {"estimator_loss": estimator_loss, "swap_loss": swap_loss}
            context = torch.cat([row[key] for key in CONTEXT_FIELDS], -1)
            seed = (row["state"], row["actor_history"], row["command"], context,
                    row["prev_action"], row["prev_prev_action"], row["episode_step"])
            update = update_policy(policy, model, value, task, seed, (actor_opt, critic_opt), rng,
                                   horizon, args.tail_weight,
                                   critic_all_steps=args.critic_target in ("all_steps", "horizon_all_steps"),
                                   critic_horizon_conditioned=critic_horizon_conditioned,
                                   tail_cap=args.tail_gradient_ratio, target_value=target_value,
                                   survival_model=survival_model, reference_policy=reference_policy,
                                   anchor_weight=args.actor_anchor)
            if target_value is not None:
                with torch.no_grad():
                    for target_param, param in zip(target_value.parameters(), value.parameters(), strict=True):
                        target_param.lerp_(param, args.critic_target_tau)
            update.update(iteration=iteration, world_model_loss=sum(losses) / len(losses), **him_metrics)
            sync()
            update.update(collection_seconds=collection_seconds,
                          update_seconds=time.monotonic() - collection_start - collection_seconds,
                          training_transitions=replay.total_steps * env.num_envs,
                          physics_substeps=replay.total_steps * env.num_envs * cfg.decimation,
                          elapsed_seconds=time.monotonic() - training_start,
                          collection_transitions_per_second=args.collect_steps * env.num_envs / collection_seconds)
            if str(env.device).startswith("cuda"):
                update["torch_peak_reserved_mib"] = torch.cuda.max_memory_reserved(env.device) / 2**20
            update["real_reward_per_step"] = reward_sum.item() / (args.collect_steps * env.num_envs)
            log_metrics(writer, update, UPDATE_TAGS)
            report["updates"].append(update)
            print(json.dumps(update), flush=True)
            if iteration % args.eval_every == 0 or iteration == args.iterations:
                # Evaluation physically resets the simulator. Episode IDs protect older replay.
                rng_state = torch.get_rng_state()
                cuda_state = torch.cuda.get_rng_state(env.device) if str(env.device).startswith("cuda") else None
                np_state = np.random.get_state()
                metrics = evaluation(env, adapter, policy, args.eval_steps)
                with torch.inference_mode():
                    metrics["fixed_history_action_rmse"] = (policy(fixed_history) - initial_actions).square().mean().sqrt().item()
                torch.set_rng_state(rng_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state(cuda_state, env.device)
                np.random.set_state(np_state)
                report["evaluations"].append({"iteration": iteration, **metrics})
                log_metrics(writer, report["evaluations"][-1], EVAL_TAGS)
                writer.flush()
                print(json.dumps(report["evaluations"][-1]), flush=True)
                worse_falls = worse_falls + 1 if metrics["falls"] > baseline["falls"] else 0
                if args.guard_patience:
                    violations = []
                    if metrics["falls_per_robot_minute"] > args.guard_falls_per_robot_minute:
                        violations.append("falls")
                    if metrics["linear_rmse_m_s"] > baseline["linear_rmse_m_s"] * args.guard_linear_factor:
                        violations.append("linear_rmse")
                    if metrics["yaw_rmse_rad_s"] > baseline["yaw_rmse_rad_s"] * args.guard_yaw_factor:
                        violations.append("yaw_rmse")
                    if metrics["reward_per_step"] < baseline["reward_per_step"] - args.guard_reward_drop:
                        violations.append("reward")
                    guard_breaches = guard_breaches + 1 if violations else 0
                    report["evaluations"][-1]["guard_violations"] = violations
                path = output / f"checkpoint_{iteration:04d}.pt"
                save_checkpoint(path, iteration)
                if score(metrics) > best:
                    best = score(metrics)
                    report.update(best_iteration=iteration, best_checkpoint=path.name,
                                  best_evaluation=metrics)
                report.update(completed_iterations=iteration,
                              training_transitions=replay.total_steps * env.num_envs,
                              physics_substeps=replay.total_steps * env.num_envs * cfg.decimation,
                              elapsed_seconds_including_eval=time.monotonic() - training_start)
                if args.guard_patience and guard_breaches >= args.guard_patience:
                    report["stop_reason"] = "physical_guard"
                    report["stop_violations"] = violations
                    save_report()
                    break
                if args.fall_stop_patience and worse_falls >= args.fall_stop_patience:
                    report["stop_reason"] = "fall_count_above_baseline_at_consecutive_evaluations"
                    save_report()
                    break
                save_report()
        save_checkpoint(output / "agent.pt", iteration)
        save_report()
    finally:
        if writer is not None:
            writer.close()
        env.close()


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("policy", "world-model", "critic", "replay", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=411)
    parser.add_argument("--num-envs", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--collect-steps", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=320)
    parser.add_argument("--actor-lr", type=float, default=5e-6)
    parser.add_argument("--tail-weight", type=float, default=0.1)
    parser.add_argument("--critic-target", choices=("start", "all_steps", "horizon_all_steps"), default="start",
                        help="Choose start-only, all-step, or remaining-horizon-conditioned Critic targets.")
    parser.add_argument("--tail-gradient-ratio", type=float,
                        help="Cap tail Actor gradient norm relative to reward gradient; log both components.")
    parser.add_argument("--critic-target-tau", type=float, default=0,
                        help="EMA rate for a separate Critic bootstrap target; zero uses the live Critic.")
    parser.add_argument("--termination-model", type=Path)
    parser.add_argument("--model-updates", type=int, default=2)
    parser.add_argument("--freeze-him", action="store_true")
    parser.add_argument("--actor-anchor", type=float, default=0,
                        help="Weight of squared seed-action deviation from the initial HIM policy.")
    parser.add_argument("--replay-capacity", type=int, default=192)
    parser.add_argument("--fall-stop-patience", type=int, default=2,
                        help="Stop after this many worse fall evaluations. Zero runs the full budget.")
    parser.add_argument("--guard-patience", type=int, default=0,
                        help="Stop after this many consecutive physical-metric breaches; zero disables the guard.")
    parser.add_argument("--guard-falls-per-robot-minute", type=float, default=0.1)
    parser.add_argument("--guard-linear-factor", type=float, default=1.25)
    parser.add_argument("--guard-yaw-factor", type=float, default=1.25)
    parser.add_argument("--guard-reward-drop", type=float, default=0.003)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.num_envs, args.iterations, args.eval_every, args.eval_steps) < 1 or args.collect_steps < HORIZON:
        parser.error("Positive counts required, with at least 8 collection steps")
    if args.replay_capacity < HORIZON or args.fall_stop_patience < 0:
        parser.error("Replay capacity must be at least 8 and fall-stop-patience must be nonnegative")
    if (args.guard_patience < 0 or args.guard_falls_per_robot_minute < 0
        or args.guard_linear_factor < 1 or args.guard_yaw_factor < 1
        or args.guard_reward_drop < 0):
        parser.error("Invalid physical guard setting")
    if not (0 < args.actor_lr < float("inf") and 0 <= args.tail_weight <= 1):
        parser.error("Expected positive finite actor-lr and tail-weight in [0, 1]")
    if (args.model_updates < 1 or not 0 <= args.critic_target_tau <= 1
        or not 0 <= args.actor_anchor < float("inf") or
        (args.tail_gradient_ratio is not None and not 0 <= args.tail_gradient_ratio <= float("inf"))):
        parser.error("Invalid experiment setting")
    if not (HIM_ROOT / "isaaclab_migration/himloco_lab/config.py").is_file():
        parser.error("Initialize third_party/MyHIMLoco")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    app = AppLauncher(args).app
    try:
        run(args, args.output)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    sys.stdout.flush()
    app.close()


if __name__ == "__main__":
    main()
