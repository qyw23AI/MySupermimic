#!/usr/bin/env python3
"""Train WBLWBL HIM with a learned model and BPTT from random initialization."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from train_online import GAMMA, HORIZON, HIM_ROOT, ONLINE_FIELDS, ROOT, optimize


def arguments():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ground-usd", type=Path, help="Local replacement for Isaac Lab's ground USD")
    parser.add_argument("--seed", type=int, default=511)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--collect-steps", type=int, default=8)
    parser.add_argument("--replay-capacity", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--model-updates", type=int, default=8)
    parser.add_argument("--model-lr", type=float, default=2e-5)
    parser.add_argument("--actor-lr", type=float, default=5e-6)
    parser.add_argument("--estimator-lr", type=float, default=1e-5)
    parser.add_argument("--anchor-weight", type=float, default=0.01)
    parser.add_argument("--max-action-p95", type=float, default=0.02)
    parser.add_argument("--exploration-std", type=float, help="Defaults to the HIM policy initial exploration std")
    parser.add_argument("--cold-start-min", type=int, default=20)
    parser.add_argument("--cold-start-max", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=320)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if (args.num_envs < 4 or min(args.iterations, args.collect_steps, args.batch_size,
                               args.model_updates, args.eval_every, args.eval_steps, args.cold_start_min) < 1
            or args.replay_capacity < max(HORIZON, args.collect_steps)
            or args.cold_start_max < args.cold_start_min):
        parser.error("Invalid environment, replay, iteration or evaluation counts")
    for name in ("model_lr", "actor_lr", "estimator_lr", "max_action_p95"):
        if not 0 < getattr(args, name) < float("inf"):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("anchor_weight", "exploration_std"):
        value = getattr(args, name)
        if value is not None and not 0 <= value < float("inf"):
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.ground_usd is not None:
        args.ground_usd = args.ground_usd.resolve(strict=True)
    return args


def run(args, start):
    import numpy as np
    import torch
    from unittest.mock import patch
    from torch.utils.tensorboard import SummaryWriter

    from himloco_lab import env as upstream_env
    from himloco_lab.adapter import LegacyRunnerAdapter
    from himloco_lab.config import class_to_dict, load_config
    from rsl_rl.modules import HIMActorCritic
    from him_supermimic.dynamics import CONTEXT_FIELDS, WorldModel, state_difference
    from him_supermimic.evaluation import evaluation, schedules
    from him_supermimic.flat_task import FlatTask
    from him_supermimic.him_policy import DifferentiableHIM
    from him_supermimic.himlab import RecordingHIMLabEnv, make_flat_env_cfg
    from him_supermimic.imagination import imagine
    from him_supermimic.replay import EpisodeReplay
    from him_supermimic.termination import heuristic_survival

    if not Path(upstream_env.__file__).resolve().is_relative_to(HIM_ROOT):
        raise RuntimeError("Expected the MyHIMLoco submodule environment")
    plan = json.loads((ROOT / "him_supermimic/configs/wblwbl_dog_v1.plan.json").read_text())
    legacy, training = load_config(task="wblwbl_dog_v1", stage=3, num_envs=args.num_envs)
    profile = plan["controlled_flat"]
    for term in profile["omitted_terms"]:
        setattr(legacy.rewards.scales, term, 0.0)
    for name, value in profile["domain_rand_overrides"].items():
        setattr(legacy.domain_rand, name, value)
    legacy.commands.curriculum = profile["command_curriculum"]
    joints = plan["contract"]["policy_joint_names"]
    effective_config = class_to_dict(legacy)
    task = FlatTask(effective_config, joints)
    out = args.output
    cfg = make_flat_env_cfg(legacy, args.device, out / "asset_cache")
    cfg.seed = args.seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.ground_usd is None:
        env = RecordingHIMLabEnv(cfg)
    else:
        ground_factory = upstream_env.sim_utils.GroundPlaneCfg
        with patch.object(upstream_env.sim_utils, "GroundPlaneCfg",
                          lambda **kw: ground_factory(usd_path=str(args.ground_usd), **kw)):
            env = RecordingHIMLabEnv(cfg)
    try:
        if env.core.dof_names != joints or abs(env.step_dt - task.dt) > 1e-8:
            raise RuntimeError("WBLWBL joint order or control timestep mismatch")
        adapter = LegacyRunnerAdapter(env)
        # Match the PPO runner's architecture without constructing a PPO optimizer or runner.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(args.seed)
            legacy_policy = HIMActorCritic(adapter.num_obs, adapter.num_privileged_obs,
                                           adapter.num_one_step_obs, adapter.num_actions,
                                           **training["policy"])
        policy = DifferentiableHIM(legacy_policy).to(env.device)
        del legacy_policy
        model = WorldModel().to(env.device)
        actor_opt = torch.optim.Adam(policy.actor.parameters(), lr=args.actor_lr)
        model_opt = torch.optim.Adam(model.parameters(), lr=args.model_lr)
        rng = torch.Generator(device=env.device).manual_seed(args.seed + 1)
        replay = EpisodeReplay(args.replay_capacity, env.num_envs, fields=ONLINE_FIELDS)
        with torch.inference_mode():
            adapter.reset()
        env.replay = replay
        std = training["policy"]["init_noise_std"] if args.exploration_std is None else args.exploration_std
        initial = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        initial_hash = hashlib.sha256(b"".join(v.numpy().tobytes() for v in initial.values())).hexdigest()
        torch.save(initial, out / "initial_policy.pt")
        (out / "config.json").write_text(json.dumps(
            {"args": vars(args), "legacy": effective_config, "training": training,
             "profile": "core_reward_v1", "initial_policy_sha256": initial_hash, "exploration_std": std,
             "termination_proxy": "height_and_tilt"}, indent=2, default=str) + "\n")

        def sample(starts):
            return starts[torch.randint(len(starts), (args.batch_size,), generator=rng, device=env.device)]

        def batch(starts):
            steps, ids = starts.unbind(-1)
            slots = (steps[:, None] + torch.arange(HORIZON, device=env.device)) % replay.capacity
            data = replay.data
            return (data["state"][slots, ids[:, None]], data["next_state_pre_reset"][slots, ids[:, None]],
                    data["executed_action"][slots, ids[:, None]], data["prev_action"][slots, ids[:, None]],
                    torch.cat([data[key][slots, ids[:, None]] for key in CONTEXT_FIELDS], -1))

        def validate(starts):
            with torch.no_grad():
                states, target, action, previous, ctx = batch(sample(starts))
                prediction = model.rollout(states[:, 0], action, previous, ctx)
                persistence = states[:, :1].expand_as(target)
                ratio = []
                for component in (slice(7, 9), slice(12, 13)):
                    mse = (prediction[..., component] - target[..., component]).square().mean()
                    baseline = (persistence[..., component] - target[..., component]).square().mean()
                    ratio.append((mse / baseline.clamp_min(1e-10)).item())
                return ratio

        def update_actor(rows):
            ctx = torch.cat([rows[key] for key in CONTEXT_FIELDS], -1)
            seed = (rows["state"], rows["actor_history"], rows["command"], ctx,
                    rows["prev_action"], rows["prev_prev_action"], rows["episode_step"])
            policy.estimator.requires_grad_(False)
            model.requires_grad_(False)
            bounds, _ = task.observation_noise_bounds(seed[0])
            noise = (2 * torch.rand((len(seed[0]), HORIZON, 45), device=env.device, generator=rng) - 1) * bounds
            with torch.no_grad():
                reference = imagine(policy, model, task, seed, HORIZON, observation_noise=noise)
                before = reference.actions
                histories = rows["actor_history"]
                initial_action = task.clip_action(policy(histories))
            weights = torch.cat((torch.ones_like(before[:, :1, 0]),
                                 heuristic_survival(reference.states[:, 1:-1]).cumprod(1)), 1)
            discount = GAMMA ** torch.arange(HORIZON, device=env.device)
            actor_before = deepcopy(policy.actor.state_dict())
            opt_before = deepcopy(actor_opt.state_dict())
            # Hold the reference continuation fixed during the local optimization.
            grad = 0.0
            for _ in range(2):
                traj = imagine(policy, model, task, seed, HORIZON, observation_noise=noise)
                objective = (traj.rewards * weights * discount).sum(1).mean()
                objective -= args.anchor_weight * (traj.actions - before).square().mean()
                grad = optimize(-objective, actor_opt, policy.actor.parameters())
            with torch.no_grad():
                shift = (task.clip_action(policy(histories)) - initial_action).abs().flatten().quantile(0.95).item()
            accepted = shift <= args.max_action_p95
            if not accepted:
                policy.actor.load_state_dict(actor_before)
                actor_opt.load_state_dict(opt_before)
            return {"actor_grad_norm": grad, "action_shift_p95": shift, "actor_update_accepted": accepted}

        normalized = False
        actor_updates = 0
        eval_transitions = 0
        reset_transitions = env.num_envs
        report = {}
        with SummaryWriter(str(out / "tensorboard")) as writer, (out / "progress.jsonl").open("x") as log:
            for iteration in range(1, args.iterations + 1):
                with torch.inference_mode():
                    reward_sum = torch.zeros((), device=env.device)
                    for _ in range(args.collect_steps):
                        action = policy(adapter.get_observations())
                        noise = torch.randn(action.shape, device=env.device, generator=rng) * std
                        reward_sum += adapter.step(action + noise)[2].sum()
                transitions = iteration * env.num_envs * args.collect_steps
                report = {"iteration": iteration, "training_transitions": transitions,
                          "real_reward_per_step": reward_sum.item() / (env.num_envs * args.collect_steps)}
                if replay.size >= HORIZON:
                    starts = replay.valid_starts(HORIZON)
                    # Reserve complete environment streams so a held-out episode cannot leak across windows.
                    holdout = starts[:, 1].remainder(4) == 0
                    train, valid = starts[~holdout], starts[holdout]
                    report.update(train_windows=len(train), validation_windows=len(valid))
                    if len(train) and len(valid):
                        states, target, actions, previous, ctx = batch(sample(train))
                        if not normalized:
                            model.fit_normalization(states, target, actions, previous, ctx)
                            normalized = True
                        model.requires_grad_(True)
                        for _ in range(args.model_updates):
                            states, target, actions, previous, ctx = batch(sample(train))
                            prediction = model.rollout(states[:, 0], actions, previous, ctx)
                            loss = (state_difference(prediction, target) / model.delta_std).square().mean()
                            optimize(loss, model_opt, model.parameters())
                        linear, yaw = validate(valid)
                        report.update(world_model_loss=loss.item(), linear_mse_ratio=linear, yaw_mse_ratio=yaw)
                        if iteration >= args.cold_start_min and max(linear, yaw) < 0.9:
                            pairs = replay.valid_starts(1)
                            ids = pairs[:, 1]
                            ticks = replay.data["episode_step"][pairs[:, 0] % replay.capacity, ids]
                            eligible = pairs[(ids.remainder(4) != 0) &
                                             ((ticks.remainder(task.command_period) + HORIZON) < task.command_period) &
                                             (ticks + HORIZON < int(env.core.max_episode_length))]
                            if len(eligible):
                                step, ids = sample(eligible).unbind(-1)
                                rows = {k: v[step % replay.capacity, ids] for k, v in replay.data.items()}
                                policy.estimator.requires_grad_(True)
                                estimator_loss, swap_loss = policy.estimator.update(
                                    rows["actor_history"], rows["next_critic_obs_pre_reset"], lr=args.estimator_lr)
                                if not np.isfinite([estimator_loss, swap_loss]).all():
                                    raise ValueError("Nonfinite HIM auxiliary loss")
                                report.update(estimator_loss=estimator_loss, swap_loss=swap_loss)
                                report.update(update_actor(rows))
                                actor_updates += int(report["actor_update_accepted"])
                report["accepted_actor_updates"] = actor_updates
                if not actor_updates and iteration >= args.cold_start_max:
                    report["stop_reason"] = "no_accepted_update_before_cold_start_budget"
                finished = iteration == args.iterations or "stop_reason" in report
                if finished or iteration % args.eval_every == 0:
                    suite, _, _ = schedules(env, 801)
                    # Evaluation resets all environments. Discard prior windows on return to training.
                    with torch.random.fork_rng(devices=[torch.device(env.device).index or 0]
                                               if str(env.device).startswith("cuda") else []):
                        np_state = np.random.get_state()
                        try:
                            report["evaluation"] = {name: evaluation(env, adapter, policy, args.eval_steps,
                                                                      seeds=(801,), schedule=schedule)
                                                    for name, schedule in suite.items()}
                        finally:
                            np.random.set_state(np_state)
                    eval_transitions += sum(v["transitions"] for v in report["evaluation"].values())
                    if not finished:
                        env.replay = None
                        with torch.inference_mode():
                            adapter.reset()
                        reset_transitions += env.num_envs
                        replay = EpisodeReplay(args.replay_capacity, env.num_envs, fields=ONLINE_FIELDS)
                        env.replay = replay
                    torch.save({"iteration": iteration, "model_state_dict": policy.state_dict(),
                                "world_model_state_dict": model.state_dict(),
                                "actor_optimizer_state_dict": actor_opt.state_dict(),
                                "estimator_optimizer_state_dict": policy.estimator.optimizer.state_dict(),
                                "world_model_optimizer_state_dict": model_opt.state_dict()},
                               out / f"checkpoint_{iteration:04d}.pt")
                report.update(elapsed_seconds=time.monotonic() - start, evaluation_transitions=eval_transitions,
                              reset_transitions=reset_transitions)
                for key, value in report.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(key, value, iteration)
                for name, metrics in report.get("evaluation", {}).items():
                    for key in ("reward_per_step", "linear_rmse_m_s", "yaw_rmse_rad_s", "falls_per_robot_minute"):
                        writer.add_scalar(f"evaluation/{name}/{key}", metrics[key], iteration)
                line = json.dumps(report, allow_nan=False)
                log.write(line + "\n")
                log.flush()
                print(json.dumps({k: v for k, v in report.items() if k != "evaluation"}), flush=True)
                if finished:
                    break
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    finally:
        env.close()


def main():
    start = time.monotonic()
    from isaaclab.app import AppLauncher

    args = arguments()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    app = AppLauncher(args).app
    try:
        run(args, start)
    finally:
        app.close()


if __name__ == "__main__":
    main()
