#!/usr/bin/env python3
"""HIM-PPO with an AMP transition reward learned from WBLWBL motion clips."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HIM = ROOT / "third_party/MyHIMLoco"
sys.path[:0] = [str(ROOT), str(HIM / "isaaclab_migration"), str(HIM / "rsl_rl")]


def run(args):
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from himloco_lab.config import load_config, class_to_dict
    from himloco_lab.adapter import LegacyRunnerAdapter
    from rsl_rl.runners import HIMOnPolicyRunner
    from him_supermimic.himlab import RecordingHIMLabEnv, make_flat_env_cfg
    from him_supermimic.replay import EpisodeReplay
    from him_supermimic.evaluation import evaluation

    legacy, training = load_config("wblwbl_dog_v1", args.num_envs, stage=3)
    plan = json.loads((ROOT / "him_supermimic/configs/wblwbl_dog_v1.plan.json").read_text())
    for name in plan["controlled_flat"]["omitted_terms"]:
        setattr(legacy.rewards.scales, name, 0.0)
    for name, value in plan["controlled_flat"]["domain_rand_overrides"].items():
        setattr(legacy.domain_rand, name, value)
    legacy.commands.curriculum = False
    legacy.commands.heading_command = False
    legacy.commands.ranges.lin_vel_x = [-0.5, 0.8]
    legacy.commands.ranges.lin_vel_y = [-0.2, 0.2]
    legacy.commands.ranges.ang_vel_yaw = [-1.0, 1.0]
    training["runner"]["num_steps_per_env"] = args.rollout_steps
    cfg = make_flat_env_cfg(legacy, args.device, args.output / "asset_cache")
    cfg.seed = args.seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = RecordingHIMLabEnv(cfg)
    try:
        def resample_commands(ids):
            for column, bounds in enumerate(((-0.5, 0.8), (-0.2, 0.2), (-1.0, 1.0))):
                env.core.commands[ids, column] = torch.empty(len(ids), device=env.device).uniform_(*bounds)
            env.core.commands[ids, :2] *= (env.core.commands[ids, :2].norm(dim=-1) > 0.2)[:, None]
        env.core._resample_commands = resample_commands
        adapter = LegacyRunnerAdapter(env)
        runner = HIMOnPolicyRunner(adapter, training, log_dir=str(args.output), device=env.device)
        policy = runner.alg.actor_critic
        joints = env.core.dof_names
        # AMP observes joint configuration and velocity in the policy joint order.
        # Clip endpoints never form a transition across motion files.
        clips = []
        manifests = []
        for group in ("walk", "turn"):
            group_clips = []
            for manifest in sorted((args.motions / group).glob("*/train.json")):
                metadata = json.loads(manifest.read_text())
                order = [metadata["joint_names"].index(name) for name in joints]
                fps = float(metadata["source_fps"])
                for name in metadata["files"]:
                    path = (manifest.parent / metadata.get("root", ".") / name).resolve()
                    motion = np.loadtxt(path, delimiter=",")
                    if motion.ndim != 2 or motion.shape[1] != 19 or len(motion) < 4 or not np.isfinite(motion).all():
                        raise ValueError(f"Invalid reference {path}")
                    source_t = np.arange(len(motion)) / fps
                    times = np.arange(0, source_t[-1], env.step_dt)
                    q = np.stack([np.interp(times, source_t, motion[:, 7 + index]) for index in order], -1)
                    if len(q) < 4:
                        raise ValueError(f"Reference too short {path}")
                    dq = np.gradient(q, env.step_dt, axis=0)
                    features = torch.tensor(np.concatenate((q, dq), -1), dtype=torch.float32, device=env.device)
                    group_clips.append(features)
                    manifests.append(str(path))
            if not group_clips:
                raise ValueError(f"No {group} reference clips")
            clips.append(group_clips)
        all_features = torch.cat([clip for group in clips for clip in group])
        mean = all_features.mean(0)
        std = all_features.std(0, unbiased=False).clamp_min(0.05)

        def normalize(x):
            return (x - mean) / std

        module_path = ROOT / "third_party/AMP/source/robot_lab/robot_lab/third_party/rsl_rl_amp/algorithms/amp_discriminator.py"
        spec = importlib.util.spec_from_file_location("upstream_amp_discriminator", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        discriminator = module.AMPDiscriminator(48, args.amp_weight, [256, 128], env.device)
        disc_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
        rng = torch.Generator(device=env.device).manual_seed(args.seed + 41)

        reference_pairs, weights = [], []
        for group in clips:
            for clip in group:
                pairs = torch.cat((normalize(clip[:-1]), normalize(clip[1:])), -1)
                reference_pairs.append(pairs)
                weights.append(torch.full((len(pairs),), 1 / (2 * len(group) * len(pairs)), device=env.device))
        reference_pairs = torch.cat(reference_pairs)
        weights = torch.cat(weights)

        def expert_pairs(count):
            return reference_pairs[torch.multinomial(weights, count, replacement=True, generator=rng)]

        config = {"algorithm": "amp_hie_ppo", "legacy": class_to_dict(legacy), "training": training, "args": vars(args),
                  "motion_files": manifests, "amp_features": "policy_order_joint_position_velocity",
                  "reward": "task_reward + amp_weight * dt * style_score"}
        (args.output / "config.json").write_text(json.dumps(config, indent=2, default=str))
        env.replay = EpisodeReplay(1, env.num_envs,
                                   fields=("state", "next_state_pre_reset", "reward_terms", "terminated"))
        obs, privileged = adapter.get_observations(), adapter.get_privileged_observations()
        torch.save({"model_state_dict": policy.state_dict()}, args.output / "checkpoint_0000.pt")
        episode_sums = torch.zeros(env.num_envs, len(env.reward_term_names) + 3, device=env.device)
        episode_lengths = torch.zeros(env.num_envs, device=env.device)
        start = time.monotonic()
        with SummaryWriter(str(args.output / "tensorboard")) as writer, (args.output / "progress.jsonl").open("w") as log:
            for iteration in range(1, args.iterations + 1):
                iteration_start = time.monotonic()
                negatives = []
                completed_returns, completed_lengths = [], []
                fall_count = 0
                sums = torch.zeros(len(env.reward_term_names) + 3, device=env.device)
                for _ in range(args.rollout_steps):
                    with torch.no_grad():
                        action = runner.alg.act(obs, privileged)
                        obs, privileged, task_reward, done, info, ids, terminal_obs = adapter.step(action)
                        row = {k: v[0] for k, v in env.replay.data.items()}
                        pair = torch.cat((normalize(row["state"][:, 13:37]),
                                          normalize(row["next_state_pre_reset"][:, 13:37])), -1)
                        score = discriminator(pair).squeeze(-1)
                        style = (1 - 0.25 * (score - 1).square()).clamp_min(0)
                        amp_reward = args.amp_weight * env.step_dt * style
                        reward = task_reward + amp_reward
                        next_obs = privileged.clone()
                        next_obs[ids] = terminal_obs
                        runner.alg.process_env_step(reward, done, info, next_obs)
                        negatives.append(pair.detach())
                        episode_sums += torch.cat((row["reward_terms"], torch.stack((task_reward, amp_reward, reward), -1)), -1)
                        episode_lengths += 1
                        if done.any():
                            completed_returns.append(episode_sums[done].clone())
                            completed_lengths.append(episode_lengths[done].clone())
                            episode_sums[done] = 0
                            episode_lengths[done] = 0
                        fall_count += int(row["terminated"].sum())
                        sums[:len(env.reward_term_names)] += row["reward_terms"].mean(0)
                        sums[-3:] += torch.stack((task_reward.mean(), amp_reward.mean(), reward.mean()))
                rollout_seconds = time.monotonic() - iteration_start
                with torch.no_grad():
                    runner.alg.compute_returns(privileged)
                losses = runner.alg.update()
                negative = torch.cat(negatives)
                for _ in range(args.discriminator_updates):
                    expert = expert_pairs(args.amp_batch)
                    index = torch.randint(len(negative), (args.amp_batch,), generator=rng, device=env.device)
                    fake = negative[index]
                    disc_loss = 0.5 * ((discriminator(expert) - 1).square().mean() +
                                       (discriminator(fake) + 1).square().mean())
                    penalty = discriminator.compute_grad_pen(expert[:, :24].clone(), expert[:, 24:].clone())
                    disc_opt.zero_grad()
                    (disc_loss + penalty).backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0, error_if_nonfinite=True)
                    disc_opt.step()
                record = {"iteration": iteration, "elapsed_seconds": time.monotonic() - start,
                          "training_transitions": iteration * args.num_envs * args.rollout_steps}
                names = [f"reward_terms/{name}" for name in env.reward_term_names]
                names += ["reward/task", "reward/amp", "reward/total"]
                for name, value in zip(names, (sums / args.rollout_steps).tolist()):
                    writer.add_scalar(name, value, iteration)
                    record[name] = value
                for name, value in zip(("value", "surrogate", "estimation", "swap"), losses):
                    writer.add_scalar(f"loss/{name}", value, iteration)
                writer.add_scalar("loss/discriminator", disc_loss.item(), iteration)
                writer.add_scalar("loss/gradient_penalty", penalty.item(), iteration)
                if completed_returns:
                    for name, value in zip(names, torch.cat(completed_returns).mean(0).tolist()):
                        writer.add_scalar("episode/" + name, value, iteration)
                    writer.add_scalar("episode/length_steps", torch.cat(completed_lengths).mean().item(), iteration)
                writer.add_scalar("train/falls", fall_count, iteration)
                writer.add_scalar("train/action_std", policy.std.mean().item(), iteration)
                writer.add_scalar("amp/expert_score", discriminator(expert).mean().item(), iteration)
                writer.add_scalar("amp/policy_score", discriminator(fake).mean().item(), iteration)
                record["rollout_seconds"] = rollout_seconds
                record["train_iteration_seconds"] = time.monotonic() - iteration_start
                writer.add_scalar("performance/transitions_per_second", args.num_envs * args.rollout_steps / record["train_iteration_seconds"], iteration)
                writer.add_scalar("performance/elapsed_seconds", record["elapsed_seconds"], iteration)
                writer.add_scalar("performance/training_transitions", record["training_transitions"], iteration)
                if iteration % args.save_every == 0 or iteration == args.iterations:
                    torch.save({"model_state_dict": policy.state_dict(), "iteration": iteration,
                                "optimizer_state_dict": runner.alg.optimizer.state_dict(),
                                "estimator_optimizer": policy.estimator.optimizer.state_dict(),
                                "discriminator": discriminator.state_dict(), "normalization_mean": mean,
                                "normalization_std": std, "discriminator_optimizer": disc_opt.state_dict()},
                               args.output / f"checkpoint_{iteration:04d}.pt")
                    record["evaluation"] = {}
                    torch_rng, cuda_rng, numpy_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state()
                    for name, command in (("forward", (0.5, 0., 0.)), ("left", (0.2, 0., 0.6)),
                                          ("right", (0.2, 0., -0.6))):
                        schedule = lambda step, c=command: torch.tensor(c, device=env.device).expand(env.num_envs, -1)
                        metrics = evaluation(env, adapter, policy.act_inference, args.eval_steps,
                                             seeds=(801,), schedule=schedule)
                        record["evaluation"][name] = metrics
                        for key in ("linear_rmse_m_s", "yaw_rmse_rad_s", "falls_per_robot_minute", "reward_per_step"):
                            writer.add_scalar(f"evaluation/{name}/{key}", metrics[key], iteration)
                    torch.set_rng_state(torch_rng)
                    torch.cuda.set_rng_state_all(cuda_rng)
                    np.random.set_state(numpy_rng)
                    obs, privileged = adapter.reset()
                    episode_sums.zero_()
                    episode_lengths.zero_()
                log.write(json.dumps(record) + "\n")
                log.flush()
                writer.flush()
                print(json.dumps(record), flush=True)
    finally:
        env.close()


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motions", type=Path, default=ROOT / "amp_supermimic/data/wblwbl")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--amp-weight", type=float, default=1.0)
    parser.add_argument("--amp-batch", type=int, default=256)
    parser.add_argument("--discriminator-updates", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=511)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.num_envs, args.iterations, args.rollout_steps, args.amp_batch,
           args.discriminator_updates, args.save_every, args.eval_steps) < 1:
        parser.error("Counts must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    app = AppLauncher(args).app
    try:
        run(args)
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
