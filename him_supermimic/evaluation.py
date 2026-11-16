"""Shared pre-reset physics evaluation for PPO and A5."""

EVAL_SEEDS = (701, 702, 703)
EVAL_FIELDS = ("command_pre_reset", "next_state_pre_reset", "reward", "terminated", "truncated", "reward_terms", "episode_step")


def set_command(env, schedule, step):
    if schedule is None:
        return
    env.command_override = schedule(step)
    core = env.core
    core.commands[:, :3] = env.command_override
    core.obs_buf[:, :3] = (env.command_override * core.commands_scale).clamp(
        -core.cfg.normalization.clip_observations, core.cfg.normalization.clip_observations)


def reset_episode_batch(env, adapter, seed, schedule=None):
    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    env.command_override = None
    core = env.core
    for buffer in (core.obs_buf, core.privileged_obs_buf, core.actions, core.last_actions,
                   core.last_last_actions, core.commands):
        buffer.zero_()
    env.pending_force.zero_()
    set_command(env, schedule, 0)
    adapter.reset()


def evaluation(env, adapter, policy, steps, *, seeds=EVAL_SEEDS, schedule=None):
    import torch

    from him_supermimic.replay import EpisodeReplay

    prior_replay = env.replay
    results = []
    try:
        with torch.no_grad():
            for seed in seeds:
                env.replay = None
                reset_episode_batch(env, adapter, seed, schedule)
                env.replay = EpisodeReplay(1, env.num_envs, fields=EVAL_FIELDS)
                totals = torch.zeros(7 + len(env.reward_term_names), dtype=torch.float64, device=env.device)
                for step in range(steps):
                    set_command(env, schedule, step)
                    adapter.step(policy(adapter.get_observations()))
                    command, state, reward, terminated, truncated, terms, episode_step = (env.replay.data[key][0] for key in EVAL_FIELDS)
                    totals[:7] += torch.stack(((state[..., 7:9] - command[..., :2]).square().sum(),
                                           (state[..., 12] - command[..., 2]).square().sum(),
                                           reward.sum(), terminated.sum(), truncated.sum(),
                                           ((episode_step + 1) * terminated).sum(),
                                           ((episode_step + 1) * truncated).sum()))
                    totals[7:] += terms.sum(dim=0)
                if not torch.isfinite(totals).all():
                    raise ValueError("Nonfinite physical evaluation")
                linear_sq, yaw_sq, mean_reward, falls, timeouts, fall_steps, timeout_steps, *components = totals.tolist()
                count = steps * env.num_envs
                linear_sq /= count
                yaw_sq /= count
                falls = int(falls)
                results.append({"seed": seed, "linear_mse": linear_sq, "yaw_mse": yaw_sq,
                                "reward_per_step": mean_reward / count, "falls": falls, "timeouts": int(timeouts),
                                "mean_fall_episode_seconds": fall_steps * env.step_dt / falls if falls else None,
                                "mean_timeout_episode_seconds": timeout_steps * env.step_dt / timeouts if timeouts else None,
                                "reward_terms_per_step": dict(zip(env.reward_term_names,
                                    (value / count for value in components), strict=True))})
    finally:
        env.replay = prior_replay
        env.command_override = None
    count = steps * env.num_envs * len(results)
    falls = sum(row["falls"] for row in results)
    return {
        "per_seed": results,
        "reward_terms_per_step": {
            name: sum(row["reward_terms_per_step"][name] for row in results) / len(results)
            for name in env.reward_term_names
        },
        "transitions": count, "timeouts": sum(row["timeouts"] for row in results),
        "linear_rmse_m_s": (sum(row["linear_mse"] for row in results) / len(results)) ** 0.5,
        "yaw_rmse_rad_s": (sum(row["yaw_mse"] for row in results) / len(results)) ** 0.5,
        "reward_per_step": sum(row["reward_per_step"] for row in results) / len(results),
        "falls": falls, "falls_per_robot_minute": falls * 60 / (count * env.step_dt),
    }


def schedules(env, seed):
    import torch

    ranges = env.core.command_ranges

    def signed(key, sign):
        return 0.6 * ranges[key][int(sign > 0)]

    x = signed("lin_vel_x", 1)
    y = [signed("lin_vel_y", sign) for sign in (-1, 1)]
    yaw = [signed("ang_vel_yaw", sign) for sign in (-1, 1)]
    commands = {
        "stand": [(0, 0, 0)], "forward": [(x, 0, 0)],
        "lateral": [(0, v, 0) for v in y], "turn": [(0, 0, v) for v in yaw],
        "arc": [(x, 0, v) for v in yaw], "step": [(0, 0, 0), (x, 0, 0), (0, 0, yaw[1])],
    }
    generator = torch.Generator(device=env.device).manual_seed(seed)
    bounds = torch.tensor([ranges[key] for key in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")],
                          device=env.device)
    # Keep the random suite within moderate training speeds, with zero lateral command at high vx.
    random = bounds[:, 0] + torch.rand((32, env.num_envs, 3), generator=generator,
                                       device=env.device) * (bounds[:, 1] - bounds[:, 0])
    random[..., 1].masked_fill_(random[..., 0].abs() >= 1.0, 0)
    random[..., :2] *= random[..., :2].norm(dim=-1, keepdim=True) > 0.2
    banks = {name: torch.tensor(values, device=env.device) for name, values in commands.items()}
    indices = torch.arange(env.num_envs, device=env.device)
    result = {name: (lambda _, values=values: values[indices % len(values)])
              for name, values in banks.items() if name != "step"}
    result["step"] = lambda step: banks["step"][(step // 200) % 3].expand(env.num_envs, -1)
    result["random"] = lambda step: random[(step // 200) % len(random)]
    return result, commands, random.cpu().tolist()
