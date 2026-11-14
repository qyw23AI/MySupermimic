"""Exercise reset and time alignment through real Isaac Lab steps."""

import torch

from .replay import EpisodeReplay


@torch.inference_mode()
def verify_replay(env, adapter, policy, *, replay_path=None):
    adapter.reset()
    replay = EpisodeReplay(24, env.num_envs)
    env.replay = replay
    checked = {"terminations": 0, "timeouts": 0, "command_resamples": 0, "partial_resets": 0}
    errors = {"state_continuity": 0.0, "reward_terms": 0.0, "terminal_joint_position": 0.0}
    timeout_step, fall_step, command_step = 17, 21, 26
    command_env = env.num_envs - 1
    last = None
    tested_fall = False
    try:
        for step in range(40):
            core = env.core
            if step == timeout_step:
                env.episode_length_buf[0] = core.max_episode_length
            if step == fall_step:
                # Place one freshly reset robot upside down with its torso near the plane.
                # The actual contact sensor must cause termination.
                ids = torch.tensor([1], device=env.device)
                env._reset_idx(ids)
                root = core.root_states[ids].clone()
                root[:, 2] = 0.19
                root[:, 3:7] = root.new_tensor([1, 0, 0, 0])
                root[:, 7:] = 0
                root[:, 9] = -1.0
                env.write_root(root, ids)
            if step == command_step:
                period = int(core.cfg.commands.resampling_time / env.step_dt)
                env.episode_length_buf[command_env] = period - 1
            action = policy(adapter.get_observations()).clone()
            if step == timeout_step:
                action[0, 0] = 2 * core.cfg.normalization.clip_actions

            history = adapter.get_observations().clone()
            obs, critic, reward, done, _, ids, terminal = adapter.step(action)
            rows = replay.data
            row = {name: value[step % replay.capacity] for name, value in rows.items()}
            torch.testing.assert_close(row["actor_history"], history, atol=0, rtol=0)
            torch.testing.assert_close(row["raw_action"], action, atol=0, rtol=0)
            clip = core.cfg.normalization.clip_actions
            torch.testing.assert_close(row["executed_action"], action.clamp(-clip, clip), atol=0, rtol=0)
            torch.testing.assert_close(row["actor_history_after_step"], obs, atol=0, rtol=0)
            # Each actuator received a prefix of previous actions followed by current actions.
            substeps = torch.arange(env.cfg.decimation, device=env.device)
            delays = torch.arange(env.cfg.decimation, device=env.device)
            candidates = torch.where(
                (substeps[None, :] >= delays[:, None])[None, :, :, None],
                row["executed_action"][:, None, None, :],
                row["prev_action"][:, None, None, :],
            )
            matches = torch.isclose(row["substep_actions"][:, None], candidates, atol=1e-6, rtol=1e-6)
            assert matches.all(dim=-1).all(dim=-1).any(dim=-1).all()
            torch.testing.assert_close(row["reward"], reward, atol=0, rtol=0)
            torch.testing.assert_close(row["terminated"] | row["truncated"], done)
            torch.testing.assert_close(row["next_critic_obs_pre_reset"][ids], terminal, atol=0, rtol=0)
            torch.testing.assert_close(row["next_critic_obs_pre_reset"][~done], critic[~done], atol=0, rtol=0)
            error = (row["reward_terms"].sum(dim=-1) - reward).abs().max().item()
            errors["reward_terms"] = max(errors["reward_terms"], error)
            # CUDA reduction and the upstream sequential sum accumulate in different orders.
            torch.testing.assert_close(row["reward_terms"].sum(dim=-1), reward, atol=1e-6, rtol=1e-6)
            if last is not None:
                same = row["episode_id"] == last["episode_id"]
                if same.any():
                    error = (row["state"][same] - last["next_state_pre_reset"][same]).abs().max().item()
                    errors["state_continuity"] = max(errors["state_continuity"], error)
                    assert error < 1e-5, error
                    torch.testing.assert_close(row["actor_history"][same], last["actor_history_after_step"][same])
                    torch.testing.assert_close(row["prev_action"][same], last["executed_action"][same])

            checked["terminations"] += int(row["terminated"].sum())
            checked["timeouts"] += int(row["truncated"].sum())
            checked["command_resamples"] += int(row["command_resampled"].sum())
            checked["partial_resets"] += int(done.any() and not done.all())
            if step == timeout_step:
                assert row["truncated"][0]
            if step >= fall_step and not tested_fall and row["terminated"][1]:
                tested_fall = True
                assert row["terminal_contact"][1] > 1
                pre = row["next_state_pre_reset"][1]
                post = env.physical_state()[1]
                assert (pre[:3] - post[:3]).norm().item() > 0.1
                # Terminal joint observation is independently computed by upstream
                # PPO; the first leg's q block must agree up to configured sensor noise.
                from_critic = terminal[ids == 1, 9:21].squeeze(0)
                expected = (pre[13:25] - core.default_dof_pos) * core.obs_scales.dof_pos
                joint_error = (from_critic - expected).abs().max().item()
                errors["terminal_joint_position"] = joint_error
                assert joint_error <= (float(core.noise_scale_vec[9:21].amax()) if core.add_noise else 0) + 1e-5
            if step == command_step:
                assert row["command_resampled"][command_env]
                # The legacy dead zone may leave vx/vy at zero after a valid resample.
                # WBLWBL's separately sampled heading is not affected by that dead zone.
                assert row["command"][command_env, 3] != row["command_pre_reset"][command_env, 3]
            last = row

        assert tested_fall, "The inverted robot did not produce a contact termination"
        horizon = 8
        starts = replay.valid_starts(horizon)
        rows = replay.ordered()
        # Independent enumeration over the retained rows, including rollover.
        retained = replay.total_steps - replay.size
        expected = []
        episode = rows["episode_id"].tolist()
        end = (rows["terminated"] | rows["truncated"]).tolist()
        command = rows["command_resampled"].tolist()
        for env_id in range(env.num_envs):
            for start in range(replay.size - horizon + 1):
                indices = range(start, start + horizon)
                if (all(episode[i][env_id] == episode[start][env_id] and not command[i][env_id] for i in indices)
                    and not any(end[i][env_id] for i in range(start, start + horizon - 1))):
                    expected.append((start + retained, env_id))
        assert set(map(tuple, starts.tolist())) == set(expected)
        batch = replay.sample(64, horizon)
        assert tuple(batch["states"].shape) == (64, 9, 37)
        assert tuple(batch["executed_action"].shape) == (64, 8, 12)
        torch.testing.assert_close(batch["states"][:, :-1], batch["state"], atol=1e-5, rtol=0)
        assert all(checked.values()), checked
        if replay_path is not None:
            torch.save({
                "transitions": {name: value.cpu() for name, value in rows.items()},
                "schema_version": 1,
                "reward_term_names": env.reward_term_names,
            }, replay_path)
        return {"status": "passed", **checked, "retained_steps": replay.size,
                "simulated_steps": replay.total_steps, "eligible_8_step_windows": len(starts),
                "max_abs_error": errors}
    finally:
        env.replay = None
