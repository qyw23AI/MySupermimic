"""Compare one-step model directional derivatives with paired Isaac Lab steps."""
import json

import torch

from .dynamics import CONTEXT_FIELDS, WorldModel
from .flat_task import FlatTask, rotate_xyzw
from .replay import EpisodeReplay


def probe(args, env, config, output):
    rows = torch.load(args.probe_replay, map_location=env.device, weights_only=True)["transitions"]
    source = json.loads((args.probe_replay.parent / "source.json").read_text())
    if source["asset"]["joint_names"] != env.core.dof_names:
        raise ValueError("Probe replay joint order differs from simulator")
    checkpoint = torch.load(args.probe_world_model, map_location=env.device, weights_only=True)
    hidden = checkpoint.get("world_model_hidden", checkpoint.get("hidden"))
    weights = checkpoint.get("world_model_state_dict", checkpoint.get("model_state_dict"))
    model = WorldModel(hidden).to(env.device).eval().requires_grad_(False)
    model.load_state_dict(weights)
    task = FlatTask(config, env.core.dof_names)
    rng = torch.Generator(device=env.device).manual_seed(args.seed)
    count = env.num_envs // 2
    ids = torch.arange(env.num_envs, device=env.device)
    eligible = ~rows["command_resampled"] & ~rows["truncated"] & (rows["episode_step"] > 0)
    report = {}

    for group, mask in (("healthy", ~rows["terminated"]), ("contact_terminal", rows["terminated"])):
        candidates = (eligible & mask).nonzero()
        if not len(candidates):
            report[group] = {"windows": 0}
            continue
        chosen = candidates[torch.randperm(len(candidates), generator=rng, device=env.device)[:count]]
        windows = len(chosen)
        chosen = torch.cat((chosen, chosen[:1].expand(count - windows, -1)))
        t, e = chosen.unbind(1)

        def row(key):
            return rows[key][t, e]

        state = row("state")
        action = row("executed_action")
        previous = row("prev_action")
        older = row("prev_prev_action")
        command = row("command")
        context = torch.cat([row(key) for key in CONTEXT_FIELDS], dim=-1)
        direction = torch.nn.functional.normalize(
            torch.randn(action.shape, device=env.device, generator=rng), dim=-1)

        def metrics(next_state, executed):
            update = task.update_command(command, next_state, row("episode_step"), command)
            reward = sum(task.reward_terms(state, next_state, update, executed, previous, older).values())
            return torch.cat((next_state[:, 7:9], next_state[:, 12:13], reward[:, None]), dim=-1)

        def prediction(a):
            executed = task.clip_action(a)
            return metrics(model(state, executed, previous, context), executed)

        _, directional = torch.autograd.functional.jvp(prediction, action, direction)
        results = {}
        for epsilon in (0.0, 0.02, 0.04):
            with torch.inference_mode():
                env._reset_idx(ids)
                root = state[:, :13].repeat(2, 1)
                root[:, :3] += env.core.env_origins
                root[:, 7:10] = rotate_xyzw(root[:, 3:7], root[:, 7:10])
                root[:, 10:13] = rotate_xyzw(root[:, 3:7], root[:, 10:13])
                env.write_root(root, ids)
                env.robot.write_joint_state_to_sim(state[:, 13:25].repeat(2, 1), state[:, 25:37].repeat(2, 1),
                                                   joint_ids=env.joint_ids, env_ids=ids)
                env.core.commands.copy_(command.repeat(2, 1))
                env.core.last_actions.copy_(previous.repeat(2, 1))
                env.core.last_last_actions.copy_(older.repeat(2, 1))
                env.episode_length_buf.copy_(row("episode_step").repeat(2))
                env.replay = EpisodeReplay(1, env.num_envs, fields=("state", "next_state_pre_reset", "terminated"))
                a = torch.cat((action + epsilon * direction, action - epsilon * direction), dim=0)
                env.step(a)
                actual_start = env.replay.data["state"][0]
                plus, minus = env.replay.data["next_state_pre_reset"][0].chunk(2)
                labels = env.replay.data["terminated"][0].reshape(2, count)
                difference = metrics(plus, task.clip_action(a[:count])) - metrics(minus, task.clip_action(a[count:]))
                entry = {"reconstructed_start_max_abs": (actual_start - state.repeat(2, 1)).abs().max().item(),
                         "physics_pair_rms": difference[:windows].square().mean(0).sqrt().tolist(),
                         "pairs_with_different_terminal": int((labels[0, :windows] != labels[1, :windows]).sum())}
                if epsilon:
                    physical = difference[:windows] / (2 * epsilon)
                    predicted = directional[:windows]
                    entry.update(rmse=(physical - predicted).square().mean(0).sqrt().tolist(),
                                 sign_agreement=(physical.sign() == predicted.sign()).float().mean(0).tolist(),
                                 physics_directional_rms=physical.square().mean(0).sqrt().tolist())
                results[str(epsilon)] = entry
        report[group] = {"windows": windows, "perturbations": results}
    (output / "report.json").write_text(json.dumps(
        {"channels": ["body_vx", "body_vy", "yaw_rate", "reconstructed_reward"], "groups": report},
        indent=2, allow_nan=False) + "\n")
