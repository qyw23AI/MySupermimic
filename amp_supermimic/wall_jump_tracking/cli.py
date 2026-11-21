"""Run the vendored SuperMimic tracker with WBL wall-jump references."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .robot import HEIGHT, JOINTS, LEGS, NAME, THICKNESS, WIDTH, prepare_followup, prepare_library, prepare_robot, register

STAGES = ("first", "second", "full")


def write_json(path, contents):
    path.write_text(json.dumps(contents, indent=2, allow_nan=False) + "\n")


def configure(args):
    asset = (args.work / "wbl_tracking.xml").resolve()
    if not asset.is_file():
        raise FileNotFoundError(f"Run prepare before training: {asset}")
    register(asset)
    return asset


def train(args):
    import torch
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer

    configure(args)
    run = args.output.resolve()
    if run.exists():
        raise FileExistsError(run)
    cfg = TrainConfig(run_name=run.name, log_root=run.parent, robot=NAME, seed=args.seed,
                      device=args.device, tf32=True)
    cfg.data.npz_dir = (args.work / args.stage).resolve(strict=True)
    cfg.env.num_envs = args.envs
    cfg.env.episode_length_s = 3.5
    cfg.env.replay_capacity = 256
    cfg.env.reset_noise = False
    cfg.env.domain_randomization = False
    cfg.env.obs_noise = False
    cfg.env.action_noise_std = args.explore
    cfg.env.sampling_mode = args.sampling
    cfg.env.anchor_obs = True
    cfg.world_model.hidden = (256, 256, 256)
    cfg.policy.encoder_hidden = (256, 128)
    cfg.policy.decoder_hidden = (256, 128)
    cfg.wm_trainer.compile = cfg.policy_trainer.compile = None
    cfg.wm_trainer.batch_size = cfg.policy_trainer.batch_size = args.batch
    cfg.wm_trainer.batches_per_update = 2
    cfg.wm_trainer.window = 6
    cfg.policy_trainer.horizon = args.horizon
    cfg.policy_trainer.critic_epochs = 2
    cfg.policy_trainer.obs_noise = False
    cfg.schedule.warmup_iterations = 32
    cfg.schedule.total_iterations = args.iterations
    cfg.schedule.log_interval = 50
    cfg.schedule.checkpoint_interval = args.save_every
    if args.teacher:
        if args.stage != "first" or args.warm_start:
            raise ValueError("A joint-target teacher is only available to first-stage training")
        cfg.policy_trainer.lr = min(cfg.policy_trainer.lr, 2.e-5)
        cfg.policy_trainer.demonstration_weight = 20.
        cfg.schedule.policy_start_iterations = 180
    if args.resume:
        cfg.resume = args.resume.resolve(strict=True)
    trainer = Trainer(cfg)
    if args.stage == "first" and args.sampling == "return":
        from types import MethodType

        command = trainer.command
        ordinary_sampling = command._sample_frames
        # Train the approach and the push as well as the crest hold. These
        # resets affect collection only. Evaluation always starts at frame zero.
        early = torch.tensor([0, 0, 0, 6, 10, 12, 14, 16], device=args.device)
        clip_start = command.library.clip_start[0]

        def sample_first_jump(_command, count):
            base = ordinary_sampling(count)
            index = torch.randint(len(early), (count,), device=args.device)
            choose_early = torch.rand(count, device=args.device) < .65
            return torch.where(choose_early, clip_start + early[index], base)

        command._sample_frames = MethodType(sample_first_jump, command)
    try:
        if args.teacher:
            warm_start_first_jump(trainer, args.work / "first_joint_targets.npy", clone=args.resume is None)
        if args.warm_start:
            saved = torch.load(args.warm_start, map_location=args.device, weights_only=False)
            if saved["config"]["robot"] != NAME:
                raise ValueError("Expected a WBL wall checkpoint")
            trainer.policy.load_state_dict(saved["policy"], strict=True)
            # The wall, actor, and action/feature contract are identical across stages.
            trainer.wm.load_state_dict(saved["wm"], strict=True)
        trainer.run()
    finally:
        trainer.env.close()


def contact_reader(env):
    """Positive-force contacts only. Crest detections also require the top-face normal."""
    import torch

    wall_sensor = env.scene["wall_feet"]
    floor_sensor = env.scene["floor_feet"]

    def ordered(sensor):
        return [next(i for i, name in enumerate(sensor.primary_names) if name.endswith(leg + "_foot_ball"))
                for leg in LEGS]

    wi, fi = ordered(wall_sensor), ordered(floor_sensor)
    names = [leg + "_foot_ball" for leg in LEGS]
    entity = env.scene["robot"]
    foot_geom_ids, _ = entity.find_geoms(names, preserve_order=True)
    sim_geom_ids = entity.indexing.geom_ids[foot_geom_ids]
    radii = torch.as_tensor(env.sim.mj_model.geom_size[sim_geom_ids.cpu().numpy(), 0],
                            device=env.device, dtype=torch.float32)

    def link_hit(name):
        s = env.scene[name].data
        return ((s.found > 0) & (s.force[..., 0].abs() > 2.)).any(dim=-1)

    def read():
        wall, floor = wall_sensor.data, floor_sensor.data
        wall_loaded = (wall.found[:, wi] > 0) & (wall.force[:, wi, 0].abs() > 2.)
        floor_loaded = (floor.found[:, fi] > 0) & (floor.force[:, fi, 0].abs() > 2.)
        feet = env.sim.data.geom_xpos[:, sim_geom_ids, :]
        bounds = (feet[..., 0].abs() + radii <= THICKNESS / 2) & (
            feet[..., 1].abs() + radii <= WIDTH / 2)
        atop = wall_loaded & (wall.normal[:, wi, 2].abs() > .7) & bounds
        atop &= (feet[..., 2] - radii - HEIGHT).abs() < .018
        bad = link_hit("wall_links") | link_hit("floor_links")
        no_support = ~(wall_loaded | floor_loaded).any(dim=-1) & ~bad
        clear = (feet[..., 0] - radii > THICKNESS / 2).all(dim=-1)
        soles = feet.clone()
        soles[..., 2] -= radii
        rear_load = torch.where(floor_loaded[:, :2], floor.force[:, fi[:2], 0].abs()
                                * floor.normal[:, fi[:2], 2].abs(), 0.).sum(dim=-1)
        return atop, floor_loaded, no_support, bad, clear, soles, rear_load

    return read

class Score:
    """Keep task acceptance independent of the training tracking reward."""
    def __init__(self, count, stage):
        self.stage = stage
        self.top_run = np.zeros(count, dtype=int)
        self.brace_run = np.zeros(count, dtype=int)
        self.max_brace_run = np.zeros(count, dtype=int)
        self.air_run = np.zeros(count, dtype=int)
        self.max_top_run = np.zeros(count, dtype=int)
        self.first_air = np.zeros(count, dtype=bool)
        self.crest = np.zeros(count, dtype=bool)
        self.rear_ready = np.zeros(count, dtype=bool)
        self.second_air = np.zeros(count, dtype=bool)
        self.landed_run = np.zeros(count, dtype=int)
        self.hit = np.zeros(count, dtype=bool)

    def step(self, active, atop, floor, airborne, hit, clear, state):
        # A second-only reset is diagnostic and cannot qualify for AMP export.
        self.hit |= active & hit
        self.air_run = np.where(active & airborne, self.air_run + 1, 0)
        self.first_air |= active & ~self.crest & (self.air_run >= 2)
        pair = atop[:, 2:].all(axis=1)
        self.top_run = np.where(active & pair, self.top_run + 1, 0)
        self.max_top_run = np.maximum(self.max_top_run, self.top_run)
        self.crest |= active & (self.top_run >= 5) & (self.first_air | (self.stage == "second"))
        rear_supported = floor[:, :2].all(axis=1)
        qx, qy = state[:, 4], state[:, 5]
        up = 1. - 2. * (qx**2 + qy**2)
        body_velocity = state[:, 19:22]
        settled = (up > .80) & (np.linalg.norm(body_velocity, axis=-1) < .35)
        braced = active & ~self.hit & pair & rear_supported & settled & (self.first_air | (self.stage == "second"))
        self.brace_run = np.where(braced, self.brace_run + 1, 0)
        self.max_brace_run = np.maximum(self.max_brace_run, self.brace_run)
        self.rear_ready |= self.brace_run >= 5
        self.second_air |= active & self.rear_ready & (self.air_run >= 2)
        landed = self.second_air & clear & floor.all(axis=1) & settled
        self.landed_run = np.where(active & landed, self.landed_run + 1, 0)
        # Stage 1 must be braced at the moment we declare success and hold rear support.
        first = self.brace_run >= 5
        return active & ~self.hit & (first if self.stage == "first" else self.landed_run >= 5)



def warm_start_first_jump(trainer, targets_path, *, clone):
    """Record a verified force-supported teacher rollout and clone its joint targets."""
    from types import MethodType

    import torch

    env, command, policy = trainer.env, trainer.command, trainer.policy
    targets = torch.as_tensor(np.load(targets_path, allow_pickle=False), dtype=torch.float32, device=env.device)
    original_sampler = command._sample_frames
    action_term = env.action_manager.get_term("joint_pos")
    previous_noise = action_term.noise_std
    obs_history, goals, actions = [], [], []
    count = env.num_envs
    success = np.zeros(count, dtype=bool)
    active = np.ones(count, dtype=bool)
    score = Score(count, "first")
    from supermimic.envs.state_io import StateIO

    state_io = StateIO(env)
    read_contacts = contact_reader(env)

    def sample_start(motion, size):
        return motion.library.clip_start[0].expand(size)

    try:
        command._sample_frames = MethodType(sample_start, command)
        action_term.noise_std = 0.
        with torch.no_grad():
            obs, _ = env.reset()
            for target in targets:
                raw = (target - command.robot.data.default_joint_pos) / trainer.spec.action_scale
                obs_history.append(obs["policy"][0].clone())
                goals.append(obs["goal"][0].clone())
                actions.append(raw[0].clone())
                obs, _, term, timeout, _ = env.step(raw)
                atop, floor, air, hit, clear, soles, _ = read_contacts()
                air &= (soles[..., 2] > .003).all(dim=-1)
                passed = score.step(active, *(x.cpu().numpy() for x in (atop, floor, air, hit, clear)),
                                    state_io.read().cpu().numpy())
                success |= passed & ~term.cpu().numpy()
                active &= ~(term | timeout).cpu().numpy()
                if not active[0]:
                    break
    finally:
        command._sample_frames = original_sampler
        action_term.noise_std = previous_noise
    if not success[0]:
        raise RuntimeError("Teacher did not pass the first jump in the training environment")
    policy_obs, goal, action = (torch.stack(items) for items in (obs_history, goals, actions))
    trainer.policy_trainer.set_demonstration(policy_obs, goal, action)
    if not clone:
        return
    optimizer = torch.optim.Adam(policy.parameters(), lr=8.e-4)
    for _ in range(650):
        index = torch.randint(len(action), (128,), device=env.device)
        batch = policy_obs[index]
        prediction = policy(batch + .008 * torch.randn_like(batch), goal[index])
        loss = (prediction - action[index]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print(f"Teacher crossed first wall, stable brace {score.max_brace_run[0] * .02:.2f} s. "
          f"Clone MSE {float(loss.detach()):.6f}", flush=True)
    trainer.save("ckpt_teacher.pt")


def evaluate(args):
    import torch
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.envs import SuperMimicEnvCfgArgs
    from supermimic.envs.tracking_env_cfg import make_env_cfg
    from supermimic.envs.state_io import StateIO
    from mjlab.envs import ManagerBasedRlEnv
    from supermimic.robots import get_robot_spec

    configure(args)
    if args.policy != "pd":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for a learned policy")
        actor, saved = load_policy_from_checkpoint(args.checkpoint, args.device)
        if saved["robot"] != NAME:
            raise ValueError("Checkpoint does not match WBL jump model")
        future, anchor = saved["env"]["future_frames"], saved["env"]["anchor_obs"]
    else:
        actor, future, anchor = None, 10, False
    files = [str((args.work / args.stage / "direct.npz").resolve(strict=True))]
    options = SuperMimicEnvCfgArgs(npz_files=files, robot=NAME, num_envs=args.envs, seed=args.seed,
                                  sampling_mode="start", episode_length_s=3.5, future_frames=future,
                                  anchor_obs=anchor, record_transitions=False, action_noise_std=0.,
                                  reset_noise=False, domain_randomization=False, obs_noise=False)
    cfg = make_env_cfg(options)
    cfg.auto_reset = False
    # Build evaluations in batches; all environments start at the same reference,
    # with optional bounded XY/yaw reset perturbations.
    if args.perturb:
        cmd = cfg.commands["motion"]
        cmd.pose_range.update(x=(-.016, .016), y=(-.012, .012), yaw=(-.025, .025))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        with torch.inference_mode():
            state_io = StateIO(env)
            reference = env.command_manager.get_term("motion")
            scale = get_robot_spec(NAME).action_scale
            obs, _ = env.reset()
            read_contacts = contact_reader(env)
            initial = state_io.read().cpu().numpy()
            history = [initial]
            actions = []
            contact_history = [[x.cpu().numpy() for x in read_contacts()]]
            stop = np.zeros(args.envs, dtype=int)
            success = np.zeros(args.envs, dtype=bool)
            active = np.ones(args.envs, dtype=bool)
            score = Score(args.envs, args.stage)
            for step in range(1, 177):
                action = actor(obs["policy"], obs["goal"]) if actor else (
                    (reference.joint_pos - reference.robot.data.default_joint_pos) / scale
                )
                actions.append(action.detach().cpu().numpy().copy())
                obs, _, terminated, timeout, _ = env.step(action)
                state = state_io.read().cpu().numpy()
                history.append(state)
                measured = [x.cpu().numpy() for x in read_contacts()]
                contact_history.append(measured)
                air = measured[2] & (measured[5][..., 2] > .003).all(axis=-1)
                passed = score.step(active, *measured[:2], air, *measured[3:5], state)
                done = (terminated | timeout).cpu().numpy()
                finite = np.isfinite(state).all(axis=1)
                passed &= finite & ~terminated.cpu().numpy()
                end = active & (done | passed | ~finite)
                success |= passed
                stop[end] = step
                active &= ~end
                if not active.any():
                    break
                if done.any():
                    ids = torch.as_tensor(np.flatnonzero(done), device=env.device)
                    obs, _ = env.reset(env_ids=ids)
            stop[active] = len(history) - 1
    finally:
        env.close()

    states = np.stack(history, axis=1)
    report = {"stage": args.stage, "policy": args.policy, "seed": args.seed, "perturb": args.perturb,
              "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
              "trials": args.envs, "successes": int(success.sum()), "success_rate": float(success.mean()),
              "paired_crest_reached": int(score.crest.sum()), "second_takeoff_detected": int(score.second_air.sum()),
              "non_foot_ground_or_wall_contacts": int(score.hit.sum()),
              "maximum_consecutive_crest_support_s": float(score.max_top_run.max() * .02),
              "maximum_consecutive_stable_brace_s": float(score.max_brace_run.max() * .02),
              "actual_crossing_amp_clips": 0}
    crest, ground, airborne, link_hit, cleared, soles, rear_force = (
        np.stack(values, axis=1) for values in zip(*contact_history)
    )
    diagnostics = []
    for trial, last in enumerate(stop):
        frame = slice(0, last + 1)
        front = soles[trial, frame, 2:]
        height = front[..., 2].min(axis=-1)
        peak = int(height.argmax())
        high = height >= HEIGHT - .005
        diagnostics.append({
            "simultaneous_front_sole_peak_m": float(height[peak]),
            "front_x_at_peak_m": front[peak, :, 0].tolist(),
            "front_crest_height_reached": bool(high.any()),
            "closest_front_pair_to_wall_at_crest_height_m": (
                float(np.abs(front[high, :, 0]).max(axis=1).min()) if high.any() else None),
            "peak_base_upward_speed_m_s": float(states[trial, frame, 21].max()),
            "peak_rear_ground_vertical_force_n": float(rear_force[trial, frame].max()),
            "airborne_samples": int(airborne[trial, frame].sum()),
        })
    report["takeoff_diagnostics"] = diagnostics
    if args.export_amp and args.stage == "full" and args.policy == "learned" and success.any():
        exported = output / "amp"
        exported.mkdir()
        files = []
        for trial in np.flatnonzero(success):
            path = exported / f"wall_direct_{trial:04d}.csv"
            # Read back simulator generalized poses, convert wxyz to the AMP xyzw convention.
            np.savetxt(path, states[trial, :stop[trial] + 1, :19][:, [0, 1, 2, 4, 5, 6, 3, *range(7, 19)]],
                       fmt="%.9g", delimiter=",")
            files.append(path.name)
        write_json(exported / "train.json", {"version": 1, "root": ".", "source_fps": 50,
                   "quaternion_order": "xyzw", "joint_names": list(JOINTS), "files": files})
        report["actual_crossing_amp_clips"] = len(files)
    np.savez_compressed(output / "rollouts.npz", states=states, stop=stop, success=success,
                        soles=soles, rear_ground_force=rear_force, crest=crest, ground=ground,
                        airborne=airborne, link_hit=link_hit, cleared=cleared,
                        actions=np.stack(actions, axis=1))
    write_json(output / "evaluation.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True, help="WBL asset and reference NPZ directory")
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--first-teacher", type=Path)
    followup = sub.add_parser("prepare-followup")
    followup.add_argument("--evaluation", type=Path, required=True)
    followup.add_argument("--checkpoint", type=Path, required=True)
    training = sub.add_parser("train")
    evaluation = sub.add_parser("evaluate")
    for cmd in (training, evaluation):
        cmd.add_argument("--stage", choices=STAGES, default="first")
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--envs", type=int, default=256 if cmd is training else 16)
        cmd.add_argument("--seed", type=int, default=511 if cmd is training else 801)
        cmd.add_argument("--device", default="cuda:0")
    training.add_argument("--batch", type=int, default=256)
    training.add_argument("--horizon", type=int, default=16)
    training.add_argument("--save-every", type=int, default=500)
    training.add_argument("--iterations", type=int, default=6000)
    training.add_argument("--explore", type=float, default=.13)
    training.add_argument("--teacher", action="store_true")
    training.add_argument("--sampling", choices=("return", "uniform", "start"), default="return")
    restore = training.add_mutually_exclusive_group()
    restore.add_argument("--warm-start", type=Path)
    restore.add_argument("--resume", type=Path)
    evaluation.add_argument("--policy", choices=("learned", "pd"), default="learned")
    evaluation.add_argument("--checkpoint", type=Path)
    evaluation.add_argument("--perturb", action="store_true")
    evaluation.add_argument("--export-amp", action="store_true")
    args = parser.parse_args()
    if args.mode in ("train", "evaluate") and args.envs <= 0:
        parser.error("--envs must be positive")
    if args.mode == "prepare":
        asset = (args.work / "wbl_tracking.xml").resolve()
        prepare_robot(asset)
        prepare_library(args.work, asset, args.first_teacher)
        write_json(args.work / "receipt.json", {
            "wall_thickness_width_height_m": [THICKNESS, WIDTH, HEIGHT],
            "wbl_asset_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
            "first_teacher_sha256": hashlib.sha256(args.first_teacher.read_bytes()).hexdigest() if args.first_teacher else None,
            "stage_clips": {stage: hashlib.sha256((args.work / stage / "direct.npz").read_bytes()).hexdigest()
                            for stage in STAGES},
        })
        print(args.work.resolve())
    elif args.mode == "prepare-followup":
        asset = configure(args)
        report = json.loads((args.evaluation / "evaluation.json").read_text())
        digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        if (report["stage"], report["policy"], report["checkpoint_sha256"]) != ("first", "learned", digest):
            raise ValueError("First-jump evaluation does not match this policy")
        trial, frames = prepare_followup(args.work, asset, args.evaluation / "rollouts.npz")
        write_json(args.work / "followup.json", {"first_checkpoint_sha256": digest,
                   "first_eval": str(args.evaluation.resolve()), "first_trial": trial,
                   "first_frames": frames, "stage_clips": {
                       stage: hashlib.sha256((args.work / stage / "direct.npz").read_bytes()).hexdigest()
                       for stage in ("second", "full")}})
    elif args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
