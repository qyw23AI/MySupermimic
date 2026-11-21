"""Train and evaluate the calibrated WBL calf-supported route with SuperMimic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from types import MethodType

import mujoco as mj
import numpy as np

from .cli import write_json
from .robot import HEIGHT, JOINTS, LEGS, NAME, THICKNESS, register

DT = .02
BENCHMARK_TRIALS = 16
EXPORT_MINIMUM = 12
HOME = np.array([-.1, 1., -1.5, .1, 1., -1.5, -.1, .8, -1.5, .1, .8, -1.5])
STAGES = ("second", "full")
OFFSETS = np.array([.002, .002, np.deg2rad(.5)])


def cohort(seed=301, count=BENCHMARK_TRIALS):
    """Same RNG, box and ordering used in the preceding MuJoCo replay."""
    return np.random.default_rng(seed).uniform(-OFFSETS, OFFSETS, (count, 3))


def write_clip(asset, path, states, joint_targets, release, entry):
    from quadmotion.clip import body_states_from_pose

    model = mj.MjModel.from_xml_path(str(asset))
    data = mj.MjData(model)
    pose = states[:, :model.nq]
    padded = np.concatenate((pose, np.repeat(pose[-1:], 15, axis=0)))
    clip = body_states_from_pose(model, padded[:, :3], padded[:, 3:7], padded[:, 7:])
    ids = [model.body(name).id for name in clip.body_names]
    jpos = np.empty((3, model.nv))
    jrot = np.empty_like(jpos)
    for frame, row in enumerate(states):
        data.qpos[:] = row[:model.nq]
        data.qvel[:] = row[model.nq:]
        mj.mj_forward(model, data)
        clip.joint_vel[frame] = data.qvel[6:]
        for index, body in enumerate(ids):
            mj.mj_jacBody(model, data, jpos, jrot, body)
            clip.body_lin_vel_w[frame, index] = jpos @ data.qvel
            clip.body_ang_vel_w[frame, index] = jrot @ data.qvel
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "motion.npz", **clip.to_npz_dict())
    np.savez_compressed(path.parent / f"{path.name}-teacher.npz", states=states,
                        targets=joint_targets, release=release, entry=entry)


def prepare(args):
    from .brace_experiment import make_model
    from .brace_replay import replay

    args.work.mkdir(parents=True, exist_ok=True)
    asset = args.work / "wbl_tracking.xml"
    if asset.resolve() != args.asset.resolve():
        shutil.copyfile(args.asset, asset)
    model = make_model(asset)
    metrics, record, initial, controls = replay(model, args.calf, "calf")
    if not metrics["second"]["stable_landing"] or not metrics["second"]["validated_unload_before_second"]:
        raise ValueError("Expected a re-verified calf unload and continuous successful second jump")
    stop = metrics["accepted_cut_frame"]
    states = np.vstack((initial[1:38], record["state"][:stop]))
    release = metrics["second"]["start_frame"]
    if args.warp_teacher:
        with np.load(args.warp_teacher, allow_pickle=False) as refined:
            states = refined["states"].copy()
            controls = refined["targets"].copy()
            if states.shape != (len(controls)+1,37) or len(controls) <= release + 30:
                raise ValueError("The refined teacher must contain a complete measured trajectory")
            if np.max(np.abs(states[0] - initial[1:38])) > .007:
                raise ValueError("Refined teacher start differs from the comparison state")
        stop = len(controls)
    write_clip(asset, args.work / "second", states, controls[:stop], release, 0)
    if args.first:
        with np.load(args.first, allow_pickle=False) as previous:
            ok = np.flatnonzero(previous["success"])
            if not len(ok):
                raise ValueError("The approach must come from a successful first jump")
            index = ok[0]
            steps = int(previous["stop"][index])
            approach = previous["states"][index, :steps+1]
            targets = previous["actions"][index, :steps] * .25 + HOME
        gap = np.max(np.abs(approach[-1] - states[0]))
        if gap > .007:
            raise ValueError(f"Approach and calf states disagree at the handoff (max abs {gap:.5f})")
        write_clip(asset, args.work / "full", np.vstack((approach[:-1], states)),
                   np.vstack((targets, controls[:stop])), release + steps, steps)
    write_json(args.work / "source.json", {
        "calf": str(args.calf.resolve()),
        "calf_sha256": hashlib.sha256(args.calf.read_bytes()).hexdigest(),
        "first": str(args.first.resolve()) if args.first else None,
        "warp_teacher": str(args.warp_teacher.resolve()) if args.warp_teacher else None,
        "benchmark_offsets_m_m_rad": cohort().tolist(),
    })
    print(f"Prepared calf motion. Second-jump demo {stop} controls, unload at {release}.")


def contact_measurements(env):
    import torch

    wall = env.scene["wall_feet"]
    floor = env.scene["floor_feet"]
    links = env.scene["wall_links"]
    floor_links = env.scene["floor_links"]
    tail = [leg + "_foot_ball" for leg in LEGS]
    wi = [next(i for i, n in enumerate(wall.primary_names) if n.endswith(name)) for name in tail]
    fi = [next(i for i, n in enumerate(floor.primary_names) if n.endswith(name)) for name in tail]
    ci = [next(i for i, n in enumerate(links.primary_names) if n.endswith(name))
          for name in ("FR_calf_collision", "FL_calf_collision")]
    other = [i for i in range(len(links.primary_names)) if i not in ci]
    entity = env.scene["robot"]
    ids, _ = entity.find_geoms(tail, preserve_order=True)
    ids = entity.indexing.geom_ids[ids]
    radii = torch.as_tensor(env.sim.mj_model.geom_size[ids.cpu().numpy(), 0],
                            dtype=torch.float32, device=env.device)

    def sensor_force(data):
        return torch.where(data.found > 0, data.force[..., 0].abs(), 0.)

    def read():
        w, f, c, ground_links = (s.data for s in (wall, floor, links, floor_links))
        cf = sensor_force(c)
        wf = sensor_force(w)[:, wi]
        ff = sensor_force(f)[:, fi]
        cn = c.normal[:, ci, 2].abs()
        wn = w.normal[:, wi, 2].abs()
        calf_top = (cn >= .75) & (c.pos[:, ci, 2] >= HEIGHT - .013)
        toe_top = (wn >= .75) & (w.pos[:, wi, 2] >= HEIGHT - .012)
        bad = torch.maximum(sensor_force(ground_links).amax(-1), cf[:, other].amax(-1))
        bad = torch.maximum(bad, torch.where(~calf_top, cf[:, ci], 0.).amax(-1))
        feet = env.sim.data.geom_xpos[:, ids, :]
        clear = (feet[..., 0] - radii > THICKNESS / 2).all(-1)
        no_support = (wf.amax(-1) < 5.) & (ff.amax(-1) < 5.) & (cf.amax(-1) < 5.)
        airborne = no_support & (feet[..., 2] - radii > .005).all(-1)
        calf_any = cf[:, ci]
        if c.force_history is not None:
            peak = c.force_history[..., 0].abs().amax(-1)
            calf_any = torch.maximum(calf_any, peak[:, ci])
            bad = torch.maximum(bad, peak[:, other].amax(-1))
            bad = torch.maximum(bad, ground_links.force_history[..., 0].abs().amax((-1, -2)))
        return {
            "calf": torch.where(calf_top, cf[:, ci] * cn, 0.),
            "calf_any": calf_any,
            "toe": torch.where(toe_top, wf * wn, 0.),
            "ground": ff * f.normal[:, fi, 2].abs(),
            "bad_n": bad,
            "feet": feet.clone(),
            "clear": clear,
            "air": airborne,
        }
    return read


class CalfScore:
    """Stateful contact milestones, independent of the tracking objective."""
    def __init__(self, count, stage, initial_state, initial_feet):
        self.stage = stage
        self.base = initial_state[:, 0].copy()
        self.rear = initial_feet[:, :2, 0].copy()
        for key in ("first_air", "first_brace", "calf_loaded", "advanced", "released", "second_air", "hit", "recontact"):
            setattr(self, key, np.zeros(count, dtype=bool))
        for key in ("first_air_run", "first_brace_run", "load_run", "advance_run", "release_run", "air_run", "landing_run"):
            setattr(self, key, np.zeros(count, dtype=np.int32))
        if stage == "second":
            self.first_air[:] = True
            self.first_brace[:] = True

    def step(self, active, state, contact):
        def count(name, mask):
            value = np.where(active & mask, getattr(self, name) + 1, 0)
            setattr(self, name, value)
            return value
        self.hit |= active & (contact["bad_n"] >= 5.)
        self.recontact |= active & self.released & (contact["calf_any"].max(-1) >= 5.)
        feet = contact["feet"]
        g = contact["ground"]
        toe = contact["toe"][:, 2:]
        calf = contact["calf"]
        air = contact["air"]
        up = 1. - 2. * (state[:, 4]**2 + state[:, 5]**2)
        speed = np.linalg.norm(state[:, 19:22], axis=-1)
        stable = (up > .72) & (speed < .35) & (g[:, :2].min(-1) > 7.)
        planted = stable & (toe.min(-1) > 7.)
        self.first_air |= count("first_air_run", air & ~self.first_brace) >= 2
        new_brace = (count("first_brace_run", self.first_air & planted & ~self.first_brace) >= 5)
        self.base[new_brace] = state[new_brace, 0]
        self.rear[new_brace] = feet[new_brace, :2, 0]
        self.first_brace |= new_brace

        bearing = self.first_brace & (calf.min(-1) > 7.) & (calf.sum(-1) > toe.sum(-1))
        self.calf_loaded |= count("load_run", bearing) >= 5
        forward = (state[:, 0] - self.base > .045) & ((feet[:, :2, 0] - self.rear).min(-1) > .075)
        self.advanced |= count("advance_run", self.calf_loaded & bearing & stable & forward) >= 5
        unload = self.advanced & stable & forward & (contact["calf_any"].max(-1) < 2.)
        self.released |= count("release_run", unload) >= 5
        self.second_air |= count("air_run", self.released & air) >= 2
        land = (self.second_air & contact["clear"] & (g.min(-1) > 5.) & (up > .82)
                & (state[:, 2] > .22) & (speed < .38)
                & (np.linalg.norm(state[:, 22:25], axis=-1) < 2.))
        return (count("landing_run", land) >= 10) & ~self.hit & ~self.recontact


def install_offsets(command, values):
    """Apply exact world-frame perturbations at the existing motion command reset."""
    import torch

    from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

    previous = command._write_reference_state_to_sim

    def write(_command, ids, pos, ori, lin, ang, joint, joint_vel):
        offset = values[ids]
        pos = pos.clone()
        pos[:, :2] += offset[:, :2]
        zr = torch.zeros_like(offset[:, 2])
        ori = quat_mul(quat_from_euler_xyz(zr, zr, offset[:, 2]), ori)
        previous(ids, pos, ori, lin, ang, joint, joint_vel)

    command._write_reference_state_to_sim = MethodType(write, command)
    return previous


def make_evaluation(work, stage, device, envs, seed, future_frames=10, strict_contact=False):
    from mjlab.envs import ManagerBasedRlEnv
    from supermimic.envs import SuperMimicEnvCfgArgs
    from supermimic.envs.tracking_env_cfg import make_env_cfg

    register(work / "wbl_tracking.xml", calf_route=True, physics_history=strict_contact)
    cfg = make_env_cfg(SuperMimicEnvCfgArgs(
        npz_files=[str(work / stage / "motion.npz")], robot=NAME, num_envs=envs, seed=seed,
        sampling_mode="start", episode_length_s=4.2, future_frames=future_frames,
        anchor_obs=True, record_transitions=False, action_noise_std=0., reset_noise=False,
        domain_randomization=False, obs_noise=False,
    ))
    cfg.auto_reset = False
    return ManagerBasedRlEnv(cfg, device=device)


def as_numpy(contact):
    return {key: value.detach().cpu().numpy().copy() for key, value in contact.items()}


class TouchdownLeveler:
    """Synchronize paired foot heights during descent after all feet clear the wall."""

    def __init__(self, env, config):
        import torch

        self.env = env
        self.front_gain, self.max_height, self.limit, self.rear_gain = map(float, config)
        robot = env.scene["robot"]
        names = [f"{leg}_{part}_joint" for leg in LEGS for part in ("thigh", "calf")]
        indices, _ = robot.find_joints(names, preserve_order=True)
        self.joints = robot.indexing.joint_ids[indices]
        self.action_columns = [JOINTS.index(name) for name in names]
        self.cleared = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, ids):
        self.cleared[ids] = False

    def apply(self, action, state, contact):
        import torch

        self.cleared |= contact["clear"]
        active = self.cleared & (state[:, 2] < self.max_height)
        active &= (state[:, 2] > .25) & (state[:, 21] < -.40)
        feet = contact["feet"]
        paired_z = feet[..., 2].reshape(self.env.num_envs, 2, 2)
        paired_diff = paired_z - paired_z.mean(dim=-1, keepdim=True)
        gain = feet.new_tensor([self.rear_gain, self.front_gain])
        desired_dz = (-paired_diff * gain[None, :, None]).reshape(self.env.num_envs, 4)
        data = self.env.sim.data
        axes = data.xaxis[:, self.joints, :].reshape(self.env.num_envs, 4, 2, 3)
        anchors = data.xanchor[:, self.joints, :].reshape_as(axes)
        jac_z = torch.cross(axes, feet[:, :, None, :] - anchors, dim=-1)[..., 2]
        dq = jac_z * desired_dz[..., None] / (jac_z.square().sum(dim=-1, keepdim=True) + .012)
        dq = dq.clamp(-self.limit, self.limit) * active[:, None, None]
        result = action.clone()
        result[:, self.action_columns] += dq.reshape(self.env.num_envs, 8) / .25
        return result


def run_evaluation(args):
    import torch
    from supermimic.envs.state_io import StateIO
    from supermimic.eval.benchmark import load_policy_from_checkpoint

    actor = None
    if args.policy == "learned":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for a learned evaluation")
        actor, cfg = load_policy_from_checkpoint(args.checkpoint, args.device)
        if cfg["robot"] != NAME or not cfg["env"]["anchor_obs"]:
            raise ValueError("Expected the WBL policy with wall-frame anchor goals")
    touchdown_settings = None
    if args.touchdown_profile:
        if actor is None:
            raise ValueError("Touchdown feedback requires an actor")
        profile = json.loads(args.touchdown_profile.read_text())
        if profile["stage"] != args.stage:
            raise ValueError("Touchdown profile must match the evaluated stage")
        touchdown_settings = np.asarray(profile["candidate"]["gains"], dtype=float)
        limits = np.asarray([2., .90, .5, 1.])
        if (touchdown_settings.shape != (4,) or not np.isfinite(touchdown_settings).all()
                or (touchdown_settings < 0.).any() or (touchdown_settings > limits).any()):
            raise ValueError("Invalid touchdown feedback profile")
    gain_profile = None
    if args.phase_gains:
        if actor is None or not actor.cfg.guide_frames:
            raise ValueError("Phase gains require a learned actor with an action guide")
        gain_profile = json.loads(args.phase_gains.read_text())
        if gain_profile["stage"] != args.stage:
            raise ValueError("Residual calibration must match the evaluated stage")
        gains = np.asarray(gain_profile["candidate"]["gains"], dtype=float)
        boundaries = np.asarray(gain_profile["phase_end_frames"], dtype=int)
        if gains.shape != (4,) or not np.isfinite(gains).all() or (gains < 0).any() or (gains > 2.).any():
            raise ValueError("Expected four finite residual gains between 0 and 2")
        if boundaries.shape != (3,) or boundaries[0] < 0 or not (np.diff(boundaries) > 0).all():
            raise ValueError("Expected three increasing phase boundaries")
    offsets = cohort(seed=args.cohort_seed, count=args.envs) if not args.nominal else np.zeros((args.envs, 3))
    env = make_evaluation(args.work, args.stage, args.device, args.envs, args.seed,
                          strict_contact=args.strict_contact)
    try:
        command = env.command_manager.get_term("motion")
        install_offsets(command, torch.as_tensor(offsets, dtype=torch.float32, device=env.device))
        with np.load(args.work / f"{args.stage}-teacher.npz", allow_pickle=False) as demo:
            targets = torch.tensor(demo["targets"], device=env.device, dtype=torch.float32)
        if args.targets:
            with np.load(args.targets, allow_pickle=False) as custom:
                targets = torch.tensor(custom["targets"], device=env.device, dtype=torch.float32)
        io = StateIO(env)
        read = contact_measurements(env)
        leveler = TouchdownLeveler(env, touchdown_settings) if touchdown_settings is not None else None
        with torch.inference_mode():
            obs, _ = env.reset()
            states = [io.read().cpu().numpy()]
            all_contacts = [as_numpy(read())]
            score = CalfScore(args.envs, args.stage, states[0], all_contacts[0]["feet"])
            success = np.zeros(args.envs, dtype=bool)
            stop = np.zeros(args.envs, dtype=int)
            active = np.ones(args.envs, dtype=bool)
            for step in range(1, len(targets) + 14):
                if actor is None:
                    desired = targets[min(step - 1, len(targets) - 1)]
                    action = (desired - command.robot.data.default_joint_pos) / .25
                else:
                    action = actor(obs["policy"], obs["goal"])
                    if gain_profile is not None:
                        local_frame = command.frame - command.library.clip_start[0]
                        guide = actor.guide_actions[local_frame.clamp(0, len(actor.guide_actions)-1)]
                        phase = int(np.searchsorted(boundaries, step-1, side="right"))
                        action = guide + (action - guide) * gains[phase]
                if leveler is not None:
                    action = leveler.apply(action, io.read(), read())
                obs, _, terminated, timeout, _ = env.step(action)
                states.append(io.read().cpu().numpy())
                all_contacts.append(as_numpy(read()))
                passed = score.step(active, states[-1], all_contacts[-1])
                broken = terminated.cpu().numpy() | ~np.isfinite(states[-1]).all(-1)
                passed &= ~broken
                success |= passed
                end = active & (broken | passed | timeout.cpu().numpy())
                stop[end] = step
                active &= ~end
                if not active.any():
                    break
                if (terminated | timeout).any():
                    ids = torch.nonzero(terminated | timeout, as_tuple=True)[0]
                    obs, _ = env.reset(env_ids=ids)
                    if leveler is not None:
                        leveler.reset(ids)
            stop[active] = len(states) - 1
    finally:
        env.close()
    trajectory = np.stack(states, axis=1)
    stacked = {key: np.stack([frame[key] for frame in all_contacts], axis=1) for key in all_contacts[0]}
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "rollouts.npz", states=trajectory, stop=stop, success=success,
                        offsets=offsets, **stacked)
    outcome = {key: int(getattr(score, key).sum()) for key in (
        "first_air", "first_brace", "calf_loaded", "advanced", "released", "second_air", "hit", "recontact",
    )}
    report = {"stage": args.stage, "policy": args.policy, "nominal": args.nominal,
              "cohort_seed": args.cohort_seed, "trials": args.envs, "successes": int(success.sum()),
              "strict_contact": args.strict_contact, "export_threshold": EXPORT_MINIMUM,
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
              "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
              "offsets_m_m_rad": offsets.tolist(), **outcome}
    if gain_profile is not None:
        report["phase_gain_profile"] = gain_profile["candidate"]
        report["phase_end_frames"] = boundaries.tolist()
    if touchdown_settings is not None:
        report["touchdown_feedback"] = {"front_gain": float(touchdown_settings[0]),
            "max_activation_height_m": float(touchdown_settings[1]),
            "joint_delta_cap_rad": float(touchdown_settings[2]), "rear_gain": float(touchdown_settings[3])}
    report["amp_exported"] = 0
    if (args.export_amp and args.stage == "full" and args.policy == "learned" and not args.nominal
            and args.strict_contact and args.cohort_seed == 301 and args.envs == 16 and success.sum() >= EXPORT_MINIMUM):
        export = args.output / "amp"
        export.mkdir()
        names = []
        for trial in np.flatnonzero(success):
            name = f"wall_calf_{trial:04d}.csv"
            pose = trajectory[trial, :stop[trial]+1, :19]
            np.savetxt(export / name, pose[:, [0, 1, 2, 4, 5, 6, 3, *range(7, 19)]],
                       delimiter=",", fmt="%.9g")
            names.append(name)
        write_json(export / "train.json", {"root": ".", "version": 1, "source_fps": 50,
                   "quaternion_order": "xyzw", "joint_names": list(JOINTS), "files": names})
        report["amp_exported"] = len(names)
    write_json(args.output / "evaluation.json", report)
    print({key: value for key, value in report.items() if key != "offsets_m_m_rad"}, flush=True)


def training_config(args):
    from supermimic.config import TrainConfig

    cfg = TrainConfig(run_name=args.output.name, log_root=args.output.parent, robot=NAME,
                      device=args.device, seed=args.seed, tf32=True)
    cfg.data.npz_dir = (args.work / args.stage).resolve(strict=True)
    cfg.env.num_envs = args.envs
    cfg.env.episode_length_s = 4.2
    cfg.env.replay_capacity = 256
    cfg.env.reset_noise = False
    cfg.env.obs_noise = False
    cfg.env.domain_randomization = False
    cfg.env.anchor_obs = True
    cfg.env.action_noise_std = args.explore
    cfg.env.sampling_mode = "uniform"
    cfg.policy.encoder_hidden = (256, 128)
    cfg.policy.decoder_hidden = (256, 128)
    if args.guide:
        with np.load(args.work / args.stage / "motion.npz", allow_pickle=False) as clip:
            cfg.policy.guide_frames = len(clip["joint_pos"])
        cfg.policy.residual_limit = args.residual_limit
    cfg.world_model.hidden = (256, 256, 256)
    cfg.wm_trainer.compile = cfg.policy_trainer.compile = None
    cfg.wm_trainer.batch_size = cfg.policy_trainer.batch_size = args.batch
    cfg.wm_trainer.batches_per_update = 2
    cfg.wm_trainer.window = 6
    cfg.policy_trainer.horizon = args.horizon
    cfg.policy_trainer.critic_epochs = 2
    cfg.policy_trainer.obs_noise = False
    cfg.policy_trainer.demonstration_weight = args.anchor
    cfg.policy_trainer.lr = args.actor_lr
    cfg.schedule.warmup_iterations = 32
    cfg.schedule.policy_start_iterations = 160
    cfg.schedule.total_iterations = args.iterations
    cfg.schedule.checkpoint_interval = args.save_every
    cfg.schedule.log_interval = 50
    return cfg


def balance_signal(state, feet_z):
    """Differential calf extension for roll, pitch and unloaded stance legs."""
    import torch

    qw, qx, qy, qz = state[..., 3:7].unbind(-1)
    roll = torch.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx.square()+qy.square()))
    pitch = torch.asin((2*(qw*qy - qz*qx)).clamp(-1., 1.))
    p, q = state[..., 22:24].unbind(-1)
    right = state.new_tensor([1., -1., 1., -1.])
    front = state.new_tensor([-1., -1., 1., 1.])
    adjustment = 1.25 * ((roll+.11*p)[..., None]*right + (pitch+.11*q)[..., None]*front)
    adjustment += 4. * (feet_z - .025).clamp(0., .075)
    return adjustment.clamp(-.26, .26)


def collect_teacher(trainer, filename, rounds, balance_start):
    import torch
    from supermimic.envs.state_io import StateIO

    env, command = trainer.env, trainer.command
    io = StateIO(env)
    with np.load(filename, allow_pickle=False) as record:
        targets = torch.tensor(record["targets"], dtype=torch.float32, device=env.device)
        states = torch.tensor(record["states"], dtype=torch.float32, device=env.device)
        entry = int(record["entry"])
    if balance_start is not None:
        # Subtract the nominal feedback already baked into the validated
        # targets, then apply feedback using the live attitude and foot heights.
        model = mj.MjModel.from_xml_path(str(filename.parent / "wbl_tracking.xml"))
        data = mj.MjData(model)
        geom_ids = [model.geom(leg + "_foot_ball").id for leg in LEGS]
        height = []
        for pose in states[:, :19].cpu().numpy():
            data.qpos[:] = pose
            mj.mj_fwdPosition(model, data)
            height.append(data.geom_xpos[geom_ids, 2].copy())
        nominal_balance = balance_signal(states, torch.as_tensor(np.asarray(height), dtype=states.dtype, device=env.device))
        read = contact_measurements(env)
        balance_frame = entry + balance_start
    offsets = torch.zeros((env.num_envs, 3), device=env.device)
    previous = install_offsets(command, offsets)
    joint_action = env.action_manager.get_term("joint_pos")
    prev_noise = joint_action.noise_std
    original = command._sample_frames
    obs_data, goal_data, action_data = [], [], []
    try:
        with torch.no_grad():
            command._sample_frames = MethodType(lambda cmd, n: cmd.library.clip_start[0].expand(n), command)
            joint_action.noise_std = 0.
            for index in range(rounds):
                offsets[:] = torch.rand_like(offsets) * 2 - 1
                offsets *= torch.tensor(OFFSETS, device=env.device) * (.75 + .35 * index)
                offsets[0] = 0.
                obs, _ = env.reset()
                active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
                for t, target in enumerate(targets):
                    state = io.read()
                    errors = states[t, 7:19] - state[:, 7:19]
                    rate_error = states[t, 25:37] - state[:, 25:37]
                    feedback = (.38 * errors + .008 * rate_error).clamp(-.16, .16)
                    gain = 0. if index == 0 else min(.40 + .24 * index, 1.)
                    desired = target + gain * feedback
                    if balance_start is not None and t > balance_frame:
                        delta = balance_signal(state, read()["feet"][..., 2]) - nominal_balance[t]
                        weight = min(1., (t - balance_frame)/9.)
                        desired[:, [2,5,8,11]] += weight * delta
                        desired[:, [1,4,7,10]] -= .5 * weight * delta
                    action = (desired - command.robot.data.default_joint_pos) / .25
                    if index > 1:
                        action = action + .020 * torch.randn_like(action)
                    # Exclude states from a reset or states far outside the teacher trajectory.
                    keep = active & (errors.abs().amax(-1) < .95) & ((state[:, 2] - states[t, 2]).abs() < .20)
                    if keep.any():
                        obs_data.append(obs["policy"][keep].clone())
                        goal_data.append(obs["goal"][keep].clone())
                        action_data.append(action[keep].clone())
                    obs, _, term, timeout, _ = env.step(action)
                    active &= ~(term | timeout)
                    if not active.any():
                        break
    finally:
        command._sample_frames = original
        command._write_reference_state_to_sim = previous
        joint_action.noise_std = prev_noise
    if not obs_data:
        raise RuntimeError("No usable teacher transitions")
    return tuple(torch.cat(values) for values in (obs_data, goal_data, action_data))


def train(args):
    import torch
    from supermimic.training.loop import Trainer

    if args.output.exists():
        raise FileExistsError(args.output)
    register(args.work / "wbl_tracking.xml", calf_route=True)
    trainer = Trainer(training_config(args))
    try:
        if args.warm:
            saved = torch.load(args.warm, map_location=args.device, weights_only=False)
            policy_weights = {key:value for key,value in saved["policy"].items() if not key.startswith("guide_")}
            missing, extra = trainer.policy.load_state_dict(policy_weights, strict=False)
            allowed = {"guide_keys", "guide_actions"} if args.guide else set()
            if set(missing) != allowed or extra:
                raise ValueError(f"Warm-start architecture mismatch: missing={missing}, extra={extra}")
            trainer.wm.load_state_dict(saved["wm"], strict=True)
        filename = args.work / f"{args.stage}-teacher.npz"
        if args.guide:
            with np.load(filename, allow_pickle=False) as record:
                targets = torch.as_tensor(record["targets"], dtype=torch.float32, device=args.device)
            length = trainer.cfg.policy.guide_frames
            command = trainer.command
            frames = command.library.clip_start[0] + torch.arange(length, device=args.device)
            features = command.library.goal_window(frames, 2)
            # A reset writes pose 0, then the command manager advances the
            # goal to frame 1 before the first policy call. Control 0 belongs
            # to that goal; this also stays consistent in imagined rollouts.
            indices = (torch.arange(length, device=args.device) - 1).clamp(0, len(targets)-1)
            targets = targets[indices]
            actions = (targets - command.robot.data.default_joint_pos[0]) / .25
            trainer.policy.set_action_guide(features, actions)
        policy_obs, goal, action = collect_teacher(trainer, filename, args.teacher_rounds, args.landing_feedback)
        trainer.policy_trainer.set_demonstration(policy_obs, goal, action)
        optimizer = torch.optim.Adam(trainer.policy.parameters(), lr=4.e-4)
        for index in range(args.pretrain):
            ind = torch.randint(len(action), (256,), device=args.device)
            output = trainer.policy(policy_obs[ind] + .004 * torch.randn_like(policy_obs[ind]), goal[ind])
            loss = (output - action[ind]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if index in (0, args.pretrain-1) or (index+1) % 400 == 0:
                print(f"clone {index+1}/{args.pretrain}: mse={loss.item():.6f}, samples={len(action)}", flush=True)
        trainer.save("ckpt_clone.pt")
        trainer.run()
    finally:
        trainer.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--asset", required=True, type=Path)
    prep.add_argument("--calf", required=True, type=Path)
    prep.add_argument("--first", type=Path)
    prep.add_argument("--warp-teacher", type=Path)
    fit = sub.add_parser("train")
    check = sub.add_parser("evaluate")
    for cmd in (fit, check):
        cmd.add_argument("--stage", choices=STAGES, default="second")
        cmd.add_argument("--device", default="cuda:0")
        cmd.add_argument("--envs", type=int, default=128 if cmd is fit else 16)
        cmd.add_argument("--seed", type=int, default=511 if cmd is fit else 801)
        cmd.add_argument("--output", type=Path, required=True)
    fit.add_argument("--iterations", type=int, default=900)
    fit.add_argument("--batch", type=int, default=128)
    fit.add_argument("--horizon", type=int, default=8)
    fit.add_argument("--save-every", type=int, default=300)
    fit.add_argument("--teacher-rounds", type=int, default=4)
    fit.add_argument("--pretrain", type=int, default=1600)
    fit.add_argument("--anchor", type=float, default=25.)
    fit.add_argument("--actor-lr", type=float, default=2.e-5)
    fit.add_argument("--explore", type=float, default=.06)
    fit.add_argument("--warm", type=Path)
    fit.add_argument("--guide", action="store_true")
    fit.add_argument("--residual-limit", type=float, default=.35)
    fit.add_argument("--landing-feedback", type=int, metavar="FRAME",
                     help="Distill stance feedback from this frame of the calf segment")
    check.add_argument("--policy", choices=("learned", "replay"), default="learned")
    check.add_argument("--checkpoint", type=Path)
    check.add_argument("--targets", type=Path)
    check.add_argument("--nominal", action="store_true")
    check.add_argument("--cohort-seed", type=int, default=301)
    check.add_argument("--export-amp", action="store_true")
    check.add_argument("--strict-contact", action="store_true")
    check.add_argument("--phase-gains", type=Path)
    check.add_argument("--touchdown-profile", type=Path)
    args = parser.parse_args()
    if args.command != "prepare" and args.envs < 1:
        parser.error("--envs must be positive")
    if args.command == "evaluate" and args.targets and args.policy != "replay":
        parser.error("--targets only applies to open-loop replay")
    {"prepare": prepare, "train": train, "evaluate": run_evaluation}[args.command](args)


if __name__ == "__main__":
    main()
