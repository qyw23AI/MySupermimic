"""Train and independently evaluate the front-led first wall jump."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import calf_tracking as tracker
from .cli import write_json


def prepare(args):
    import shutil

    args.work.mkdir(parents=True)
    asset = args.work / "wbl_tracking.xml"
    shutil.copyfile(args.asset, asset)
    source = args.reference if args.reference.is_file() else args.reference / "replay.npz"
    with np.load(source, allow_pickle=False) as demo:
        if "targets" in demo:
            states, targets = demo["states"], demo["targets"]
        else:
            states = np.vstack((demo["initial"][1:38], demo["state"]))
            targets = demo["controls"]
    if states.shape != (len(targets) + 1, 37):
        raise ValueError("Reference state and control lengths disagree")
    tracker.write_clip(asset, args.work / "first", states, targets, 0, 0)


def train(args):
    tracker.train(SimpleNamespace(
        **vars(args), stage="first", guide=True, residual_limit=.22, warm=None,
        explore=.025, anchor=55., actor_lr=5.e-6, batch=min(args.envs, 512),
        horizon=8, teacher_rounds=3, pretrain=1200, landing_feedback=None,
        save_every=100,
    ))


class CompliantCatch:
    """Per-environment gain and pose changes gated by flight and shin contact."""

    def __init__(self, env, profiles):
        import torch
        from .robot import JOINTS

        self.p = torch.as_tensor(profiles, dtype=torch.float32, device=env.device)
        self.model = env.sim.model
        mj_model = env.sim.mj_model
        staged = {int(joint):i for i,(joint,_) in enumerate(mj_model.actuator_trnid)}
        self.ids = [staged[mj_model.joint("robot/"+name).id] for name in JOINTS]
        env.sim.expand_model_fields(("actuator_gainprm","actuator_biasprm"))
        self.kp = self.model.actuator_gainprm[0,self.ids,0].clone()
        self.kv = -self.model.actuator_biasprm[0,self.ids,2].clone()
        self.default = env.scene["robot"].data.default_joint_pos
        self.air = torch.zeros(len(profiles),dtype=torch.bool,device=env.device)
        self.age = torch.zeros(len(profiles),device=env.device)

    def apply(self, action, contact):
        import torch

        air = torch.as_tensor(contact["air"],device=action.device)
        front = torch.as_tensor(contact["calf"],device=action.device)
        self.air |= air
        self.age += (self.air & ((front.min(-1).values>7.) | (self.age>0))).to(self.age.dtype)
        front_gain, rear_gain, tuck, reach, pull, rear_bend, restore, rear_damping = self.p.unbind(-1)
        ramp = (self.age / restore).clamp(0,1)
        # When either shin touches, keep the flight compliance until both shins support.
        f_scale = torch.where(self.air, front_gain+(1-front_gain)*ramp, 1.)
        r_scale = torch.where(self.air, rear_gain+(1-rear_gain)*ramp, 1.)
        scale = torch.cat((r_scale[:,None].expand(-1,6),f_scale[:,None].expand(-1,6)),-1)
        damping = torch.cat((torch.where(self.air,rear_damping,1.)[:,None].expand(-1,6),
                             scale[:,6:].sqrt()),-1)
        self.model.actuator_gainprm[:,self.ids,0] = scale*self.kp
        self.model.actuator_biasprm[:,self.ids,1] = -scale*self.kp
        self.model.actuator_biasprm[:,self.ids,2] = -damping*self.kv
        desired = action*.25+self.default
        airborne = self.air.to(action.dtype)*(1-ramp)
        desired[:,[8,11]] -= tuck[:,None]*airborne[:,None]
        desired[:,[7,10]] += reach[:,None]*airborne[:,None]
        desired[:,[2,5]] -= rear_bend[:,None]*airborne[:,None]
        # Increase fore knee flexion gradually once both shins take load.
        desired[:,[8,11]] -= (ramp*pull)[:,None]
        return (desired-self.default)/.25


def evaluate(args):
    import torch
    from supermimic.envs.state_io import StateIO
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from .four_stage_tracking import PhysicsAudit

    actor = None
    if args.checkpoint:
        actor, _ = load_policy_from_checkpoint(args.checkpoint, args.device)
    seeds = args.cohort_seeds
    offsets = np.vstack([tracker.cohort(seed, args.envs) for seed in seeds])
    cohort_size = len(offsets)
    compliance = np.load(args.compliance, allow_pickle=False) if args.compliance else None
    if args.repeat_profiles and (compliance is None or args.single):
        raise ValueError("--repeat-profiles requires a compliance population")
    if args.single and compliance is None:
        raise ValueError("--single requires --compliance")
    if compliance is not None:
        if args.profiles or args.profile_file or not args.checkpoint:
            raise ValueError("Compliance profiles require a learned checkpoint and no trajectory profile")
        if compliance.ndim != 2 or compliance.shape[1] != 8 or not len(compliance) or not np.isfinite(compliance).all():
            raise ValueError("Expected N×8 compliance profiles")
        if (compliance[:,6] <= 0).any() or (compliance[:,[0,1,7]] <= 0).any():
            raise ValueError("Gain scales and ramp duration must be positive")
        if args.single:
            compliance = np.repeat(compliance[:1],cohort_size,axis=0)
        elif args.repeat_profiles:
            compliance = np.repeat(compliance,cohort_size,axis=0)
        offsets = offsets[np.arange(len(compliance)) % len(offsets)]
    profiles = np.load(args.profiles, allow_pickle=False) if args.profiles else None
    if args.profile_file:
        if args.checkpoint or profiles is not None:
            raise ValueError("A single replay profile cannot be combined with a checkpoint or population")
        profiles = np.repeat(np.load(args.profile_file,allow_pickle=False)[:1],len(offsets),axis=0)
    if profiles is not None:
        if profiles.ndim != 3 or profiles.shape[1:] != (2,12) or not np.isfinite(profiles).all():
            raise ValueError("Profiles must have shape N×2×12 and contain finite offsets")
        offsets = offsets[np.arange(len(profiles)) % len(offsets)]
    count = len(offsets)
    env = tracker.make_evaluation(args.work, "first", args.device, count, 801)
    targets = np.load(args.work / "first-teacher.npz")["targets"]
    if profiles is not None:
        target_tensor = torch.as_tensor(targets, device=env.device, dtype=torch.float32)
        deltas = torch.as_tensor(profiles, device=env.device, dtype=torch.float32)
    try:
        command = env.command_manager.get_term("motion")
        tracker.install_offsets(command, torch.as_tensor(offsets, dtype=torch.float32, device=env.device))
        io = StateIO(env)
        audit = PhysicsAudit(env)
        controller = CompliantCatch(env, compliance) if compliance is not None else None
        frames, readings, samples = [], [], []
        first_end = np.full(count, len(targets)+8, dtype=int)
        with torch.inference_mode():
            obs, _ = env.reset()
            frames.append(io.read().cpu().numpy())
            readings.append(tracker.as_numpy(audit.read_base()))
            for step in range(1, len(targets)+9):
                audit.frames.clear()
                action = (actor(obs["policy"], obs["goal"]) if actor is not None else
                          (torch.tensor(targets[min(step-1,len(targets)-1)],device=env.device,dtype=torch.float32)
                           - command.robot.data.default_joint_pos)/.25)
                if profiles is not None:
                    alpha = min(1., max(0., (step - 15) / 8.))
                    beta = min(1., max(0., (step - 23) / 6.))
                    desired = target_tensor[min(step-1,len(targets)-1)] + alpha*deltas[:,0] + beta*deltas[:,1]
                    action = (desired-command.robot.data.default_joint_pos)/.25
                if controller is not None:
                    action = controller.apply(action, readings[-1])
                obs, _, term, timeout, _ = env.step(action)
                frames.append(io.read().cpu().numpy())
                contact, sample = audit.control_frame()
                readings.append({key: contact[key] for key in readings[0]})
                samples.append(sample)
                ended = (term | timeout).cpu().numpy()
                first_end[ended] = np.minimum(first_end[ended], step)
                if ended.any():
                    obs, _ = env.reset(env_ids=torch.nonzero(term | timeout, as_tuple=True)[0])
    finally:
        env.close()
    states = np.stack(frames, axis=1)
    contacts = {key: np.stack([row[key] for row in readings], axis=1) for key in readings[0]}
    physics = {key: np.concatenate([row[key] for row in samples], axis=1) for key in samples[0]}
    rows = []
    for i, end in enumerate(first_end):
        valid = slice(0,end*4)
        front = physics["top"][i,valid,:2]
        toe = physics["toe"][i,valid,2:]
        ground = physics["ground"][i,valid]
        bad = physics["bad"][i,valid]
        def first(mask):
            return int(np.flatnonzero(mask)[0]) if mask.any() else None
        shin_hits = [first(front[:, leg] > 5.) for leg in range(2)]
        both_shins = None if None in shin_hits else max(shin_hits)
        shin_on = first((front > 5.).any(-1))
        toe_on = first((toe > 5.).any(-1))
        airborne = (ground.max(-1) < 5.) & (front.max(-1) < 5.) & (toe.max(-1) < 5.) & (bad < 5.)
        air = first(np.lib.stride_tricks.sliding_window_view(airborne,8).all(-1)) if end>=2 else None
        state = states[i,1:end+1]
        speed = np.linalg.norm(state[:,19:22],axis=-1)
        spin = np.linalg.norm(state[:,22:25],axis=-1)
        up = 1-2*(state[:,4]**2+state[:,5]**2)
        support = (front.min(-1)>7.) & (ground[:,:2].min(-1)>7.) & (front.sum(-1)>toe.sum(-1)) & (bad<5.)
        # Test the 100 ms window on 200 Hz boundaries. Control-state values
        # are held across each 20 ms interval for the attitude/speed gate.
        stable = support & np.repeat((speed<.43) & (spin<2.5) & (up>.63),4)
        if air is not None:
            stable[:air+7] = False
        longest = 0
        streak = 0
        for supported in stable:
            streak = streak + 1 if supported else 0
            longest = max(longest, streak)
        start = first(np.lib.stride_tricks.sliding_window_view(stable,20).all(-1)) if len(stable)>=20 else None
        hold_end = None if start is None else start+20
        direct = both_shins is not None and (toe_on is None or both_shins < toe_on)
        impulse = ground[max(0,air-20):air].sum(0)*.005 if air else np.zeros(4)
        share = float(impulse[2:].sum()/max(impulse.sum(),1e-9))
        knee = np.abs(np.rad2deg(states[i,:, [9,12]]))
        min_push = float(knee[:,:air//4+1].min()) if air else None
        min_until_wall = float(knee[:,:shin_on//4+1].min()) if shin_on is not None else None
        checks = dict(front_led=air is not None and share>=.75,
                      flexed=min_push is not None and min_push>=88 and min_until_wall is not None and min_until_wall>=83,
                      shin_first=direct, held=hold_end is not None and air is not None and hold_end>air,
                      clean=hold_end is not None and not (bad[:hold_end]>=5.).any())
        assess_start = min(end, (both_shins or 84)//4+1)
        assess_end = min(end, assess_start+7)
        pose = states[i,assess_start:assess_end+1]
        feet = contacts["feet"][i,assess_start:assess_end+1]
        rear_stagger = float(np.mean(abs(feet[:,0,0]-feet[:,1,0])))
        advance = float(pose[-1,0]-pose[0,0])
        flexion = float(np.rad2deg(abs(pose[-1,[15,18]])-abs(pose[0,[15,18]])).min())
        pre = None if both_shins is None else float(states[i,max(0,both_shins//4-1),21])
        post = None if both_shins is None else float(states[i,both_shins//4:min(end+1,both_shins//4+8),21].max())
        bend = None if both_shins is None else float(np.rad2deg(abs(states[i,max(0,both_shins//4-1),[15,18]])).min())
        peak = None if shin_on is None else float(front[shin_on:min(len(front),shin_on+20)].max())
        rows.append(dict(pre_vz=pre, post_vz=post, bend_deg=bend, front_contact_peak_n=peak, rear_stagger_m=rear_stagger, advance_m=advance, front_flex_delta_deg=flexion,trial=i%args.envs+1, seed=seeds[(i//args.envs)%len(seeds)], passed=all(checks.values()),
                         checks=checks, hold_samples=longest, front_share_last_100ms=share, rear_min_push_deg=min_push,
                         rear_min_until_wall_deg=min_until_wall, toe_sample=toe_on,
                         shin_sample=shin_on, both_shin_sample=both_shins, hold_end_sample=hold_end,
                         max_nm=float(np.abs(physics["torque"][i,valid]).max())))
    ok = np.array([row["passed"] for row in rows])
    if args.repeat_profiles:
        groups = [rows[i:i+cohort_size] for i in range(0,count,cohort_size)]
        metrics = [(sum(row["passed"] for row in group),
                    float(np.mean([row["post_vz"] if row["post_vz"] is not None else 3. for row in group])),
                    float(np.mean([row["advance_m"] for row in group]))) for group in groups]
        best = max(range(len(groups)),key=lambda index:(metrics[index][0],-metrics[index][1],metrics[index][2]))
        args.output.mkdir(parents=True)
        np.save(args.output/"top-profiles.npy",compliance[best*cohort_size:best*cohort_size+1])
        write_json(args.output/"evaluation.json",dict(trials=count, metrics=metrics, best=best))
        print("Repeated profiles best",best,metrics[best],flush=True)
        return
    if args.profiles or (args.compliance and not args.single):
        def rank(i):
            row = rows[i]
            if args.compliance:
                quality = (-(row["post_vz"] if row["post_vz"] is not None else 3.)
                           + .004*(row["bend_deg"] or 0.) - row["rear_stagger_m"] + 4*row["advance_m"])
                return (row["passed"], row["checks"]["shin_first"], quality, row["hold_samples"])
            return (row["passed"], row["checks"]["shin_first"], row["hold_samples"],
                    row["front_share_last_100ms"])
        best = sorted(range(count), key=rank, reverse=True)[:32]
        best_index = best[0]
        stop = rows[best_index]["hold_end_sample"]
        if rows[best_index]["passed"] and compliance is None:
            steps = (stop+3)//4
            tick = np.arange(1,steps+1)
            alpha = np.clip((tick-15)/8,0,1)[:,None]
            beta = np.clip((tick-23)/6,0,1)[:,None]
            selected = targets[np.minimum(tick-1,len(targets)-1)] + alpha*profiles[best_index,0] + beta*profiles[best_index,1]
            args.output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output/"best-teacher.npz", targets=selected, states=states[best_index,:steps+1])
        args.output.mkdir(parents=True, exist_ok=True)
        np.save(args.output/"top-profiles.npy", (profiles if compliance is None else compliance)[best])
        write_json(args.output/"evaluation.json", dict(trials=count, successes=int(ok.sum()),
                   per_trial=[rows[i] for i in best]))
        print("Candidate results",int(ok.sum()),"/",count, "best",rows[best[0]],flush=True)
        return
    args.output.mkdir(parents=True)
    write_json(args.output / "evaluation.json", dict(stage="first", trials=count, successes=int(ok.sum()), per_trial=rows))
    np.savez_compressed(args.output / "rollouts.npz", states=states, stop=first_end,
                        success=ok, offsets=offsets, **contacts)
    np.savez_compressed(args.output / "physics_200hz.npz", **physics)
    print([(seed,sum(r["passed"] for r in rows if r["seed"]==seed)) for seed in seeds], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--asset", type=Path, required=True)
    prep.add_argument("--reference", type=Path, required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--iterations", type=int, default=400)
    fit.add_argument("--seed", type=int, default=607)
    check = sub.add_parser("evaluate")
    check.add_argument("--checkpoint", type=Path)
    check.add_argument("--profiles", type=Path)
    check.add_argument(
        "--compliance", type=Path,
        help="N×8 .npy rows: front and rear gain scales, front tuck and reach, fore pull, rear bend, ramp steps, rear damping scale",
    )
    check.add_argument("--repeat-profiles", action="store_true")
    check.add_argument("--single", action="store_true", help="Apply the first compliance profile to all cohort starts")
    check.add_argument("--profile-file", type=Path)
    check.add_argument("--cohort-seeds", type=int, nargs="+", default=[301,1621])
    for name, option in (("train",fit),("evaluate",check)):
        option.add_argument("--output", type=Path, required=True)
        option.add_argument("--device", default="cuda:0")
        option.add_argument("--envs", type=int, default=4096 if name=="train" else 16)
    args = parser.parse_args()
    {"prepare":prepare, "train":train, "evaluate":evaluate}[args.mode](args)


if __name__ == "__main__":
    main()
