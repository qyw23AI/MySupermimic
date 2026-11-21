"""Closed-loop SuperMimic training for rear-shin support transfer and full wall crossing."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import mujoco as mj
import numpy as np

from . import calf_tracking as tracker
from .brace_experiment import CONTROL_DT, SUBSTEPS, make_model, simulate
from .cli import write_json
from .robot import HEIGHT, JOINTS, LEGS, NAME

STAGES = ("descent", "full")
SUB_DT = CONTROL_DT / SUBSTEPS


def prepare(args):
    from mujoco.rollout import Rollout

    report = json.loads((args.source / "evaluation.json").read_text())["continuous"]
    if not report["four_stage_pass"]:
        raise ValueError("The source must pass the continuous four-stage replay")
    with np.load(args.source / "continuous-replay.npz", allow_pickle=False) as data:
        initial, targets = data["initial"], data["controls"]
        poses = data["qpos"]
    args.work.mkdir(parents=True, exist_ok=True)
    asset = args.work / "wbl_tracking.xml"
    shutil.copyfile(args.asset, asset)
    model = make_model(asset)
    # Re-run physics to recover real velocities instead of differentiating poses.
    with Rollout(nthread=1) as runner:
        raw, _ = simulate(runner, model, [mj.MjData(model)], initial, targets[None])
    states = np.vstack((initial[1:38], raw[0,SUBSTEPS-1::SUBSTEPS,1:]))
    np.testing.assert_allclose(states[:,:19], poses, atol=2.e-6, rtol=0.)
    catch = report["rear_shin_catch_frame"]
    descent_states, descent_targets = states[catch:], targets[catch:]
    if args.descent_teacher:
        with np.load(args.descent_teacher, allow_pickle=False) as refined:
            descent_states, descent_targets = refined["states"], refined["targets"]
        expected = states[catch,:19].copy()
        # The Warp reset clips joint positions to the articulation limits.
        expected[7:] = np.clip(expected[7:],model.jnt_range[1:,0],model.jnt_range[1:,1])
        if (descent_states.shape != (len(descent_targets)+1,37) or
                np.max(np.abs(descent_states[0,:19] - expected)) > .004):
            raise ValueError("Refined descent must preserve the verified split-level start")
    tracker.write_clip(asset, args.work/"descent", descent_states, descent_targets, 0, 0)
    if args.full_teacher:
        with np.load(args.full_teacher, allow_pickle=False) as refined:
            revised_states, revised_targets = refined["states"], refined["targets"]
        if (revised_states.shape != (len(revised_targets)+1,37) or
                not np.isfinite(revised_states).all() or
                np.max(np.abs(revised_states[0,:7] - states[0,:7])) > .001):
            raise ValueError("Refined full teacher must keep the nominal wall approach")
        states, targets = revised_states, revised_targets
    tracker.write_clip(asset, args.work/"full", states, targets, report["unload_frame"], report["first_handoff_frame"])
    write_json(args.work/"reference.json", {
        "source": str(args.source.resolve()), "catch_frame": catch, "steps": len(targets),
        "descent_teacher": str(args.descent_teacher.resolve()) if args.descent_teacher else None,
        "full_teacher": str(args.full_teacher.resolve()) if args.full_teacher else None,
        "cohort_offsets_m_m_rad": tracker.cohort().tolist(),
    })
    print(f"Prepared descent ({len(targets)-catch} controls) and full route ({len(targets)} controls)", flush=True)


class TransferScore(tracker.CalfScore):
    """Require the two shins to catch, then a supported transfer including simultaneous footfalls."""

    def __init__(self, count, stage, initial_state, feet, stand_frames=10):
        super().__init__(count, stage, initial_state, feet)
        self.caught = np.zeros(count, dtype=bool)
        self.catch_run = np.zeros(count, dtype=np.int32)
        self.final_run = self.catch_run.copy()
        self.unsupported_run = self.catch_run.copy()
        self.extra_hop = self.caught.copy()
        self.first_catch = self.catch_run.copy()
        self.step_number = 0
        self.stand_frames = stand_frames

    def step(self, active, state, c):
        self.step_number += 1
        if self.stage == "full":
            super().step(active, state, c)
        else:
            self.hit |= active & (c["bad_n"] >= 5.)
        floor, feet = c["ground"], c["feet"]
        front = ((floor[:,2:] > 5.) & (feet[:,2:,0] > .08)).all(-1)
        speed = np.linalg.norm(state[:,19:22], axis=-1)
        spin = np.linalg.norm(state[:,22:25], axis=-1)
        up = 1. - 2. * (state[:,4]**2 + state[:,5]**2)
        rear_support = c["rear_calf"].min(-1) > 7.
        settle = (up > .65) & (state[:,2] > .27) & (speed < .42) & (spin < 2.5)
        catch_now = active & front & rear_support & settle & ~self.hit & ~self.recontact
        if self.stage == "full":
            catch_now &= self.second_air
        self.catch_run = np.where(catch_now, self.catch_run + 1, 0)
        newly_caught = ~self.caught & (self.catch_run >= (6 if self.stage == "full" else 2))
        self.caught |= newly_caught
        self.first_catch[newly_caught] = self.step_number
        self.unsupported_run = np.where(active & self.caught & c["air"], self.unsupported_run + 1, 0)
        self.extra_hop |= self.unsupported_run >= 2
        finished = (active & self.caught & ~self.hit & ~self.recontact & ~self.extra_hop
                    & c["clear"] & (floor.min(-1) > 5.) & (speed < .35) & (spin < 2.)
                    & (up > .82) & (state[:,2] > .22))
        self.final_run = np.where(finished, self.final_run + 1, 0)
        return self.final_run >= self.stand_frames


class PhysicsAudit:
    """Read simulated actuator torque and contact loads at each 200 Hz step."""

    def __init__(self, env):
        import torch
        self.torch = torch
        self.env = env
        self.wall_links = env.scene["wall_links"]
        self.floor_links = env.scene["floor_links"]
        self.wall = env.scene["wall_feet"]
        self.floor = env.scene["floor_feet"]
        self.front = self.indices(self.wall_links, [f"{leg}_calf_collision" for leg in LEGS[2:]])
        self.rear = self.indices(self.wall_links, [f"{leg}_calf_collision" for leg in LEGS[:2]])
        self.other = [i for i in range(len(self.wall_links.primary_names)) if i not in self.front + self.rear]
        self.fi = self.indices(self.floor, [f"{leg}_foot_ball" for leg in LEGS])
        self.wi = self.indices(self.wall, [f"{leg}_foot_ball" for leg in LEGS])
        model = env.sim.mj_model
        self.dof = [int(model.jnt_dofadr[model.joint("robot/"+j).id]) for j in JOINTS]
        self.frames = []
        self.read_base = tracker.contact_measurements(env)
        substep = env.metrics_manager.compute_substep

        def record():
            substep()
            self.frames.append(self.read())
        env.metrics_manager.compute_substep = record

    @staticmethod
    def indices(sensor, names):
        return [next(i for i, actual in enumerate(sensor.primary_names) if actual.endswith(n)) for n in names]

    @staticmethod
    def force(contact):
        return contact.force[...,0].abs() * (contact.found > 0)

    def read(self):
        torch = self.torch
        links, floor_links, floor, wall = [
            s._extract_sensor_data() for s in (self.wall_links, self.floor_links, self.floor, self.wall)
        ]
        link_force = self.force(links)
        calf_indices = self.front + self.rear
        calf_normal = links.normal[:, calf_indices, 2].abs()
        top = (calf_normal >= .75) & (links.pos[:,calf_indices,2] >= HEIGHT-.013)
        calf_force = link_force[:,calf_indices]
        top_load = torch.where(top, calf_force*calf_normal, 0.)
        bad = torch.maximum(self.force(floor_links).amax(-1), link_force[:,self.other].amax(-1))
        bad = torch.maximum(bad, torch.where(~top,calf_force,0.).amax(-1))
        floor_force = self.force(floor)[:,self.fi] * floor.normal[:,self.fi,2].abs()
        toe_force = self.force(wall)[:,self.wi]
        # DOFs are one-to-one with the twelve position actuators. These are
        # actual simulated generalized drive forces in Nm, not targets or PD estimates.
        torque = self.env.sim.data.qfrc_actuator[:,self.dof].clone()
        return dict(bad=bad, calf=calf_force, top=top_load, ground=floor_force, toe=toe_force, torque=torque)

    def control_frame(self):
        read = self.read_base()
        samples = {k: self.torch.stack([row[k] for row in self.frames],1) for k in self.frames[0]}
        current = self.read()
        read["bad_n"] = self.torch.maximum(current["bad"], samples["bad"].amax(1))
        read["calf_any"] = self.torch.maximum(current["calf"][:,:2],samples["calf"][:,:,:2].amax(1))
        read["rear_calf"] = current["top"][:,2:]
        read["air"] &= (current["calf"].max(-1).values < 5.)
        return tracker.as_numpy(read), tracker.as_numpy(samples)


def impact(samples, onset):
    n = np.flatnonzero((samples.max(-1)>12.) & (np.arange(len(samples))>=onset))
    if not n.size:
        return None
    first=int(n[0])
    win = samples[first:first+round(.10/SUB_DT)]
    return {"onset_s": first*SUB_DT, "peak_pair_n":float(win.sum(-1).max()),
            "peak_per_leg_n":win.max(0).tolist(), "impulse_ns_100ms": float(win.sum()*SUB_DT)}


class LandingStabilizer:
    """After wall clearance, smoothly reduce stepping and damp the standing pose."""

    def __init__(self, env, profiles):
        import torch

        parameters = torch.as_tensor(profiles, dtype=torch.float32, device=env.device)
        self.weight, self.height, self.damping, self.ramp, self.contacts = parameters.unbind(-1)
        robot = env.scene["robot"]
        indices,_ = robot.find_joints(list(JOINTS),preserve_order=True)
        self.ids = robot.indexing.joint_ids[indices]
        self.data = env.sim.data
        self.age = torch.zeros(env.num_envs,device=env.device)
        self.anchor = torch.zeros(env.num_envs,4,3,device=env.device)
        self.default = robot.data.default_joint_pos

    def apply(self, action, state, contact, eligible):
        import torch

        feet = torch.as_tensor(contact["feet"],device=action.device)
        grounded = (contact["ground"]>5.).sum(-1)
        entry = torch.as_tensor(eligible & contact["clear"],device=action.device)
        entry &= torch.as_tensor(grounded,device=action.device)>=self.contacts
        new = entry & (self.age==0)
        self.anchor[new] = feet[new]
        self.anchor[new,:,2] = .0238
        self.age += ((self.age>0)|entry).to(self.age.dtype)
        position = state[:,:3]
        correction = self.anchor-feet
        correction[...,:2] += .045*state[:,None,19:21]
        correction[...,2] -= (self.height-position[:,2]).clamp(-.06,.04)[:,None]
        posture = torch.stack((-2*state[:,4],-2*state[:,5],state[:,5]*0),-1)
        posture -= .07*state[:,22:25]
        correction -= torch.cross(posture[:,None].expand_as(feet),feet-position[:,None],dim=-1)
        axes = self.data.xaxis[:,self.ids,:].reshape(-1,4,3,3)
        pivots = self.data.xanchor[:,self.ids,:].reshape_as(axes)
        jac = torch.cross(axes,feet[:,:,None]-pivots,dim=-1).transpose(-1,-2)
        damping = .014*torch.eye(3,device=action.device)
        dq = (jac.transpose(-1,-2) @ torch.linalg.solve(
            jac@jac.transpose(-1,-2)+damping,correction.clamp(-.09,.09)[...,None])).squeeze(-1)
        target = state[:,7:19] + dq.flatten(1).clamp(-.25,.25) - self.damping[:,None]*state[:,25:37]
        mix = (self.age/self.ramp).clamp(0,1)*self.weight
        return torch.lerp(action,(target-self.default)/.25,mix[:,None])


class RearShinBalance:
    """Adjust second-flight placement and correct side-to-side load during the catch."""

    def __init__(self, profiles, device):
        import torch

        self.parameters = torch.as_tensor(profiles, dtype=torch.float32, device=device)
        self.age = torch.zeros(len(profiles), device=device)
        self.lateral_anchor = torch.zeros_like(self.age)

    def apply(self, action, state, loads, airborne, eligible):
        import torch

        flight = torch.as_tensor(airborne, device=action.device)
        eligible = torch.as_tensor(eligible, device=action.device)
        entered = flight & (self.age == 0)
        self.lateral_anchor[entered] = state[entered, 1]
        self.age += (flight | (self.age > 0)).to(self.age.dtype)
        age = self.age
        flight_window = (age/6).clamp(0,1) * ((25-age)/9).clamp(0,1)
        load = torch.as_tensor(loads, device=action.device)
        asymmetry = ((load[:,0]-load[:,1])/100.).clamp(-1,1)
        grounded = (load.max(-1).values > 7.).to(action.dtype)
        hip, thigh, knee, roll_gain, load_gain, front_gain = self.parameters[:,:6].unbind(-1)
        roll = 2*(state[:,3]*state[:,4] + state[:,5]*state[:,6])
        correction = ((roll_gain*roll + load_gain*asymmetry*grounded)*((58-age)/12).clamp(0,1)).clamp(-.25,.25)
        enabled = eligible * (age>0)
        adjustment = torch.stack((hip*flight_window, thigh*flight_window,
                                  knee*flight_window+correction, front_gain*roll), -1)
        adjustment = adjustment*enabled[:,None]/.25
        result = action.clone()
        result[:,[0,1,2,8]] -= adjustment
        result[:,[3,4,5,11]] += adjustment
        if self.parameters.shape[1] == 10:
            rear_y, front_y, left_load, velocity = self.parameters[:,6:].unbind(-1)
            window = (age/6).clamp(0,1)*((58-age)/12).clamp(0,1)*enabled
            shift = ((state[:,1]-self.lateral_anchor)*5 + state[:,20]*velocity).clamp(-.5,.5)
            result[:,[0,3]] += (window*rear_y*shift/.25)[:,None]
            result[:,[6,9]] += (window*front_y*shift/.25)[:,None]
            result[:,5] += window*grounded*asymmetry.clamp(0,1)*left_load/.25
        return result


def rear_catch_metrics(top, state, release):
    start = release * SUBSTEPS
    loaded = top[start:,2:] > 7.
    union = np.flatnonzero(loaded.any(-1))
    if not len(union):
        return None
    onset = start + int(union[0])
    window = top[onset:onset+60,2:]
    per_leg = window.sum(axis=0) * .005
    dual = window.min(-1) > 7.
    edges = np.flatnonzero(np.diff(np.r_[False,dual,False]))
    longest_dual = np.diff(edges)[::2].max(initial=0)
    lo, hi = onset//SUBSTEPS, (onset+60)//SUBSTEPS+1
    stance = state[lo:hi]
    w,x,y,z = stance[:,3:7].T
    roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    return dict(
        onset_s=onset*.005,
        impulse_right_left_ns=per_leg.tolist(),
        impulse_imbalance=float(abs(per_leg[0]-per_leg[1])/max(per_leg.sum(),1.e-6)),
        left_impulse_share=float(per_leg[1]/max(per_leg.sum(),1.e-6)),
        dual_contact_s=float(dual.sum()*.005),
        longest_dual_s=float(longest_dual*.005),
        roll_peak_deg=float(np.rad2deg(abs(roll).max())),
        side_shift_m=float(abs(stance[:,1]-stance[0,1]).max()),
        left_shift_m=float((stance[:,1]-stance[0,1]).max()),
    )


def evaluate(args):
    import torch
    from supermimic.envs.state_io import StateIO
    from supermimic.eval.benchmark import load_policy_from_checkpoint

    actor = None
    if args.checkpoint:
        actor, cfg = load_policy_from_checkpoint(args.checkpoint, args.device)
        if cfg["robot"] != NAME or not cfg["env"]["anchor_obs"]:
            raise ValueError("Checkpoint is incompatible with the WBL wall environment")
    descent_actor = None
    if args.descent_checkpoint:
        if args.stage != "full" or actor is None:
            raise ValueError("A descent handoff requires a full-route actor")
        descent_actor, descent_cfg = load_policy_from_checkpoint(args.descent_checkpoint, args.device)
        if descent_cfg["robot"] != NAME or not descent_cfg["env"]["anchor_obs"]:
            raise ValueError("Incompatible descent actor")
    offsets = np.zeros((args.envs,3)) if args.nominal else tracker.cohort(args.cohort_seed,args.envs)
    profiles = None
    cohort_size = args.envs
    rear_profiles = None
    if args.rear_profiles:
        if actor is None or args.stage != "full" or args.landing_profiles or args.export_amp:
            raise ValueError("Rear profiles require a full-route actor and no other sweep or export")
        rear_profiles = np.load(args.rear_profiles, allow_pickle=False)
        if (rear_profiles.ndim != 2 or rear_profiles.shape[1] not in (6,11,15) or not len(rear_profiles)
                or not np.isfinite(rear_profiles).all()):
            raise ValueError("Expected N×6, N×11 or N×15 finite rear profiles")
        if rear_profiles.shape[1] >= 11:
            landing = rear_profiles[:,-5:]
            if (args.stabilize_landing or (landing[:,0]<0).any() or (landing[:,0]>1).any()
                    or (landing[:,1]<=0).any() or (landing[:,2]<0).any()
                    or (landing[:,3]<1).any() or not np.isin(landing[:,4],(2,3,4)).all()):
                raise ValueError("Invalid embedded landing profiles")
        offsets = np.tile(offsets,(len(rear_profiles),1))
        args.envs = len(offsets)
    if args.landing_profiles:
        profiles = np.load(args.landing_profiles,allow_pickle=False)
        if args.stabilize_landing or args.export_amp:
            raise ValueError("--landing-profiles is incompatible with --stabilize-landing and --export-amp")
        if (profiles.ndim != 2 or profiles.shape[1] != 5 or not len(profiles)
                or not np.isfinite(profiles).all() or (profiles[:,3]<1).any()
                or ((profiles[:,0]<0)|(profiles[:,0]>1)).any()
                or (profiles[:,1]<=0).any() or (profiles[:,2]<0).any()
                or not np.isin(profiles[:,4],(2,3,4)).all()):
            raise ValueError("Invalid landing profile values")
        offsets = np.tile(offsets,(len(profiles),1))
        args.envs = len(offsets)
    if not np.isfinite(args.stand_seconds) or not .2 <= args.stand_seconds <= .8:
        raise ValueError("--stand-seconds must be between 0.2 and 0.8")
    if args.stand_seconds > .2 and args.export_amp:
        raise ValueError("Extended stance verification is incompatible with AMP export")
    stand_frames = int(np.ceil(args.stand_seconds / CONTROL_DT))
    env = tracker.make_evaluation(args.work,args.stage,args.device,args.envs,args.seed)
    try:
        if stand_frames > 10:
            # Keep the final goal active while testing standing past the clip boundary.
            env.termination_manager.get_term_cfg("clip_end").func = lambda env, **kwargs: env.episode_length_buf < 0
        command=env.command_manager.get_term("motion")
        if descent_actor is not None:
            from supermimic.envs.mdp.motion_library import MotionLibrary
            from supermimic.world_model.heads import anchor_relative_obs

            descent_library = MotionLibrary(
                [args.work/"descent"/"motion.npz"], command.library.body_names,
                command.robot.data.default_joint_pos[0], device=env.device,
                root_body=command.library.root_body, robot=NAME,
            )
            start = descent_library.clip_start[0]
            last = descent_library.clip_last[0]

        tracker.install_offsets(command,torch.as_tensor(offsets,dtype=torch.float32,device=env.device))
        with np.load(args.work/f"{args.stage}-teacher.npz") as teacher:
            targets=torch.as_tensor(teacher["targets"],dtype=torch.float32,device=env.device)
            release=int(teacher["release"])
        if args.targets:
            if actor is not None:
                raise ValueError("--targets only applies to open-loop replay")
            with np.load(args.targets, allow_pickle=False) as custom:
                targets=torch.as_tensor(custom["targets"],dtype=torch.float32,device=env.device)
        audit=PhysicsAudit(env)
        io=StateIO(env)
        config = profiles
        if config is None and args.stabilize_landing:
            config = np.array([[args.stabilize_landing,.32,.007,5.,3.]])
        if rear_profiles is not None and rear_profiles.shape[1] >= 11:
            config = rear_profiles[:,-5:]
        stabilizer = LandingStabilizer(env,np.repeat(config,args.envs//len(config),axis=0)) if config is not None else None
        rear_balance = RearShinBalance(np.repeat(rear_profiles[:,:10 if rear_profiles.shape[1] == 15 else 6],cohort_size,axis=0),env.device) if rear_profiles is not None else None
        if stabilizer and (actor is None or args.stage!="full" or descent_actor is not None):
            raise ValueError("Landing stabilization requires a full-route actor")
        with torch.inference_mode():
            obs,_=env.reset()
            states=[io.read().cpu().numpy()]
            contact=tracker.as_numpy(audit.read_base())
            start_contact=audit.read()
            contact["rear_calf"]=start_contact["top"][:,2:].cpu().numpy()
            contact["bad_n"]=start_contact["bad"].cpu().numpy()
            history=[contact]
            subphysics=[]
            scorer=TransferScore(args.envs,args.stage,states[0],contact["feet"],stand_frames)
            success=np.zeros(args.envs,dtype=bool)
            active=~success.copy()
            stop=np.zeros(args.envs,dtype=int)
            for step in range(1,len(targets)+14+max(0,stand_frames-10)):
                audit.frames.clear()
                action=(actor(obs["policy"],obs["goal"]) if actor is not None else
                        (targets[min(len(targets)-1,step-1)]-command.robot.data.default_joint_pos)/.25)
                if descent_actor is not None and (active & scorer.caught).any():
                    ages = torch.as_tensor(step - scorer.first_catch, device=env.device)
                    frame = (start + ages.clamp_min(1)).clamp_max(last)
                    ref = descent_library.ref(frame)
                    current = io.read()
                    goal = torch.cat((descent_library.goal_window(frame, command.cfg.future_frames),
                                     anchor_relative_obs(current[:,:3], current[:,3:7],
                                                         ref.root_pos_w, ref.root_quat_w)), dim=-1)
                    down = descent_actor(obs["policy"],goal)
                    transfer = torch.as_tensor(active & scorer.caught,device=env.device)
                    mix = (ages / max(1,args.handoff_frames)).clamp(0.,1.) * transfer
                    action = torch.lerp(action, down, mix[:,None])
                if rear_balance:
                    action = rear_balance.apply(action,io.read(),history[-1]["rear_calf"],scorer.second_air,
                                                active & ~scorer.hit & ~history[-1]["clear"])
                if stabilizer:
                    action = stabilizer.apply(action,io.read(),history[-1],
                                              active & scorer.caught & ~scorer.hit & ~scorer.recontact)
                obs,_,term,timeout,_=env.step(action)
                states.append(io.read().cpu().numpy())
                c, sample=audit.control_frame()
                history.append(c)
                subphysics.append(sample)
                actual_done=(term|timeout).cpu().numpy()
                passed=scorer.step(active,states[-1],c)&~term.cpu().numpy()
                success |= passed
                end=active & (passed|actual_done|~np.isfinite(states[-1]).all(-1))
                stop[end]=step
                active &= ~end
                if not active.any():
                    break
                if actual_done.any():
                    obs,_=env.reset(env_ids=torch.nonzero(term|timeout,as_tuple=True)[0])
            stop[active]=len(states)-1
    finally:
        env.close()
    states=np.stack(states,axis=1)
    recorded={k:np.stack([value[k] for value in history],1) for k in history[0]}
    physics={k:np.concatenate([value[k] for value in subphysics],1) for k in subphysics[0]}
    trials=[]
    for i, cut in enumerate(stop):
        end=int(cut)*SUBSTEPS
        torque=np.abs(physics["torque"][i,:end]).max(0)
        floor=physics["ground"][i,:end]
        catch=int(scorer.first_catch[i])
        trials.append({"trial":i+1,"pass":bool(success[i]),"rear_catch_at_s":catch*CONTROL_DT if catch else None,
                       "rear_balance":rear_catch_metrics(physics["top"][i,:end],states[i,:cut+1],release) if args.stage=="full" else None,
                       "front_ground_impact":impact(floor[:,2:],release*SUBSTEPS) if args.stage=="full" else None,
                       "rear_shin_wall_impact":impact(physics["top"][i,:end,2:],release*SUBSTEPS) if args.stage=="full" else None,
                       "rear_ground_impact":impact(floor[:,:2],catch*SUBSTEPS) if catch else None,
                       "peak_joint_torque_nm":dict(zip(JOINTS,torque.tolist())),
                       "joint_limit_near_samples":int((np.abs(physics["torque"][i,:end])>=23.69).any(-1).sum())})
    args.output.mkdir(parents=True, exist_ok=False)
    if not args.summary_only:
        np.savez_compressed(args.output/"rollouts.npz",states=states,stop=stop,success=success,offsets=offsets,**recorded)
        np.savez_compressed(args.output/"physics_200hz.npz",**physics)
    report={"stage":args.stage,"policy":"learned" if actor else "replay","trials":args.envs,
            "successes":int(success.sum()),"cohort_seed":args.cohort_seed,"nominal":args.nominal,
            "strict_contact":True,"physics_hz":200,"stand_frames":stand_frames,"amp_exported":0,"per_trial":trials,
            "checkpoint_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None}
    if profiles is not None:
        report["landing_sweep"] = []
        for i,row in enumerate(profiles):
            subset=slice(i*cohort_size,(i+1)*cohort_size)
            report["landing_sweep"].append({
                "profile":row.tolist(),"passed":int(success[subset].sum()),
                "caught":int(scorer.caught[subset].sum()),"hit":int(scorer.hit[subset].sum())})
    if rear_profiles is not None:
        report["rear_sweep"] = []
        for i, row in enumerate(rear_profiles):
            subset = slice(i*cohort_size,(i+1)*cohort_size)
            readings = [trial["rear_balance"] for trial in trials[subset] if trial["rear_balance"] is not None]
            summary = {key:float(np.median([item[key] for item in readings])) if readings else None
                       for key in ("impulse_imbalance","left_impulse_share","dual_contact_s","longest_dual_s","roll_peak_deg","side_shift_m","left_shift_m")}
            report["rear_sweep"].append(dict(profile=row.tolist(), passed=int(success[subset].sum()),
                    caught=int(scorer.caught[subset].sum()), hit=int(scorer.hit[subset].sum()),
                    extra_hop=int(scorer.extra_hop[subset].sum()), **summary))
    if args.stabilize_landing:
        report["landing_stabilizer_weight"] = args.stabilize_landing
    if args.descent_checkpoint:
        report["descent_checkpoint_sha256"] = hashlib.sha256(args.descent_checkpoint.read_bytes()).hexdigest()
        report["handoff_blend_frames"] = args.handoff_frames
    for field in ("first_brace","released","second_air","caught","hit","recontact","extra_hop"):
        report[field]=int(getattr(scorer,field).sum())
    if (args.export_amp and args.stage=="full" and actor and not args.nominal and args.envs==16
            and args.cohort_seed==301 and success.sum()>=tracker.EXPORT_MINIMUM):
        export=args.output/"amp"
        export.mkdir()
        files=[]
        for i in np.flatnonzero(success):
            name=f"wall_four_stage_{i:04d}.csv"
            np.savetxt(export/name,states[i,:stop[i]+1,:19][:,[0,1,2,4,5,6,3,*range(7,19)]],delimiter=",",fmt="%.9g")
            files.append(name)
        write_json(export/"train.json",{"root":".","version":1,"source_fps":50,"quaternion_order":"xyzw",
                                        "joint_names":list(JOINTS),"files":files})
        report["amp_exported"]=len(files)
    write_json(args.output/"evaluation.json",report)
    print({k:v for k,v in report.items() if k!="per_trial"},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work",type=Path,required=True)
    sub=p.add_subparsers(dest="command",required=True)
    prep=sub.add_parser("prepare")
    prep.add_argument("--source",type=Path,required=True)
    prep.add_argument("--asset",type=Path,required=True)
    prep.add_argument("--descent-teacher",type=Path)
    prep.add_argument("--full-teacher",type=Path)
    fit=sub.add_parser("train")
    play=sub.add_parser("evaluate")
    for cmd in (fit,play):
        cmd.add_argument("--stage",choices=STAGES,default="descent")
        cmd.add_argument("--output",type=Path,required=True)
        cmd.add_argument("--envs",type=int,default=96 if cmd is fit else 16)
        cmd.add_argument("--seed",type=int,default=511 if cmd is fit else 801)
        cmd.add_argument("--device",default="cuda:0")
    fit.add_argument("--warm",type=Path)
    fit.add_argument("--actor-lr",type=float,default=4.e-6)
    fit.add_argument("--anchor",type=float,default=90.)
    fit.add_argument("--residual-limit",type=float,default=.35)
    fit.add_argument("--balance-start",type=int,dest="landing_feedback")
    fit.add_argument("--env-batch",type=int,dest="batch",default=96)
    fit.add_argument("--horizon",type=int,default=8)
    fit.add_argument("--iterations",type=int,default=550)
    fit.add_argument("--save-every",type=int,default=200)
    fit.add_argument("--teacher-rounds",type=int,default=4)
    fit.add_argument("--pretrain",type=int,default=1300)
    fit.add_argument("--explore",type=float,default=.028)
    fit.set_defaults(guide=True)
    play.add_argument("--cohort-seed",type=int,default=301)
    play.add_argument("--nominal",action="store_true")
    play.add_argument("--checkpoint",type=Path)
    play.add_argument("--targets",type=Path)
    play.add_argument("--descent-checkpoint",type=Path)
    play.add_argument("--handoff-frames",type=int,default=6)
    play.add_argument("--export-amp",action="store_true")
    play.add_argument("--stabilize-landing",type=float,choices=(0.,.5,.8,1.),default=0.)
    play.add_argument(
        "--landing-profiles", type=Path,
        help="N×5 .npy profiles: blend, target height (m), joint damping (s), ramp (steps), minimum grounded feet",
    )
    play.add_argument("--rear-profiles", type=Path, help="N×6 rear balance N×11 rear and landing, or N×15 with lateral feedback")
    play.add_argument("--summary-only", action="store_true", help="Save evaluation metrics without raw trajectories")
    play.add_argument("--stand-seconds", type=float, default=.2, help="Required consecutive stable four-foot stance")
    args=p.parse_args()
    {"prepare":prepare,"train":tracker.train,"evaluate":evaluate}[args.command](args)


if __name__=="__main__":
    main()
