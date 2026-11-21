"""Compare two WBL second-jump strategies from an identical wall brace."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco as mj
from mujoco.rollout import Rollout
import numpy as np

from .brace_experiment import FEET, SUBSTEPS, make_model, simulate
from .contact_route import DT, PERTURBATION, add_sensors, first, measure, score
from .robot import HEIGHT, THICKNESS

VARIANTS = ("shins-push", "reach-then-push")
WINDOW = round(.10 / DT)


def launch_metrics(m, start):
    """Measure contact impulses and ordering at 200 Hz before the second flight."""
    n, ticks, _ = m["floor_all"].shape
    future = np.arange(ticks)[None, :] >= start * SUBSTEPS
    floor = m["floor_all"]
    shins = m["shin_all"][..., 2:]
    toe = m["toe_all"][..., 2:]
    air = first(m["air"][:, start:], 2)
    air_start = (start + air - 2) * SUBSTEPS
    # Find the last loaded physics sample before the first 50 Hz airborne frame.
    # This avoids including contact-free time in the pre-takeoff window.
    index = np.arange(ticks)[None, :]
    supports = floor.max(-1) + shins.max(-1) + toe.max(-1)
    last_support = np.where(future & (index < air_start[:, None] + SUBSTEPS) &
                            (supports >= 5.), index, -1).max(-1)
    launched = (air > 0) & (last_support >= start * SUBSTEPS)
    window = (index <= last_support[:, None]) & (
        index > np.maximum(last_support - WINDOW, start * SUBSTEPS - 1)[:, None]
    ) & launched[:, None]
    front_impulse = (shins * window[..., None]).sum(1) * DT
    rear_impulse = (floor[..., :2] * window[..., None]).sum(1) * DT
    toe_impulse = (toe * window[..., None]).sum(1) * DT
    vz = m["vz_all"]
    rising = vz > .18
    pairs = (shins.min(-1) > 10.) & (floor[..., :2].min(-1) > 10.)
    overlap = (pairs & window & rising).sum(-1) * DT
    last_dual_shin = np.where((shins.min(-1)>10.) & window,index,-1).max(-1)
    shin_to_release = np.where(last_dual_shin>=0,(last_support-last_dual_shin)*DT,-1.)
    reach = ((m["foot_all"][..., 2:, 0] > THICKNESS / 2 + .032) &
             (m["foot_all"][..., 2:, 2] > HEIGHT + .010)).all(-1)
    quiet_front = (shins.max(-1) < 6.) & (toe.max(-1) < 6.)
    # After the feet clear the crest, both hind feet must still produce
    # appreciable normal force during the upward motion.
    rear_drive = (floor[..., :2].min(-1) > 18.) & (floor[..., :2].sum(-1) > 125.) & rising
    cleared = first(reach & quiet_front & window, 2)
    drive = first(rear_drive & reach & quiet_front & window & (cleared[:, None] > 0) &
                  (index + 1 > cleared[:, None]), 2)
    toe_share = toe_impulse.sum(-1) / np.maximum(front_impulse.sum(-1) + rear_impulse.sum(-1)
                                               + toe_impulse.sum(-1), .01)
    no_hit = np.where(index <= last_support[:, None], m["bad"], 0.).max(-1) < 5.
    release_index = np.maximum(last_support,0)[:,None]
    before_air = np.take_along_axis(vz,release_index,1)[:,0]
    forward_release = np.take_along_axis(m["vx_all"],release_index,1)[:,0]
    early_air = (index>last_support[:,None]) & (index<=last_support[:,None]+round(.25/DT))
    launch_x = np.take_along_axis(m["x_all"],release_index,1)[:,0]
    flight_x = np.where(early_air,m["x_all"],launch_x[:,None]).max(-1)-launch_x
    directional = (forward_release>=.38) & (flight_x>=.10)
    coordination = {
        "shins-push": (overlap >= .010) & (front_impulse.min(-1) >= .7) &
                      (rear_impulse.min(-1) >= .7) & (toe_share <= .10),
        "reach-then-push": (cleared > 0) & (drive > cleared) &
                           (rear_impulse.min(-1) >= 1.) & (toe_share <= .05),
    }
    return dict(launched=launched, air=air, last_support=last_support,
                front=front_impulse, rear=rear_impulse, toe=toe_impulse, toe_share=toe_share,
                overlap=overlap, dual_shin_to_release_s=shin_to_release,
                reach=cleared, drive=drive, no_hit=no_hit,
                upward_at_release=before_air, forward_speed_at_release=forward_release,
                first_quarter_second_advance_m=flight_x,
                **{k: launched & no_hit & directional & v for k, v in coordination.items()})


def read(model, raw, sensors):
    measured = measure(model, raw, sensors)
    from .brace_experiment import signal
    from .four_stage import crest_load, force_signals
    measured["toe_all"] = crest_load(force_signals(model, sensors, "wall", FEET))
    measured["vz_all"] = raw[..., 22]
    measured["vx_all"] = raw[..., 20]
    measured["x_all"] = raw[..., 1]
    measured["foot_all"] = np.stack(
        [signal(model, sensors, f"position_{foot}") for foot in FEET], axis=-2)
    return measured


def catch_impact(m, d):
    ticks = np.arange(m["vz_all"].shape[1])[None,:]
    floor = m["floor_all"][...,2:]
    beyond_wall = m["foot_all"][...,2:,0]>THICKNESS/2+.025
    landed = ((floor>20.) & beyond_wall).any(-1) & (ticks>d["last_support"][:,None])
    touch = first(landed)-1
    window = (touch[:,None]>=0) & (ticks>=touch[:,None]) & (ticks<touch[:,None]+round(.18/DT))
    peak = np.where(window,floor.sum(-1),0.).max(-1)
    rise = np.where(window,m["vz_all"],0.).max(-1)
    incoming = np.take_along_axis(m["vz_all"],np.maximum(touch-1,0)[:,None],1)[:,0]
    after_touch = (touch[:,None]>=0) & (ticks>=touch[:,None])
    later_rise = np.where(after_touch,m["vz_all"],0.).max(-1)
    return touch,peak,rise,np.maximum(-incoming,0.),later_rise


def objective(m, start, variant, launch_only=False, soft_catch=False):
    d = launch_metrics(m, start)
    catch, catch_event = score(m, 2, start)
    # Guide the knee and foot path even in candidates that fail to get airborne.
    torso = m["state"][:, start:]
    flight_guide = .8*np.clip(torso[..., 0].max(-1)+.23, 0, .55)/.55
    if variant == "shins-push":
        details = (3.0*np.clip(d["overlap"]/.025, 0, 1) +
                   1.1*np.clip(d["front"].min(-1)/2., 0, 1) +
                   .6*np.clip(d["rear"].min(-1)/2., 0, 1))
    else:
        details = (2.1*(d["reach"]>0) + 2.3*(d["drive"]>0) +
                   .7*np.clip(d["rear"].min(-1)/4., 0, 1))
    clean = np.exp(-np.maximum.accumulate(m["bad_control"],axis=1)[:,start:]/75.)
    vx = np.maximum(torso[...,19],0.)
    vz = np.maximum(torso[...,21],0.)
    near = np.exp(-((torso[...,0]+.15)/.19)**2)
    if variant == "shins-push":
        front = np.clip(m["front"][:,start:].min(-1)/36.,0,1)
        rear = np.clip(m["floor"][:,start:,:2].min(-1)/30.,0,1)
        drive = (front*rear*near*(.2+np.clip(vz/.45,0,1)+.5*np.clip(vx/.65,0,1))*clean).max(-1)
    else:
        feet = m["feet"][:,start:,2:]
        clear = np.clip((feet[...,0].min(-1)+.04)/.22,0,1)
        above = np.clip((feet[...,2].min(-1)-.27)/.07,0,1)
        rear = np.clip(m["floor"][:,start:,:2].min(-1)/30.,0,1)
        front_free = np.exp(-(m["front"][:,start:].max(-1)+m["toe"][:,start:,2:].max(-1))/35.)
        drive = (clear*above*rear*front_free*near*(.5+np.clip(vz/.55,0,1))*clean).max(-1)
    # Search first for a different thrust pattern. The later catch reward
    # cannot substitute for the specified launch.
    consequence = np.where(d[variant],catch,.16*catch) if not launch_only else .04*catch
    forward_flight = (1.8*np.clip(d["forward_speed_at_release"]/.65,0,1)
                      + 2.4*np.clip(d["first_quarter_second_advance_m"]/.20,0,1))*d["launched"]
    value = (consequence + flight_guide + details + 2.7*drive + .4*d["launched"] + forward_flight
             + 14.*d[variant] + 8.*(d[variant] & (catch_event>0))*(not launch_only)
             - 3.*d["toe_share"] - .35*np.log1p(m["bad"].max(-1)/100.))
    if soft_catch:
        touch,peak,rise,incoming,later_rise = catch_impact(m,d)
        impact_score = (1.8*np.log1p(peak/220.) + 3.2*rise + .65*incoming)
        foot = m["floor"][:,start:,2:].min(-1)>5.
        shin = m["rear"][:,start:].min(-1)>7.
        posture = np.exp(-((m["pitch"][:,start:]-.29)/.25)**2)
        hold = (foot*shin*posture*np.exp(-3.*m["speed"][:,start:]**2)*clean)
        window = np.lib.stride_tricks.sliding_window_view(hold,6,axis=1).mean(-1).max(-1)
        touch_frames = np.maximum(touch,0)//SUBSTEPS
        height = np.take_along_axis(m["state"][...,2],touch_frames[:,None],1)[:,0]
        before_touch = np.arange(m["bad"].shape[1])[None,:]<=touch[:,None]
        prior_collision = np.where(before_touch,m["bad"],0.).max(-1)
        landing_score = np.where((touch>=0) & (height>.22) & (peak>=120.) & (prior_collision<5.),
                                 6.-impact_score,-15.)
        value += (landing_score + 5.5*window - .012*m["bad"].max(-1) - 5.5*later_rise)*d[variant]
    return value, d, catch_event


def optimize(model, initial, prefix, template, variant, args):
    start = len(prefix)
    duration = len(template)
    skip = 0
    if args.keep_launch:
        with Rollout(nthread=1) as runner:
            raw,sensors = simulate(runner,model,[mj.MjData(model)],initial,np.vstack((prefix,template))[None])
        data = launch_metrics(read(model,raw,sensors),start)
        if not data[variant][0]:
            raise ValueError("Warm start does not meet the requested launch conditions")
        skip = int(data["air"][0])
    if args.freeze_through:
        skip = max(skip,args.freeze_through-start)
    if skip>=duration-2:
        raise ValueError("Frozen prefix must leave at least three controls")
    grid = np.unique(np.r_[skip, skip+1, np.arange(skip+3,duration,3),duration-1])
    basis = np.array([np.interp(np.arange(duration),grid,row) for row in np.eye(len(grid))]).T
    centre = template[grid].copy()
    rng = np.random.default_rng(args.seed)
    sigma = np.tile([.11,.30,.38], (len(grid),4)) * args.spread
    lower, upper = model.jnt_range[1:].T
    workers = [mj.MjData(model) for _ in range(args.threads)]
    best = -np.inf
    saved = np.vstack((prefix, template))
    with Rollout(nthread=args.threads) as runner:
        for it in range(args.iters):
            draw = rng.normal(size=(args.batch,*centre.shape))*sigma*max(.22,1.-it/(args.iters*1.12))
            n = args.batch//3
            draw[:n,:,[3,4,5,9,10,11]] = draw[:n,:,[0,1,2,6,7,8]]*[-1,1,1,-1,1,1]
            knots = centre+draw
            knots[0] = centre
            proposals = np.empty((args.batch,start+duration,12))
            proposals[:,:start] = prefix
            proposals[:,start:] = np.clip(basis@knots,lower-.18,upper+.18)
            proposals[:,start:start+skip] = template[:skip]
            proposals[0] = saved
            raw, sensors = simulate(runner,model,workers,initial,proposals)
            m = read(model,raw,sensors)
            value, data, catch = objective(m,start,variant,args.launch_only,args.soft_catch)
            value -= .02*(np.diff(knots,n=2,axis=1)**2/sigma[1:-1]**2).mean(axis=(1,2))
            top = np.argsort(value)[-max(6,args.batch//14):]
            centre = .22*centre+.78*knots[top].mean(0)
            k = top[-1]
            if value[k]>best:
                best = value[k]
                saved = proposals[k].copy()
            if it%10==0 or (data[variant][k] and catch[k]>0):
                print(variant,it,round(float(best),2),"launch",bool(data[variant][k]),
                      "catch",int(catch[k]),"overlap",round(float(data["overlap"][k]),3),
                      "J",np.r_[data["front"][k],data["rear"][k],data["toe"][k]].round(1),
                      "reach,drive",int(data["reach"][k]),int(data["drive"][k]),flush=True)
            if data[variant][k] and catch[k]>0 and not args.launch_only and not args.soft_catch:
                break
            del raw,sensors,m
    return saved


def run(args):
    with np.load(args.source/"replay.npz",allow_pickle=False) as f:
        initial, previous = f["initial"], f["controls"]
    reference = json.loads((args.source/"evaluation.json").read_text())
    brace = reference["stage_frames"]["hind_crouch"]
    freeze = previous[:brace]
    warm = previous[brace:reference["stage_frames"]["rear_shins"]+9].copy()
    model = make_model(args.asset, configure=add_sensors)
    if args.warm:
        with np.load(args.warm,allow_pickle=False) as f:
            np.testing.assert_array_equal(f["controls"][:brace], freeze)
            warm = f["controls"][brace:].copy()
    if args.extra_frames and not args.replay_only:
        warm = np.pad(warm,((0,args.extra_frames),(0,0)),mode="edge")
    targets = np.vstack((freeze,warm)) if args.replay_only else optimize(model,initial,freeze,warm,args.variant,args)
    with Rollout(nthread=1) as runner:
        raw,sensors = simulate(runner,model,[mj.MjData(model)],initial,targets[None])
    m = read(model,raw,sensors)
    _, details, catch = objective(m,brace,args.variant,soft_catch=args.soft_catch)
    touch, peak, rise, incoming, later_rise = catch_impact(m,details)
    out = args.output
    out.mkdir(parents=True,exist_ok=True)
    established = int(catch[0]) if details[args.variant][0] else -1
    stop = established if established>0 else len(targets)
    report = {
        "variant":args.variant,"qualified_launch":bool(details[args.variant][0]),
        "catch_hold_frame":established if established>0 else None,
        "catch_qualified":established>0,
        "baseline_launch":{},
        "second_launch_preview":True,
        "soft_catch":args.soft_catch,
        "front_touchdown": {
            "sim_time_s":float((touch[0]+1)*DT) if touch[0]>=0 else None,
            "peak_pair_load_n":float(peak[0]),
            "peak_body_upward_speed_m_s_first_180ms":float(rise[0]),
            "peak_body_upward_speed_m_s_after_touchdown":float(later_rise[0]),
            "incoming_downward_speed_m_s":float(incoming[0]),
        },
        "launch":{key: values[0].tolist() for key,values in details.items()},
        "peak_unplanned_contact_n":float(m["bad"][0].max()),
        "stage_frames":{"front_shins":reference["stage_frames"]["front_shins"],"hind_crouch":brace},
        "passed":False,
    }
    if established>0:
        report["stage_frames"]["rear_shins"] = established
    with Rollout(nthread=1) as runner:
        baseline_raw, baseline_sensors = simulate(runner,model,[mj.MjData(model)],initial,previous[None])
    baseline = launch_metrics(read(model,baseline_raw,baseline_sensors),brace)
    report["baseline_launch"] = {key:values[0].tolist() for key,values in baseline.items()}
    report["perturbations"] = {}
    for seed in (301,1621):
        offsets = np.random.default_rng(seed).uniform(-PERTURBATION,PERTURBATION,(16,3))
        trials = np.tile(initial[:38],(16,1))
        trials[:,1:3] += offsets[:,:2]
        trials[:,4] = np.cos(offsets[:,2]/2.)
        trials[:,7] = np.sin(offsets[:,2]/2.)
        workers = min(args.threads,16)
        with Rollout(nthread=workers) as runner:
            trial_raw, trial_sensors = runner.rollout(
                model,[mj.MjData(model) for _ in range(workers)],trials,
                np.broadcast_to(np.repeat(targets,SUBSTEPS,axis=0),(16,len(targets)*SUBSTEPS,12)),
            )
        trial = read(model,trial_raw,trial_sensors)
        # Stage two must finish by the fixed switch to the tested takeoff.
        prefix = {k:v[:,:brace*(SUBSTEPS if k.endswith("_all") or k=="bad" else 1)] for k,v in trial.items()}
        first_jump = score(prefix,0,front_led=True)[1]>0
        prepared = first_jump & (score(prefix,1,reference["stage_frames"]["front_shins"],front_led=True)[1]>0)
        qualified = prepared & launch_metrics(trial,brace)[args.variant]
        settle = score(trial,2,brace)[1]>0
        report["perturbations"][str(seed)] = {
            "brace":int(prepared.sum()),"launch":int(qualified.sum()),"catch":int((qualified & settle).sum())}
    states=m["state"][0,:stop]
    np.savez_compressed(out/"replay.npz",initial=initial,controls=targets[:stop],
                        qpos=np.vstack((initial[1:20],states[:,:19])),
                        **{key:m[key][0,:stop] for key in ("state","floor","front","rear","pitch")})
    (out/"evaluation.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--variant",choices=VARIANTS,required=True)
    p.add_argument("--warm",type=Path)
    p.add_argument("--launch-only",action="store_true")
    p.add_argument("--replay-only",action="store_true")
    p.add_argument("--soft-catch",action="store_true")
    p.add_argument("--extra-frames",type=int,default=0)
    p.add_argument("--freeze-through",type=int,default=0)
    p.add_argument("--keep-launch",action="store_true",
                   help="Preserve a qualified takeoff and optimize only the flight and catch")
    p.add_argument("--iters",type=int,default=240)
    p.add_argument("--batch",type=int,default=112)
    p.add_argument("--threads",type=int,default=6)
    p.add_argument("--seed",type=int,default=721)
    p.add_argument("--spread",type=float,default=1.)
    options = p.parse_args()
    if options.iters < 1 or options.batch < 16 or options.threads < 1 or options.spread <= 0:
        p.error("Use positive iterations, threads, spread, and a batch of at least 16")
    if options.keep_launch and not options.warm:
        p.error("--keep-launch requires --warm")
    run(options)
