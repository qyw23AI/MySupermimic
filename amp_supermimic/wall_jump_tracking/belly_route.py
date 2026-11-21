"""Test a second jump to belly support, then a front-foot led wall descent."""
import argparse
import json
from pathlib import Path

import mujoco as mj
from mujoco.rollout import Rollout
import numpy as np

from .brace_experiment import COLLIDERS, FEET, SUBSTEPS, make_model, simulate
from .contact_route import DT, PERTURBATION, add_sensors, bump, first, strength
from .four_stage import crest_load, force_signals
from .second_launch import read
from .robot import HEIGHT, THICKNESS

STAGES = ("belly_support", "front_pull", "four_foot_landing")


def inspect(model, raw, sensors):
    m = read(model, raw, sensors)
    shell = force_signals(model, sensors, "wall", ("base_link_collision",))
    shell_load = crest_load(shell)[..., 0]
    shell_side = np.where(shell_load > 0., 0., abs(shell[...,0,1]))
    floor_bodies = abs(force_signals(model,sensors,"floor",COLLIDERS)[...,1]).max(-1)
    wall_legs = force_signals(model,sensors,"wall",COLLIDERS[1:])
    # Shins may brush the crest as the robot changes support. Wall-side,
    # thigh and floor body loads count against the intended motion.
    legal = crest_load(wall_legs)>0
    legal[...,::2] = False
    wrong = np.maximum.reduce((shell_side,floor_bodies,np.where(legal,0.,abs(wall_legs[...,1])).max(-1)))
    toe = force_signals(model,sensors,"wall",FEET)
    wrong = np.maximum(wrong,np.where(crest_load(toe)>0.,0.,abs(toe[...,1])).max(-1))
    m.update(belly=shell_load[:,SUBSTEPS-1::SUBSTEPS], belly_all=shell_load, wrong=wrong,
             belly_point=shell[:,SUBSTEPS-1::SUBSTEPS,0,4:7])
    return m


def assessment(m, start, stage, brace):
    end = m["state"].shape[1]
    t = np.arange(end)[None,:]
    state = m["state"]
    feet = m["feet"]
    speed = m["speed"]
    pelvis = state[...,0]
    floor = m["floor"]
    belly = m["belly"]
    clear = feet[...,0]>THICKNESS/2+.024
    front_floor = ((floor[...,2:]>6.) & clear[...,2:]).all(-1)
    supported = belly>12.
    bad = m["wrong"].reshape(len(state),end,SUBSTEPS).max(-1)
    max_bad = np.maximum.accumulate(bad,1)
    okay = max_bad<12.
    safe = np.exp(-max_bad/160.)
    foot_ahead = clear[...,2:].all(-1)
    vbody = np.exp(-2.5*speed**2-.08*m["spin"]**2)
    air = m["air"] & (belly<5.) & (bad<8.)
    flew = first(air & (t>=brace),2)
    prior_flight = (flew[:,None]>0) & (t+1>=flew[:,None])
    strength_front = strength(floor[...,2:],22.)
    torso = np.clip(belly/85.,0,1)
    # Calf angles measure flexion. Keep the folded hind legs as a soft
    # objective until it is time for them to lower.
    folded = np.clip((m["bend"][...,:2].min(-1)-.8)/1.1,0,1)
    if stage == 0:
        ready = (supported & prior_flight & foot_ahead & (m["up"]>.70) & (speed<.50)
                 & (m["spin"]<3.) & okay & (t>=start) & (m["bend"][...,:2].min(-1)>.9))
        event = first(ready,5)
        geometry = bump(pelvis,-.025,.17)*bump(state[...,2],.38,.10)*bump(m["pitch"],.12,.45)
        reach = np.clip(feet[...,2:,0].min(-1)/.30,0,1)
        target = (1.6*geometry+.8*reach+torso*(1.6+.8*reach+1.5*vbody+2.4*folded)
                  +.65*folded+2.4*ready)*safe
        target *= np.where(prior_flight,1.,.45)
    elif stage == 1:
        origin = pelvis[:,start-1,None]
        distance = pelvis-origin
        no_hop = np.maximum.accumulate(np.pad(
            np.lib.stride_tricks.sliding_window_view(air & (t>=start),4,axis=1).all(-1),
            ((0,0),(3,0))),axis=1)==0
        knees_folded = m["bend"][...,:2].min(-1)>.80
        ready = (front_floor & (distance>.12) & (pelvis>.085) & (speed<.44)
                 & (m["up"]>.72) & okay & (t>=start) & no_hop & knees_folded)
        event = first(ready,5)
        progress = np.clip(distance/.22,-.5,1.)
        front_reach = np.clip((feet[...,2:,0].min(-1)-.10)/.20,0,1)
        lower_front = bump(feet[...,2:,2],.039,.135).min(-1)*foot_ahead
        reachable = (1.5*lower_front+2.2*lower_front*torso+1.1*front_floor)*np.clip((pelvis+.16)/.20,0,1)
        target = (2.3*progress + 1.7*strength_front + .8*folded +
                  .8*torso + 1.6*strength_front*vbody + .85*front_reach +
                  1.8*strength_front*progress*vbody+reachable)*safe*np.where(no_hop,1.,.34)
        target += (3.8*ready + 1.6*strength_front*folded*vbody*safe*no_hop)
        target -= .7*np.clip(np.abs(state[...,21])-.35,0,2)
    else:
        event = first((floor.min(-1)>6.) & clear.all(-1) & (belly<6.) &
                      (m["up"]>.80) & (speed<.32) & (m["spin"]<2.0) &
                      (state[...,2]>.22) & okay & (t>=start),10)
        progress = np.clip((feet[...,:2,0].min(-1)+.05)/.35,0,1)
        rear_reach = bump(feet[...,:2,2],.035,.18).min(-1)*clear[...,:2].all(-1)
        ready = (floor.min(-1)>6.) & clear.all(-1) & (m["up"]>.8) & (speed<.32)
        target = (1.7*progress + 2.3*strength(floor,17.)*vbody + 1.2*rear_reach +
                  .8*strength_front + .62*np.clip((pelvis-.08)/.35,0,1)+2.3*ready)*safe
    tail = target[:,start:]
    window = np.lib.stride_tricks.sliding_window_view(tail,5,axis=1).mean(-1).max(-1)
    value = .4*tail.mean(-1)+.7*tail.max(-1)+2.2*window+11.*(event>0)
    value -= .24*np.log1p(bad.max(-1)/20.)
    crest_peak = m["belly_all"][:,start*SUBSTEPS:].max(-1)
    value -= (.65 if stage==0 else .9)*np.log1p(crest_peak/200.)
    if stage == 0:
        value += .9*(flew>0)-.28*np.log1p(m["belly_all"].max(-1)/200.)
    return value,event


def landing_seed(model, initial, prefix):
    """Seed a staggered forefoot pull and lower the rear feet on the far side."""
    data = mj.MjData(model)
    with Rollout(nthread=1) as pool:
        raw,_=simulate(pool,model,[data],initial,prefix[None])
    pose=raw[0,-1,1:20].copy()
    data.qpos[:]=pose
    mj.mj_forward(model,data)
    ids=[model.geom(foot).id for foot in FEET]
    feet=data.geom_xpos[ids].copy()
    target=feet.copy()
    target[:2,0]=pose[0]+.13
    target[2:,0]=np.maximum(feet[2:,0]+.22,pose[0]+.29)
    target[:,2]=.016
    lower,upper=model.jnt_range[1:].T
    jac=np.zeros((3,model.nv))
    result=[]
    joints=pose[7:].copy()

    def ease(t,start,width):
        x=np.clip((t-start)/width,0,1)
        return x*x*(3-2*x)

    for t in range(68):
        travel=ease(t,2,36)
        goals=feet.copy()
        for i in range(4):
            if i<2:
                forward=ease(t,5+4*i,24)
                down=ease(t,27+4*i,16)
                goals[i,:2]=(1-forward)*feet[i,:2]+forward*target[i,:2]
                goals[i,2]=(1-down)*max(feet[i,2],.34)+down*target[i,2]
            else:
                step=ease(t,1+(i-2)*13,13)
                goals[i]=(1-step)*feet[i]+step*target[i]
                goals[i,2]+=.052*np.sin(np.pi*step)
        data.qpos[:3]=pose[:3]+[.22*travel,0.,.09*ease(t,13,30)]
        quat=(1-travel)*pose[3:7]+travel*np.array([1.,0,0,0])
        data.qpos[3:7]=quat/np.linalg.norm(quat)
        for _ in range(10):
            data.qpos[7:]=joints
            mj.mj_kinematics(model,data)
            mj.mj_comPos(model,data)
            for i,geom in enumerate(ids):
                leg=slice(3*i,3*i+3)
                mj.mj_jacGeom(model,data,jac,None,geom)
                j=jac[:,6+3*i:9+3*i]
                err=goals[i]-data.geom_xpos[geom]
                delta=j.T @ np.linalg.solve(j@j.T+.003*np.eye(3),err)
                joints[leg]=np.clip(joints[leg]+np.clip(delta,-.20,.20),lower[leg]+.01,upper[leg]-.01)
        result.append(joints.copy())
    return np.asarray(result)


def solve(model, initial, prefix, warm, brace, stage, args, *, assessor=assessment, stages=STAGES):
    start = len(prefix)
    frames = len(warm)
    frozen = min(args.freeze_landing,frames-4) if stage==2 else 0
    steps = np.unique(np.r_[frozen,frozen+1,np.arange(frozen+3,frames,3),frames-1])
    basis = np.array([np.interp(np.arange(frames),steps,col) for col in np.eye(len(steps))]).T
    mean = warm[steps].copy()
    deviation = np.tile([.13,.31,.42],(len(steps),4))*args.spread
    rng=np.random.default_rng(args.seed+stage)
    limits = model.jnt_range[1:]
    workers = [mj.MjData(model) for _ in range(args.threads)]
    best = -np.inf
    best_u = np.vstack((prefix,warm))
    qualified = -1
    with Rollout(nthread=args.threads) as pool:
        for iteration in range(args.iters[stage]):
            noise = rng.normal(size=(args.batch,*mean.shape))*deviation
            symmetric = args.batch//3
            noise[:symmetric,:,[3,4,5,9,10,11]] = noise[:symmetric,:,[0,1,2,6,7,8]]*[-1,1,1,-1,1,1]
            knots = mean+noise*max(.18,1.-iteration/(args.iters[stage]*1.12))
            knots[0] = mean
            trials = np.empty((args.batch,start+frames,12))
            trials[:,:start] = prefix
            trials[:,start:] = np.clip(basis@knots,limits[:,0]-.12,limits[:,1]+.12)
            trials[:,start:start+frozen] = warm[:frozen]
            trials[0] = best_u
            raw,sensors = simulate(pool,model,workers,initial,trials)
            m = inspect(model,raw,sensors)
            score,event = assessor(m,start,stage,brace)
            score -= .028*np.mean(np.diff(knots,n=2,axis=1)**2/deviation[1:-1]**2,axis=(1,2))
            top=np.argsort(score)[-max(6,args.batch//14):]
            k=top[-1]
            mean=.2*mean+.8*knots[top].mean(0)
            if score[k]>best:
                best, best_u, qualified = score[k],trials[k].copy(),int(event[k])
                np.savez_compressed(args.output/f"{stages[stage]}.npz",initial=initial,controls=best_u,event=qualified,start=start)
            if iteration%12==0 or qualified>0:
                print(stages[stage],iteration,round(float(best),2),qualified,
                      "x",round(float(m["state"][k,:,0].max()),2),"belly",round(float(m["belly"][k].max())),
                      "bad",round(float(m["wrong"][k].max())),flush=True)
            if qualified>0:
                break
    return best_u,qualified


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    reference=json.loads((args.source/"evaluation.json").read_text())
    brace=reference["stage_frames"]["hind_crouch"]
    with np.load(args.source/"replay.npz") as src:
        initial=src["initial"]
        old=src["controls"]
    model=make_model(args.asset,configure=add_sensors)
    prefix=old[:brace]
    events=[]
    with np.load(args.takeoff/"replay.npz") as warm:
        reach=warm["controls"]
    for stage in range(3):
        if stage == 0:
            warm = reach[brace:brace+53].copy()
        elif stage == 1:
            warm = np.tile(prefix[-1],(76,1))
            progress=np.linspace(0,1,len(warm))
            warm[:,[1,4]] += .22*progress[:,None]
            warm[:,[2,5]] -= .23*progress[:,None]
            warm[:,[7,10]] += .55*progress[:,None]
            warm[:,[8,11]] -= .37*progress[:,None]
        else:
            warm=landing_seed(model,initial,prefix)
        saved = args.output/f"{STAGES[stage]}.npz"
        compatible = False
        if args.resume and saved.exists():
            with np.load(saved) as f:
                trajectory, event = f["controls"],int(f["event"])
                compatible = (np.array_equal(f["initial"],initial)
                              and np.array_equal(trajectory[:len(prefix)],prefix))
            if compatible:
                warm=trajectory[len(prefix):]
                if stage==1 and event<0 and len(warm)<76:
                    original_length = len(warm)
                    warm=np.pad(warm,((0,76-original_length),(0,0)),mode="edge")
                    t=np.clip(np.arange(len(warm)-original_length)/12.,0,1)
                    warm[original_length:,[8,11]] += .40*t[:,None]
                    warm[original_length:,[7,10]] -= .36*t[:,None]
                if stage==2 and event<0 and len(warm)<80:
                    warm=np.pad(warm,((0,80-len(warm)),(0,0)),mode="edge")
                if event>0:
                    prefix=trajectory[:event]
                    events.append(event)
                    continue
        if args.replay_only:
            if not compatible:
                raise ValueError(f"No saved trajectory matches the prefix for {STAGES[stage]}")
            prefix=trajectory
            break
        trajectory,event=solve(model,initial,prefix,warm,brace,stage,args)
        if event<0:
            prefix=trajectory
            break
        prefix=trajectory[:event]
        events.append(event)
    with Rollout(nthread=1) as pool:
        raw,sensors=simulate(pool,model,[mj.MjData(model)],initial,prefix[None])
    m=inspect(model,raw,sensors)
    for stage,(start,end) in enumerate(zip([brace,*events],events)):
        _,result=assessment(m,start,stage,brace)
        if result[0]!=end:
            raise RuntimeError(f"Stage {stage+1} changed on replay")
    belly=m["belly"][0]
    front_ground=(m["floor"][...,2:].min(-1)>6.) & (m["feet"][...,2:,0].min(-1)>THICKNESS/2+.024)
    touch=first(front_ground & (np.arange(len(prefix))>=brace))[0]
    start_belly=first(m["belly"]>12.)[0]
    report=dict(passed=len(events)==3, first_shins_frame=reference["stage_frames"]["front_shins"],
                brace_frame=brace, events=dict(zip(STAGES,events)),
                belly_contact_frame=int(start_belly) if start_belly>0 else None,
                front_floor_frame=int(touch) if touch>0 else None,
                peak_belly_load_n=float(m["belly_all"][0].max()),
                peak_unplanned_contact_n=float(m["wrong"][0].max()),
                belly_and_dual_front_load_s=float(np.count_nonzero((belly>12.) & front_ground[0])*SUBSTEPS*DT),
                belly_contact_forward_m=float(m["state"][0,np.flatnonzero(belly>12.)[-1],0]
                                             -m["state"][0,start_belly-1,0]) if start_belly>0 else None)
    if events:
        supported = (m["floor"][0].max(-1)>6.) | (belly>12.) | (m["rear"][0].max(-1)>7.) | (m["front"][0].max(-1)>7.) | (m["toe"][0].max(-1)>7.)
        unsupported = ~supported[events[0]:]
        edges = np.flatnonzero(np.diff(np.r_[False,unsupported,False]))
        report["unsupported_intervals_after_belly_support_s"] = (
            (edges[1::2]-edges[::2])*SUBSTEPS*DT).tolist()
    if touch>0 and len(events)>1:
        report["base_advance_after_front_ground_m"] = float(m["state"][0,events[1]-1,0]-m["state"][0,touch-1,0])
        report["front_foot_advance_after_first_support_m"] = (
            m["feet"][0,events[1]-1,2:,0]-m["feet"][0,touch-1,2:,0]).tolist()
    report["perturbations"] = {}
    for seed in (301,1621):
        offsets = np.random.default_rng(seed).uniform(-PERTURBATION,PERTURBATION,(16,3))
        initial_batch = np.tile(initial[:38],(16,1))
        initial_batch[:,1:3] += offsets[:,:2]
        initial_batch[:,4] = np.cos(offsets[:,2]/2)
        initial_batch[:,7] = np.sin(offsets[:,2]/2)
        workers=min(args.threads,16)
        with Rollout(nthread=workers) as pool:
            trial_raw,trial_sensors=pool.rollout(
                model,[mj.MjData(model) for _ in range(workers)],initial_batch,
                np.broadcast_to(np.repeat(prefix,SUBSTEPS,axis=0),(16,len(prefix)*SUBSTEPS,12)))
        trial=inspect(model,trial_raw,trial_sensors)
        passed=np.ones(16,dtype=bool)
        counts={}
        for stage,start in enumerate([brace,*events][:3]):
            if len(prefix)-start<10:
                break
            _,event=assessment(trial,start,stage,brace)
            passed &= event>0
            if stage<len(events):
                passed &= event<=events[stage]
            counts[STAGES[stage]]=int(passed.sum())
        report["perturbations"][str(seed)]=counts
    frames=m["state"][0]
    np.savez_compressed(args.output/"replay.npz",initial=initial,controls=prefix,
                        qpos=np.vstack((initial[1:20],frames[:,:19])),
                        **{k:m[k][0] for k in ("floor","front","rear","pitch","belly","wrong","feet")})
    (args.output/"evaluation.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(report,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("asset","source","takeoff","output"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    parser.add_argument("--iters",nargs=3,type=int,default=(220,200,180))
    parser.add_argument("--batch",type=int,default=112)
    parser.add_argument("--threads",type=int,default=8)
    parser.add_argument("--seed",type=int,default=901)
    parser.add_argument("--spread",type=float,default=1.)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--freeze-landing",type=int,default=0)
    parser.add_argument("--replay-only",action="store_true",help="Recheck saved stage trajectories without search")
    args=parser.parse_args()
    if min(args.iters)<1 or args.threads<1 or args.batch<16 or args.spread<=0 or args.freeze_landing<0:
        parser.error("Invalid search budget")
    run(args)
