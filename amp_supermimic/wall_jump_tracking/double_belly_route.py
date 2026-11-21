"""Evaluate front-body and rear-body wall support across the two WBL jumps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco as mj
from mujoco.rollout import Rollout
import numpy as np

from .belly_route import assessment, inspect, landing_seed, solve
from .brace_experiment import FEET, SUBSTEPS, make_model, simulate
from .contact_route import DT, PERTURBATION, add_sensors, bump, first, strength
from .robot import THICKNESS

STAGES = ("front_body_mount", "rear_body_catch", "four_foot_landing")


def local_contact_x(m):
    w,x,y,z=np.moveaxis(m["state"][...,3:7],-1,0)
    axis=np.stack((1.-2.*(y*y+z*z),2.*(x*y+w*z),2.*(x*z-w*y)),axis=-1)
    return ((m["belly_point"]-m["state"][...,:3])*axis).sum(-1)


def evaluate(m,start,stage,_):
    if stage==2:
        reward,event=assessment(m,start,stage,0)
        tail=m["state"][:,start:]
        ground=m["floor"][:,start:]
        quiet=np.exp(-4.*m["speed"][:,start:]**2-.14*m["spin"][:,start:]**2)
        even = quiet * strength(ground,15.)
        settled=np.lib.stride_tricks.sliding_window_view(even,10,axis=1).mean(-1).max(-1)
        drift=np.maximum(np.abs(tail[...,1])-.27,0.).max(-1)
        lift=np.maximum(tail[...,21]-.55,0.).max(-1)
        return reward+4.*settled-1.8*drift-1.4*lift,event
    state=m["state"]
    x,z=state[...,0],state[...,2]
    ticks=np.arange(state.shape[1])[None,:]
    local=local_contact_x(m)
    floor=m["floor"]
    feet=m["feet"]
    torso=np.clip(m["belly"]/65.,0.,1.)
    bad=m["wrong"].reshape(*x.shape,SUBSTEPS).max(-1)
    safe=np.exp(-np.maximum.accumulate(bad,axis=1)/130.)
    clean=np.maximum.accumulate(bad,axis=1)<12.
    air=m["air"] & (m["belly"]<5.) & (bad<8.)
    flight=first(air & (ticks>=max(0,start-2) if stage==0 else ticks>=start),2)
    after_flight=(flight[:,None]>0)&(ticks+1>=flight[:,None])
    forward_feet=(feet[...,2:,0]>THICKNESS/2+.024).all(-1)
    settled=(m["speed"]<.52)&(m["spin"]<3.)&(m["up"]>.68)
    calm=np.exp(-2.5*m["speed"]**2-.11*m["spin"]**2)
    if stage==4:
        front=(floor[...,2:].min(-1)>7.)&forward_feet
        progress=np.clip((x-x[:,start-1,None])/.20,0,1)
        folded=np.clip(m["bend"][...,:2].min(-1)/1.6,0,1)
        stance=strength(floor[...,2:],17.)*forward_feet
        ready=front&(x>.22)&(m["speed"]<.40)&(m["up"]>.72)&clean&(ticks>=start)
        event=first(ready,5)
        value=(2.5*progress+2.*stance+1.8*progress*stance+1.2*torso*stance+
               .8*folded+.7*calm+3.5*ready)*safe
        value-=.6*np.clip(np.abs(state[...,21])-.35,0,2)
        tail=value[:,start:]
        return .4*tail.mean(-1)+3.*np.lib.stride_tricks.sliding_window_view(tail,5,axis=1).mean(-1).max(-1)+11.*(event>0),event
    if stage==3:
        rear_floor=(floor[...,:2].min(-1)>7.) & (feet[...,:2,0].max(-1)<-.065)
        near=feet[...,:2,0].min(-1)
        flex=m["bend"][...,:2].min(-1)
        ready=(m["belly"]>12.)&(local>.02)&rear_floor&(near>-.48)&(flex>1.40)&(m["speed"]<.38)&clean
        ready &= ticks>=start
        event=first(ready,5)
        target=(1.6*torso+1.4*bump(near,-.37,.23)+.7*np.clip(flex/1.8,0,1)+
                1.4*strength(floor[...,:2],22.)+2.3*torso*calm+3.*ready)*safe
        tail=target[:,start:]
        return .5*tail.mean(-1)+2.8*np.lib.stride_tricks.sliding_window_view(tail,5,axis=1).mean(-1).max(-1)+12.*(event>0),event
    if stage==0:
        rear_floor=(floor[...,:2].min(-1)>7.)&(feet[...,:2,0].max(-1)<-.065)
        ready=(m["belly"]>12.)&(local>.035)&rear_floor&settled
        # Reject a prior sustained front-shin brace. Transient grazing the
        # wall top while getting the torso over the edge is allowed.
        brace=first(m["front"].min(-1)>7.,4)
        clean &= (brace[:,None]<0)|(ticks+1<brace[:,None])
        setup=bump(x,-.16,.14)*bump(z,.36,.105)
        guide=np.clip((feet[...,2:,0].min(-1)+.05)/.20,0,1)
        support=torso*np.clip((local+.015)/.12,0,1)*strength(floor[...,:2],20.)
        value=(1.4*setup+.8*guide+2.3*support+1.6*support*calm)*safe
    else:
        front=(floor[...,2:].min(-1)>7.)&forward_feet
        ready=(m["belly"]>12.)&(local<-.035)&front&settled
        push=(ticks>=start)&(ticks+1<flight[:,None])&forward_feet&(floor[...,:2].min(-1)>10.)&(state[...,21]>.08)
        prior_push=first(push)
        ready &= (prior_push[:,None]>0)&(ticks+1>prior_push[:,None])
        body=bump(x,.135,.16)*bump(z,.34,.125)*bump(m["pitch"],.35,.30)
        front_load=strength(floor[...,2:],20.)*forward_feet
        front_guide=bump(feet[...,2:,2],.035,.135).min(-1)*forward_feet
        support=torso*np.clip((.015-local)/.13,0,1)
        folded=np.clip((m["bend"][...,:2].min(-1)-.7)/1.55,0,1)
        value=(1.5*body+front_guide+1.15*front_load+2.3*support+
               2.0*front_load*support*calm+.5*folded+.75*push)*safe
    ready &= clean & after_flight & (ticks>=start)
    event=first(ready,5)
    value=(value*(.5+.5*after_flight)+2.6*ready)[:,start:]
    rolling=np.lib.stride_tricks.sliding_window_view(value,5,axis=1).mean(-1).max(-1)
    peak=m["belly_all"][:,start*SUBSTEPS:].max(-1)
    return .55*value.max(-1)+.25*value.mean(-1)+2.4*rolling+13.*(event>0)-.7*np.log1p(peak/200.),event


def second_seed(model,initial,prefix):
    """Aim the rear push backward and down while keeping both forefeet across."""
    data=mj.MjData(model)
    with Rollout(nthread=1) as pool:
        raw,_=simulate(pool,model,[data],initial,prefix[None])
    data.qpos[:]=raw[0,-1,1:20]
    pose=data.qpos.copy()
    mj.mj_forward(model,data)
    ids=[model.geom(foot).id for foot in FEET]
    positions=data.geom_xpos[ids].copy()
    lower,upper=model.jnt_range[1:].T
    jac=np.empty((3,model.nv))
    q=pose[7:].copy()
    warm=[]
    for i in range(65):
        drive=np.clip((i-2)/8,0,1)
        flight=np.clip((i-9)/20,0,1)
        feet=positions.copy()
        feet[:2,0]-=.12*drive
        feet[:2,2]-=.10*drive
        feet[:2,0]+=.49*flight
        feet[:2,2]+=.48*flight
        feet[2:,0]=np.maximum(feet[2:,0],.26+.08*flight)
        feet[2:,2]=(1-drive)*feet[2:,2]+drive*(.20-.19*flight)
        data.qpos[:3]=pose[:3]+[.27*flight,0,.095*np.sin(np.pi*flight)]
        quat=(1-flight)*pose[3:7]+flight*np.array([.987,0,.16,0])
        data.qpos[3:7]=quat/np.linalg.norm(quat)
        for _ in range(10):
            data.qpos[7:]=q
            mj.mj_kinematics(model,data)
            mj.mj_comPos(model,data)
            for j,geom in enumerate(ids):
                leg=slice(3*j,3*j+3)
                mj.mj_jacGeom(model,data,jac,None,geom)
                block=jac[:,6+3*j:9+3*j]
                delta=block.T @ np.linalg.solve(block@block.T+.004*np.eye(3),feet[j]-data.geom_xpos[geom])
                q[leg]=np.clip(q[leg]+np.clip(delta,-.20,.20),lower[leg]+.02,upper[leg]-.02)
        warm.append(q.copy())
    return np.asarray(warm)


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    with np.load(args.first/"replay.npz",allow_pickle=False) as f:
        initial=f["initial"]
        first_jump=f["controls"]
    with np.load(args.belly/"replay.npz",allow_pickle=False) as f:
        prior=f["controls"]
    model=make_model(args.asset,configure=add_sensors)
    prefix=first_jump[:args.freeze_first]
    starts=[]
    endpoints=[]
    hind_ready=None
    front_advance=None
    for stage,name in enumerate(STAGES):
        if stage==1:
            path=args.output/"hind_setup.npz"
            ready_controls=None
            if args.resume and path.exists():
                with np.load(path,allow_pickle=False) as f:
                    if np.array_equal(f["initial"],initial) and np.array_equal(f["controls"][:len(prefix)],prefix):
                        ready_controls=f["controls"].copy()
                        ready_frame=int(f["event"])
            if ready_controls is None or (ready_frame<0 and not args.replay_only):
                if args.replay_only:
                    raise ValueError("Missing matching hind setup")
                if ready_controls is not None:
                    warm=ready_controls[len(prefix):]
                else:
                    warm=np.tile(prefix[-1],(55,1))
                    mix=np.clip(np.arange(55)/18.,0,1)
                    aim=np.array([-.1,1.5,-2.05,.1,1.5,-2.05,-.15,.15,-.78,.15,.15,-.78])
                    warm=(1-mix[:,None])*warm+mix[:,None]*aim
                ready_controls,ready_frame=solve(model,initial,prefix,warm,0,3,args,assessor=evaluate,stages=(*STAGES,"hind_setup"))
            if ready_frame<0:
                prefix=ready_controls
                break
            prefix=ready_controls[:ready_frame]
            hind_ready=ready_frame
        if stage==2:
            path=args.output/"front_advance.npz"
            advance_controls=None
            if args.resume and path.exists():
                with np.load(path,allow_pickle=False) as f:
                    if np.array_equal(f["initial"],initial) and np.array_equal(f["controls"][:len(prefix)],prefix):
                        advance_controls=f["controls"].copy()
                        advance_frame=int(f["event"])
            if advance_controls is None or (advance_frame<0 and not args.replay_only):
                if args.replay_only:
                    raise ValueError("Missing matching front advance")
                if advance_controls is None:
                    warm=np.tile(prefix[-1],(62,1))
                    mix=np.clip(np.arange(62)/23.,0,1)
                    warm[:,[7,10]] += .62*mix[:,None]
                    warm[:,[8,11]] += .20*mix[:,None]
                    warm[:,[2,5]] -= .30*mix[:,None]
                else:
                    warm=advance_controls[len(prefix):]
                advance_controls,advance_frame=solve(model,initial,prefix,warm,0,4,args,
                                                      assessor=evaluate,stages=(*STAGES,"hind_setup","front_advance"))
            if advance_frame<0:
                prefix=advance_controls
                break
            prefix=advance_controls[:advance_frame]
            front_advance=advance_frame
        starts.append(len(prefix))
        path=args.output/f"{name}.npz"
        compatible=False
        if args.resume and path.exists():
            with np.load(path,allow_pickle=False) as f:
                controls,event=f["controls"],int(f["event"])
                compatible=(np.array_equal(controls[:len(prefix)],prefix)
                            and np.array_equal(f["initial"],initial))
            if compatible and event>0:
                prefix=controls[:event]
                endpoints.append(event)
                continue
        if args.replay_only:
            if not compatible:
                raise ValueError(f"Missing matching replay for {name}")
            prefix=controls
            break
        if compatible and not (stage==1 and args.seed_second):
            warm=controls[len(prefix):]
            if stage==2 and len(warm)<96:
                length=len(warm)
                warm=np.pad(warm,((0,96-length),(0,0)),mode="edge")
                mix=np.clip((np.arange(len(warm))-length)/18.,0,1)
                stand=np.array([-.12,1.,-1.55,.12,1.,-1.55,-.12,.9,-1.55,.12,.9,-1.55])
                warm=(1-mix[:,None])*warm+mix[:,None]*stand
        elif stage==0:
            warm=np.vstack((first_jump[args.freeze_first:],prior[36:73]))
        elif stage==1:
            if args.seed_second:
                warm=second_seed(model,initial,prefix)
            else:
                warm=prior[36:108].copy()
                warm[:11] += np.linspace(.9,0,11)[:,None]*(prefix[-1]-warm[0])
        else:
            warm=landing_seed(model,initial,prefix)
        controls,event=solve(model,initial,prefix,warm,0,stage,args,assessor=evaluate,stages=STAGES)
        if event<0:
            prefix=controls
            break
        prefix=controls[:event]
        endpoints.append(event)
    with Rollout(nthread=1) as pool:
        raw,sensors=simulate(pool,model,[mj.MjData(model)],initial,prefix[None])
    m=inspect(model,raw,sensors)
    local=local_contact_x(m)[0]
    for stage,(start,end) in enumerate(zip(starts,endpoints)):
        _,verified=evaluate(m,start,stage,0)
        if verified[0]!=end:
            raise RuntimeError(f"Stage {STAGES[stage]} changed on independent replay")
    for auxiliary,start,stop in ((3,endpoints[0] if endpoints else None,hind_ready),
                                 (4,endpoints[1] if len(endpoints)>1 else None,front_advance)):
        if stop is not None:
            _,verified=evaluate(m,start,auxiliary,0)
            if verified[0]!=stop:
                raise RuntimeError("An auxiliary stage changed on independent replay")
    report={"passed":len(endpoints)==len(STAGES),"events":dict(zip(STAGES,endpoints)),
            "double_belly_route":True,"hind_setup_frame":hind_ready,"front_advance_frame":front_advance,
            "frozen_first_jump_frames":args.freeze_first,
            "peak_belly_load_n":float(m["belly_all"].max()),
            "peak_unplanned_contact_n":float(m["wrong"].max())}
    for stage,end in enumerate(endpoints[:2]):
        window=slice(end-5,end)
        report[STAGES[stage]]={
            "body_local_contact_x_m":local[window].tolist(),
            "crest_vertical_load_n":m["belly"][0,window].tolist(),
            "front_foot_floor_load_n":m["floor"][0,window,2:].tolist(),
            "peak_crest_load_n":float(m["belly_all"][0,(0 if stage==0 else hind_ready*SUBSTEPS):end*SUBSTEPS].max()),
        }
    if endpoints and len(prefix)>endpoints[0]:
        report["rear_body_and_front_support_total_s"]=float(np.count_nonzero(
            (m["belly"][0]>12.)&(local<-.035)&(m["floor"][0,:,2:].min(-1)>7.))*SUBSTEPS*DT)
    if front_advance is not None:
        report["trunk_advance_m"]=float(m["state"][0,front_advance-1,0]-m["state"][0,endpoints[1]-1,0])
    if front_advance is not None:
        grounded=(m["floor"][0,front_advance:].min(-1)>6.) & (
            m["feet"][0,front_advance:,:,0].min(-1)>THICKNESS/2+.024)
        state=m["state"][0,front_advance:]
        stable=grounded&(m["speed"][0,front_advance:]<.32)&(m["spin"][0,front_advance:]<2.)&(
            m["up"][0,front_advance:]>.80)&(state[:,2]>.22)
        edges=np.flatnonzero(np.diff(np.r_[False,stable,False]))
        report["longest_stable_four_foot_window_s"]=float((edges[1::2]-edges[::2]).max(initial=0)*SUBSTEPS*DT)
        report["all_four_feet_reached_far_floor"]=bool(grounded.any())
        report["peak_vertical_speed_during_descent_m_s"]=float(state[:,21].max())
    report["perturbations"]={}
    for seed in (301,1621):
        offsets=np.random.default_rng(seed).uniform(-PERTURBATION,PERTURBATION,(16,3))
        trial=np.tile(initial[:38],(16,1))
        trial[:,1:3]+=offsets[:,:2]
        trial[:,4]=np.cos(offsets[:,2]/2.)
        trial[:,7]=np.sin(offsets[:,2]/2.)
        n=min(args.threads,16)
        controls=np.broadcast_to(np.repeat(prefix,SUBSTEPS,axis=0),(16,len(prefix)*SUBSTEPS,12))
        with Rollout(nthread=n) as pool:
            r,s=pool.rollout(model,[mj.MjData(model) for _ in range(n)],trial,controls)
        measured=inspect(model,r,s)
        valid=np.ones(16,dtype=bool)
        results={}
        for stage,start in enumerate(starts):
            if len(prefix)-start<10:
                break
            if stage==1 and hind_ready is not None:
                _,setup=evaluate(measured,endpoints[0],3,0)
                valid &= (setup>0)&(setup<=hind_ready)
                results["hind_setup"]=int(valid.sum())
            if stage==2 and front_advance is not None:
                _,advance=evaluate(measured,endpoints[1],4,0)
                valid &= (advance>0)&(advance<=front_advance)
                results["front_advance"]=int(valid.sum())
            _,event=evaluate(measured,start,stage,0)
            valid &= event>0
            if stage<len(endpoints):
                valid &= event<=endpoints[stage]
            results[STAGES[stage]]=int(valid.sum())
        report["perturbations"][str(seed)]=results
    trajectory=m["state"][0]
    np.savez_compressed(args.output/"replay.npz",initial=initial,controls=prefix,
                        qpos=np.vstack((initial[1:20],trajectory[:,:19])),
                        **{key:m[key][0] for key in ("floor","front","rear","pitch","belly")})
    (args.output/"evaluation.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(report,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("asset","first","belly","output"):
        p.add_argument(f"--{name}",type=Path,required=True)
    p.add_argument("--iters",nargs=5,type=int,default=(260,260,200,220,240))
    p.add_argument("--batch",type=int,default=112)
    p.add_argument("--threads",type=int,default=8)
    p.add_argument("--seed",type=int,default=1001)
    p.add_argument("--spread",type=float,default=.9)
    p.add_argument("--freeze-first",type=int,default=7)
    p.add_argument("--freeze-landing",type=int,default=0)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--seed-second",action="store_true")
    p.add_argument("--replay-only",action="store_true")
    a=p.parse_args()
    if min(a.iters)<1 or a.threads<1 or a.batch<16 or a.spread<=0 or not 6<=a.freeze_first<=12 or a.freeze_landing<0:
        p.error("Invalid search budget or freeze length")
    run(a)
