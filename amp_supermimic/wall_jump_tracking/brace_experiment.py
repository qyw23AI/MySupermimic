"""Optimize and verify a WBL wall-support transfer in free-base MuJoCo."""
from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
from mujoco.rollout import Rollout
import numpy as np

from .robot import HEIGHT as WALL_HEIGHT, LEGS, THICKNESS, WIDTH
CONTROL_DT = .02
TOE_SIZE = .0238
SUBSTEPS = 4
HOME = np.array([-.1,1.,-1.5,.1,1.,-1.5,-.1,.8,-1.5,.1,.8,-1.5])
FULL_PHYSICS_SIZE = 38  # time + qpos + qvel for this WBL model
COLLIDERS = ("base_link_collision", *[f"{leg}_{part}_collision" for leg in LEGS for part in ("thigh","calf")])
FEET = tuple(f"{leg}_foot_ball" for leg in LEGS)
FRONT_CALVES = ("FR_calf_collision", "FL_calf_collision")


def make_model(asset, *, configure=None):
    spec = mj.MjSpec.from_file(str(asset))
    spec.option.timestep = CONTROL_DT / SUBSTEPS
    spec.worldbody.add_geom(name="floor", type=mj.mjtGeom.mjGEOM_PLANE, size=(0, 0, 1),
                           friction=(.8,.005,.0001), rgba=(.24,.28,.33,1.))
    spec.worldbody.add_geom(name="wall", type=mj.mjtGeom.mjGEOM_BOX, size=(THICKNESS / 2, WIDTH / 2, WALL_HEIGHT / 2),
                           pos=(0., 0., WALL_HEIGHT / 2), friction=(.8,.005,.0001), rgba=(.22,.48,.52,1.))
    sensors = [*FEET, *FRONT_CALVES]
    for name in sensors:
        spec.add_sensor(name=f"position_{name}", type=mj.mjtSensor.mjSENS_FRAMEPOS,
                        objtype=mj.mjtObj.mjOBJ_GEOM, objname=name)
    for leg in LEGS:
        for axis in ("hip", "thigh", "calf"):
            actuator=spec.add_actuator(name=f"{leg}_{axis}", target=f"{leg}_{axis}_joint",
                                      trntype=mj.mjtTrn.mjTRN_JOINT, forcerange=(-23.7,23.7), forcelimited=True)
            actuator.set_to_position(40.,kv=1.)
    spec.visual.headlight.ambient = (.28,.28,.28)
    spec.visual.headlight.diffuse = (.57,.57,.57)
    if configure is not None:
        configure(spec)
    root = ET.fromstring(spec.to_xml())
    sensor_xml = root.find("sensor")
    for obstacle in ("floor","wall"):
        for geom in FEET + COLLIDERS:
            ET.SubElement(sensor_xml,"contact",name=f"{obstacle}_{geom}",geom1=geom,geom2=obstacle,
                          data="found force pos normal",reduce="maxforce",num="1")
    return mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def signal(model, values, name):
    sensor = model.sensor(name)
    return values[...,sensor.adr[0]:sensor.adr[0]+sensor.dim[0]]


def measurements(model, sc, trajectory, mode):
    def stack(prefix,names):
        return np.stack([signal(model,sc,prefix+name) for name in names],axis=-2)
    step = slice(SUBSTEPS-1,None,SUBSTEPS)
    feet = stack("position_",FEET)[:,step]
    calves = stack("position_",FRONT_CALVES)[:,step]
    floor_feet = stack("floor_",FEET)
    wall_feet = stack("wall_",FEET)
    wall_calves = stack("wall_",FRONT_CALVES)
    wall_forbidden = tuple(g for g in COLLIDERS if mode=="toe" or g not in FRONT_CALVES)
    bad_force=np.maximum(abs(stack("floor_",COLLIDERS)[...,1]).max(axis=-1),
                         abs(stack("wall_",wall_forbidden)[...,1]).max(axis=-1))
    if mode=="calf":
        non_top=(wall_calves[...,0]>.5)&((abs(wall_calves[...,9])<.75)|(wall_calves[...,6]<.287))
        bad_force=np.maximum(bad_force,np.where(non_top,abs(wall_calves[...,1]),0.).max(axis=-1))
    def crest(c):
        return (c[...,0]>.5)&(abs(c[...,9])>=.75)&(c[...,6]>=.288)
    toe_force=np.where(crest(wall_feet),abs(wall_feet[...,1]*wall_feet[...,9]),0.)[:,step,2:]
    calf_force=np.where(crest(wall_calves),abs(wall_calves[...,1]*wall_calves[...,9]),0.)[:,step]
    rear_force=abs(floor_feet[...,1]*floor_feet[...,9])[:,step,:2]
    state=trajectory[:,step,1:]
    return state,feet,calves,toe_force,calf_force,rear_force,bad_force


def windows(mask,n):
    return np.lib.stride_tricks.sliding_window_view(mask,n,axis=1).all(-1)


def evaluate(model, sensors, raw, initial, mode):
    state,feet,calves,toe,calf,rear,hit=measurements(model,sensors,raw,mode)
    base_delta=state[...,0]-initial[1]
    rear_delta=(feet[...,:2,0]-initial[-2:]).min(-1)
    speed=np.linalg.norm(state[...,19:22],axis=-1)
    tilt=1.-2.*(state[...,4]**2+state[...,5]**2)
    bad=np.maximum.accumulate(hit,axis=1)[:,SUBSTEPS-1::SUBSTEPS]>5.
    stable=(speed<.35)&(tilt>.72)&(rear.min(-1)>7.)&~bad
    support=calf if mode=="calf" else toe
    on=support.min(-1)>7.
    if mode=="calf":
        on &= (calf.sum(-1)>toe.sum(-1))  # front load is carried principally by the shins
    braced=windows(stable & on,5)
    progressed=windows(stable & on & (base_delta>.045) & (rear_delta>.075),5)
    ever_braced=np.maximum.accumulate(np.pad(braced,((0,0),(4,0))),axis=1)
    ever_progressed=np.maximum.accumulate(np.pad(progressed,((0,0),(4,0))),axis=1)
    unloaded=windows(ever_progressed & stable & (calf.max(-1)<2.) & (base_delta>.045) & (rear_delta>.075),5) if mode=="calf" else progressed
    return dict(state=state,feet=feet,calves=calves,toe=toe,calf=calf,rear=rear,
                raw_bad_force=hit,stable=stable,on=on,braced=braced,progressed=progressed,
                unload=unloaded,ever_braced=ever_braced,ever_progressed=ever_progressed,
                base_delta=base_delta,rear_delta=rear_delta,speed=speed)


def objective(m, mode):
    state,feet=m['state'],m['feet']
    speed=m['speed']
    tilt=1.-2.*(state[...,4]**2+state[...,5]**2)
    progress=np.clip(m['base_delta']/.06,-1,1)+np.clip(m['rear_delta']/.11,-1,1)
    support=m['calf'] if mode=="calf" else m['toe']
    support_strength=np.clip(support.min(-1)/16.,0.,1.)
    rear_strength=np.clip(m['rear'].min(-1)/12.,0.,1.)
    steadiness=np.exp(-3*speed**2)*np.clip((tilt-.6)/.27,0,1)
    roll=abs(state[...,4])
    # Smooth distance guide towards the far side of the crest while the shins cross top height.
    if mode=="calf":
        foot_x=feet[...,2:,0]; foot_z=feet[...,2:,2]
        guide=(np.exp(-((foot_x-.082)/.10)**2-((foot_z-.304)/.07)**2)).min(-1)
    else:
        guide=(np.exp(-(feet[...,2:,0]/.07)**2-((feet[...,2:,2]-.3238)/.06)**2)).min(-1)
    safe_prefix=np.exp(-.013*np.cumsum(np.maximum(m['raw_bad_force']-3.,0.)**.65,axis=1)[:,SUBSTEPS-1::SUBSTEPS])
    frame_score=(.8*support_strength+.22*guide+.32*rear_strength+.28*steadiness-.2*roll)*safe_prefix
    steadied=np.lib.stride_tricks.sliding_window_view(frame_score,5,axis=1).mean(-1)
    score=steadied.max(-1)+.35*steadied.mean(-1)
    score+=2.5*m['braced'].any(-1)+4*m['progressed'].any(-1)
    if mode=="calf":
        forward=(support_strength*.65+.35*guide)*(.50+.5*rear_strength)*steadiness*safe_prefix
    else:
        forward=(.65+.35*support_strength)*rear_strength*steadiness*safe_prefix
    score+=(forward*np.clip(progress,0,2)).max(-1)*.9 + .26*progress[:,-1]
    score-=.24*np.log1p(m['raw_bad_force'].max(-1)/9.)
    return score


def simulate(roller,model,datas,initial,control):
    n, length,_=control.shape
    raw=np.empty((n,length*SUBSTEPS,1+model.nq+model.nv))
    sensors=np.empty((n,length*SUBSTEPS,model.nsensordata))
    roller.rollout(model,datas,np.broadcast_to(initial[:FULL_PHYSICS_SIZE],(n,FULL_PHYSICS_SIZE)),np.repeat(control,SUBSTEPS,axis=1),
                   state=raw,sensordata=sensors)
    return raw,sensors


def optimize(model,initial,mode,seed,iterations,batch,threads,folder,prior=None):
    rng=np.random.default_rng(seed)
    prefix=None
    if prior is not None:
        with np.load(prior) as warm:
            if not np.array_equal(warm['initial'],initial):
                raise ValueError("The continuation must preserve the initial state")
            progress_frames = np.flatnonzero(warm['ever_progressed'])
            if not progress_frames.size:
                raise ValueError("The continuation requires a verified support and rear-foot advance")
            prefix_end=int(progress_frames[0])+5
            prefix=warm['controls'][:prefix_end]
    frames=81
    grid=np.arange(0,frames,5) if prefix is None else np.r_[len(prefix)-1,np.arange(len(prefix)+4,frames,5),frames-1]
    grid=np.unique(grid)
    interp=np.vstack([np.interp(np.arange(frames),grid,row) for row in np.eye(len(grid))]).T
    mean=np.broadcast_to(initial[8:20],(len(grid),12)).copy()
    length=46 if mode=="calf" else 81
    if mode=="calf":
        # Front feet move slightly over the crest, exposing the distal shins to its upper edge.
        start=np.clip((grid-3)/24,0.,1.)
        for idx in (7,10):
            mean[:,idx]-=.48*start
        for idx in (8,11):
            mean[:,idx]-=.19*start
    for idx in (1,4):
        mean[:,idx]-=.23*np.clip((grid-16)/27,0.,1.)
    for idx in (2,5):
        mean[:,idx]-=.46*np.clip((grid-16)/27,0.,1.)
    if prefix is not None:
        mean[:]=prefix[-1]
        phase=np.clip((grid-len(prefix))/25.,0.,1.)
        mean[:,[7,10]]+=.38*phase[:,None]
        mean[:,[8,11]]+=.16*phase[:,None]
        length=frames
    mean[0]=prefix[-1] if prefix is not None else initial[8:20]
    spread=np.tile([.095,.25,.31],(len(grid),4))
    data=[mj.MjData(model) for _ in range(threads)]
    bounds=model.jnt_range[1:]
    best=-np.inf
    experiment=folder/f'{mode}-best.npz'
    with Rollout(nthread=threads) as roller:
        for it in range(iterations):
            variations = rng.normal(size=(batch,*mean.shape))
            mirror = np.concatenate((variations[:,:,0:3]*[-1,1,1],variations[:,:,6:9]*[-1,1,1]),axis=-1)
            variations[:batch//2,:,[3,4,5,9,10,11]]=mirror[:batch//2]
            samples=mean + variations*spread*max(.28,1.-it/(iterations*1.25))
            samples[:,0]=mean[0]
            samples[0]=mean
            controls=np.clip(interp[:length]@samples,bounds[:,0]-.2,bounds[:,1]+.2)
            if prefix is not None:
                controls[:,:len(prefix)]=prefix
            raw,s=simulate(roller,model,data,initial,controls)
            measured=evaluate(model,s,raw,initial,mode)
            score=objective(measured,mode)
            if prefix is not None:
                phase=slice(len(prefix)+4,None)
                release=(np.exp(-measured['calf'].max(-1)[:,phase]/30.)
                         *np.clip(measured['rear'].min(-1)[:,phase]/22.,0.,1.)
                         *np.exp(-3*measured['speed'][:,phase]**2)
                         *np.clip(measured['base_delta'][:,phase]/.07,0,1)
                         *np.clip(measured['rear_delta'][:,phase]/.10,0,1))
                unsafe=np.maximum.accumulate(measured['raw_bad_force'],axis=1)[:,3::4][:,phase]
                release*=np.exp(-unsafe/30.)
                score+=4*np.lib.stride_tricks.sliding_window_view(release,5,axis=1).mean(-1).max(-1)
                score+=12*measured['unload'].any(-1)
            score-=.01*np.mean((samples[:,2:]-2*samples[:,1:-1]+samples[:,:-2])**2/spread[1:-1]**2,axis=(1,2))
            ranking=np.argsort(score)
            good=ranking[-max(8,batch//13):]
            mean=.18*mean+.82*samples[good].mean(0)
            selected=good[-1]
            if score[selected]>best:
                best=score[selected]
                np.savez_compressed(experiment,controls=controls[selected],initial=initial,
                                    **{k:v[selected] for k,v in measured.items()})
            if it%5==0 or (measured['progressed'][selected].any() and it%2==0):
                print(mode,it,round(float(best),4), round(float(score[selected]),4),
                      'brace',measured['braced'][selected].sum(),'move',measured['progressed'][selected].sum(),
                      'unload',measured['unload'][selected].sum(),
                      'x',np.round(np.max(measured['base_delta'][selected]),3),
                      'rear',np.round(np.max(measured['rear_delta'][selected]),3),
                      'hit',np.round(np.max(measured['raw_bad_force'][selected]),1),
                      'calf',np.round(measured['calf'][selected].min(-1).max(),1),flush=True)
    return experiment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('toe','calf'),required=True)
    p.add_argument('--seed',type=int,default=502)
    p.add_argument('--iters',type=int,default=120)
    p.add_argument('--batch',type=int,default=256)
    p.add_argument('--threads',type=int,default=6)
    p.add_argument('--prior',type=Path)
    args=p.parse_args()
    if args.threads < 1 or args.iters < 1 or args.batch < 16:
        p.error("--threads and --iters must be positive and --batch must be at least 16")
    if args.prior and args.mode != "calf":
        p.error("--prior continues the calf-support search into unloading")
    args.output.mkdir(parents=True,exist_ok=True)
    model=make_model(args.work/'input/wbl_tracking.xml')
    with np.load(args.work/'first-nominal-eval/rollouts.npz') as record:
        index=int(np.flatnonzero(record['success'])[0])
        t=int(record['stop'][index])
        sample=record['states'][index,t]
        initial=np.r_[0.,sample,record['soles'][index,t,:2,0]]
    optimize(model,initial,args.mode,args.seed,args.iters,args.batch,args.threads,args.output,args.prior)


if __name__=='__main__':
    main()
