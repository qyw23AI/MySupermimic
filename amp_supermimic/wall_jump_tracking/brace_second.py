from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import mujoco as mj
from mujoco.rollout import Rollout
from .brace_experiment import (
    FEET, FRONT_CALVES, HOME, SUBSTEPS, THICKNESS, TOE_SIZE,
    evaluate, make_model, signal, simulate, windows,
)

LANDING_WINDOW = 10


def score_jump(model,sc,raw,initial,prefix_length):
    m=evaluate(model,sc,raw,initial,'calf')
    state=m['state'][:,prefix_length:]
    feet=m['feet'][:,prefix_length:]
    force=np.stack([signal(model,sc[ :,3::4][:,prefix_length:], f'{surface}_{foot}')[...,1]
                    for surface in ('floor','wall') for foot in FEET],axis=-1)
    floor=np.abs(force[...,:4])>5
    wall=np.abs(force[...,4:])>5
    on_calves=m['calf'][:,prefix_length:].max(-1)>5
    calf_contact = np.stack([
        abs(signal(model, sc[:, prefix_length * SUBSTEPS:], "wall_" + name)[..., 1])
        for name in FRONT_CALVES
    ], axis=-1).max(axis=-1)
    recontact = np.maximum.accumulate(calf_contact, axis=1)[:, SUBSTEPS - 1::SUBSTEPS]
    any_bad = np.maximum(
        np.maximum.accumulate(m['raw_bad_force'], axis=1)[:, SUBSTEPS - 1::SUBSTEPS][:, prefix_length:],
        recontact,
    )
    clearance=feet[...,0].min(-1)>THICKNESS / 2 + TOE_SIZE
    air=~(floor.any(-1)|wall.any(-1)|on_calves)&(feet[...,2].min(-1)>.029)
    flew=np.maximum.accumulate(np.pad(windows(air,2),((0,0),(1,0))),axis=1)
    up=1-2*(state[...,4]**2+state[...,5]**2)
    stable=(up>.82)&(state[...,2]>.22)&(np.linalg.norm(state[...,19:22],axis=-1)<.38)&(np.linalg.norm(state[...,22:25],axis=-1)<2.0)
    success=windows(flew&clearance&floor.all(-1)&stable&(any_bad<5.),LANDING_WINDOW)
    no_contact=np.exp(-.018*np.power(any_bad,.68))
    rear_x=feet[...,:2,0].min(-1)
    rear_z=feet[...,:2,2].min(-1)-TOE_SIZE
    position=np.clip((rear_x+.48)/.52,0,1.0)
    height=np.clip(rear_z/.32,0,1.0)
    raising=np.clip((state[...,2]-.32)/.32,0,1)
    forward=np.clip((state[...,0]+.23)/.44,0,1.0)
    score=(1.25*position+.65*height+.4*forward+.10*raising+.23*flew)*no_contact
    score=score.max(-1)+.35*score.mean(-1)
    clean_pass=(flew&clearance&(any_bad<5.)).any(-1)
    land=(flew&clearance&floor.all(-1)&(up>.70)&(any_bad<5.)).any(-1)
    leading=np.clip((feet[...,2:,0].min(-1)+.04)/.34,0,1.0)
    score+=.4*(leading*position*no_contact).max(-1)
    stability = (np.exp(-.75*np.linalg.norm(state[...,19:22],axis=-1)**2
                        -.16*np.linalg.norm(state[...,22:25],axis=-1)**2)
                 * np.clip((up-.60)/.37,0.,1.))
    upright=np.clip((state[...,2]-.12)/.20,0.,1.)
    soak = stability*floor.mean(-1)**2*flew*clearance*no_contact*(.18+.82*upright)
    score+=1.1*clean_pass+2.2*land+15.*success.any(-1)
    score+=4.*np.lib.stride_tricks.sliding_window_view(soak,5,axis=1).mean(-1).max(-1)
    score+=2.8*soak[:,-18:].mean(-1)
    score+=1.5*np.lib.stride_tricks.sliding_window_view(soak,10,axis=1).mean(-1).max(-1)
    score-=.40*np.log1p(recontact.max(-1)/8.)
    return score,dict(success=success,air=air,cleared=clearance,landed=land,contacts=floor,
                      pass_air=flew.any(-1),rear_x=rear_x,rear_z=rear_z,
                      calf_recontact=recontact, second_forbidden=any_bad, **m)


def optimize(model,args):
    with np.load(args.prior) as demo:
        result=np.flatnonzero(demo['unload'])
        if not len(result):
            raise ValueError('Second jump requires verified calf unload')
        prefix=demo['controls'][:result[0]+5].copy()
        initial=demo['initial'].copy()
    n=len(prefix)
    validator = [mj.MjData(model)]
    with Rollout(nthread=1) as roller:
        raw, sensors = simulate(roller, model, validator, initial, prefix[None])
        checked = evaluate(model, sensors, raw, initial, "calf")
    if not checked["unload"][0, -1]:
        raise ValueError("The calf unload did not pass a fresh replay from the common initial state")
    length=n+80
    fixed = prefix
    if args.freeze_until:
        if args.warm is None or not n <= args.freeze_until <= length - 12:
            raise ValueError("--freeze-until needs a warm sequence and must leave at least 12 frames")
        with np.load(args.warm) as candidate:
            fixed = candidate["controls"][:args.freeze_until].copy()
        if not np.array_equal(fixed[:n],prefix) or len(fixed) != args.freeze_until:
            raise ValueError("Warm sequence does not preserve the verified unloading prefix")
    fixed_count = len(fixed)
    grid=np.unique(np.r_[fixed_count-1,np.arange(fixed_count+3,length,4),length-1])
    W=np.vstack([np.interp(np.arange(length),grid,row) for row in np.eye(len(grid))]).T
    q=fixed[-1]
    mean=np.broadcast_to(q,(len(grid),12)).copy()
    rear_thigh=[1,4]; rear_knee=[2,5]
    time=grid-(n-1)
    def adjust(cols,knot_frames,value):
        mean[:,cols]+=np.interp(time,knot_frames,value)[:,None]
    adjust(rear_thigh,[0,5,10,17,29,52],[0,-.14,-.35,-.10,.30,-.21])
    adjust(rear_knee,[0,5,10,17,29,52],[0,.25,.65,-.38,-.38,.05])
    adjust([7,10],[0,6,13,24,36,52],[0,-.40,-.25,-.36,-.30,-.22])
    adjust([8,11],[0,6,13,24,36,52],[0,-.07,.25,.62,.77,.71])
    if args.warm:
        with np.load(args.warm) as warm:
            known=warm['controls']
            extras=np.linspace(known[-1],HOME,max(length-len(known),0)+1)[1:]
            mean=np.concatenate((known,extras))[grid].copy()
            home_blend=np.clip((grid-fixed_count-34)/30.,0.,.85)
            mean=mean*(1.-home_blend[:,None])+HOME*home_blend[:,None]
    spread=np.tile([.13,.30,.37],(len(grid),4))
    bounds=model.jnt_range[1:]
    rng=np.random.default_rng(args.seed)
    data=[mj.MjData(model) for _ in range(args.threads)]
    best=-np.inf
    with Rollout(nthread=args.threads) as roller:
        for iteration in range(args.iters):
            changes=rng.normal(size=(args.batch,*mean.shape))*spread
            changes[:args.batch//2,:,[3,4,5,9,10,11]]=changes[:args.batch//2,:,[0,1,2,6,7,8]]*[-1,1,1,-1,1,1]
            knots=mean+changes*max(.30,1-iteration/(args.iters*1.2))
            knots[:,0]=q
            knots[0]=mean
            u=np.clip(W@knots,bounds[:,0]-.2,bounds[:,1]+.2)
            u[:,:fixed_count]=fixed
            states,sensors=simulate(roller,model,data,initial,u)
            score,m=score_jump(model,sensors,states,initial,n)
            score-=.012*np.mean(((knots[:,2:]-2*knots[:,1:-1]+knots[:,:-2])/spread[1:-1])**2,axis=(1,2))
            top=np.argsort(score)[-max(10,args.batch//16):]
            mean=.22*mean+.78*knots[top].mean(0)
            idx=top[-1]
            if score[idx]>best:
                best=score[idx]
                np.savez_compressed(args.output,controls=u[idx],initial=initial,prefix=n,
                                    **{k:v[idx] for k,v in m.items()})
            if iteration%4==0:
                print(iteration,round(float(best),4),'air',m['pass_air'][idx],
                      'rear_x',round(float(m['rear_x'][idx].max()),3),
                      'rear_z',round(float(m['rear_z'][idx].max()),3),
                      'hit',round(float(m['raw_bad_force'][idx].max()),1),'recontact',round(float(m['calf_recontact'][idx].max()),1),'land',m['landed'][idx],
                      'pass',m['success'][idx].sum(),flush=True)
            if m['success'][idx].any():
                break

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warm',type=Path)
    parser.add_argument('--freeze-until',type=int)
    parser.add_argument('--seed',type=int,default=853)
    parser.add_argument('--iters',type=int,default=130)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--batch',type=int,default=240)
    args=parser.parse_args()
    if args.threads < 1 or args.iters < 1 or args.batch < 16:
        parser.error("--threads and --iters must be positive and --batch must be at least 16")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    optimize(make_model(args.work/'input/wbl_tracking.xml'),args)
