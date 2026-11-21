"""Refine a four-stage reference in Warp on training-only physical perturbations."""
from pathlib import Path
import argparse
import numpy as np
import torch
from . import calf_tracking as tr
from .four_stage_tracking import PhysicsAudit, TransferScore
from supermimic.envs.state_io import StateIO



def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("run",type=Path); p.add_argument("--iterations",type=int,default=28)
    p.add_argument("--stage",choices=["descent","full"],default="descent")
    p.add_argument("--edited-from",type=int)
    p.add_argument("--prior",type=Path)
    p.add_argument("--out",default="descent-optimized.npz")
    p.add_argument("--phase",choices=("transfer","catch"),default="transfer")
    args=p.parse_args()
    with np.load(args.run/"input"/f"{args.stage}-teacher.npz") as f:
        targets=f["targets"].copy()
        release=int(f["release"])
    if args.prior:
        with np.load(args.prior) as f:
            targets=f["targets"].copy()
    if args.phase=="catch" and args.stage!="full":
        p.error("The rear-shin catch is part of the full stage")
    b, trials=36, 4
    env=tr.make_evaluation(args.run/"input",args.stage,"cuda:0",b*trials,1861)
    cmd=env.command_manager.get_term("motion")
    xyz=tr.cohort(707,trials); xyz[0]=0.
    tr.install_offsets(cmd,torch.tensor(np.tile(xyz,(b,1)),dtype=torch.float32,device=env.device))
    audit=PhysicsAudit(env)
    io=StateIO(env)
    start=args.edited_from if args.edited_from is not None else (release if args.phase=="catch" else 40)
    if not 0 <= start < len(targets)-6:
        p.error("The editable region must have at least seven control frames")
    grid=np.unique(np.r_[start,np.arange(start+3,len(targets),4),len(targets)-1])
    interp=np.stack([np.interp(np.arange(len(targets)),grid,x) for x in np.eye(len(grid))],axis=-1)
    rng=np.random.default_rng(1932)
    mean=np.zeros((len(grid),12))
    spread=np.array([.035,.10,.13]*4)
    best=-np.inf
    try:
        with torch.inference_mode():
            for iteration in range(args.iterations):
                knots=mean+rng.standard_normal((b,*mean.shape))*spread*max(.22,1-iteration/(args.iterations*1.25))
                knots[:,0]=0
                knots[0]=mean
                candidates=targets+interp@knots
                actions=torch.tensor(np.repeat(candidates,trials,axis=0),device=env.device,dtype=torch.float32)
                nominal_state=[]
                env.reset()
                initial=io.read().cpu().numpy()
                nominal_state.append(initial[::trials].copy())
                score=TransferScore(b*trials,args.stage,initial,tr.as_numpy(audit.read_base())["feet"])
                active=np.ones(b*trials,dtype=bool)
                success=~active.copy()
                greatest_run=np.zeros(b*trials)
                peak_invalid=np.zeros(b*trials)
                steady=[]
                stop=np.zeros(b*trials,dtype=int)
                for t in range(len(targets)):
                    audit.frames.clear()
                    _,_,term,timeout,_=env.step((actions[:,t]-cmd.robot.data.default_joint_pos)/.25)
                    state=io.read().cpu().numpy()
                    nominal_state.append(state[::trials].copy())
                    contact,_=audit.control_frame()
                    accepted=score.step(active,state,contact)&~term.cpu().numpy()
                    success |= score.caught if args.phase=="catch" else accepted
                    safe=(~score.hit & ~score.recontact & ~score.extra_hop & active & score.caught)
                    peak_invalid=np.maximum(peak_invalid,np.where(active,contact["bad_n"],0.))
                    stable=(np.clip(contact["ground"].min(-1)/12.,0,1)
                            * np.exp(-2*np.linalg.norm(state[:,19:22],axis=-1)**2
                                     -.10*np.linalg.norm(state[:,22:25],axis=-1)**2)
                            *contact["clear"]*(state[:,2]>.23))
                    if args.phase=="catch":
                        safe=~score.hit & ~score.recontact & active & score.second_air
                        stable=(np.clip(contact["rear_calf"].min(-1)/12.,0,1)*
                                np.clip(contact["ground"][:,2:].min(-1)/12.,0,1)*
                                np.exp(-np.linalg.norm(state[:,19:22],axis=-1)**2
                                       -.085*np.linalg.norm(state[:,22:25],axis=-1)**2))
                    run=score.catch_run if args.phase=="catch" else score.final_run
                    steady.append(stable*safe)
                    greatest_run=np.maximum(greatest_run,np.where(active,run,0))
                    done=(term|timeout).cpu().numpy()
                    stop[active & (accepted | done)]=t+1
                    active &= ~done
                    if done.any():
                        env.reset(env_ids=torch.nonzero(term|timeout,as_tuple=True)[0])
                hold=np.lib.stride_tricks.sliding_window_view(np.asarray(steady).T,10,axis=1).mean(-1).max(-1)
                value=(3*success + .12*greatest_run + 2.5*hold + .20*score.caught
                       - .14*np.log1p(peak_invalid/5) - .22*score.extra_hop - .7*score.recontact)
                value=value.reshape(b,trials)
                objective=.72*value.mean(-1)+.28*value.min(-1)
                objective-=.022*((knots/spread)**2).mean(axis=(1,2))
                elite=np.argsort(objective)[-6:]
                i=int(elite[-1])
                mean=.25*mean+.75*knots[elite].mean(0)
                if objective[i]>best:
                    best=float(objective[i])
                    np.savez_compressed(args.run/args.out,targets=candidates[i],states=np.stack(nominal_state,1)[i],
                                        training_success=success.reshape(b,trials)[i],release=release)
                if iteration%3==0:
                    print(iteration,best,success.reshape(b,trials)[i].tolist(),int(greatest_run.reshape(b,trials)[i].max()),flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
