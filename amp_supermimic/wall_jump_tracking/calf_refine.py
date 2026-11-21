"""Search for a wall- and landing-safe teacher across a training-only perturbation set."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .calf_tracking import CalfScore, cohort, contact_measurements, install_offsets, make_evaluation


def trial_values(state, c, release):
    cut = slice(release + 1, None)
    air = c["air"][:, cut]
    valid_air = np.maximum.accumulate(np.pad(air[:, 1:] & air[:, :-1], ((0,0),(1,0))), axis=1)
    bad = np.maximum(c["bad_n"][:, cut], c["calf_any"][:, cut].max(-1))
    bad = np.maximum.accumulate(bad, axis=1)
    safe = np.exp(-bad / 70.)
    clear = c["clear"][:, cut]
    x = state[:, cut, 0]
    rear = c["feet"][:, cut, :2]
    rear_height = np.clip((rear[..., 2].min(-1) - .0238) / .32, 0, 1)
    rear_dist = np.clip((rear[..., 0].min(-1) + .47) / .53, 0, 1)
    clean_flight = (clear & valid_air & (bad < 5.)).any(-1)
    clean = (1.2 * rear_dist + .4 * rear_height + .55 * np.clip((x+.22)/.45,0,1)) * safe
    speed = np.linalg.norm(state[:, cut, 19:22], axis=-1)
    rot = np.linalg.norm(state[:, cut, 22:25], axis=-1)
    height = state[:, cut, 2]
    up = 1. - 2. * (state[:, cut, 4]**2 + state[:, cut, 5]**2)
    feet_down = (c["ground"][:, cut] > 5.).mean(-1)
    settled = (np.exp(-speed**2 - .12*rot**2) * feet_down**2 * safe * valid_air * clear
               * np.clip((height-.17)/.13,0,1) * np.clip((up-.65)/.3,0,1))
    if settled.shape[1] >= 10:
        hold = np.lib.stride_tricks.sliding_window_view(settled, 10, axis=1).mean(-1).max(-1)
    else:
        hold = settled.mean(-1)
    return (clean.max(-1) + 1.1 * clean_flight + 3.8 * hold
            - 1.3 * np.log1p(bad[:, :min(16,bad.shape[1])].max(-1)/22.))


def optimize(args):
    from supermimic.envs.state_io import StateIO

    with np.load(args.work / "second-teacher.npz", allow_pickle=False) as teacher:
        reference = teacher["targets"].copy()
        release = int(teacher["release"])
    if args.prior:
        with np.load(args.prior, allow_pickle=False) as previous:
            current = previous["targets"]
        reference[:len(current)] = current
    end = len(reference) if args.phase == "land" else min(release+34,len(reference))
    start = release if args.phase == "release" else args.landing_start
    if not release <= start < end - 12:
        raise ValueError("Need at least 12 editable frames after the accepted unload")
    grid = np.unique(np.r_[start, np.arange(start+3,end,4),end-1])
    interpolation = np.vstack([np.interp(np.arange(end), grid, row) for row in np.eye(len(grid))]).T
    mean = np.zeros((len(grid),12))
    spread = np.array([.045,.14,.21]*4)
    if args.phase == "land":
        spread *= 1.25
    rng = np.random.default_rng(args.seed)
    offsets = cohort(seed=733,count=args.repeats)
    offsets[0] = 0.
    count = args.population * args.repeats
    env = make_evaluation(args.work, "second", args.device, count, args.seed)
    command = env.command_manager.get_term("motion")
    install_offsets(command, torch.tensor(np.tile(offsets,(args.population,1)),dtype=torch.float32,device=env.device))
    home = command.robot.data.default_joint_pos
    io = StateIO(env)
    measure = contact_measurements(env)
    best = -np.inf
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        with torch.inference_mode():
            for iteration in range(args.iterations):
                d = rng.normal(size=(args.population,*mean.shape)) * spread
                symmetry = d[:args.population//3,:,[0,1,2,6,7,8]] * [-1,1,1,-1,1,1]
                d[:args.population//3,:,[3,4,5,9,10,11]] = symmetry
                knots = mean + d * max(.25, 1 - iteration/(args.iterations*1.2))
                knots[:,0] = 0.
                knots[0] = mean
                candidate = reference[:end] + interpolation @ knots
                u = torch.as_tensor(np.repeat(candidate,args.repeats,axis=0),
                                    device=env.device,dtype=torch.float32)
                env.reset()
                state = [io.read()]
                readings = [measure()]
                for frame in range(end):
                    _, _, term, timeout, _ = env.step((u[:,frame]-home)/.25)
                    state.append(io.read())
                    readings.append(measure())
                    if (term | timeout).any():
                        env.reset(env_ids=torch.nonzero(term | timeout, as_tuple=True)[0])
                trajectory = torch.stack(state,1).cpu().numpy()
                contact = {key:torch.stack([x[key] for x in readings],1).cpu().numpy() for key in readings[0]}
                raw = trial_values(trajectory,contact,release).reshape(-1,args.repeats)
                score = .75*raw.mean(1)+.25*raw.min(1)
                score -= .04 * ((knots/spread)**2).mean((1,2))
                elite = np.argsort(score)[-max(4,args.population//10):]
                mean = .25*mean+.75*knots[elite].mean(0)
                idx = int(elite[-1])
                rows = slice(idx*args.repeats,(idx+1)*args.repeats)
                if score[idx] > best:
                    best = float(score[idx])
                    offset = rows.start
                    selected = reference.copy()
                    selected[:end] = candidate[idx]
                    np.savez_compressed(args.output,targets=selected,states=trajectory[offset],release=release,
                                        training_offsets=offsets,score=best,rollout_ends_at=end)
                if iteration % 3 == 0:
                    peak = contact["calf_any"][rows,release+1:].max((1,2)).round(1)
                    print(f"{args.phase} {iteration}: score={best:.3f}, calf recontacts N={peak.tolist()}",flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase",choices=("release","land"),default="release")
    parser.add_argument("--prior",type=Path)
    parser.add_argument("--landing-start",type=int,default=68)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--population",type=int,default=64)
    parser.add_argument("--repeats",type=int,default=4)
    parser.add_argument("--iterations",type=int,default=32)
    parser.add_argument("--seed",type=int,default=2827)
    args=parser.parse_args()
    optimize(args)


if __name__ == "__main__":
    main()
