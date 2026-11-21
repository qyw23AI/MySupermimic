"""Search and verify the WBL rear-shin catch and step-down on the 30 cm wall."""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco as mj
from mujoco.rollout import Rollout
import numpy as np

from .brace_experiment import (
    COLLIDERS, CONTROL_DT, FEET, HOME, SUBSTEPS, TOE_SIZE,
    evaluate, make_model, signal, simulate, windows,
)
from .robot import HEIGHT, THICKNESS

REAR_CALVES = ("RR_calf_collision", "RL_calf_collision")
SUB_DT = CONTROL_DT / SUBSTEPS
REAR_HOLD = 6
FINAL_HOLD = 10


def force_signals(model, sensors, surface, names):
    return np.stack([signal(model, sensors, f"{surface}_{name}") for name in names], axis=-2)


def crest_load(contact):
    top = (contact[..., 0] > .5) & (abs(contact[..., 9]) >= .75) & (contact[..., 6] >= HEIGHT - .013)
    return np.where(top, abs(contact[..., 1] * contact[..., 9]), 0.)


def ever(mask, size):
    return np.maximum.accumulate(np.pad(windows(mask, size), ((0,0),(size - 1,0))), axis=1)


def measurements(model, raw, sc, initial, release):
    after = release * SUBSTEPS
    frame = slice(SUBSTEPS - 1, None, SUBSTEPS)
    # The physics frames before release have their own front-calf gate.
    earlier = sc[:, :after]
    raw_earlier = raw[:, :after]
    prefix = evaluate(model, earlier, raw_earlier, initial, "calf")
    released = prefix["unload"][:, -1]
    sc = sc[:, after:]
    state = raw[:, after:][:, frame, 1:]
    feet = np.stack([signal(model, sc[:, frame], f"position_{name}") for name in FEET], axis=-2)
    floor_s = force_signals(model, sc, "floor", FEET)
    wall_s = force_signals(model, sc, "wall", FEET)
    rear_s = force_signals(model, sc, "wall", REAR_CALVES)
    floor_force = abs(floor_s[..., 1] * floor_s[..., 9])
    rear_load = crest_load(rear_s)
    wall_force = abs(wall_s[..., 1])
    disallowed_wall = [g for g in COLLIDERS if g not in REAR_CALVES]
    bad = abs(force_signals(model, sc, "wall", disallowed_wall)[..., 1]).max(-1)
    bad = np.maximum(bad, abs(force_signals(model, sc, "floor", COLLIDERS)[..., 1]).max(-1))
    side = (rear_s[..., 0] > .5) & ((abs(rear_s[..., 9]) < .75) | (rear_s[..., 6] < HEIGHT - .013))
    bad = np.maximum(bad, np.where(side, abs(rear_s[..., 1]), 0.).max(-1))
    # Use force observations from all four physics steps for invalid contacts.
    peak_bad = np.maximum.accumulate(bad, axis=1)[:, frame]
    okay = peak_bad < 5.
    rear = rear_load[:, frame]
    floor = floor_force[:, frame]
    airborne = ((floor.max(-1) < 5.) & (wall_force[:, frame].max(-1) < 5.)
                & (rear.max(-1) < 5.) & (feet[..., 2].min(-1) > .029))
    flew = ever(airborne, 2)
    front_down = ((floor[..., 2:] > 5.) & (feet[..., 2:, 0] > .08)).all(-1)
    up = 1. - 2. * (state[..., 4]**2 + state[..., 5]**2)
    speed = np.linalg.norm(state[..., 19:22], axis=-1)
    angular = np.linalg.norm(state[..., 22:25], axis=-1)
    chest_safe = (up > .65) & (state[..., 2] > .27)
    braced = (rear.min(-1) > 7.) & front_down & chest_safe & (speed < .42) & (angular < 2.5)
    catch = windows(braced & flew & okay & released[:, None], REAR_HOLD)
    caught = ever(braced & flew & okay & released[:, None], REAR_HOLD)
    far = (feet[..., 0].min(-1) > THICKNESS/2 + TOE_SIZE)
    # Each rear foot has to land on the far floor. The footfalls must
    # be separated, and the body must not become fully airborne again.
    far_rear_ground = ((floor[..., :2] > 5.) & (feet[..., :2,0] > THICKNESS/2 + TOE_SIZE))
    far_rear_ground &= caught[..., None] & okay[..., None]
    queued = [ever(far_rear_ground[...,leg],2) for leg in range(2)]
    earlier = [np.pad(flag[:,:-3],((0,0),(3,0))) for flag in queued]
    spaced = (earlier[0] & far_rear_ground[...,1]) | (earlier[1] & far_rear_ground[...,0])
    one_down = np.maximum.accumulate(spaced,axis=1)
    support = (floor.max(-1) > 5.) | (rear.max(-1) > 5.) | (wall_force[:, frame].max(-1) > 5.)
    extra_hop = ever(caught & ~support,2)
    done = (caught & one_down & ~extra_hop & okay & far & (floor.min(-1)>5.) & (up>.82)
            & (state[..., 2]>.22) & (speed < .35) & (angular < 2.))
    landed = windows(done, FINAL_HOLD)
    return dict(state=state,feet=feet,floor=floor,rear=rear,front_down=front_down,airborne=airborne,
                flew=flew,catch=catch,caught=caught,one_down=one_down,landed=landed,
                speed=speed,angular=angular,up=up,bad=bad,peak_bad=peak_bad,
                extra_hop=extra_hop, far_rear_ground=far_rear_ground, support=support,
                floor_force=floor_force,rear_load=rear_load,raw=raw[:, after:],released=released)


def reward(measured, phase, fixed_from=0):
    m = measured
    feet, state = m["feet"], m["state"]
    okay = np.exp(-m["peak_bad"]/110.)
    # Foot centres near the far rim are a smooth proxy for the rear shins'
    # placement. Acceptance is based on force and normal at the shins themselves.
    rear_guide = np.exp(-((feet[..., :2, 0]-.075)/.155)**2
                        -((feet[..., :2, 2]-.320)/.13)**2).mean(-1)
    front_guide = np.exp(-((feet[..., 2:, 0]-.63)/.30)**2
                         -((feet[..., 2:, 2]-.024)/.15)**2).mean(-1)
    shin = np.clip(m["rear"].min(-1)/23.,0,1)
    front = np.clip(m["floor"][..., 2:].min(-1)/15.,0,1)
    stopped = np.exp(-m["speed"]**2-.08*m["angular"]**2)
    posture = np.clip((m["up"]-.6)/.35,0,1)
    if phase == "catch":
        target = (.85*rear_guide+.55*front_guide+.75*shin+.35*front
                  +.85*shin*front*stopped*posture) * okay * m["flew"]
        hold = np.lib.stride_tricks.sliding_window_view(target, REAR_HOLD, axis=1).mean(-1).max(-1)
        return hold + .6*target.max(-1) + 4.*m["catch"].any(-1) - .12*np.log1p(m["bad"].max(-1)/12.)
    progress = np.clip((feet[..., :2, 0].min(-1)+.025)/.37,0,1)
    support = np.clip(m["floor"].min(-1)/13.,0,1)
    standing_front = np.clip(m["floor"][...,2:].max(-1)/14.,0,1)
    floating_rear = np.maximum(0.,feet[...,:2,2].max(-1)-.34)
    target = (1.05*progress + 1.25*support*stopped*posture +
              .55*np.clip((state[...,0]-.24)/.44,0,1) +
              .72*standing_front - .7*floating_rear)*okay
    target *= np.where(m["extra_hop"],.32,1.)
    target = target[:, fixed_from:]
    # Reward a low-rate four-foot support window during the final transfer,
    # even if a single sensor sample still falls short of a hard gate.
    stance = (support*stopped*posture*okay*np.exp(-((state[...,2]-.34)/.09)**2))[:,fixed_from:]
    stand_window = np.lib.stride_tricks.sliding_window_view(stance, FINAL_HOLD, axis=1).mean(-1).max(-1)
    hold = np.lib.stride_tricks.sliding_window_view(target, FINAL_HOLD, axis=1).mean(-1).max(-1)
    return (hold + .55*target.mean(-1) + 2.2*stand_window + 3.*m["one_down"].any(-1)
            + 8.*m["landed"].any(-1) - .20*np.log1p(m["bad"].max(-1)/12.)
            - .85*m["extra_hop"].any(-1))


def search(args):
    model = make_model(args.asset)
    with np.load(args.direct, allow_pickle=False) as file:
        initial = file["initial"]
        seed_controls = file["controls"]
        release = int(file["prefix"])
    if args.phase == "catch":
        fixed = seed_controls[:release]
        length = release + 57
    else:
        with np.load(args.catch, allow_pickle=False) as file:
            first_catch = np.flatnonzero(file["catch"])
            if not first_catch.size:
                raise ValueError("Step-down requires a prior verified dual rear-shin catch")
            fixed = file["controls"][:release + int(first_catch[0]) + REAR_HOLD]
            if not np.array_equal(file["initial"], initial):
                raise ValueError("The catch and direct control must share their starting state")
        length = len(fixed) + 64
        start = fixed[-1]
        t = np.arange(1,65)
        seed = np.broadcast_to(start,(64,12)).copy()
        # Keep one rear shin loaded while the opposite hind leg unfolds;
        # the front pair is guided to stay planted during this transfer.
        alpha = np.clip((t-6)/32,0.,1.)
        alpha = alpha**2*(3.-2.*alpha)
        seed[:,6:] += alpha[:,None]*(HOME[6:]-start[6:])
        for leg, delay in ((0,4),(1,17)):
            alpha = np.clip((t-delay)/24,0.,1.)
            alpha = alpha**2*(3.-2.*alpha)
            slot = slice(3*leg,3*leg+3)
            seed[:,slot] += alpha[:,None]*(HOME[slot]-start[slot])
        seed_controls = np.vstack((fixed,seed))
    if args.freeze_until:
        if not args.warm:
            raise ValueError("--freeze-until requires --warm")
        with np.load(args.warm,allow_pickle=False) as candidate:
            segment = candidate["controls"][:args.freeze_until].copy()
        if not np.array_equal(segment[:len(fixed)],fixed) or len(segment)!=args.freeze_until:
            raise ValueError("Warm start does not preserve the previously qualified control prefix")
        fixed = segment
        length = max(length,len(fixed)+34)
    end = len(fixed)
    grid = np.unique(np.r_[end-1,np.arange(end+2,length,4),length-1])
    blend = np.array([np.interp(np.arange(length),grid,row) for row in np.eye(len(grid))]).T
    mean = np.pad(seed_controls,((0,max(0,length-len(seed_controls))),(0,0)),mode="edge")[grid].copy()
    # Start the search near the actual second jump, then move the target toward
    # the split-level stance. The physical rollout decides what is achievable.
    if args.phase == "catch" and args.warm is None:
        t = grid-end
        fold = np.clip((t-9)/13,0,1)
        mean[:,[1,4]] -= .17*fold[:,None]
        mean[:,[2,5]] -= .22*fold[:,None]
    if args.warm:
        with np.load(args.warm,allow_pickle=False) as file:
            warm = file["controls"]
        mean = np.pad(warm,((0,max(0,length-len(warm))),(0,0)),mode="edge")[grid].copy()
    mean[0] = fixed[-1]
    # Symmetric variations are useful for the catch. The step-down still
    # searches separate left and right motions to obtain an intermediate footfall.
    rng = np.random.default_rng(args.seed)
    spread = np.tile([.10,.28,.34],(len(grid),4))
    if args.freeze_until:
        spread *= .58
    workers = [mj.MjData(model) for _ in range(args.threads)]
    best = -np.inf
    best_path = args.output / f"{args.phase}.npz"
    args.output.mkdir(parents=True,exist_ok=True)
    with Rollout(nthread=args.threads) as roller:
        for iteration in range(args.iters):
            perturb = rng.normal(size=(args.batch,*mean.shape))*spread
            n_sym = args.batch//3 if args.phase=="catch" else args.batch//8
            perturb[:n_sym,:,[3,4,5,9,10,11]] = perturb[:n_sym,:,[0,1,2,6,7,8]]*[-1,1,1,-1,1,1]
            knots = mean + perturb*max(.27,1.-iteration/(args.iters*1.20))
            knots[0] = mean
            knots[:,0] = fixed[-1]
            u = np.clip(blend@knots,model.jnt_range[1:,0]-.2,model.jnt_range[1:,1]+.2)
            u[:,:end] = fixed
            raw, sensors = simulate(roller,model,workers,initial,u)
            m = measurements(model,raw,sensors,initial,release)
            value = reward(m,args.phase,max(0,end-release))
            value -= .015*np.mean(((knots[:,2:]-2*knots[:,1:-1]+knots[:,:-2])/spread[1:-1])**2,axis=(1,2))
            elite = np.argsort(value)[-max(8,args.batch//14):]
            mean = .20*mean + .80*knots[elite].mean(0)
            index = elite[-1]
            if value[index] > best:
                best = value[index]
                np.savez_compressed(best_path,controls=u[index],initial=initial,release=release,
                                    **{key: value[index] for key,value in m.items()})
            if iteration%4==0 or (m["landed"].any() and args.phase=="step"):
                print(iteration,round(float(best),3),"catch",int(m["catch"][index].sum()),
                      "one_down",bool(m["one_down"][index].any()),"land",int(m["landed"][index].sum()),
                      "both_shins_n",round(float(m["rear"][index].min(-1).max()),1),
                      "bad_n",round(float(m["bad"][index].max()),1),"extra_hop",bool(m["extra_hop"][index].any()),flush=True)
            if args.phase=="step" and m["landed"][index].any():
                break


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset",type=Path,required=True)
    p.add_argument("--direct",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--phase",choices=("catch","step"),default="catch")
    p.add_argument("--catch",type=Path)
    p.add_argument("--warm",type=Path)
    p.add_argument("--freeze-until",type=int)
    p.add_argument("--iters",type=int,default=150)
    p.add_argument("--batch",type=int,default=224)
    p.add_argument("--threads",type=int,default=8)
    p.add_argument("--seed",type=int,default=409)
    args=p.parse_args()
    if args.phase=="step" and not args.catch:
        p.error("--phase step requires --catch")
    if args.iters < 1 or args.threads < 1 or args.batch < 16:
        p.error("Use positive iterations/threads and at least 16 candidates")
    search(args)
