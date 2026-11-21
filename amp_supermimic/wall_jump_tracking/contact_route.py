"""Optimize and independently replay the four-stage WBL wall maneuver in MuJoCo."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco as mj
from mujoco.rollout import Rollout
import numpy as np

from .brace_experiment import COLLIDERS, CONTROL_DT, FEET, FRONT_CALVES, SUBSTEPS, make_model, signal, simulate
from .four_stage import REAR_CALVES, crest_load, force_signals
from .four_stage_replay import impact_series
from .robot import HEIGHT, LEGS, THICKNESS

PERTURBATION = np.array([.002, .002, np.deg2rad(.5)])

DT = CONTROL_DT / SUBSTEPS
STEPS = ("front_shins", "hind_crouch", "rear_shins", "four_foot_landing")


def add_sensors(spec):
    for leg in LEGS:
        spec.add_sensor(
            name=f"knee_{leg}", type=mj.mjtSensor.mjSENS_FRAMEPOS,
            objtype=mj.mjtObj.mjOBJ_XBODY, objname=f"{leg}_calf",
        )
    for leg in LEGS:
        for axis in ("hip", "thigh", "calf"):
            joint = f"{leg}_{axis}"
            spec.add_sensor(name=f"effort_{joint}",type=mj.mjtSensor.mjSENS_ACTUATORFRC,
                            objtype=mj.mjtObj.mjOBJ_ACTUATOR,objname=joint)
    spec.add_sensor(
        name="robot_com", type=mj.mjtSensor.mjSENS_SUBTREECOM,
        objtype=mj.mjtObj.mjOBJ_BODY, objname="base_link",
    )


def first(mask, steps=1):
    if mask.shape[1] < steps:
        return np.full(mask.shape[0], -1, dtype=int)
    hit = np.lib.stride_tricks.sliding_window_view(mask, steps, axis=1).all(-1)
    return np.where(hit.any(1), hit.argmax(1) + steps, -1)


def measure(model, raw, sensors):
    sample = slice(SUBSTEPS - 1, None, SUBSTEPS)
    state = raw[:, sample, 1:]
    feet = np.stack([signal(model, sensors[:, sample], f"position_{g}") for g in FEET], axis=-2)
    knees = np.stack([signal(model, sensors[:, sample], f"knee_{leg}") for leg in LEGS], axis=-2)
    com = signal(model, sensors[:, sample], "robot_com")
    shin = force_signals(model, sensors, "wall", REAR_CALVES + FRONT_CALVES)
    toe = force_signals(model, sensors, "wall", FEET)
    floor = force_signals(model, sensors, "floor", FEET)
    floor_all = abs(floor[..., 1] * floor[..., 9])
    shin_all = crest_load(shin)
    toe_all = crest_load(toe)
    bad_ground = abs(force_signals(model, sensors, "floor", COLLIDERS)[..., 1]).max(-1)
    bodies = tuple(g for g in COLLIDERS if g not in REAR_CALVES + FRONT_CALVES)
    bad_wall = abs(force_signals(model, sensors, "wall", bodies)[..., 1]).max(-1)
    shin_side = np.where((shin[...,0] > .5) & (shin_all == 0), abs(shin[...,1]), 0.).max(-1)
    toe_side = np.where((toe[...,0] > .5) & (toe_all == 0), abs(toe[...,1]), 0.).max(-1)
    bad = np.maximum.reduce((bad_ground, bad_wall, shin_side, toe_side))
    bad_control = bad.reshape(len(state), len(state[0]), SUBSTEPS).max(-1)
    normal = shin_all[:, sample]
    floor = floor_all[:, sample]
    front, rear = normal[..., 2:], normal[..., :2]
    rear_contact = shin[:, sample, :2, 4:7]
    rear_axis = feet[..., :2, :] - knees[..., :2, :]
    delta = rear_contact - knees[..., :2, :]
    fraction = (delta * rear_axis).sum(-1) / np.maximum((rear_axis**2).sum(-1), 1.e-6)
    w, x, y, z = np.moveaxis(state[..., 3:7], -1, 0)
    # Positive pitch lowers the forward end of the robot.
    pitch = np.arcsin(np.clip(2*(w*y-z*x), -1., 1.))
    up = 1.-2.*(x*x+y*y)
    speed = np.linalg.norm(state[..., 19:22], axis=-1)
    spin = np.linalg.norm(state[..., 22:25], axis=-1)
    air = ((floor.max(-1) < 5.) & (normal.max(-1) < 5.) &
           (toe_all[:, sample].max(-1) < 5.) & (bad_control < 5.) & (feet[...,2].min(-1) > .029))
    bend = -state[..., [9, 12, 15, 18]]
    return dict(
        state=state, feet=feet, knees=knees, com=com, front=front, rear=rear, floor=floor,
        floor_all=floor_all, shin_all=shin_all, toe=toe_all[:,sample],
        toe_peak=toe_all.reshape(len(state),len(state[0]),SUBSTEPS,4).max(2),
        fraction=fraction, pitch=pitch, up=up, speed=speed, spin=spin, air=air, bend=bend,
        bad=bad, bad_control=bad_control,
    )


def first_balanced(mask, front_load, rear_load, frames):
    window = np.lib.stride_tricks.sliding_window_view
    support = window(mask,frames,axis=1).all(-1)
    front = window(front_load.sum(-1),frames,axis=1).sum(-1)
    rear = window(rear_load.sum(-1),frames,axis=1).sum(-1)
    stable = support & (front >= .52*(front+rear))
    return np.where(stable.any(1),stable.argmax(1)+frames,-1)


def bump(value, target, width):
    return np.exp(-((value-target)/width)**2)


def strength(load, scale):
    return np.clip(load.min(-1)/scale, 0., 1.)


def first_push(m):
    """Measure vertical impulse through the first confirmed flight, at 200 Hz."""
    confirmed = first(m["air"],2)
    takeoff = np.where(confirmed>0, confirmed-1,0)
    ticks = np.arange(m["floor_all"].shape[1])[None,:]
    cutoff = takeoff[:,None]*SUBSTEPS
    launch = (ticks<cutoff) & (ticks>=np.maximum(0,cutoff-round(.10/DT)))
    impulse = np.einsum("bt,btl->bl",launch,m["floor_all"])*DT
    full_impulse = np.einsum("bt,btl->bl",ticks<cutoff,m["floor_all"])*DT
    share = full_impulse[:,2:].sum(-1)/np.maximum(full_impulse.sum(-1),1.e-6)
    in_phase = np.arange(m["bend"].shape[1])[None,:]<confirmed[:,None]
    rear_bend = np.where(in_phase[...,None],m["bend"][...,:2],np.inf).min(1)
    upward = np.where(in_phase,m["state"][...,21],0.).max(-1)
    return confirmed, full_impulse, share, rear_bend, upward, impulse


def touchdown_rebound(m, start, landing):
    """Measure rebound after both rear feet first load the far ground."""
    rear_loaded = (m["floor"][:,start:,:2].min(-1)>5.) & (m["feet"][:,start:,:2,0].min(-1)>THICKNESS/2+.0238)
    contact = first(rear_loaded)
    frames = np.arange(start,len(m["air"][0]))[None,:]+1
    after_touch = (contact[:,None]>0) & (frames>=start+contact[:,None])
    during_settle = after_touch & ((landing[:,None]<0) | (frames<=landing[:,None]))
    upward = np.where(during_settle,m["state"][:,start:,21],0.).max(-1)
    unsupported = (m["floor"][:,start:].sum(-1) < 8.) & during_settle
    return contact, upward, unsupported.sum(-1)


def score(m, stage, start=0, *, max_hops=1, front_led=False, absorb_landing=False):
    length = m["air"].shape[1]
    index = np.arange(length)[None,:]
    future = index >= start
    front, rear, floor = m["front"], m["rear"], m["floor"]
    feet, knees, bend, state = m["feet"], m["knees"], m["bend"], m["state"]
    angle, speed, spin = m["pitch"], m["speed"], m["spin"]
    front_offset = feet[...,2:,0].mean(-1)-m["com"][...,0]
    clean = np.maximum.accumulate(m["bad_control"],axis=1) < 5.
    soft_clean = np.exp(-np.maximum.accumulate(m["bad_control"],axis=1)/95.)
    low_motion = np.exp(-2*speed**2-.1*spin**2)
    front_strength = strength(front,20.)
    hind_strength = strength(floor[...,:2],18.)
    first_air = first(m["air"],2)
    crossed_air = (first_air[:,None] > 0) & (index+1 >= first_air[:,None])
    first_touch = first(m["toe_peak"][...,2:].max(-1)>7.)
    shin_landing = (crossed_air & clean & (front.min(-1)>7.) & (floor[...,:2].min(-1)>7.) &
                    (speed<.43) & (spin<2.5) & (m["up"]>.63))
    catch_front = first(shin_landing & future,5)
    direct = (catch_front>0) & ((first_touch<0) | (first_touch>=catch_front-4))
    hind_x = feet[...,:2,0].min(-1)
    prepared = (shin_landing & (bend[...,:2].min(-1)>1.7) &
                (hind_x>-.40) & (feet[...,:2,0].max(-1)<-.066) & (speed<.30))
    front_floor = (floor[...,2:].min(-1)>5.) & (feet[...,2:,0].min(-1)>.08)
    second_air = first(m["air"] & future,2)
    after_jump = (second_air[:,None] > 0) & (index+1>=second_air[:,None])
    proximal = ((m["fraction"]>=-.08)&(m["fraction"]<=.45)).all(-1)
    balance = (front_floor & (rear.min(-1)>7.) & proximal & (angle>.13) & (angle<.55) &
               (bend[...,2:].min(-1)>.65) & (bend[...,2:].max(-1)<1.55) &
               (front_offset<.25) & (front_offset>-.10) &
               (speed<.58) & (spin<2.5) & (m["up"]>.67))
    landed = (clean & (floor.min(-1)>5.) & (feet[...,0].min(-1)>THICKNESS/2+.0238) &
              (m["up"]>.82) & (speed<.35) & (spin<2.) & (state[...,2]>.22))
    if stage == 0:
        # Front shin centre and knee/foot heights guide the search until contact.
        centre = .5*(knees[...,2:,:]+feet[...,2:,:])
        guide = (bump(centre[...,0],-.012,.12)*bump(centre[...,2],HEIGHT+.04,.10)).min(-1)
        support = front_strength*hind_strength*low_motion
        frame = (.55*guide+.5*front_strength+.25*hind_strength*front_strength+1.45*support)
        frame *= crossed_air * soft_clean
        sustained = np.lib.stride_tricks.sliding_window_view(frame,5,axis=1).mean(-1).max(-1)
        ready = np.lib.stride_tricks.sliding_window_view(shin_landing,5,axis=1).mean(-1).max(-1)
        value = .5*frame.max(-1) + .5*frame.mean(-1) + 2.*sustained + 1.7*ready + 8.*direct + 1.5*(catch_front>0)
        value -= 1.1*(first_touch>0)*(~direct) + .15*np.log1p(m["bad"].max(-1)/12.)
        if front_led:
            air, _, share, bend_min, upward, _ = first_push(m)
            flexion = np.clip((bend_min.min(-1)-1.1)/.55,0,1)
            forward = np.clip(m["com"][...,0].max(-1)-m["com"][:,0,0],0,.28)/.28
            before_flight = (index<air[:,None]) & (air[:,None]>0)
            rear_bend = np.rad2deg(bend[...,:2])
            # Allow the rear knees to open under load, with a smooth preference
            # for moderate flexion instead of a rapid tuck during the push.
            tucked = np.maximum(rear_bend-128.,0.).mean(-1)
            excess = (tucked*before_flight).sum(-1)/np.maximum(air,1)
            shin_contact = first(front.min(-1)>7.)
            before_wall = (index < np.where(shin_contact>0,shin_contact,length)[:,None])
            rear_full = np.where(before_wall[...,None],rear_bend,np.inf).min(axis=(1,2))
            similar = np.abs(rear_bend[...,0]-rear_bend[...,1])
            asymmetry = (similar*before_flight).sum(-1)/np.maximum(air,1)
            tuck_max = np.where(before_flight[...,None],rear_bend,0.).max(axis=(1,2))
            launch_quality = (3.6*share + 1.6*flexion + .65*np.clip(upward/1.3,0,1)
                              + .5*forward - .12*excess - .055*asymmetry - .035*np.maximum(tuck_max-135.,0.)
                              + .025*np.clip(rear_full-70.,0,30))
            qualifies = ((air>0) & (share>=.65) & (bend_min.min(-1)>=np.deg2rad(88.))
                         & (rear_full>=83.) & (tuck_max<=138.))
            value = value-6.*direct + launch_quality*(air>0) + 12.*(direct&qualifies)
            direct &= qualifies
        return value, np.where(direct,catch_front,-1)
    if stage == 1:
        fold = bump(bend[...,:2],2.12,.46).min(-1)
        advance = bump(hind_x,-.32,.16)
        frame = front_strength*hind_strength*(.7+fold+.65*advance+.75*low_motion)*soft_clean
        if front_led:
            delta = feet[...,:2,0] - feet[:,start-1:start,:2,0]
            step_gain = np.clip(delta.min(-1)/.12,0,1)
            prepared &= step_gain >= .25
            # Give the search a gradient while one rear foot is off the
            # floor. Qualification still requires all four support points.
            near = bump(state[...,0],-.24,.09)*bump(state[...,2],.37,.11)
            settled = np.exp(-9.*speed**2-.12*spin**2)
            combined = front_strength*hind_strength*step_gain*settled
            frame += (1.3*step_gain*(.4+.6*front_strength) + .65*near
                      + .6*front_strength*low_motion + 2.4*combined) * soft_clean
        event=first(prepared & future,5)
        tail = frame[:,start:]
        rolling = np.lib.stride_tricks.sliding_window_view(tail,5,axis=1).mean(-1).max(-1)
        value = .5*tail.max(-1)+.4*tail.mean(-1)+2.*rolling+7.*(event>0)
        if front_led:
            ready = np.lib.stride_tricks.sliding_window_view(prepared & future,5,axis=1).mean(-1).max(-1)
            value += 2.8*ready
        return value, event
    if stage == 4:
        # Recover a position from which the hind legs can push forward after
        # the fore shins lift. Keep actual rear-floor support during release.
        yoke = front.max(-1)<3.
        hind_span = state[...,0,None]-feet[...,:2,0]
        geometry = (bump(state[...,0],-.18,.08)*bump(state[...,2],.40,.07)*
                    bump(angle,-.47,.30)*bump(hind_span,.21,.115).min(-1))
        front_clear = ((feet[...,2:,0]>.045)&(feet[...,2:,2]>HEIGHT-.008)).all(-1)
        steady = (floor[...,:2].min(-1)>7.) & (speed<.72) & (state[...,19] > -.08) & (spin<2.9)
        released = (yoke & clean & steady & front_clear & (state[...,0]>-.23) &
                    (state[...,2]>.37) & (hind_span.min(-1)>.14) &
                    (bend[...,:2].min(-1)>1.5))
        event = first(released & future,3)
        frame = (geometry*(.4+.6*hind_strength)+.50*hind_strength*low_motion+
                 .50*yoke*geometry+.3*front_clear)*soft_clean
        tail = frame[:,start:]
        rolling = np.lib.stride_tricks.sliding_window_view(tail,4,axis=1).mean(-1).max(-1)
        return .5*tail.max(-1)+2.*rolling+9.*(event>0), event
    if stage == 2:
        point = .75*knees[...,:2,:]+.25*feet[...,:2,:]
        shin_guide = (bump(point[...,0],.02,.12)*bump(point[...,2],HEIGHT+.035,.12)).min(-1)
        front_guide = (bump(feet[...,2:,0],.46,.27)*bump(feet[...,2:,2],.025,.13)).min(-1)
        close = bump(front_offset,.17,.17)
        posture = bump(angle,.27,.24)*bump(bend[...,2:],1.0,.46).min(-1)*close
        actual = strength(rear,23.)*strength(floor[...,2:],17.)
        near_knee = bump(m["fraction"],.24,.25).min(-1)
        frame = (.70*shin_guide+.45*front_guide+.55*posture+actual*(1.+1.0*near_knee+.8*low_motion+.85*posture))
        frame *= after_jump*soft_clean
        event=first_balanced(balance & clean & after_jump,floor[...,2:],rear,6)
        tail = frame[:,start:]
        rolling = np.lib.stride_tricks.sliding_window_view(tail,6,axis=1).mean(-1).max(-1)
        readiness = np.lib.stride_tricks.sliding_window_view(
            balance & clean & after_jump,6,axis=1).mean(-1).max(-1)
        value=.5*tail.max(-1)+.3*tail.mean(-1)+2.*rolling+2.*readiness+9.*(event>0)
        return value, event
    after = slice(start,None)
    landing = first(landed & future,10)
    airborne = m["air"][:,after]
    # Stepping and a short forward hop are both admissible. A long second flight
    # or a high-speed touchdown loses score.
    before_finish = (landing[:,None]<0) | (index[:,after] < landing[:,None])
    air = airborne & before_finish
    sustained = air[:,1:] & air[:,:-1]
    starts = np.diff(np.pad(sustained.astype(np.int8),((0,0),(1,0))),axis=1) > 0
    hops = starts.sum(-1)
    too_long = first(air,10)>0
    hop_ok = ~too_long & (hops<=max_hops)
    tail = (strength(floor,14.)*low_motion*np.clip((m["up"]-.65)/.30,0,1)*soft_clean)[:,after]
    advance = np.clip((feet[...,0].min(-1) - .025)/.40,0,1)[:,after]
    shift = np.clip((m["com"][...,0]-m["com"][:,start-1,0,None])/.31,0,1)[:,after]
    stance = np.lib.stride_tricks.sliding_window_view(tail,10,axis=1).mean(-1).max(-1)
    contact_peak = m["floor_all"][:,start*SUBSTEPS:,:2].sum(-1).max(-1)
    clearance = np.clip((feet[..., :2,0].min(-1)-.09)/.26,0,1)[:,after]
    front_support = strength(floor[...,2:],17.)[:,after]
    rear_support = strength(floor[...,:2],17.)[:,after]
    guides = (.55*front_support+.85*clearance+.70*rear_support)*soft_clean[:,after]
    guidance = np.lib.stride_tricks.sliding_window_view(guides,8,axis=1).mean(-1).max(-1)
    readiness = np.lib.stride_tricks.sliding_window_view(
        landed[:,after],10,axis=1).mean(-1).max(-1)
    height = bump(state[:,after,2],.32,.13)
    stable = (tail*height).max(-1)
    value = (2.5*stance + 1.6*guidance + .85*advance.mean(-1) + .80*shift.mean(-1) +
             2.8*readiness + .7*stable + 9.*((landing>0)&hop_ok))
    value -= .24*np.log1p(contact_peak/100.) + .8*too_long + .65*np.maximum(0,hops-max_hops)
    if absorb_landing:
        contact, upward, unsupported = touchdown_rebound(m,start,landing)
        no_bounce = (contact>0) & (upward<=.40) & (unsupported==0)
        value += 1.5*(contact>0) - 5.4*upward - .38*unsupported
        value -= 9.*((landing>0)&hop_ok&~no_bounce)
        value -= .7*np.log1p(contact_peak/180.)
        hop_ok &= no_bounce
    return value, np.where(hop_ok,landing,-1)


def descent_template(model, initial, previous):
    data = mj.MjData(model)
    with Rollout(nthread=1) as runner:
        raw, _ = simulate(runner,model,[data],initial,previous[None])
    data.qpos[:] = raw[0,-1,1:20]
    mj.mj_forward(model,data)
    start = data.qpos.copy()
    feet = np.array([data.geom(g).xpos for g in FEET])
    targets = feet.copy()
    targets[:2,0] = np.maximum(feet[:2,0]+.23,.37)
    targets[:2,2] = .016
    targets[:2,1] = start[1] + [-.18,.18]
    targets[2:,1] = start[1] + [-.22,.22]
    root = start[:3].copy()
    root[0] += .24
    root[2] = .32
    q = start[7:].copy()
    geom_ids = [model.geom(g).id for g in FEET]
    lower, upper = model.jnt_range[1:].T
    jac = np.empty((3,model.nv))
    seed = []
    for tick in range(72):
        move = np.clip(tick/30.,0,1)
        move = move*move*(3-2*move)
        down = np.clip((tick-8)/24.,0,1)
        down = down*down*(3-2*down)
        goal = feet+(targets-feet)*move
        goal[:2,2] = feet[:2,2]+(targets[:2,2]-feet[:2,2])*down+.043*np.sin(np.pi*down)
        data.qpos[:3] = (1-move)*start[:3]+move*root
        quat = (1-move)*start[3:7]+move*np.array([1.,0.,0.,0.])
        data.qpos[3:7] = quat/np.linalg.norm(quat)
        data.qpos[7:] = q
        for _ in range(12):
            mj.mj_kinematics(model,data)
            mj.mj_comPos(model,data)
            for leg, geom in enumerate(geom_ids):
                k = slice(3*leg,3*leg+3)
                pos = data.geom_xpos[geom]
                mj.mj_jacGeom(model,data,jac,None,geom)
                j = jac[:,6+3*leg:9+3*leg]
                delta = j.T @ np.linalg.solve(j@j.T+.003*np.eye(3),goal[leg]-pos)
                q[k] = np.clip(q[k]+np.clip(delta,-.18,.18),lower[k]+.02,upper[k]-.02)
            data.qpos[7:] = q
        seed.append(q.copy())
    result = np.asarray(seed)
    result[:5] += np.linspace(.65,0.,5)[:,None]*(previous[-1]-result[0])
    return result


def optimize(model, initial, previous, template, stage, iterations, batch, threads, seed, dest, *,
             refine=False, max_hops=1, front_led=False, absorb_landing=False):
    start = len(previous)
    length = start+len(template)
    frozen = min(22 if absorb_landing else 18,len(template)-13) if stage==3 and refine else 0
    grid = np.unique(np.r_[frozen,frozen+1,np.arange(frozen+3,len(template),3),len(template)-1])
    blend = np.array([np.interp(np.arange(len(template)),grid,row) for row in np.eye(len(grid))]).T
    mean = template[grid].copy()
    # Insert the incumbent exactly, alongside interpolated perturbations.
    incumbent = template
    rng = np.random.default_rng(seed)
    best = -np.inf
    ranges = model.jnt_range[1:]
    base_spread = np.tile([.10,.30,.40], (len(grid),4))
    if stage == 3:
        base_spread *= .62
    if stage == 0 and front_led:
        base_spread *= .62
    if refine:
        base_spread *= .55
    datas = [mj.MjData(model) for _ in range(threads)]
    stage_name = (*STEPS, "takeoff_setup")[stage]
    result = dest / f"{stage_name}.npz"
    with Rollout(nthread=threads) as runner:
        for it in range(iterations):
            noise = rng.normal(size=(batch,*mean.shape))*base_spread
            count = batch//3
            noise[:count,:,[3,4,5,9,10,11]] = noise[:count,:,[0,1,2,6,7,8]]*[-1,1,1,-1,1,1]
            knots = mean+noise*max(.19,1.-it/(1.16*iterations))
            knots[0] = mean
            candidates = np.empty((batch,length,12))
            candidates[:,:start] = previous
            candidates[:,start:] = np.clip(blend@knots,ranges[:,0]-.18,ranges[:,1]+.18)
            candidates[0,start:] = incumbent
            if frozen:
                candidates[:,start:start+frozen] = template[:frozen]
            raw, sensors = simulate(runner,model,datas,initial,candidates)
            m = measure(model,raw,sensors)
            value, event = score(m,stage,start,max_hops=max_hops,front_led=front_led,absorb_landing=absorb_landing)
            value -= .024*np.mean(np.diff(knots,n=2,axis=1)**2/base_spread[1:-1]**2,axis=(1,2))
            elite = np.argsort(value)[-max(6,batch//14):]
            mean = .22*mean + .78*knots[elite].mean(0)
            k = elite[-1]
            if value[k] > best:
                best = value[k]
                np.savez_compressed(result,initial=initial,controls=candidates[k],event=event[k],start=start)
            if it%5 == 0 or event[k]>0:
                diagnostics = ()
                if stage==0 and front_led:
                    _, _, share, bend_min, _, _ = first_push(m)
                    diagnostics = ("front %",round(float(100*share[k]),1),
                                   "min rear bend",round(float(np.rad2deg(bend_min[k].min())),1))
                print(stage_name,it,"score",round(float(best),3),"event",event[k],
                      "max both shin loads",np.round([m["front"][k].min(-1).max(),m["rear"][k].min(-1).max()],1),
                      "bad",round(float(m["bad"][k].max()),1),*diagnostics,flush=True)
            if event[k]>0:
                break
            del raw, sensors, m
    with np.load(result, allow_pickle=False) as f:
        event = int(f["event"])
        return f["controls"][:event] if event>0 else None


def through_frame(measured, frame):
    physics = {"floor_all", "shin_all", "bad"}
    return {key:value[:,:frame * (SUBSTEPS if key in physics else 1)]
            for key,value in measured.items()}


def run(args):
    model = make_model(args.asset, configure=add_sensors)
    args.output.mkdir(parents=True,exist_ok=True)
    with np.load(args.baseline,allow_pickle=False) as f:
        initial = f["initial"].copy()
        old = f["controls"].copy()
    initial[1] += args.closer
    t = np.arange(43)
    warm_front = old[:43].copy()
    fold = np.clip((t-7)/9,0.,1.)*np.clip((34-t)/13,0.,1.)
    warm_front[:,[8,11]] -= .62*fold[:,None]
    if args.first_warm:
        with np.load(args.first_warm,allow_pickle=False) as f:
            warm_front = f["controls"][:43].copy()
        if len(warm_front)<20:
            raise ValueError("The warm start needs at least 20 controls")
        if len(warm_front)<43:
            warm_front = np.pad(warm_front,((0,43-len(warm_front)),(0,0)),mode="edge")
    previous = np.empty((0,12))
    events = []
    release_frame = None
    stages = 1 if args.first_only else len(STEPS)
    for phase in range(stages):
        if phase == 0:
            seed = warm_front
        elif phase == 1:
            seed = np.broadcast_to(previous[-1],(34,12)).copy()
            alpha = np.clip(np.arange(34)/17.,0.,1.)
            seed[:,[2,5]] -= .42*alpha[:,None]
            seed[:,[1,4]] += .16*alpha[:,None]
            if args.front_led:
                with Rollout(nthread=1) as runner:
                    raw, _ = simulate(runner,model,[mj.MjData(model)],initial,previous[None])
                actual = raw[0,-1,8:20]
                target = np.array([-.08,1.52,-2.14,.08,1.52,-2.14,
                                   .15,.98,-2.48,-.15,.98,-2.48])
                t = np.arange(46)
                alpha = np.clip((t-2)/15.,0.,1.)
                seed = (1.-alpha[:,None])*actual + alpha[:,None]*target
        elif phase == 2:
            release_seed = old[42:72].copy()
            release_seed[:10] += np.linspace(1.,0.,10)[:,None]*(previous[-1]-release_seed[0])
            setup_path = args.output/"takeoff_setup.npz"
            setup = None
            if args.resume and setup_path.exists():
                with np.load(setup_path,allow_pickle=False) as f:
                    stop = int(f["event"])
                    if stop>0 and np.array_equal(f["initial"],initial) and np.array_equal(f["controls"][:len(previous)],previous):
                        setup = f["controls"][:stop]
            if setup is None:
                setup = optimize(model,initial,previous,release_seed,4,args.iters[2],args.batch,args.threads,args.seed+4,args.output)
            if setup is None:
                with np.load(setup_path,allow_pickle=False) as f:
                    previous = f["controls"]
                break
            previous = setup
            release_frame = len(previous)
            seed = old[72:115].copy()
            seed[:5] += np.linspace(.7,0.,5)[:,None]*(previous[-1]-seed[0])
            # Let the fore knees flex on landing and tuck the hind shins up early.
            f = np.clip((np.arange(len(seed))-16)/9.,0.,1.)
            seed[:,[8,11]] -= .25*f[:,None]
        else:
            seed = descent_template(model,initial,previous)
        output = args.output / f"{STEPS[phase]}.npz"
        candidate = None
        refine = False
        if args.resume and output.exists():
            with np.load(output,allow_pickle=False) as f:
                event = int(f["event"])
                compatible = (np.array_equal(f["initial"],initial) and
                              np.array_equal(f["controls"][:len(previous)],previous))
                if compatible and event>0:
                    candidate = f["controls"][:event]
                elif compatible:
                    seed = f["controls"][len(previous):]
                    refine = True
        if candidate is None:
            candidate = optimize(model,initial,previous,seed,phase,args.iters[phase],
                                 args.batch,args.threads,args.seed+phase,args.output,
                                 refine=refine,max_hops=args.max_hops,front_led=args.front_led,absorb_landing=args.absorb_landing)
        if candidate is None:
            break
        previous = candidate
        events.append(len(previous))
    # Always keep a replay, including the best failed phase for diagnosis.
    if len(events)<stages and not (len(events)==2 and release_frame is None):
        with np.load(args.output/f"{STEPS[len(events)]}.npz",allow_pickle=False) as f:
            previous = f["controls"]
    with Rollout(nthread=1) as runner:
        raw,sensors = simulate(runner,model,[mj.MjData(model)],initial,previous[None])
    m = measure(model,raw,sensors)
    starts = [0, *events]
    if release_frame is not None:
        starts[2] = release_frame
    for phase, end in enumerate(events):
        _, event = score(m,phase,starts[phase],max_hops=args.max_hops,front_led=args.front_led,absorb_landing=args.absorb_landing)
        if event[0]!=end:
            raise RuntimeError(f"Independent replay changed stage {phase+1}: {event[0]} vs {end}")
    if args.front_led and release_frame is not None:
        _, release = score(through_frame(m,release_frame),4,events[1])
        if release[0]!=release_frame:
            raise RuntimeError("Independent replay changed the second takeoff setup")
    frames = m["state"][0]
    qpos = np.vstack((initial[1:20],frames[:,:19]))
    cut = events[2]-1 if len(events)>2 else None
    shots = {}
    if cut is not None:
        shots = dict(rear_shin_knee_fraction=m["fraction"][0,cut].tolist(),
                     rear_shin_distance_from_knee_m=np.linalg.norm(
                         force_signals(model,sensors,"wall",REAR_CALVES)[0,cut*SUBSTEPS+SUBSTEPS-1,:,4:7]
                         -m["knees"][0,cut,:2],axis=-1).tolist(),
                     nose_down_deg=float(np.rad2deg(m["pitch"][0,cut])),
                     front_knee_bend_deg=np.rad2deg(m["bend"][0,cut,2:]).tolist(),
                     front_share_during_hold=float(m["floor"][0,cut-5:cut+1,2:].sum() /
                                       (m["floor"][0,cut-5:cut+1,2:].sum()+m["rear"][0,cut-5:cut+1].sum())),
                     com_behind_front_feet_m=float(m["feet"][0,cut,2:,0].mean()-m["com"][0,cut,0]))
    report = {
        "passed":len(events)==4,"stage_frames":dict(zip(STEPS,events)),
        "max_post_catch_hops":args.max_hops,
        "first_jump_passed":len(events)>0,
        "first_jump_only":args.first_only,
        "first_jump_front_led_required":args.front_led,
        "initial_base_x_m":float(initial[1]),"initial_wall_offset_change_m":args.closer,
        "first_takeoff_frame":int(first(m["air"],2)[0]),
        "second_takeoff_setup_frame":release_frame,
        "peak_unplanned_contact_n":float(m["bad"].max()),
        "first_front_toe_crest_contact_frame":(
            int(first(m["toe_peak"][...,2:].max(-1)>7.)[0])),
        "rear_catch": shots,
    }
    air, push, front_share, knee_min, upward, last_100ms = first_push(m)
    if air[0]>0:
        report["first_jump_push"] = {
            "front_vertical_impulse_ns":float(push[0,2:].sum()),
            "rear_vertical_impulse_ns":float(push[0,:2].sum()),
            "per_foot_vertical_impulse_ns":push[0].tolist(),
            "front_share":float(front_share[0]),
            "front_share_last_100ms":float(last_100ms[0,2:].sum()/max(last_100ms[0].sum(),1.e-6)),
            "rear_maximum_knee_bend_deg_until_first_flight":np.rad2deg(m["bend"][0,:air[0],:2].max(0)).tolist(),
            "rear_minimum_knee_bend_deg_first_stage":np.rad2deg(m["bend"][0,:events[0] if events else None,:2].min(0)).tolist(),
            "rear_minimum_knee_bend_deg":np.rad2deg(knee_min[0]).tolist(),
            "peak_base_upward_speed_m_s":float(upward[0]),
        }
    if len(events)>1:
        t = events[1]-1
        step_start = events[0]-1
        report["hind_crouch"] = {
            "rear_foot_advance_m": (m["feet"][0,t,:2,0]-m["feet"][0,step_start,:2,0]).tolist(),
            "rear_feet_x_m":m["feet"][0,t,:2,0].tolist(),
            "front_shin_load_n":m["front"][0,t].tolist(),
            "rear_knee_bend_deg": np.rad2deg(m["bend"][0,t,:2]).tolist(),
            "rear_foot_load_n": m["floor"][0,t,:2].tolist(),
        }
    if len(events)>1 and release_frame is not None:
        begin=release_frame*SUBSTEPS
        report["front_floor_impact"] = impact_series(raw[0,begin:],m["floor_all"][0,begin:,2:])
        report["rear_wall_impact"] = impact_series(raw[0,begin:],m["shin_all"][0,begin:,:2])
    if len(events)>2:
        touch, rebound, gap = touchdown_rebound(m,events[2],np.array([events[3] if len(events)>3 else -1]))
        report["touchdown_rebound"] = {
            "first_dual_rear_touchdown_frame":int(events[2]+touch[0]) if touch[0]>0 else None,
            "peak_upward_speed_m_s":float(rebound[0]),
            "unsupported_steps_after_touchdown":int(gap[0]),
        }
        report["rear_floor_impact"] = impact_series(raw[0,events[2]*SUBSTEPS:],m["floor_all"][0,events[2]*SUBSTEPS:,:2])
        airborne = m["air"][0,events[2]:]
        edges = np.flatnonzero(np.diff(np.r_[False,airborne,False]))
        durations = (edges[1::2]-edges[::2])*CONTROL_DT
        report["post_catch_airborne_durations_s"] = durations.tolist()
        report["single_hop_gate_passed"] = bool(len(events)==4 and np.count_nonzero(durations>=2*CONTROL_DT)<=1)
    effort_names = [f"{leg}_{axis}" for leg in LEGS for axis in ("hip","thigh","calf")]
    effort = np.concatenate([signal(model,sensors[0],f"effort_{name}") for name in effort_names],axis=-1)
    report["torque"] = {
        "peak_nm_by_actuator": dict(zip(effort_names,abs(effort).max(0).tolist())),
        "time_any_near_limit_s": float(np.count_nonzero((abs(effort)>=.99*23.7).any(-1))*DT),
    }
    report["perturbations"] = {}
    report["perturbation_trials"] = {}
    if args.front_led and len(events)>2:
        report["second_jump_diagnostics"] = {}
    for seed in (301, 1621):
        offsets = np.random.default_rng(seed).uniform(-PERTURBATION, PERTURBATION, (16,3))
        trials = np.broadcast_to(initial,(16,len(initial))).copy()
        trials[:,1:3] += offsets[:,:2]
        # The reference starts with a level body aligned with the wall.
        trials[:,4] = np.cos(.5*offsets[:,2])
        trials[:,7] = np.sin(.5*offsets[:,2])
        trial_targets = np.broadcast_to(previous,(16,*previous.shape))
        workers = min(args.threads,16)
        with Rollout(nthread=workers) as runner:
            trial_raw, trial_sensor = runner.rollout(
                model, [mj.MjData(model) for _ in range(workers)], trials[:,:38],
                np.repeat(trial_targets,SUBSTEPS,axis=1),
            )
        trial = measure(model,trial_raw,trial_sensor)
        count = {}
        per_trial = {}
        valid = np.ones(16,dtype=bool)
        for phase, name in enumerate(STEPS):
            if phase >= len(events):
                count[name] = None
                continue
            if args.front_led and phase==2:
                _, release = score(through_frame(trial,release_frame),4,events[1])
                valid &= release>0
                count["takeoff_setup"] = int(valid.sum())
                per_trial["takeoff_setup"] = np.where(valid,release,-1).tolist()
            # Use the actual contact before the next segment's scheduled switch.
            phase_trial = through_frame(trial,events[phase]) if args.front_led else trial
            _, transition = score(phase_trial,phase,starts[phase],max_hops=args.max_hops,front_led=args.front_led,absorb_landing=args.absorb_landing)
            if args.front_led and phase==2:
                window = slice(release_frame, events[2])
                flight = first(trial["air"][:,window],2)
                ticks = np.arange(events[2]-release_frame)[None,:]+1
                after_flight = (flight[:,None]>0) & (ticks>=flight[:,None])
                front_ground = ((trial["floor"][:,window,2:].min(-1)>5.)
                                & (trial["feet"][:,window,2:,0].min(-1)>.08))
                both_rear_shins = trial["rear"][:,window].min(-1)>7.
                both_shins_and_front_feet = after_flight & front_ground & both_rear_shins
                report["second_jump_diagnostics"][str(seed)] = {
                    "passed_takeoff_setup":int(valid.sum()),
                    "had_flight":int((valid & (flight>0)).sum()),
                    "had_front_feet_and_rear_shins_support":int((valid & both_shins_and_front_feet.any(-1)).sum()),
                    "passed_catch_hold":int((valid & (transition>0)).sum()),
                }
            valid &= transition>0
            count[name] = int(valid.sum())
            per_trial[name] = np.where(valid,transition,-1).tolist()
        report["perturbations"][str(seed)] = count
        report["perturbation_trials"][str(seed)] = per_trial
    np.savez_compressed(args.output/"replay.npz",initial=initial,controls=previous,qpos=qpos,
                        torque_200hz_nm=effort,
                        **{k:m[k][0] for k in ("floor","front","rear","pitch","state")})
    (args.output/"evaluation.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset",type=Path,required=True)
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--closer",type=float,default=.055)
    parser.add_argument("--iters",type=int,nargs=4,default=[120,70,180,160])
    parser.add_argument("--batch",type=int,default=96)
    parser.add_argument("--threads",type=int,default=8)
    parser.add_argument("--seed",type=int,default=610)
    parser.add_argument("--max-hops",type=int,default=1,
                        help="Allow this many post-catch flights of at least two control steps")
    parser.add_argument("--absorb-landing",action="store_true",
                        help="Require sustained support and limit upward rebound after touchdown")
    parser.add_argument("--front-led",action="store_true", help="Require front-led first takeoff")
    parser.add_argument("--first-only",action="store_true", help="Search and evaluate only the first jump")
    parser.add_argument("--first-warm",type=Path, help="Seed the first jump from a saved control sequence")
    parser.add_argument("--resume",action="store_true")
    options = parser.parse_args()
    if min(options.iters) < 1 or options.batch < 16 or options.threads < 1 or options.closer <= 0.:
        parser.error("Use positive iterations, threads and closer distance, and a batch of at least 16")
    if options.max_hops < 0:
        parser.error("--max-hops must be nonnegative")
    run(options)
