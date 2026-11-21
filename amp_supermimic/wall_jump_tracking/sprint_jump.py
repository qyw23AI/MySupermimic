"""Search procedural compression, extension and landing after a WBL sprint."""
import argparse
import json
from pathlib import Path

import mujoco as mj
from mujoco.rollout import Rollout
import numpy as np

from .brace_experiment import COLLIDERS, FEET, HOME, SUBSTEPS, make_model, signal, simulate
from .contact_route import add_sensors
from .four_stage import force_signals
from .robot import HEIGHT, LEGS, ROOT, THICKNESS
from .single_hop import batch_score, intervals, perturb, measure_hop, policy_runup


def candidates(runup, count, rng, center=None):
    # Independent front and rear hip sweep and knee extension. All targets
    # remain subject to the model's motor force limits.
    low = np.array([3, 3, 7, .1, .3, -.6, -.4, .35, .35])
    high = np.array([12, 11, 24, .65, 1., .8, 1., 1.6, 1.6])
    params = rng.uniform(low, high, (count, len(low))) if center is None else np.clip(
        rng.normal(center, .22 * (high - low), (count, len(low))), low, high)
    controls = np.broadcast_to(HOME, (count, len(runup) + 85, 12)).copy()
    controls[:, :len(runup)] = runup
    for row, p in zip(controls, params):
        compress, push, flight = np.round(p[:3]).astype(int)
        initial = runup[-1]
        crouch = initial + np.tile([0., p[3], -p[4]], 4)
        extend = crouch + np.repeat([[0., p[5], p[7]], [0., p[6], p[8]]], 2, axis=0).ravel()
        tuck = HOME + np.tile([0., .48, -.75], 4)
        t = len(runup)
        row[t:t+compress] = np.linspace(initial, crouch, compress + 1)[1:]
        t += compress
        row[t:t+push] = extend
        t += push
        row[t:t+flight] = tuck
        t += flight
        row[t:t+8] = np.linspace(tuck, HOME, 9)[1:]
    return controls, params


def wall_assist(model, raw, sensors, initial, runup_frames, *, launch_window=24):
    m = measure_hop(model, raw, sensors)
    rear = force_signals(model, sensors, "wall", FEET[:2])
    top = (rear[..., 6] >= HEIGHT - .012) & (abs(rear[..., 9]) > .65)
    load = np.where(top, abs(rear[..., 1] * rear[..., 9]), 0.)
    calves = tuple(f"{leg}_calf_collision" for leg in LEGS)
    non_calf_links = tuple(link for link in COLLIDERS if link not in calves)
    calf_force = abs(force_signals(model, sensors, "wall", calves)[..., 1])
    knee_x = np.stack([signal(model, sensors, f"knee_{leg}")[...,0] for leg in LEGS], axis=-1)
    knee_qpos = [model.joint(f"{leg}_calf_joint").qposadr[0] for leg in LEGS]
    # The knee origin marks the leading end of the shin before it clears the wall.
    extended_before_clearance = (knee_x <= THICKNESS / 2) & (abs(raw[..., 1 + np.array(knee_qpos)]) < .60)
    forbidden_calf = np.where(extended_before_clearance, calf_force, 0.)
    floor_links = force_signals(model, sensors, "floor", COLLIDERS)
    floor_force = abs(floor_links[..., 1])
    knee_near = {}
    for leg in LEGS[2:]:
        knee = signal(model, sensors, f"knee_{leg}")
        for part in ("thigh", "calf"):
            index = COLLIDERS.index(f"{leg}_{part}_collision")
            knee_near[index] = np.linalg.norm(floor_links[..., index, 4:7] - knee, axis=-1) <= .085
    forbidden_wall = np.maximum.reduce([
        abs(force_signals(model, sensors, "wall", FEET[2:] + non_calf_links)[..., 1]).max(-1),
        forbidden_calf.max(-1),
        np.where(top, 0., abs(rear[..., 1])).max(-1),
    ])
    score, _, details = batch_score(m, initial[1], runup_frames)
    passed = np.zeros(len(raw), dtype=bool)
    for i, (state, feet) in enumerate(zip(m["state"], m["feet"])):
        # A jump starts from floor support and rises before any wall assistance.
        launch = next(((a, b) for a, b in intervals(m["air"][i])
                       if runup_frames <= a <= runup_frames + launch_window and b-a >= 3
                       and state[a, 21] > .18 and state[a:b, 2].max() - state[a-1, 2] >= .025
                       and m["foot_floor"][i, a-1].max() > 5.), None)
        touch = np.flatnonzero(load[i].max(-1) >= 5.)
        first_touch = int(touch[0] // SUBSTEPS) if touch.size else None
        ground = floor_force[i].copy()
        allowed_knee = 0.
        if launch is not None:
            end = launch[0] * SUBSTEPS
            prep = slice(max(0, end - 90), end)
            for index, near in knee_near.items():
                selected = near[i, prep]
                allowed_knee = max(allowed_knee, float(ground[prep, index][selected].max(initial=0.)))
                ground[prep, index][selected] = 0.
        forbidden = np.maximum(forbidden_wall[i], ground.max(-1))
        peak_forbidden = float(forbidden.max())
        ordered = launch is not None and (first_touch is None or first_touch >= launch[1])
        if touch.size and not ordered:
            peak_forbidden = max(peak_forbidden, float(load[i].max()))
        far = feet[..., 0].min(-1) > THICKNESS / 2 + .024
        standing = (far & (m["floor"][i].min(-1) > 5.) & (m["up"][i] > .82)
                    & (m["speed"][i] < .35) & (m["spin"][i] < 2.) & (state[:, 2] > .22))
        stable = bool(standing[-25:].all())
        airborne_advance = 0.
        front_clearance = 0.
        released = first_touch is None
        if launch is not None:
            begin, end = launch
            airborne_advance = float(state[begin:end, 0].max() - state[begin, 0])
            front = feet[begin:end, 2:]
            # Reward both front feet approaching the far edge above the crest.
            front_clearance = float(np.max(np.min(
                np.clip((front[..., 0] + .30) / .38, 0., 1.)
                * np.clip((front[..., 2] - .20) / .135, 0., 1.), axis=1)))
            if first_touch is not None:
                last_touch = int(touch[-1] // SUBSTEPS)
                released = (last_touch < len(state)-25
                            and state[-25, 0] - state[last_touch, 0] > .03)
        stable = stable and released
        passed[i] = ordered and stable and peak_forbidden < 5.
        score[i] = (10 * np.clip((state[:,0].max() - initial[1]) / 1.8, 0, 1)
                    + 12 * ordered + 8 * (ordered and touch.size > 0)
                    + 14 * front_clearance + 12 * airborne_advance
                    + 15 * standing.mean() + 50 * passed[i]
                    - 4 * np.log1p(peak_forbidden / 20)
                    - .18 * np.count_nonzero(forbidden >= 5))
        details[i].update(passed=bool(passed[i]), rising_takeoff_frames=launch,
                          rear_wall_first_frame=first_touch, rear_wall_peak_n=float(load[i].max()),
                          rear_wall_released=bool(released),
                          forbidden_peak_n=peak_forbidden, permitted_front_knee_floor_peak_n=allowed_knee,
                          forbidden_floor_peak_n=float(ground.max()),
                          forbidden_floor_peak_s=float((ground.argmax() // ground.shape[1] + 1) * .005)
                          if ground.max() >= 5. else None,
                          forbidden_floor_link=COLLIDERS[int(np.unravel_index(ground.argmax(), ground.shape)[1])]
                          if ground.max() >= 5. else None,
                          forbidden_wall_peak_n=float(forbidden_wall[i].max()),
                          extended_calf_wall_peak_n=float(forbidden_calf[i].max()),
                          permitted_calf_wall_peak_n=float(np.where(extended_before_clearance[i], 0., calf_force[i]).max()),
                          final_stance=bool(stable))
    return m, score, passed, details


def rollout(roller, model, starts, controls):
    return roller.rollout(model, [mj.MjData(model) for _ in range(8)],
                          starts, np.repeat(controls, SUBSTEPS, axis=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=160)
    parser.add_argument('--assist', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = make_model(args.asset, configure=add_sensors)
    wall = model.geom('wall').id
    wall_x = model.geom_pos[wall, 0]
    initial = np.r_[0., -1.42, 0., .365, 1., 0., 0., 0., HOME, np.zeros(18)]
    ck = ROOT / 'amp_supermimic/outputs/trackerlab-amp/wblwbl_open_flat_10k/last.pt'
    runup, _, info = policy_runup(initial, ck, 50, 1.1)
    if len(runup) != 50:
        raise RuntimeError(info)
    rng = np.random.default_rng(2171)
    best = -np.inf
    center = None
    with Rollout(nthread=8) as roller:
        for step in range(args.rounds):
            # Establish a valid launch on flat ground before searching wall clearance.
            flat = step < args.rounds // 3
            if step == args.rounds // 3:
                best = -np.inf
            model.geom_pos[wall, 0] = 10. if flat else wall_x
            data = [mj.MjData(model) for _ in range(8)]
            controls, params = candidates(runup, 96, rng, center if step % 5 else None)
            controls = np.clip(controls, *model.jnt_range[1:].T)
            if args.assist and not flat:
                starts = np.broadcast_to(initial, (len(controls), len(initial))).copy()
                starts[:, 1] += rng.uniform(-.20, .33, len(controls))
                raw, sensors = rollout(roller, model, starts, controls)
                m, score, passed, details = wall_assist(model, raw, sensors, initial, len(runup))
            else:
                raw, sensors = simulate(roller, model, data, initial, controls)
                starts = np.broadcast_to(initial, (len(controls), len(initial)))
                m = measure_hop(model, raw, sensors)
                score, passed, details = batch_score(m, initial[1], len(runup))
            for i, state in enumerate(m['state']):
                launch = slice(len(runup), len(runup) + 27)
                vx, vz = state[launch, 19], state[launch, 21]
                score[i] += 5 * np.max(np.clip(vx, 0, 3) * np.clip(vz, 0, 3))
                for start, end in intervals(m['air'][i]):
                    if start >= len(runup) and end-start >= 3 and state[start,21] > 0:
                        score[i] += 15 * max(0, state[start:end,2].max()-state[start,2])
            i = int(np.argmax(score + 1000 * passed))
            if score[i] + 1000 * passed[i] > best:
                center = params[i]
                best = score[i] + 1000 * passed[i]
                best_controls = controls[i].copy()
                best_start = starts[i].copy()
                print(step, 'flat' if flat else 'wall', details[i], flush=True)
                np.savez_compressed(args.output / ('flat-best.npz' if flat else 'wall-best.npz'),
                                    initial=best_start, controls=best_controls, parameters=center)
            if not flat and passed[i]:
                break
    results = {}
    for seed in (None, 301, 1621):
        label = 'nominal' if seed is None else str(seed)
        starts = best_start[None] if seed is None else perturb(best_start, seed, 16)
        with Rollout(nthread=8) as roller:
            raw, sensors = rollout(roller, model, starts,
                                   np.broadcast_to(best_controls, (len(starts), *best_controls.shape)))
        if args.assist:
            m, score, passed, detail = wall_assist(model, raw, sensors, best_start, len(runup))
        else:
            m = measure_hop(model, raw, sensors)
            score, passed, detail = batch_score(m, best_start[1], len(runup))
        results[label] = dict(passes=int(passed.sum()), trials=len(passed), detail=detail)
        if seed is None:
            np.savez_compressed(args.output / 'replay-nominal.npz', states=m['state'],
                                air=m['air'], success=passed, scores=score)
    (args.output / 'evaluation.json').write_text(json.dumps(dict(runup=info, trials=results), indent=2))


if __name__ == '__main__':
    main()
