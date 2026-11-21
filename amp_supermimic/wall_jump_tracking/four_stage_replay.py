"""Independently replay the four-stage route and measure its impact loads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco as mj
from mujoco.rollout import Rollout
import numpy as np

from .brace_experiment import (
    COLLIDERS, CONTROL_DT, FEET, SUBSTEPS, evaluate, make_model, signal, simulate, windows,
)
from .brace_second import score_jump
from .four_stage import FINAL_HOLD, REAR_HOLD, SUB_DT, crest_load, force_signals, measurements


def first_event(mask, delay=0):
    events = np.flatnonzero(mask)
    return int(events[0]) + delay if events.size else None


def first_approach(model, states, sensor):
    """Check genuine first-flight and front-foot crest contact with rear feet grounded."""
    foot_floor = force_signals(model, sensor, "floor", FEET)[0]
    foot_wall = force_signals(model, sensor, "wall", FEET)[0]
    loaded = abs(foot_floor[..., 1] * foot_floor[..., 9]) > 7.
    crest = crest_load(foot_wall)[:, 2:] > 7.
    impact = np.maximum(
        abs(force_signals(model, sensor, "floor", COLLIDERS)[0, ..., 1]).max(-1),
        abs(force_signals(model, sensor, "wall", COLLIDERS)[0, ..., 1]).max(-1),
    )
    feet_z = np.stack([signal(model, sensor, f"position_{name}")[0,:,2] for name in FEET],axis=-1)
    flight = (~(loaded.any(-1) | (abs(foot_wall[...,1]) > 5.).any(-1))
              & (impact < 5.) & (feet_z.min(-1) > .029))
    start = first_event(windows(flight[None], 8)[0], 7)
    finite = np.isfinite(states).all()
    end = len(states)-1
    final = states[end, 1:]
    speed = np.linalg.norm(final[19:22])
    selected = np.arange(len(states) - 5*SUBSTEPS, len(states), SUBSTEPS) + SUBSTEPS - 1
    support = crest[selected].all(-1) & loaded[selected,:2].all(-1)
    passes = start is not None and start < selected[0] and support.all() and finite and speed < .35 and impact.max() < 5.
    return dict(passed=bool(passes), flight_start_s=start*SUB_DT if start is not None else None,
                nominal_front_support_n=crest_load(foot_wall)[end,2:].tolist(), duration_s=len(states)*SUB_DT)


def impact_series(raw, force, after=0, through=None):
    """Summed positive vertical load and impulse, using MJCF force sensors at 200 Hz."""
    force = force[:through] if through is not None else force
    points = np.flatnonzero(force[after:].max(-1) > 12.)
    if not len(points):
        return None
    first = after + int(points[0])
    values = force[first:first + round(.10 / SUB_DT)]
    return {
        "time_after_unload_s": first*SUB_DT,
        "base_vertical_speed_before_load_m_s": float(raw[max(0, first-1),22]),
        "peak_pair_sum_n_first_100ms": float(values.sum(-1).max()),
        "peak_per_leg_n_first_100ms": values.max(0).tolist(),
        "pair_vertical_impulse_ns_first_100ms": float(values.sum()*SUB_DT),
    }


def replay(model, filename, variant="catch", approach=None):
    with np.load(filename, allow_pickle=False) as file:
        initial = file["initial"].copy()
        targets = file["controls"].copy()
        release = int(file["prefix"] if "prefix" in file else file["release"])
    count = 0
    if approach:
        with np.load(approach, allow_pickle=False) as previous:
            approach_controls = previous["approach_controls"].copy()
            start = previous["approach_initial"].copy()
        count = len(approach_controls)
        targets = np.vstack((approach_controls, targets))
    else:
        start = initial
    with Rollout(nthread=1) as runner:
        raw, sensor = simulate(runner, model, [mj.MjData(model)], start, targets[None])
    after_approach = count * SUBSTEPS
    first = None
    if count:
        first = first_approach(model, raw[0,:after_approach], sensor[:,:after_approach])
        if not np.allclose(initial[:38], raw[0,after_approach-1], atol=1.e-5,rtol=0.):
            raise ValueError("The recorded foothold state does not match continuous replay")
    remaining = raw[:,after_approach:]
    sensed = sensor[:,after_approach:]
    revised = measurements(model, remaining, sensed, initial, release)
    front = evaluate(model,sensed[:,:release*SUBSTEPS],remaining[:,:release*SUBSTEPS],initial,"calf")
    brace_frame = first_event(front["braced"][0], 5)
    progress_frame = first_event(front["progressed"][0], 5)
    catch_frame = first_event(revised["catch"][0], REAR_HOLD)
    walk_frame = first_event(revised["landed"][0], FINAL_HOLD)
    if variant == "direct":
        _, original = score_jump(model, sensed, remaining, initial, release)
        walk_frame = first_event(original["success"][0], FINAL_HOLD)
    # Keep the complete trial for a failure, or stop on the frame that passes.
    after_release_frames = walk_frame or len(revised["state"][0])
    cutoff = count + release + after_release_frames
    retained = (cutoff - count - release) * SUBSTEPS
    valid = revised["flew"][0]
    air_start = first_event(valid)
    rear_floor_after = (air_start or 0) * SUBSTEPS
    ground = revised["floor_force"][0]
    top = revised["rear_load"][0]
    motion = revised["raw"][0]
    report = {
        "verified_first_takeoff_and_crest": first,
        "first_handoff_frame": count if count else None,
        "front_calf_braced_frame": count + brace_frame if brace_frame else None,
        "rear_advance_frame": count + progress_frame if progress_frame else None,
        "unload_frame": count + release,
        "front_calf_unload_verified": bool(revised["released"][0]),
        "rear_shin_catch_frame": count + release + catch_frame if catch_frame else None,
        "rear_shin_catch_hold_s": REAR_HOLD*CONTROL_DT if catch_frame else 0.,
        "rear_footfall_frames_inc_one": [
            count + release + point if point is not None else None
            for point in [first_event(revised["far_rear_ground"][0,:,leg], 1) for leg in range(2)]
        ],
        "step_down_stable": bool(walk_frame),
        "four_stage_pass": bool(first and first["passed"] and revised["released"][0] and catch_frame and walk_frame),
        "new_airborne_phase_after_catch": bool(revised["extra_hop"][0,:after_release_frames].any()),
        "forbidden_contact_peak_n": float(revised["bad"][0,:retained].max()),
        "front_foot_ground_impact": impact_series(motion,ground[:,2:],through=retained),
        "rear_shin_crest_impact": impact_series(motion,top,through=retained),
        "rear_foot_ground_impact": impact_series(motion,ground[:,:2],rear_floor_after,retained),
        "cut_frame": cutoff,
    }
    qpos = np.vstack((start[1:20],raw[0,SUBSTEPS-1::SUBSTEPS,1:20]))
    forces = dict(
        front_wall=crest_load(force_signals(model,sensor,"wall",FEET))[0,SUBSTEPS-1::SUBSTEPS,2:],
        rear_wall=crest_load(force_signals(model,sensor,"wall",("RR_calf_collision","RL_calf_collision")))[0,SUBSTEPS-1::SUBSTEPS],
        floor=abs(force_signals(model,sensor,"floor",FEET)[0,SUBSTEPS-1::SUBSTEPS,:,1]),
    )
    public = {"qpos": qpos[:cutoff+1], "controls": targets[:cutoff], "initial":start,
              **{key:value[:cutoff] for key,value in forces.items()}}
    return report, public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--approach", type=Path)
    parser.add_argument("--continuous", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.continuous) != bool(args.approach):
        parser.error("--continuous and --approach must be supplied together")
    model = make_model(args.asset)
    experiments = {"direct":(args.direct,"direct",None),"split":(args.candidate,"catch",None)}
    if args.continuous:
        experiments["continuous"]=(args.continuous,"catch",args.approach)
    args.output.mkdir(parents=True,exist_ok=True)
    results = {}
    for key, entry in experiments.items():
        report, roll = replay(model,*entry)
        results[key] = report
        np.savez_compressed(args.output / f"{key}-replay.npz",**roll)
    file = args.output / "evaluation.json"
    file.write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
    print(file)


if __name__ == "__main__":
    main()
