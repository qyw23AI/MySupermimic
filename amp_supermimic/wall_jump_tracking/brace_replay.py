"""Independently replay the saved joint targets and summarize contact milestones."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco as mj
from mujoco.rollout import Rollout
import numpy as np

from .brace_experiment import CONTROL_DT, SUBSTEPS, evaluate, make_model, simulate
from .brace_second import LANDING_WINDOW, score_jump


def first_frame(mask, window=1):
    frames = np.flatnonzero(mask)
    return int(frames[0] + window) if len(frames) else None


def summary(measured, mode, second):
    braced = first_frame(measured["braced"], 5)
    progress = first_frame(measured["progressed"], 5)
    release = first_frame(measured["unload"], 5) if mode == "calf" else None
    layout = {
        "support_mode": mode,
        "braced_frame": braced,
        "rear_advance_frame": progress,
        "unload_frame": release,
        "frame_seconds": CONTROL_DT,
    }

    def snapshot(frame):
        index = frame - 1
        force = measured["calf" if mode == "calf" else "toe"][index]
        return {
            "time_s": frame * CONTROL_DT,
            "base_advance_m": float(measured["base_delta"][index]),
            "rear_advance_each_m": measured["rear_delta_each"][index].tolist(),
            "front_support_n": force.tolist(),
            "linear_speed_m_s": float(measured["speed"][index]),
            "base_angular_speed_rad_s": float(np.linalg.norm(measured["state"][index, 22:25])),
            "rear_support_n": measured["rear"][index].tolist(),
            "peak_forbidden_contact_n_to_frame": float(measured["raw_bad_force"][:frame * SUBSTEPS].max()),
        }

    if progress:
        layout["rear_advance"] = snapshot(progress)
    if release:
        layout["unload"] = snapshot(release)
    if second is not None:
        prefix = int(second)
        accepted = first_frame(measured["success"], LANDING_WINDOW)
        clear_frames = measured["cleared"] & (measured["second_forbidden"] < 5.)
        last = accepted or len(measured["cleared"])
        final_state = measured["state"][prefix + last - 1]
        layout["second"] = {
            "start_frame": prefix,
            "validated_unload_before_second": bool(measured["unload"][prefix - 5]),
            "takeoff_detected": bool(measured["pass_air"]),
            "cleared": bool(clear_frames.any()),
            "four_foot_touchdown": bool(measured["landed"]),
            "stable_landing": accepted is not None,
            "stable_landing_frame": prefix + accepted if accepted else None,
            "minimum_stable_landing_s": LANDING_WINDOW * CONTROL_DT,
            "peak_calf_recontact_n_before_cut": float(measured["calf_recontact"][last - 1]),
            "peak_forbidden_contact_n_before_cut": float(measured["second_forbidden"][last - 1]),
            "base_height_m_at_cut": float(final_state[2]),
            "base_speed_m_s_at_cut": float(np.linalg.norm(final_state[19:22])),
        }
        cut = prefix + accepted if accepted else len(measured["state"])
    else:
        cut = release or progress or len(measured["state"])
    layout["accepted_cut_frame"] = cut
    return layout


def replay(model, file, mode):
    with np.load(file, allow_pickle=False) as data:
        control, initial = data["controls"], data["initial"]
        prefix = int(data["prefix"]) if "prefix" in data else None
    with Rollout(nthread=1) as runner:
        raw, sensors = simulate(runner, model, [mj.MjData(model)], initial, control[None])
    if prefix is None:
        values = evaluate(model, sensors, raw, initial, mode)
    else:
        _, values = score_jump(model, sensors, raw, initial, prefix)
    measured = {key: value[0] for key, value in values.items()}
    measured["rear_delta_each"] = measured["feet"][:, :2, 0] - initial[-2:]
    return summary(measured, mode, prefix), measured, initial, control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--toe", type=Path, required=True)
    parser.add_argument("--calf", type=Path, required=True)
    parser.add_argument("--second", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = make_model(args.work / "input/wbl_tracking.xml")
    toe, _, toe_initial, _ = replay(model, args.toe, "toe")
    calf, _, calf_initial, _ = replay(model, args.calf, "calf")
    if not np.array_equal(toe_initial, calf_initial):
        raise ValueError("Both methods must start at the same WBL state")
    reports = {"common_initial_state": True, "toe": toe, "calf": calf}
    if args.second:
        second, _, second_initial, _ = replay(model, args.second, "calf")
        if not np.array_equal(calf_initial, second_initial) or not second["second"]["validated_unload_before_second"]:
            raise ValueError("The second jump did not preserve the accepted calf unload")
        reports["calf_second"] = second
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2, allow_nan=False) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
