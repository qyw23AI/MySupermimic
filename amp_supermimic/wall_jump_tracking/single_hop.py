"""Search and evaluate a true one-flight WBL jump over the 30 cm wall in MuJoCo."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import mujoco as mj
from mujoco.rollout import Rollout
import numpy as np

from .brace_experiment import COLLIDERS, CONTROL_DT, FEET, SUBSTEPS, make_model, simulate
from .contact_route import add_sensors, measure
from .four_stage import force_signals
from .robot import REF, ROOT, THICKNESS

PERTURBATION = np.array([.002, .002, np.deg2rad(.5)])
FAR_X = THICKNESS / 2 + .0238


def intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def measure_hop(model, raw, sensors):
    """Measure one-hop contacts while allowing normal foot-floor support."""
    measured = measure(model, raw, sensors)
    count, physics_steps = sensors.shape[:2]
    control_steps = physics_steps // SUBSTEPS
    foot_floor = force_signals(model, sensors, "floor", FEET)
    foot_normal = abs(foot_floor[..., 1] * foot_floor[..., 9])
    body_floor = abs(force_signals(model, sensors, "floor", COLLIDERS)[..., 1]).max(-1)
    any_wall = abs(force_signals(model, sensors, "wall", FEET + COLLIDERS)[..., 1]).max(-1)
    body_floor = body_floor.reshape(count, control_steps, SUBSTEPS).max(-1)
    any_wall = any_wall.reshape(count, control_steps, SUBSTEPS).max(-1)
    measured["bad_control"] = np.maximum(body_floor, any_wall)
    measured["wall_contact"] = any_wall
    measured["foot_floor"] = foot_normal.reshape(count, control_steps, SUBSTEPS, len(FEET)).max(2)
    measured["air"] = ((measured["foot_floor"].max(-1) < 5.) & (any_wall < 5.)
                        & (body_floor < 5.) & (measured["feet"][..., 2].min(-1) > .029))
    return measured


def resample(values: np.ndarray, count: int) -> np.ndarray:
    old = np.linspace(0., 1., len(values))
    new = np.linspace(0., 1., count)
    return np.stack([np.interp(new, old, values[:, j]) for j in range(values.shape[1])], axis=1)


def policy_runup(initial: np.ndarray, checkpoint: Path, frames: int, vx: float = .60):
    """Collect closed-loop run-up joint targets from the existing WBL flat policy."""
    import mujoco as mj
    import torch

    script_dir = ROOT / "amp_supermimic/scripts"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("wbl_play_mujoco", script_dir / "play_mujoco.py")
    play = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(play)
    (ck, terrain, model, data, robot, height, qpos_ids, qvel_ids, actuator_ids,
     default_q, qlo, qhi, field, mdp, actor, _body_velocities) = play._load_policy_and_scene(
        checkpoint.resolve(), "open_flat", 0, ground_plane=True
    )
    if model.nq != 19 or model.nv != 18 or len(actuator_ids) != 12:
        raise ValueError("The flat-policy model differs from the WBL 19/18 state and 12-action contract")
    mj.mj_resetData(model, data)
    data.qpos[:] = initial[1:20]
    data.qvel[:] = initial[20:38]
    data.ctrl[actuator_ids] = initial[8:20]
    mj.mj_forward(model, data)
    xi = torch.zeros(1, robot.num_joints + 4, dtype=torch.float32)
    a_prev = torch.zeros(1, robot.num_joints, dtype=torch.float32)
    cy = float(terrain.centers[0])
    command = torch.tensor([[cy, vx, 0., 0., height, 0., 0.]], dtype=torch.float32)
    state0 = play._state_from_data(data, qpos_ids, qvel_ids)
    obs = mdp.obs(state0, a_prev, xi, command)
    hist = obs[:, None].expand(-1, actor.hist, -1).clone()
    controls, states = [], [np.r_[data.time, data.qpos.copy(), data.qvel.copy()]]
    with torch.no_grad():
        for _ in range(frames):
            state = play._state_from_data(data, qpos_ids, qvel_ids)
            obs = mdp.obs(state, a_prev, xi, command)
            hist = torch.cat((hist[:, 1:], obs[:, None]), 1)
            action = actor(hist, mdp.goal(state, command)).clamp(-mdp.action_limit, mdp.action_limit)
            target = (default_q + action[0] * torch.tensor(robot.action_scale, dtype=torch.float32)).clamp(qlo, qhi)
            controls.append(target.numpy().copy())
            data.ctrl[actuator_ids] = target.numpy()
            for _substep in range(4):
                mj.mj_step(model, data)
            a_prev = action
            states.append(np.r_[data.time, data.qpos.copy(), data.qvel.copy()])
            state = play._state_from_data(data, qpos_ids, qvel_ids)
            if bool(mdp.termination(state, command)[0]):
                break
    control_array = np.asarray(controls, dtype=np.float32)
    state_array = np.asarray(states, dtype=np.float64)
    displacement = float(state_array[-1, 1] - state_array[0, 1])
    metrics = {
        "checkpoint": str(checkpoint), "checkpoint_iteration": ck.get("iteration"),
        "command_vx_m_s": vx, "requested_frames": frames, "completed_frames": len(control_array),
        "forward_displacement_m": displacement,
        "final_forward_velocity_m_s": float(state_array[-1, 20]),
        "min_base_height_m": float(state_array[:, 3].min()),
        "final_base_height_m": float(state_array[-1, 3]),
    }
    return control_array, state_array, metrics


def seed_controls(initial: np.ndarray, horizon: int, runup_frames: int, runup: np.ndarray) -> np.ndarray:
    """Build closed-loop policy run-up, one push, a tucked flight, landing, and stance hold."""
    csv = np.loadtxt(REF / "high_wall_jump_direct.csv", delimiter=",")
    q = csv[:, 7:].copy()
    q += initial[8:20] - q[0]
    if runup.shape != (runup_frames, 12):
        raise ValueError(f"Expected {runup_frames} policy run-up controls, got {runup.shape}")
    jump_horizon = horizon - runup_frames
    # Keep the forward-jump compression and extension together.
    jump_name = "quad_jump_forward_1m_50hz"
    jump_path = ROOT / "amp_supermimic/data/wblwbl/jump" / jump_name / f"{jump_name}.csv"
    jump = np.loadtxt(jump_path, delimiter=",")[:, 7:]
    launch = jump[36:52].copy()
    hold_count, landing_count = 25, 15
    flight_count = jump_horizon - len(launch) - hold_count - landing_count
    if flight_count < 20:
        raise ValueError("The horizon is too short for run-up, one flight, landing, and 0.5 s stance")
    tuck = jump[53].copy()
    alpha = np.linspace(0., 1., flight_count)
    ease = alpha * alpha * (3. - 2. * alpha)
    flight = np.repeat(tuck[None], flight_count, axis=0)
    flight[:6] = (1. - ease[:6, None]) * launch[-1] + ease[:6, None] * tuck
    land_alpha = np.linspace(0., 1., landing_count)
    land_alpha = land_alpha * land_alpha * (3. - 2. * land_alpha)
    landing = (1. - land_alpha[:, None]) * tuck + land_alpha[:, None] * q[-1]
    hold = np.repeat(q[-1:], hold_count, axis=0)
    # Blend the policy's last joint target into the one-jump launch posture.
    blend = np.linspace(0., 1., 6)
    blend = blend * blend * (3. - 2. * blend)
    launch[:6] = (1. - blend[:, None]) * runup[-1] + blend[:, None] * launch[:6]
    result = np.vstack((runup, launch, flight, landing, hold))
    if len(result) != horizon:
        raise ValueError(f"expected {horizon} controls, got {len(result)}")
    return result


def batch_score(m: dict[str, np.ndarray], x_start: float, runup_frames: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    count, steps = m["air"].shape
    wall = m["wall_contact"]
    bad = m["bad_control"]
    scores = np.zeros(count, dtype=np.float32)
    passed = np.zeros(count, dtype=bool)
    details = []
    windows = np.lib.stride_tricks.sliding_window_view
    for i in range(count):
        sustained_air = [(a, b) for a, b in intervals(m["air"][i]) if b - a >= 6]
        state, feet = m["state"][i], m["feet"][i]
        root_x, root_z = state[:, 0], state[:, 2]
        flights = []
        for a, b in sustained_air:
            before_z = root_z[max(0, a - 1)]
            rise = float(root_z[a:b].max() - before_z)
            launch_vz = float(state[a, 21])
            supported_before = a > 0 and bool(m["foot_floor"][i, a - 1].max() > 5.)
            if launch_vz >= .15 and rise >= .06 and supported_before:
                flights.append((a, b))
        far_feet = feet[..., 0].min(-1) > FAR_X
        grounded = m["floor"][i].min(-1) > 5.
        upright = m["up"][i] > .82
        quiet = (m["speed"][i] < .35) & (m["spin"][i] < 2.)
        standing = far_feet & grounded & upright & quiet & (state[:, 2] > .22) & (bad[i] < 5.)
        stable_start = -1
        end_flight = flights[-1][1] if flights else 0
        if steps >= 25:
            stable_windows = windows(standing, 25).all(-1)
            candidates = np.flatnonzero(stable_windows & (np.arange(len(stable_windows)) >= end_flight))
            if len(candidates):
                stable_start = int(candidates[0])
        wall_free = not np.any(bad[i] >= 5.)
        exactly_one = len(flights) == 1 and len(sustained_air) == 1
        flight_start = flights[0][0] if flights else -1
        takeoff_target = runup_frames + 14
        in_takeoff_window = runup_frames <= flight_start <= runup_frames + 18
        runup_index = min(runup_frames, steps - 1)
        runup_dx = float(state[runup_index, 0] - state[0, 0])
        runup_speed = float(m["speed"][i, runup_index])
        runup_vx = float(state[runup_index, 19])
        runup_vy = float(state[runup_index, 20])
        runup_dy = float(state[runup_index, 1] - state[0, 1])
        side_drift = float(np.max(np.abs(state[:, 1] - state[0, 1])))
        has_runup = runup_dx >= .04 and runup_vx >= .20 and abs(runup_vy) < .30 and abs(runup_dy) < .08
        ok = exactly_one and in_takeoff_window and wall_free and stable_start >= 0 and has_runup and side_drift < .10
        passed[i] = ok

        crossing = m["air"][i] & (root_x >= -.12) & (root_x <= .12)
        clearance = float(root_z[crossing].max()) if crossing.any() else 0.
        progress = float(np.clip((root_x.max() - x_start) / .95, 0., 1.2))
        far_support = float((far_feet & grounded).mean())
        landing_support = float(np.max(((feet[..., 0] > FAR_X) & (m["floor"][i] > 5.)).mean(-1)))
        duration = min((flights[0][1] - flights[0][0]) / 28., 1.) if flights else 0.
        stable_any = stable_start >= 0
        peak_bad = float(bad[i].max())
        wall_frames = int(np.count_nonzero(wall[i] >= 5.))
        flight_dx = (float(root_x[flight_start:end_flight].max() - root_x[flight_start])
                     if flights else 0.)
        flight_xmax = float(root_x[flight_start:end_flight].max()) if flights else float(root_x.max())
        launch_window = state[runup_frames:min(steps, runup_frames + 20)]
        launch_shape = np.max(np.clip(launch_window[:, 21], 0., 3.)
                              * np.clip(launch_window[:, 19], 0., 2.))
        timing_score = (4. - .35 * abs(flight_start - takeoff_target)) if flights else -10.
        score = (5. * launch_shape + 5. * progress + 3. * np.clip(runup_dx / .12, 0., 1.)
                 + 5. * np.clip(runup_vx / .50, 0., 1.)
                 + 6. * np.clip((clearance - .40) / .30, 0., 1.) + 2. * duration
                 + 12. * np.clip(flight_dx / .55, 0., 1.)
                 + 8. * np.clip((flight_xmax + .25) / .70, 0., 1.)
                 + 10. * far_support + 8. * landing_support + 20. * stable_any + 30. * ok
                 + timing_score - 20. * side_drift - 5. * abs(runup_vy) - 8. * max(0, len(flights) - 1)
                 - 15. * max(0, len(sustained_air) - len(flights))
                 - 6. * wall_frames - peak_bad / 100.)
        scores[i] = score
        details.append({
            "passed": bool(ok), "flight_intervals_frames": [[a, b] for a, b in flights],
            "sustained_air_intervals_frames": [[a, b] for a, b in sustained_air],
            "positive_takeoff_flight_count": len(flights),
            "takeoff_window_pass": bool(in_takeoff_window),
            "runup_forward_m": runup_dx, "runup_speed_m_s": runup_speed,
            "runup_forward_velocity_m_s": runup_vx, "runup_lateral_velocity_m_s": runup_vy,
            "runup_lateral_drift_m": runup_dy, "max_lateral_drift_m": side_drift,
            "stable_start_frame": stable_start if stable_start >= 0 else None,
            "max_root_x_m": float(root_x.max()), "root_clearance_at_wall_m": clearance,
            "flight_forward_m": flight_dx, "flight_max_root_x_m": flight_xmax,
            "far_ground_support_fraction": far_support, "best_far_foot_support_fraction": landing_support,
            "wall_contact_frames": wall_frames,
            "peak_forbidden_contact_n": peak_bad, "final_base_speed_m_s": float(m["speed"][i, -1]),
        })
    return scores, passed, details


def perturb(initial: np.ndarray, seed: int, count: int) -> np.ndarray:
    offsets = np.random.default_rng(seed).uniform(-PERTURBATION, PERTURBATION, (count, 3))
    starts = np.broadcast_to(initial, (count, len(initial))).copy()
    starts[:, 1:3] += offsets[:, :2]
    starts[:, 4] = np.cos(.5 * offsets[:, 2])
    starts[:, 7] = np.sin(.5 * offsets[:, 2])
    return starts


def evaluate(model, initial, controls, seed, count, threads, runup_frames):
    starts = np.broadcast_to(initial, (count, len(initial))).copy() if seed is None else perturb(initial, seed, count)
    batches = np.broadcast_to(controls, (count, *controls.shape)).copy()
    workers = min(threads, count)
    with Rollout(nthread=workers) as roller:
        raw = np.empty((count, controls.shape[0] * SUBSTEPS, 1 + model.nq + model.nv))
        sensors = np.empty((count, controls.shape[0] * SUBSTEPS, model.nsensordata))
        roller.rollout(model, [mj.MjData(model) for _ in range(workers)], starts[:, :38],
                       np.repeat(batches, SUBSTEPS, axis=1), state=raw, sensordata=sensors)
    m = measure_hop(model, raw, sensors)
    scores, passed, detail = batch_score(m, float(initial[1]), runup_frames)
    return starts, raw, m, scores, passed, detail


def run(args):
    model = make_model(args.asset, configure=add_sensors)
    with np.load(args.baseline, allow_pickle=False) as f:
        initial = f["initial"].copy()
    runup, runup_states, runup_metrics = policy_runup(
        initial, args.runup_checkpoint, args.runup_frames, args.runup_vx
    )
    if len(runup) != args.runup_frames:
        raise RuntimeError(f"Flat-policy run-up terminated after {len(runup)} of {args.runup_frames} controls")
    template = seed_controls(initial, args.horizon, args.runup_frames, runup)
    lower, upper = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    grid = np.unique(np.r_[0, np.arange(4, args.horizon, 4), args.horizon - 1])
    blend = np.stack([np.interp(np.arange(args.horizon), grid, row) for row in np.eye(len(grid))], axis=1)
    mean = template[grid].copy()
    spread = np.tile(np.array([.14, .34, .44] * 4, dtype=np.float32), (len(grid), 1))
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "runup.json").write_text(json.dumps(runup_metrics, indent=2, ensure_ascii=False) + "\n")
    np.savez_compressed(args.output / "runup-policy.npz", controls=runup, states=runup_states)
    best_score = -np.inf
    best_controls = template.copy()
    history = []
    datas = [mj.MjData(model) for _ in range(args.threads)]
    with Rollout(nthread=args.threads) as roller:
        for it in range(args.iters):
            decay = max(.30, 1. - it / (args.iters * 1.15))
            knots = mean[None] + rng.normal(size=(args.batch, *mean.shape)) * spread[None] * decay
            knots[0] = mean
            if args.batch > 1:
                knots[1] = template[grid]
            symmetric = args.batch // 3
            if symmetric:
                knots[:symmetric, :, [3, 4, 5, 9, 10, 11]] = (
                    knots[:symmetric, :, [0, 1, 2, 6, 7, 8]] *
                    np.array([-1., 1., 1., -1., 1., 1.])[None, None, :]
                )
            candidates = np.clip(np.einsum("tk,bkj->btj", blend, knots), lower[None, None] - .12,
                                 upper[None, None] + .12)
            candidates[:, :args.runup_frames] = runup[None]
            candidates[1] = template
            raw, sensors = simulate(roller, model, datas, initial, candidates)
            measured = measure_hop(model, raw, sensors)
            scores, passed, detail = batch_score(measured, float(initial[1]), args.runup_frames)
            elite_count = max(6, args.batch // 12)
            elite = np.argsort(scores)[-elite_count:]
            mean = .15 * mean + .85 * knots[elite].mean(0)
            index = int(np.argmax(scores))
            if float(scores[index]) > best_score:
                best_score = float(scores[index])
                best_controls = candidates[index].copy()
                np.savez_compressed(args.output / "best-single-hop.npz", initial=initial,
                                    controls=best_controls, score=best_score, passed=bool(passed[index]),
                                    detail=json.dumps(detail[index]))
            row = {"iteration": it, "score": best_score,
                   "batch_best": float(scores[index]), "batch_passes": int(passed.sum()),
                   "best_detail": detail[index]}
            history.append(row)
            if it % 5 == 0 or passed.any():
                print(json.dumps(row, ensure_ascii=False), flush=True)
            if passed.any() and passed[index]:
                break
            del raw, sensors, measured
    (args.output / "search.json").write_text(json.dumps(history, indent=2, ensure_ascii=False) + "\n")

    reports = {"stage": "one-shot-direct", "wall_height_m": .30, "stand_frames": 25,
               "one_flight_only": True, "wall_contact_allowed": False,
               "runup_source": str(args.runup_checkpoint),
               "checkpoint_sha256": hashlib.sha256(args.runup_checkpoint.read_bytes()).hexdigest(),
               "iterations": len(history), "best_score": best_score,
               "search_pass": bool(np.load(args.output / "best-single-hop.npz")["passed"]),
               "cohorts": {}}
    for seed in (None, 301, 1621):
        label = "nominal" if seed is None else str(seed)
        starts, raw, m, scores, passed, details = evaluate(model, initial, best_controls, seed,
                                                           1 if seed is None else 16, args.threads, args.runup_frames)
        reports["cohorts"][label] = {"trials": len(passed), "successes": int(passed.sum()),
                                     "per_trial": details}
        if seed is None or seed == 301:
            np.savez_compressed(args.output / f"replay-{label}.npz", states=m["state"],
                                feet=m["feet"], ground=m["floor"], front_wall=m["front"],
                                rear_wall=m["rear"], toe_wall=m["toe"], bad=m["bad_control"],
                                air=m["air"], initial=starts, controls=best_controls,
                                success=passed, scores=scores)
    (args.output / "evaluation.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in reports.items() if k != "cohorts"}, ensure_ascii=False), flush=True)


def evaluate_saved(args):
    model = make_model(args.asset, configure=add_sensors)
    with np.load(args.baseline, allow_pickle=False) as f:
        initial = f["initial"].copy()
    with np.load(args.output / "best-single-hop.npz", allow_pickle=False) as f:
        controls = f["controls"].copy()
        best_score = float(f["score"])
    reports = {"stage": "one-shot-direct", "wall_height_m": .30, "stand_frames": 25,
               "one_flight_only": True, "wall_contact_allowed": False,
               "runup_source": str(args.runup_checkpoint),
               "checkpoint_sha256": hashlib.sha256(args.runup_checkpoint.read_bytes()).hexdigest(),
               "best_score": best_score,
               "search_pass": bool(np.load(args.output / "best-single-hop.npz")["passed"]), "cohorts": {}}
    for seed in (None, 301, 1621):
        label = "nominal" if seed is None else str(seed)
        starts, raw, m, scores, passed, detail = evaluate(model, initial, controls, seed,
                                                          1 if seed is None else 16, args.threads, args.runup_frames)
        reports["cohorts"][label] = {"trials": len(passed), "successes": int(passed.sum()),
                                     "per_trial": detail}
        if seed is None or seed == 301:
            np.savez_compressed(args.output / f"replay-{label}.npz", states=m["state"],
                                feet=m["feet"], ground=m["floor"], front_wall=m["front"],
                                rear_wall=m["rear"], toe_wall=m["toe"], bad=m["bad_control"],
                                air=m["air"], initial=starts, controls=controls,
                                success=passed, scores=scores)
    (args.output / "evaluation.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"search_pass": reports["search_pass"],
                      "cohorts": {k:(v["successes"],v["trials"]) for k,v in reports["cohorts"].items()}},
                     ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=130)
    parser.add_argument("--runup-frames", type=int, default=16)
    parser.add_argument("--runup-checkpoint", type=Path,
                        default=ROOT / "amp_supermimic/outputs/trackerlab-amp/wblwbl_open_flat_10k/last.pt")
    parser.add_argument("--runup-vx", type=float, default=.60)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--eval-only", action="store_true", help="Evaluate best-single-hop.npz without searching again")
    args = parser.parse_args()
    if args.horizon < args.runup_frames + 90 or args.runup_frames < 8 or args.iters < 1 or args.batch < 16 or args.threads < 1:
        parser.error("Use horizon >= 70, positive iterations/threads, and batch >= 16")
    if args.eval_only:
        evaluate_saved(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
