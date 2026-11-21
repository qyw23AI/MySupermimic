"""Resolve local legacy configs without loading a simulator."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

from amp_supermimic.legacy import ROOT, class_to_dict, load_task_config, resolve_repo_path
from .references import audit_manifest, prepare_pool
from .robot import Robot

DEFAULT_FRICTION_RANGE = (.3, 1.2)


def resolve(args):
    env, train = load_task_config(args.task)
    for target, fields in (
        (env.control, ("project_joint_targets",)),
        (env.env, ("num_envs", "seed", "device", "max_episode_steps", "action_limit")),
        (env.amp, ("motion_config", "amp_weight", "amp_batch", "amp_updates", "amp_history")),
        (env.terrain, ("competition_stage", "level", "source", "friction_range")),
        (env.commands, ("training_modes",)),
        (train.runner, ("iterations", "compile", "compile_critic", "fused_optimizers")),
        (train.world_model, ("wm_only_iters", "warmup_steps", "horizon", "hist", "wm_window",
                             "wm_batches", "wm_batch", "pi_batch", "buffer_steps")),
    ):
        for field in fields:
            value = getattr(args, field, None)
            if value is not None:
                if not hasattr(target, field):
                    raise ValueError(f"{field} cannot be applied to task {args.task}")
                if field == "training_modes":
                    value = tuple(value)
                setattr(target, field, value)
    validate(env, train)
    return env, train


def validate(env, train):
    if not math.isclose(env.control.physics_dt * env.control.decimation, .02, rel_tol=0., abs_tol=1e-9):
        raise ValueError("TrackerLab command and expert states require a 0.02 s control period")
    if not math.isfinite(env.terrain.friction) or env.terrain.friction < 0:
        raise ValueError("Expected a finite nonnegative ground contact friction coefficient")
    low, high = env.terrain.friction_range
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high):
        raise ValueError("Expected a finite, ordered, nonnegative ground contact friction range")
    if env.commands.curriculum.enabled and env.terrain.profile != "open_flat":
        raise ValueError("The adaptive velocity curriculum requires the open_flat profile")
    if env.amp.reference_reset_probability and (env.terrain.profile != "open_flat" or env.domain_rand.self_collisions):
        raise ValueError("PhysX reference resets require open_flat and disabled self-collision")
    if env.commands.jump_probability or env.commands.jump_height:
        raise ValueError("Explicit jump commands are not configured for PhysX")
    if env.commands.range_mode != "full":
        raise ValueError("PhysX commands use explicit samples; reference-derived ranges are unavailable")
    if (not env.commands.samples or any(len(row) != 3 or not all(map(math.isfinite, row))
                                       for row in env.commands.samples)):
        raise ValueError("Expected finite vx, vy and yaw command samples")
    modes = getattr(env.commands, "training_modes", None)
    allowed_modes = {"stop", "forward", "backward", "strafe", "turn", "mixed", "forward_turn"}
    if modes is not None and (not modes or len(set(modes)) != len(modes) or set(modes) - allowed_modes):
        raise ValueError("Command training modes must be a non-empty unique subset of the supported modes")

    lo, hi = env.commands.resample_steps
    if not 100 <= lo <= hi <= env.env.max_episode_steps:
        raise ValueError("Command interval must lie within the episode (lower bound 100 steps)")
    wm = train.world_model
    if not 2 <= env.amp.amp_history <= wm.hist:
        raise ValueError("Actor history must cover the AMP window")
    if min(env.env.num_envs, env.env.solver_iterations, train.runner.iterations, env.amp.amp_batch,
           env.amp.amp_updates, wm.horizon, wm.wm_window, wm.wm_batch, wm.wm_batches,
           wm.pi_batch, wm.warmup_steps) < 1 or not math.isfinite(env.amp.amp_weight) or env.amp.amp_weight <= 0:
        raise ValueError("Expected positive environment, training and AMP sizes")
    if not 0 < env.env.action_limit <= 20 or not 0 <= wm.wm_only_iters:
        raise ValueError("Invalid action limit or world model warmup")
    if wm.buffer_steps < max(wm.horizon, wm.wm_window, wm.hist) or wm.horizon >= env.env.max_episode_steps:
        raise ValueError("Replay and episode limits must cover model windows")
    if env.terrain.profile == "parkour":
        if env.terrain.level is not None and not 0 <= env.terrain.level < 6:
            raise ValueError("TrackerLab parkour levels are 0–5")
    elif env.terrain.profile not in ("plane", "open_flat"):
        terrain = env.terrain
        from .wmp_terrain import OBSTACLE_LEVELS

        maximum = OBSTACLE_LEVELS.get(terrain.profile, 9)
        if not 0 <= terrain.level <= maximum or terrain.competition_stage not in ("eval", "learn"):
            raise ValueError(f"{terrain.profile} supports levels 0–{maximum} in eval or learn mode")
        if terrain.competition_stage == "learn" and terrain.level != 0:
            raise ValueError("WMP learn mode starts at L0")
        if terrain.competition_window < 1 or not 0 <= terrain.competition_demote < terrain.competition_promote <= 1:
            raise ValueError("Invalid WMP curriculum thresholds")


def robot_from(env):
    return Robot.load(resolve_repo_path(env.robot.urdf), env.robot, env.control)


def motion_receipt(env, robot, output=None):
    motion = resolve_repo_path(env.amp.motion_config)
    if motion is not None:
        records = audit_manifest(motion, robot.joints, env.amp.amp_history)
        return motion, {"train": {str(path): info for path, info in records.items()}}
    receipt = prepare_pool(resolve_repo_path(env.amp.motion_root), env.amp.clips, env.amp.heldout,
                           robot.joints, env.amp.amp_history, output)
    return (output / "train.json" if output is not None else None), receipt


def snapshot(task, env, train):
    return {"task": task, "env_cfg": class_to_dict(env), "train_cfg": class_to_dict(train)}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + "\n")


def read_snapshot(path):
    document = json.loads(path.read_text())
    env = json.loads(json.dumps(document["env_cfg"]), object_hook=lambda v: SimpleNamespace(**v))
    train = json.loads(json.dumps(document["train_cfg"]), object_hook=lambda v: SimpleNamespace(**v))
    return document["task"], env, train
