"""HIMLoco-style task registry and config inheritance for SuperMimic training."""

from __future__ import annotations

import copy
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TASKS = {
    "wblwbl_dog_v1": ("wblwbl_dog_v1.wblwbl_dog_v1_config", "WblwblDogV1Cfg"),
    "wblwbl_dog_v1_open_flat": (
        "wblwbl_dog_v1.wblwbl_dog_v1_open_flat_config",
        "WblwblDogV1OpenFlatCfg",
    ),
}


_PHYSX_MODULE = "wblwbl_dog_v1.wblwbl_dog_v1_physx_config"
PHYSX_TASKS = {
    "wblwbl_dog_v1_physx_flat": (_PHYSX_MODULE, "WblwblDogV1PhysxCfg"),
    "wblwbl_dog_v1_physx_wmp_stair_gap": (_PHYSX_MODULE, "WblwblDogV1PhysxWmpStairGapCfg"),
    "wblwbl_dog_v1_physx_wmp_stair_gap_ramp": (_PHYSX_MODULE, "WblwblDogV1PhysxWmpRampCfg"),
    "wblwbl_dog_v1_physx_wmp_gap_combos": (_PHYSX_MODULE, "WblwblDogV1PhysxWmpCombosCfg"),
}
TASKS.update(PHYSX_TASKS)


def list_tasks(*, backend: str | None = None) -> tuple[str, ...]:
    if backend is None:
        return tuple(sorted(TASKS))
    if backend == "physx":
        return tuple(sorted(PHYSX_TASKS))
    if backend == "warp":
        return tuple(sorted(TASKS.keys() - PHYSX_TASKS.keys()))
    raise ValueError(f"Unknown task backend {backend!r}")


def class_to_dict(obj):
    if isinstance(obj, dict):
        return {k: class_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [class_to_dict(v) for v in obj]
    if not hasattr(obj, "__dict__"):
        return obj
    return {
        k: class_to_dict(getattr(obj, k))
        for k in dir(obj)
        if not k.startswith("_") and not callable(getattr(obj, k))
    }


def resolve_repo_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else (ROOT / p).resolve()


def load_task_config(task_name: str, num_envs: int | None = None):
    if task_name not in TASKS:
        raise KeyError(f"Unknown task {task_name!r}; available: {list_tasks()}")
    module_name, cls_name = TASKS[task_name]
    mod = import_module(f"amp_supermimic.legacy.{module_name}")
    env_cfg = copy.deepcopy(getattr(mod, cls_name)())
    train_cfg = copy.deepcopy(getattr(mod, f"{cls_name}Train")())
    if num_envs is not None:
        env_cfg.env.num_envs = int(num_envs)
    return env_cfg, train_cfg


def apply_runtime_task_patches(env_cfg) -> None:
    """Push config values into open_terrain runtime dictionaries before training."""
    import open_terrain

    cur_dict = class_to_dict(env_cfg.commands.curriculum)
    for k, v in cur_dict.items():
        if k in open_terrain.OPEN_FLAT_CURRICULUM:
            open_terrain.OPEN_FLAT_CURRICULUM[k] = v

    rew_dict = class_to_dict(env_cfg.rewards)
    for k, v in rew_dict.items():
        if k in open_terrain.OPEN_FLAT_REWARDS:
            open_terrain.OPEN_FLAT_REWARDS[k] = v


def build_upstream_train_args(env_cfg, train_cfg, cli_args: list[str]) -> list[str]:
    """Convert legacy env/train configs into train_terrain CLI flags; CLI args take precedence."""

    def has_flag(flag: str) -> bool:
        return any(arg == flag or arg.startswith(f"{flag}=") for arg in cli_args)

    def has_bool_flag(stem: str) -> bool:
        return has_flag(f"--{stem}") or has_flag(f"--no-{stem}")

    args: list[str] = []

    def add_val(flag: str, value):
        if value is not None and not has_flag(flag):
            args.extend([flag, str(value)])

    def add_bool(stem: str, value: bool, *, store_true_only: bool = False):
        if store_true_only:
            if value and not has_flag(f"--{stem}"):
                args.append(f"--{stem}")
        elif not has_bool_flag(stem):
            args.append(f"--{stem}" if value else f"--no-{stem}")

    default_run_dir = (
        train_cfg.runner.run_dir
        or f"amp_supermimic/outputs/trackerlab-amp/{train_cfg.runner.experiment_name}_{train_cfg.runner.run_name}"
    )
    add_val("--run-dir", resolve_repo_path(default_run_dir))
    add_val("--num-envs", env_cfg.env.num_envs)
    add_val("--seed", env_cfg.env.seed)
    add_val("--device", env_cfg.env.device)
    add_val("--max-episode-steps", env_cfg.env.max_episode_steps)
    add_val("--solver-iterations", env_cfg.env.solver_iterations)
    add_val("--action-limit", env_cfg.env.action_limit)
    add_val("--clearance-ratio", env_cfg.env.clearance_ratio)

    add_val("--command-range-mode", env_cfg.commands.range_mode)
    if not has_flag("--command-resample-steps"):
        args.extend(["--command-resample-steps", str(env_cfg.commands.resample_steps[0]), str(env_cfg.commands.resample_steps[1])])
    add_val("--jump-height", env_cfg.commands.jump_height)
    add_val("--jump-probability", env_cfg.commands.jump_probability)

    add_bool("physical-domain-randomization", env_cfg.domain_rand.physical_domain_randomization, store_true_only=True)
    add_bool("self-collisions", env_cfg.domain_rand.self_collisions)
    add_val("--collision-mask-version", env_cfg.domain_rand.collision_mask_version)

    add_val("--amp-motion-config", resolve_repo_path(env_cfg.amp.motion_config))
    add_val("--amp-weight", env_cfg.amp.amp_weight)
    add_val("--amp-batch", env_cfg.amp.amp_batch)
    add_val("--amp-updates", env_cfg.amp.amp_updates)
    add_val("--amp-history", env_cfg.amp.amp_history)
    add_val("--style-feature-mode", env_cfg.amp.style_feature_mode)
    add_val("--reference-reset-probability", env_cfg.amp.reference_reset_probability)
    add_val("--foot-slide-weight", env_cfg.amp.foot_slide_weight)
    add_val("--feet-close-weight", env_cfg.amp.feet_close_weight)

    add_val("--lr", train_cfg.algorithm.lr)
    add_val("--actor-lr", train_cfg.algorithm.actor_lr)
    add_val("--gamma", train_cfg.algorithm.gamma)
    add_val("--lam", train_cfg.algorithm.lam)

    add_val("--buffer-steps", train_cfg.world_model.buffer_steps)
    add_val("--utd", train_cfg.world_model.utd)
    add_val("--horizon", train_cfg.world_model.horizon)
    add_val("--hist", train_cfg.world_model.hist)
    add_val("--wm-window", train_cfg.world_model.wm_window)
    add_val("--wm-batches", train_cfg.world_model.wm_batches)
    add_val("--wm-batch", train_cfg.world_model.wm_batch)
    add_val("--pi-batch", train_cfg.world_model.pi_batch)
    add_val("--critic-epochs", train_cfg.world_model.critic_epochs)
    add_val("--critic-minibatches", train_cfg.world_model.critic_minibatches)
    add_val("--warmup-steps", train_cfg.world_model.warmup_steps)
    add_val("--wm-only-iters", train_cfg.world_model.wm_only_iters)

    add_val("--iterations", train_cfg.runner.iterations)
    add_val("--log-every", train_cfg.runner.log_every)
    add_val("--ckpt-every", train_cfg.runner.ckpt_every)
    add_val("--keep-every", train_cfg.runner.keep_every)
    add_val("--speed-probe-updates", train_cfg.runner.speed_probe_updates)
    add_bool("compile", train_cfg.runner.compile)
    add_bool("compile-critic", train_cfg.runner.compile_critic)
    add_bool("fused-optimizers", train_cfg.runner.fused_optimizers)

    args.extend(cli_args)
    return args
