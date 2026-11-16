"""Run TrackerLab command AMP for WBLWBL using the existing deployment MJCF."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party/trackerlab_supermimic/scripts/rls/supermimic"
ROBOT = "wblwbl_dog_v1"
TASK = f"locomotion/{ROBOT}/velocity"
PROFILES = ("velocity", "stairs", "mountain", "parkour", "open_terrain", "open_flat")
DEFAULT_SOURCE_CONFIG = ROOT / "him_supermimic/outputs/zero-start/b-seed511/config.json"
URDF = ROOT / f"third_party/MyHIMLoco/legged_gym/resources/robots/{ROBOT}/urdf/wblwbl_dog.urdf"
MJCF = ROOT.parent / f"robocon2027_sim_control/deploy_cpp/robot/{ROBOT}/xml/{ROBOT}.xml"


def register(mjcf, source_config):
    sys.path.insert(0, str(UPSTREAM))
    from bootstrap import configure_paths

    configure_paths()
    import trackerTask.supermimic.robots as robots
    import trackerTask.supermimic.terrain as terrain
    from trackerTask.supermimic.gait import FOOT_BODIES, FOOT_OFFSETS

    config = json.loads(source_config.read_text())
    profile_cfg = config.get("profile")
    task_name = (
        config.get("task")
        or (profile_cfg.get("task") if isinstance(profile_cfg, dict) else None)
        or config.get("legacy", {}).get("asset", {}).get("name")
    )
    if task_name != ROBOT:
        raise ValueError(f"Expected a {ROBOT} source configuration")
    legacy = config["legacy"]
    contract = json.loads((ROOT / "amp_supermimic/configs/wblwbl_dog_v1.plan.json").read_text())["contract"]
    joints = tuple(contract["policy_joint_names"])
    pose = legacy["init_state"]["default_joint_angles"]
    root = ET.parse(mjcf).getroot()
    feet = tuple(f"{leg}_calf" for leg in ("RR", "RL", "FR", "FL"))
    offsets = []
    for name in feet:
        sphere = root.find(f".//body[@name='{name}']/geom[@type='sphere']")
        if sphere is None:
            raise ValueError(f"Missing foot collision sphere on {name}")
        offsets.append(tuple(map(float, sphere.attrib["pos"].split())))
    spec = robots.RobotSpec(ROBOT, joints, ("base_link", *feet), "base_link", "base_link",
                            str(mjcf.resolve()), tuple(pose[name] for name in joints),
                            (contract["action_scale"],) * len(joints))
    limits = {joint.attrib["name"]: joint.find("limit")
              for joint in ET.parse(URDF).getroot().findall("joint")}
    motors = tuple((float(limits[name].attrib["effort"]), float(limits[name].attrib["velocity"]),
                    contract["default_kp"], contract["default_kd"], .01) for name in joints)
    get_robot = robots.get_robot
    robot_parameters = terrain.robot_parameters

    def lookup(name="g1"):
        return spec if name == ROBOT else get_robot(name)

    def parameters(name):
        return (spec, terrain.ROBOT_HEIGHTS[ROBOT], motors) if name == ROBOT else robot_parameters(name)

    robots.get_robot = lookup
    terrain.get_robot = lookup
    terrain.robot_parameters = parameters
    terrain.ROBOT_HEIGHTS[ROBOT] = legacy["init_state"]["pos"][2]
    for profile in ("velocity", *terrain.PROFILES):
        task_id = f"locomotion/{ROBOT}/{profile}"
        terrain.ALL_TASKS[task_id] = terrain.TerrainTask(task_id, ROBOT, profile)
    from open_terrain import patch_open_terrain
    patch_open_terrain(ROBOT)
    FOOT_BODIES[ROBOT] = feet
    FOOT_OFFSETS[ROBOT] = tuple(offsets)
    return spec


def prepare(replay_path, output, spec):
    import numpy as np
    import torch
    from prepare_gait_references import align_reference_floor, motion_config, reference_audit, write_json

    source = json.loads(replay_path.with_name("source.json").read_text())["asset"]
    if tuple(source["joint_names"]) != spec.joint_names:
        raise ValueError("Replay joint order differs from the WBLWBL contract")
    if source["sha256"] != hashlib.sha256(URDF.read_bytes()).hexdigest():
        raise ValueError("Replay was collected with a different URDF")
    data = torch.load(replay_path, map_location="cpu", weights_only=True)["transitions"]
    state = data["state"]
    output.mkdir(parents=True, exist_ok=False)
    train, test, sources = [], [], {}
    for env in range(state.shape[1]):
        steps = data["episode_step"][:, env]
        episodes = data["episode_id"][:, env]
        ended = data["terminated"][:, env] | data["truncated"][:, env]
        boundary = (steps[1:] != steps[:-1] + 1) | (episodes[1:] != episodes[:-1]) | ended[:-1]
        cuts = [0, *(torch.where(boundary)[0] + 1).tolist(), len(steps)]
        for start, stop in zip(cuts, cuts[1:]):
            if stop - start < 160:
                continue
            window = state[start:stop, env]
            speed = window[:, 7:9].norm(dim=-1)
            moving = (speed > .16) & (window[:, 7] > .12)
            edges = np.diff(np.r_[False, moving.numpy(), False].astype(np.int8))
            for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                if right - left < 160:
                    continue
                raw = window[left:right, :13 + spec.num_joints]
                values = torch.cat((raw[:, :7], raw[:, 13:]), -1).numpy().copy()
                values[:, :2] -= values[0, :2].copy()
                values, alignment = align_reference_floor(values, ROBOT)
                name = f"env{env:03d}_{start + left:04d}.csv"
                np.savetxt(output / name, values, delimiter=",", fmt="%.9g")
                sources[name] = {"replay": str(replay_path.resolve()), "env": env,
                                 "frames": [start + int(left), start + int(right)],
                                 "floor_alignment": alignment}
                (test if env % 5 == 0 else train).append(name)
                if len(train) >= 8 and len(test) >= 2:
                    break
            if len(train) >= 8 and len(test) >= 2:
                break
        if len(train) >= 8 and len(test) >= 2:
            break
    if not train or not test:
        raise ValueError("Replay needs separate moving training and held-out recordings")
    audit = reference_audit(output, ROBOT, train, test, sources)
    train = [name for name in train if not audit["records"][name]["failures"]]
    test = [name for name in test if not audit["records"][name]["failures"]]
    write_json(output / "candidate_audit.json", audit)
    if not train or not test:
        raise ValueError("No valid training or held-out clips after physical gait audit")
    audit = reference_audit(output, ROBOT, train, test, sources)
    if audit["status"] != "passed":
        raise ValueError(audit["failures"])
    write_json(output / "gait_audit.json", audit)
    write_json(output / "train.json", motion_config(ROBOT, train))
    write_json(output / "test.json", motion_config(ROBOT, test))
    print(f"Audited references: {len(train)} training, {len(test)} held out")


def _option_value(args, flag):
    prefix = f"{flag}="
    for idx, arg in enumerate(args):
        if arg == flag and idx + 1 < len(args):
            return args[idx + 1]
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def _train_argv(program, default_profile, upstream_args):
    if _option_value(upstream_args, "--task") is not None:
        return [program, *upstream_args]
    profile = _option_value(upstream_args, "--task_profile") or default_profile
    return [program, "--task", f"locomotion/{ROBOT}/{profile}", *upstream_args]


def main():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from amp_supermimic.legacy import (
        apply_runtime_task_patches,
        build_upstream_train_args,
        class_to_dict,
        list_tasks,
        load_task_config,
        resolve_repo_path,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list_tasks(backend="warp"),
                        help="Optional legacy task name from amp_supermimic/legacy")
    parser.add_argument("--list-tasks", action="store_true",
                        help="Print available legacy task configs and exit")
    parser.add_argument("--mjcf", type=Path, default=None)
    parser.add_argument("--source-config", type=Path, default=None)
    parser.add_argument("--profile", choices=PROFILES, default=None)
    parser.add_argument("--live-viewer", action="store_true",
                        help="Open a live MuJoCo 3D viewer showing real-time rollouts during training")
    parser.add_argument("--viewer-envs", type=int, default=16,
                        help="Number of parallel robots to render simultaneously in the live viewer")
    parser.add_argument("--viewer-env", type=int, default=0,
                        help="Highlighted environment index (shows height scan & telemetry)")
    parser.add_argument("--viewer-fps", type=float, default=35.0,
                        help="Maximum render FPS during live viewer training (0 for unlimited)")
    parser.add_argument("--no-viewer-scan", action="store_true",
                        help="Disable rendering the 187-point heightfield scan dots in the live viewer")
    parser.add_argument("--reset-warm-start-goal-enc", action="store_true",
                        help="Re-initialize actor.enc linear layers after loading --warm-start weights")
    parser.add_argument("mode", nargs="?", choices=("prepare", "train", "evaluate", "export"))
    parser.add_argument("upstream_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.list_tasks:
        print("\n".join(list_tasks(backend="warp")))
        return
    if args.mode is None:
        parser.error("the following arguments are required: mode")

    env_cfg = train_cfg = None
    if args.task is not None:
        env_cfg, train_cfg = load_task_config(args.task)
        mjcf = args.mjcf or resolve_repo_path(env_cfg.robot.mjcf) or MJCF
        source_config = args.source_config or resolve_repo_path(env_cfg.robot.source_config) or DEFAULT_SOURCE_CONFIG
        profile = args.profile or env_cfg.terrain.profile
    else:
        mjcf = args.mjcf or MJCF
        source_config = args.source_config or DEFAULT_SOURCE_CONFIG
        profile = args.profile or "velocity"

    spec = register(mjcf, source_config)
    if env_cfg is not None:
        apply_runtime_task_patches(env_cfg)
    if args.mode == "prepare":
        prepare_parser = argparse.ArgumentParser()
        prepare_parser.add_argument("--replay", type=Path, required=True)
        prepare_parser.add_argument("--output", type=Path, required=True)
        options = prepare_parser.parse_args(args.upstream_args)
        prepare(options.replay, options.output, spec)
    elif args.mode == "train":
        if args.live_viewer:
            from live_viewer import close_live_viewer, install_live_viewer

            install_live_viewer(
                env_id=args.viewer_env,
                viewer_envs=args.viewer_envs,
                max_fps=args.viewer_fps,
                show_height_scan=not args.no_viewer_scan,
            )
        import train_terrain
        if args.reset_warm_start_goal_enc and not getattr(train_terrain.adapt_warm_start_actor, "_enc_reset_patched", False):
            import torch
            orig_adapt = train_terrain.adapt_warm_start_actor

            def patched_adapt(actor, joint_names, *, reset_arms=False, normalizer_prior_count=0.):
                receipt = orig_adapt(actor, joint_names, reset_arms=reset_arms,
                                     normalizer_prior_count=normalizer_prior_count)
                for layer in actor.enc:
                    if isinstance(layer, torch.nn.Linear):
                        layer.reset_parameters()
                receipt["goal_encoder_reset"] = True
                return receipt

            patched_adapt._enc_reset_patched = True
            train_terrain.adapt_warm_start_actor = patched_adapt

        train = train_terrain.main
        upstream_args = (
            build_upstream_train_args(env_cfg, train_cfg, args.upstream_args)
            if env_cfg is not None else args.upstream_args
        )
        run_dir = _option_value(upstream_args, "--run-dir")
        if env_cfg is not None and run_dir is not None and "--describe" not in upstream_args:
            run_path = Path(run_dir)
            run_path.mkdir(parents=True, exist_ok=True)
            (run_path / "legacy_task_config.json").write_text(
                json.dumps(
                    {
                        "task": args.task,
                        "env_cfg": class_to_dict(env_cfg),
                        "train_cfg": class_to_dict(train_cfg),
                    },
                    indent=2,
                    default=str,
                ) + "\n",
                encoding="utf-8",
            )

        sys.argv = _train_argv(sys.argv[0], profile, upstream_args)
        try:
            train()
        finally:
            if args.live_viewer:
                close_live_viewer()
    else:
        if args.mode == "evaluate":
            from evaluate_terrain import main as upstream_main
        else:
            from export_command_policy import main as upstream_main

        sys.argv = [sys.argv[0], *args.upstream_args]
        upstream_main()


if __name__ == "__main__":
    main()
