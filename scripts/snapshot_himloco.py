#!/usr/bin/env python3
"""Record local source identity and pure configuration; never launch a simulator."""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

FILES = [
    "README.md",
    "isaaclab_migration/README.md",
    "isaaclab_migration/himloco_lab/__init__.py",
    "isaaclab_migration/himloco_lab/config.py",
    "isaaclab_migration/himloco_lab/env_cfg.py",
    "isaaclab_migration/himloco_lab/env.py",
    "isaaclab_migration/himloco_lab/core.py",
    "isaaclab_migration/himloco_lab/logic.py",
    "isaaclab_migration/himloco_lab/math.py",
    "isaaclab_migration/himloco_lab/adapter.py",
    "isaaclab_migration/himloco_lab/legacy/base/legged_robot_config.py",
    "isaaclab_migration/himloco_lab/legacy/base/base_config.py",
    "isaaclab_migration/himloco_lab/legacy/wblwbl_dog_v1/wblwbl_dog_v1_config.py",
    "isaaclab_migration/scripts/train.py",
    "isaaclab_migration/scripts/export_policy.py",
    "rsl_rl/rsl_rl/modules/him_actor_critic.py",
    "rsl_rl/rsl_rl/modules/him_estimator.py",
    "rsl_rl/rsl_rl/algorithms/him_ppo.py",
    "rsl_rl/rsl_rl/runners/him_on_policy_runner.py",
    "rsl_rl/rsl_rl/storage/him_rollout_storage.py",
]


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source.resolve(strict=True)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "isaaclab_migration"))
    from himloco_lab.config import asset_contract, class_to_dict, load_config

    cfg, train = load_config("wblwbl_dog_v1", num_envs=512, stage=3)
    asset, names, limits, root_link = asset_contract(cfg)
    data = {
        "schema_version": "himloco-source.v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "head": git(root, "rev-parse", "HEAD"),
        "working_tree_status": git(root, "status", "--porcelain"),
        "note": "Hashes describe the working tree, including uncommitted edits. No simulation was run.",
        "files": {p: digest(root / p) for p in FILES},
        "asset": {"path": str(asset), "sha256": digest(asset), "joint_names": names,
                  "joint_limits": limits, "root_link": root_link},
        "profile": {"task": "wblwbl_dog_v1", "stage": 3, "num_envs_override": 512,
                    "env": class_to_dict(cfg), "train": train},
        "runtime_validation": "not_run",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(f"Recorded {len(FILES)} source hashes and asset contract in {args.output}")


if __name__ == "__main__":
    main()
