"""Package retargeted WBLWBL train.json clips into a style-campaign-pool-v1 pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = ROOT / "amp_supermimic/data/wblwbl"
DEFAULT_OUTPUT = DEFAULT_DATA_ROOT / "pools/walk_turn"
DEFAULT_TRAIN_CATEGORIES = ("walk", "turn")
DEFAULT_HELDOUT = ("walk/vhdc_horse1_s1_walk_03",)


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _discover_clips(data_root: Path, categories: tuple[str, ...], heldout: set[str],
                    explicit_clips: tuple[str, ...] | None = None):
    clips = []
    if explicit_clips is not None:
        ordered = list(dict.fromkeys([*explicit_clips, *sorted(heldout)]))
        for rel_id in ordered:
            category, clip_name = rel_id.split("/", 1)
            train_cfg = data_root / rel_id / "train.json"
            if not train_cfg.is_file():
                raise FileNotFoundError(f"Missing train.json for clip {rel_id}: {train_cfg}")
            cfg = json.loads(train_cfg.read_text(encoding="utf-8"))
            files = cfg.get("files") or []
            if len(files) != 1:
                raise ValueError(f"Expected 1 file in {train_cfg}, found {files}")
            csv_path = (train_cfg.parent / cfg.get("root", ".") / files[0]).resolve()
            if not csv_path.is_file():
                raise FileNotFoundError(f"Missing CSV referenced by {train_cfg}: {csv_path}")
            split = "test" if rel_id in heldout else "train"
            clips.append((rel_id, category, clip_name, cfg, csv_path, split))
    else:
        for category in categories:
            cat_dir = data_root / category
            if not cat_dir.is_dir():
                raise FileNotFoundError(f"Missing category directory: {cat_dir}")
            for train_cfg in sorted(cat_dir.glob("*/train.json")):
                clip_dir = train_cfg.parent
                rel_id = f"{category}/{clip_dir.name}"
                cfg = json.loads(train_cfg.read_text(encoding="utf-8"))
                files = cfg.get("files") or []
                if len(files) != 1:
                    raise ValueError(f"Expected 1 file in {train_cfg}, found {files}")
                csv_path = (train_cfg.parent / cfg.get("root", ".") / files[0]).resolve()
                if not csv_path.is_file():
                    raise FileNotFoundError(f"Missing CSV referenced by {train_cfg}: {csv_path}")
                split = "test" if rel_id in heldout else "train"
                clips.append((rel_id, category, clip_dir.name, cfg, csv_path, split))
    if not any(c[-1] == "train" for c in clips) or not any(c[-1] == "test" for c in clips):
        raise ValueError("Pool must have non-empty train and test splits")
    return clips


def build_pool(data_root: Path, output: Path, categories: tuple[str, ...], heldout: tuple[str, ...],
               style: str, mjcf: Path, source_config: Path, overwrite: bool = False,
               explicit_clips: tuple[str, ...] | None = None):
    sys.path.insert(0, str(ROOT / "amp_supermimic/scripts"))
    import train as wblwbl_train

    spec = wblwbl_train.register(mjcf, source_config)
    from prepare_style_campaign import _identity, verify_pool
    from train_terrain import validate_style_references

    clips = _discover_clips(data_root, categories, set(heldout), explicit_clips=explicit_clips)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory exists (pass --overwrite to rebuild): {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    records = []
    seen_hashes = set()
    for rel_id, category, clip_name, cfg, csv_path, split in clips:
        if cfg["joint_names"] != list(spec.joint_names):
            raise ValueError(f"Joint mismatch in {rel_id}")
        if cfg["source_fps"] != 50 or cfg["quaternion_order"] != "xyzw":
            raise ValueError(f"Expected 50 Hz xyzw config in {rel_id}")
        src_hash = _sha256(csv_path)
        if src_hash in seen_hashes:
            raise ValueError(f"Duplicate motion payload detected: {csv_path}")
        seen_hashes.add(src_hash)
        values = np.loadtxt(csv_path, delimiter=",", ndmin=2)
        dest_name = f"{category}__{clip_name}.csv"
        dest_path = output / dest_name
        shutil.copyfile(csv_path, dest_path)
        records.append({
            "file": dest_name,
            "category": category,
            "sha256": _sha256(dest_path),
            "original_sha256": src_hash,
            "source_file": str(csv_path.relative_to(ROOT)),
            "source_group": rel_id,
            "frames": int(len(values)),
            "fps": 50,
            "duration_seconds": float((len(values) - 1) / 50.0),
            "split": split,
        })

    configs = {}
    for split in ("train", "test"):
        split_files = [row["file"] for row in records if row["split"] == split]
        split_cfg = {
            "version": 1,
            "root": ".",
            "source_fps": 50,
            "quaternion_order": "xyzw",
            "joint_names": list(spec.joint_names),
            "files": split_files,
        }
        cfg_path = output / f"{split}.json"
        cfg_path.write_bytes(_json_bytes(split_cfg))
        configs[split] = {"file": cfg_path.name, "sha256": _sha256(cfg_path)}

    recipe = {
        "schema": "wblwbl-retargeted-pool-recipe-v1",
        "categories": list(categories),
        "heldout": list(heldout),
        "clips": [{"source_group": row["source_group"], "original_sha256": row["original_sha256"],
                   "split": row["split"]} for row in records],
    }
    manifest = {
        "schema": "style-campaign-pool-v1",
        "status": "ready",
        "robot": spec.name,
        "style": style,
        "fps": 50,
        "quaternion_order": "xyzw",
        "joint_names": list(spec.joint_names),
        "model_sha256": _sha256(Path(spec.mjcf_path)),
        "recipe_sha256": _identity(recipe),
        "recipe": recipe,
        "files": records,
        "configs": configs,
    }
    (output / "manifest.json").write_bytes(_json_bytes(manifest))

    verify_pool(output, spec.name, style)
    receipt = validate_style_references(output / "train.json", spec.name)
    return receipt, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_TRAIN_CATEGORIES))
    parser.add_argument("--clips", nargs="+", default=None,
                        help="Optional explicit category/clip IDs to include in train split")
    parser.add_argument("--heldout", nargs="+", default=list(DEFAULT_HELDOUT))
    parser.add_argument("--style", default="walk_turn")
    parser.add_argument("--mjcf", type=Path, default=ROOT.parent / "robocon2027_sim_control/deploy_cpp/robot/wblwbl_dog_v1/xml/wblwbl_dog_v1.xml")
    parser.add_argument("--source-config", type=Path, default=ROOT / "him_supermimic/outputs/zero-start/b-seed511/config.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    categories = (tuple(dict.fromkeys(c.split("/", 1)[0] for c in args.clips))
                  if args.clips is not None else tuple(args.categories))
    receipt, records = build_pool(
        args.data_root.resolve(),
        args.output.resolve(),
        categories,
        tuple(args.heldout),
        args.style,
        args.mjcf.resolve(),
        args.source_config.resolve(),
        overwrite=args.overwrite,
        explicit_clips=tuple(args.clips) if args.clips is not None else None,
    )
    print(json.dumps({"receipt": receipt, "train_count": sum(r["split"] == "train" for r in records),
                      "test_count": sum(r["split"] == "test" for r in records),
                      "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
