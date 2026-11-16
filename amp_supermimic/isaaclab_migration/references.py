"""Build separate TrackerLab-compatible train and evaluation motion manifests."""

import csv
import hashlib
import json
import math
import shutil


def audit_manifest(manifest, joints, history):
    manifest = manifest.resolve(strict=True)
    document = json.loads(manifest.read_text())
    if (tuple(document["joint_names"]) != joints or document["source_fps"] != 50
            or document["quaternion_order"] != "xyzw" or document["version"] != 1):
        raise ValueError(f"Incompatible reference format: {manifest}")
    root = (manifest.parent / document["root"]).resolve()
    records = {}
    for name in document["files"]:
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or "rubbish" in path.parts or path in records:
            raise ValueError(f"Invalid or duplicate motion path: {path}")
        count = 0
        with path.open(newline="") as stream:
            for row in csv.reader(stream):
                values = list(map(float, row))
                norm = sum(x * x for x in values[3:7])
                if len(values) != 19 or not all(map(math.isfinite, values)) or abs(norm - 1) > .01:
                    raise ValueError(f"Invalid pose in {path} at row {count + 1}")
                count += 1
        if count < history:
            raise ValueError(f"Motion too short for {history} frames: {path}")
        records[path] = {"frames": count, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if not records:
        raise ValueError(f"Empty reference manifest: {manifest}")
    return records


def prepare_pool(source, train, heldout, joints, history, output=None):
    source = source.resolve(strict=True)
    split = {}
    for tag, clips in (("train", train), ("test", heldout)):
        records = {}
        for clip in clips:
            manifest = source / clip / "train.json"
            if not manifest.resolve().is_relative_to(source) or "rubbish" in manifest.parts:
                raise ValueError(f"Invalid reference clip: {clip}")
            records.update(audit_manifest(manifest, joints, history))
        if not records:
            raise ValueError(f"Empty {tag} reference split")
        split[tag] = records
    seen = {r["sha256"] for r in split["train"].values()}
    if seen.intersection(r["sha256"] for r in split["test"].values()):
        raise ValueError("Reference files overlap across training and held-out clips")
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        for tag, records in split.items():
            names = []
            for index, path in enumerate(records):
                name = f"{tag}_{index:02d}_{path.stem}.csv"
                shutil.copyfile(path, output / name)
                names.append(name)
            (output / f"{tag}.json").write_text(json.dumps({
                "version": 1, "root": ".", "source_fps": 50,
                "quaternion_order": "xyzw", "joint_names": list(joints), "files": names,
            }, indent=2) + "\n")
    return {tag: {str(path): info for path, info in records.items()} for tag, records in split.items()}
