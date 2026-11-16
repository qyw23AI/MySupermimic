#!/usr/bin/env python3
"""Download and validate all 40 pinned Kine2Go Go2 reference clips."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRACKER_SCRIPTS = ROOT / "third_party/trackerlab_supermimic/scripts/rls/supermimic"
sys.path.insert(0, str(TRACKER_SCRIPTS))
import prepare_kine2go as importer


def clip_names_from_ranges(path):
    names = []
    for line in path.read_text().splitlines():
        match = re.match(r'^"([^ ]+) ([^ ]+) (\d+) (-?\d+)"$', line.strip())
        if match:
            names.append(match.group(1))
    return names


def add_license_texts(source_dir):
    manifest_path = source_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    known = {record["file"] for record in manifest["files"]}
    listing = json.loads(importer.fetch(f"{importer.API}/LICENSES?limit=100"))
    for entry in listing:
        name = entry.get("path", "")
        if entry.get("type") != "file" or name in known:
            continue
        content = importer.fetch(f"{importer.BASE}/{name}")
        algorithm = "sha256" if "lfs" in entry else "git_blob_sha1"
        expected = entry["lfs"]["oid"] if "lfs" in entry else entry["oid"]
        actual = hashlib.sha256(content).hexdigest() if algorithm == "sha256" else importer.blob_hash(content)
        if actual != expected:
            raise ValueError(f"Upstream hash mismatch: {name}")
        target = source_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        manifest["files"].append({
            "file": name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "source_hash_algorithm": algorithm,
            "source_hash": expected,
        })
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    ranges_path = ROOT / "amp_supermimic/data/raw/kine2go_inward_review/PIPELINE_motion_ranges.txt"
    clips = clip_names_from_ranges(ranges_path)
    if len(clips) != 40 or len(set(clips)) != 40:
        raise ValueError(f"Expected 40 unique pinned Kine2Go motions, got {len(clips)}")

    source = ROOT / "amp_supermimic/data/raw/kine2go_all40"
    output = ROOT / "amp_supermimic/data/go2/kine2go_all40"
    config_path = ROOT / "amp_supermimic/data/configs/go2_kine2go_all40.json"
    importer.download(source, clips)
    add_license_texts(source)

    from bootstrap import configure_paths
    configure_paths()
    from trackerTask.supermimic.robots import get_robot
    spec = get_robot("go2")

    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for clip in clips:
        raw_path = source / "data" / clip / "motion.npy"
        raw = np.load(raw_path, allow_pickle=False)
        meta = json.loads(raw_path.with_name("clip.json").read_text())
        values = importer.convert_array(raw, spec.joint_names)
        validation = importer.validate_fk(raw, values, spec)
        target = output / f"{clip}.csv"
        np.savetxt(target, values, delimiter=",", fmt="%.10f")
        records.append({
            "file": target.name,
            "frames": int(len(values)),
            "duration_seconds": (len(values) - 1) / int(meta["fps"]),
            "raw_sha256": importer.digest(raw_path),
            "sha256": importer.digest(target),
            "source": meta.get("source"),
            "source_url": meta.get("source_url"),
            "license": meta.get("license"),
            "attribution": meta.get("attribution"),
            "description": meta.get("description"),
            "tags": meta.get("tags", []),
            "validation": validation,
        })

    manifest = {
        "version": 1,
        "dataset": importer.DATASET,
        "revision": importer.REVISION,
        "pipeline_repository": importer.LOCK["pipeline_repository"],
        "pipeline_revision": importer.LOCK["pipeline_revision"],
        "clips": clips,
        "source_fps": importer.LOCK["source_fps"],
        "joint_names": list(spec.joint_names),
        "quaternion_order": "xyzw",
        "reference_data_only": True,
        "conversion": "Kine2Go Go2 reference rows; raw joint order and WXYZ quaternion converted to TrackerLab Go2 order; no smoothing or crop.",
        "files": records,
        "source_files": json.loads((source / "source_manifest.json").read_text())["files"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    config = {
        "version": 1,
        "root": "../go2/kine2go_all40",
        "source_fps": importer.LOCK["source_fps"],
        "quaternion_order": "xyzw",
        "joint_names": list(spec.joint_names),
        "files": [record["file"] for record in records],
        "source_license_note": "Per-clip licenses are listed in manifest.json and raw Kine2Go clip.json files.",
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({
        "source": str(source),
        "output": str(output),
        "clips": len(records),
        "licenses": sorted({record["license"] for record in records}),
        "fk_frames": sum(record["validation"]["frames_checked"] for record in records),
    }, indent=2))


if __name__ == "__main__":
    main()
