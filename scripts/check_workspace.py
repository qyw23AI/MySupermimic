#!/usr/bin/env python3
"""Check pinned source identities; this is not a simulation or training test."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    errors = []
    for item in json.loads((root / "manifests/upstreams.json").read_text())["submodules"]:
        path = root / item["path"]
        try:
            head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
            dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip()
            indexed = subprocess.check_output(["git", "-C", str(root), "ls-files", "--stage", "--", item["path"]], text=True).split()
            if head != item["commit"] or dirty:
                errors.append(f"Submodule changed: {item['path']}")
            if indexed[:2] != ["160000", item["commit"]]:
                errors.append(f"Gitlink does not match manifest: {item['path']}")
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"Cannot inspect {item['path']}: {exc}")
    source = json.loads((root / "manifests/himloco_source.json").read_text())
    local = root / source["source_root"]
    if local.exists():
        for relative, expected in source["files"].items():
            path = local / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                errors.append(f"HIMLoco source drift: {relative}")
        asset = local / source["asset"]["path"]
        if not asset.is_file() or hashlib.sha256(asset.read_bytes()).hexdigest() != source["asset"]["sha256"]:
            errors.append("HIMLoco robot asset changed")
    else:
        errors.append(f"Missing MyHIMLoco source: {local}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Submodule pins and source fingerprints verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
