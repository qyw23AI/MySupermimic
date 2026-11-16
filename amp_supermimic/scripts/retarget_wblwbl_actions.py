"""Retarget local Go2 reference trajectories into six WBLWBL action directories."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def category(name):
    if "jump" in name:
        return "jump"
    if "strafe" in name:
        return "strafe"
    if "turn" in name or "_synth_" in name:
        return "turn"
    if "crawl" in name or "scoot" in name:
        return "crawl"
    if any(gait in name for gait in ("run", "trot", "canter")):
        return "run"
    return "walk"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    sources = ROOT / "data/go2/kine2go_all40"
    output = ROOT / "data/wblwbl"
    manifest = json.loads((sources / "manifest.json").read_text())
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")

    optimized_dir = ROOT.parent / "third_party/go2_flip_TO/datasets/go2/mocap_motions_go2"
    nju_dir = ROOT / "data/raw/nju_agility_jump"
    inputs = [(sources / entry["file"], []) for entry in manifest["files"]]
    inputs += [(source, ["--optimized-json"]) for source in sorted(optimized_dir.glob("*_50hz.txt"))]
    inputs += [(source, ["--nju-json"]) for source in sorted(nju_dir.glob("jump_*.json"))]

    def retarget(item):
        source, format_args = item
        group = "jump" if format_args else category(source.stem)
        destination = output / group / source.stem
        process = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("retarget_go2.py")),
             "--source", str(source), "--output", str(destination),
             "--rotate-foot-calibration", "--swing-boost", "0"]
            + format_args,
            env=env, capture_output=True, text=True,
        )
        if process.returncode:
            raise RuntimeError(f"{source.name}: {process.stderr}")
        report = json.loads((destination / "report.json").read_text())
        return {
            "category": group,
            "source": str(source.relative_to(ROOT.parent)),
            "motion": str((destination / source.with_suffix(".csv").name).relative_to(output)),
            "report": str((destination / "report.json").relative_to(output)),
            "frames": report["frames"],
            "ik_screen_pass": report["motion_usable_for_amp"],
            "flat_floor_joint_pd_pass": report["playback_pass"],
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        entries = list(pool.map(retarget, inputs))
    index = {"fps": 50, "motions": entries}
    (output / "action_index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"Retargeted {len(entries)} motions into {output}")


if __name__ == "__main__":
    main()
