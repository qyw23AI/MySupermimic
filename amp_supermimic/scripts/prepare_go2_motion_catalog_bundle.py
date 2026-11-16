#!/usr/bin/env python3
"""Prepare a compact, attributed catalog of public Go2 motion references."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "amp_supermimic/data/raw/go2_lvr_backward_sideways"
DEST = ROOT / "amp_supermimic/data/go2/lvr_velocity_demos"
METRICS = ROOT / "amp_supermimic/outputs/go2_motion_catalog/go2_public_motion_catalog.json"


def copy_hdf5_compact(source: Path, target: Path) -> tuple[int, int]:
    with h5py.File(source, "r") as src:
        if "last_index" not in src:
            raise ValueError(f"Simulation demo has no last_index: {source}")
        valid_rows = int(np.asarray(src["last_index"][...]).reshape(-1)[0]) + 1
        row_capacity = int(src["_indices"].shape[0]) if "_indices" in src else None
        if row_capacity is None or valid_rows > row_capacity:
            raise ValueError(f"Invalid valid row count {valid_rows}: {source}")
        with h5py.File(target, "w") as dst:
            def copy_attrs(a, b):
                for key, value in a.attrs.items():
                    b.attrs[key] = value

            copy_attrs(src, dst)
            dst.attrs["catalog_valid_rows"] = valid_rows
            dst.attrs["catalog_source_path"] = str(source.relative_to(ROOT))
            def visit(name, obj):
                if isinstance(obj, h5py.Group):
                    group = dst.require_group(name)
                    copy_attrs(obj, group)
                    return
                parent_name, _, leaf = name.rpartition("/")
                parent = dst.require_group(parent_name) if parent_name else dst
                trim = bool(obj.shape) and row_capacity is not None and obj.shape[0] == row_capacity
                data = obj[:valid_rows] if trim else obj[...]
                opts = {}
                if getattr(data, "size", 0) > 1024 and getattr(data, "dtype", None) is not None and data.dtype.kind not in "OUS":
                    opts = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
                dset = parent.create_dataset(leaf, data=data, **opts)
                copy_attrs(obj, dset)
            src.visititems(visit)
    return valid_rows, row_capacity


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compact_lvr() -> list[dict]:
    DEST.mkdir(parents=True, exist_ok=True)
    records = []
    for direction in ("backward", "forward", "sideway"):
        src = RAW / f"isaac_go2_{direction}" / "traj-1.hdf5"
        dst = DEST / f"isaac_go2_{direction}_traj-1_compact.hdf5"
        valid, capacity = copy_hdf5_compact(src, dst)
        records.append({
            "task": direction,
            "kind": "Isaac Lab simulation demonstration",
            "source": str(src.relative_to(ROOT)),
            "file": str(dst.relative_to(ROOT)),
            "valid_rows": valid,
            "preallocated_rows": capacity,
            "sha256": sha256(dst),
        })
    for direction in ("backward", "forward", "sideway"):
        src = RAW / f"unitree_go2_{direction}" / "step-250+250.hdf5"
        dst = DEST / f"unitree_go2_{direction}_step-250+250.hdf5"
        shutil.copy2(src, dst)
        records.append({
            "task": direction,
            "kind": "Real Go2 demonstration",
            "source": str(src.relative_to(ROOT)),
            "file": str(dst.relative_to(ROOT)),
            "sha256": sha256(dst),
        })
    (DEST / "manifest.json").write_text(json.dumps({
        "source_repository": "https://github.com/ruipengZ/Latent-Variation-Regularization",
        "dataset_description": "Official LVR Go2 forward, backward and sideways demonstrations. Simulation files are compacted to last_index+1 valid rows; real-world files are copied unchanged.",
        "files": records,
    }, indent=2) + "\n")
    return records


def build_catalog(records: list[dict]) -> None:
    metrics = json.loads(METRICS.read_text())
    kine = metrics["clips"][:40]
    optimized = metrics["clips"][40:]
    manifest_path = ROOT / "amp_supermimic/data/go2/kine2go_all40/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    lines = [
        "# Go2 可公开获取动作目录",
        "",
        f"更新：{__import__('datetime').date.today().isoformat()}",
        "",
        "## 视频预览",
        "",
        "[完整 45 段参考动作视频](../../outputs/go2_motion_catalog/go2_public_motion_catalog.mp4)",
        "",
        "视频为固定方位相机下的运动学参考回放，不是物理接触/PD 控制验收。40 段 Kine2Go 原始动作以 60 Hz 关节/根部参考重定向到 Go2；另 5 段为公开轨迹优化器已有的 Go2 空翻/跳跃轨迹。",
        "",
        "## 明确包含后退、侧移的动作数据",
        "",
        "LVR 官方示范数据明确提供 Go2 前进、后退、侧移三种任务，各有 Isaac Lab 仿真轨迹和真机示范。仿真轨迹 HDF5 原文件预分配 1,000,000 行，但 last_index=999；本目录将其压缩为前 1,000 个有效采样，原文件仍保留在 data/raw/go2_lvr_backward_sideways/。真机文件按来源原样复制。采样包含观测和动作，不包含视频或全局位姿轨迹。",
        "",
        "| 任务 | 仿真示范（有效行） | 真机示范 |",
        "|---|---|---|",
    ]
    by_task = {}
    for rec in records:
        by_task.setdefault(rec["task"], []).append(rec)
    for task in ("backward", "forward", "sideway"):
        sims = [r for r in by_task[task] if r["kind"].startswith("Isaac")]
        reals = [r for r in by_task[task] if r["kind"].startswith("Real")]
        lines.append(f"| {task} | [{sims[0]['valid_rows']} 行 HDF5]({sims[0]['file']}) | [HDF5 示范]({reals[0]['file']}) |")
    lines += [
        "",
        "## 40 段 Kine2Go → Go2 参考动作",
        "",
        "来源包含自然犬类动作、AI4Animation 控制器合成轨迹、Solo8 动作和马步态片段；均只是重定向到 Go2 骨架的参考动作。非真实 Go2 实机采集。Kine2Go 数据集提供 CC BY-NC 4.0、CC BY-SA 4.0 等不同授权，需逐条查看每段 clip.json 与数据目录中的许可证；使用时保留归属与授权限制。",
        "",
        "| 动作名 | 动作说明 | 时长(s) | 授权 |",
        "|---|---|---:|---|",
    ]
    for clip in kine:
        lines.append(f"| {clip['name']} | {clip.get('description','').replace('|','/')} | {clip['source_duration_seconds']:.2f} | {clip['license']} |")
    lines += [
        "",
        "## 5 段轨迹优化动作",
        "",
        "以下是公开 SE(3) 全身轨迹优化仓库中已提交的 Go2 参考轨迹，不是 Go2 实机示范：",
        "",
        "| 动作名 | 说明 |",
        "|---|---|",
    ]
    for clip in optimized:
        lines.append(f"| {clip['name']} | {clip.get('description','')} |")
    lines += [
        "",
        "该仓库还提供未预生成轨迹的求解脚本：后退斜坡行走、前进、左/右侧移斜坡、原地正/反转。它们是可运行的优化任务定义，不等同于现成专家数据；求解依赖 Pinocchio/CasADi/IPOPT 等依赖。",
        "",
        "## 本地来源位置",
        "",
        "- Kine2Go Go2 参考：amp_supermimic/data/go2/kine2go_all40/；原始片段、许可证和 clip.json：amp_supermimic/data/raw/kine2go_all40/。",
        "- LVR 仿真及真机前进/后退/侧移：amp_supermimic/data/go2/lvr_velocity_demos/；原始下载：amp_supermimic/data/raw/go2_lvr_backward_sideways/。",
        "- 已生成的优化轨迹及求解脚本：third_party/go2_flip_TO/。",
        "- 可复现的 45 段视频渲染脚本：amp_supermimic/scripts/render_go2_public_motion_catalog.py。",
        "- 来源固定版本及逐片段校验：Kine2Go manifest 记录了 revision、SHA-256、FK 误差、关节限位余量。",
        "",
        "## 重要限制",
        "",
        "- Kine2Go 的左转/右转包含根部朝向变化，侧移素材是 strafe，不应把横向平移称作 yaw 转向。",
        "- LVR 观测数据没有全局根部位姿，无法据此恢复真实世界坐标下的完整后退/侧移视频；数据仍包含 policy observation/action，可用于模仿学习实验。",
        "- 这些公共动作不能直接当作 WBLWBL 专家数据。迁移到 WBLWBL 前仍需重定向、限位检查、接触/动力学回放和质量筛选。",
        "",
    ]
    (ROOT / "amp_supermimic/data/go2/MOTION_CATALOG.md").write_text("\n".join(lines))


def build_bundle() -> Path:
    output = ROOT / "amp_supermimic/outputs/go2_motion_catalog/go2_public_motion_catalog_bundle.zip"
    include = [
        ROOT / "amp_supermimic/data/go2/MOTION_CATALOG.md",
        ROOT / "amp_supermimic/data/go2/kine2go_all40",
        ROOT / "amp_supermimic/data/go2/lvr_velocity_demos",
        ROOT / "amp_supermimic/outputs/go2_motion_catalog/go2_public_motion_catalog.mp4",
        ROOT / "amp_supermimic/outputs/go2_motion_catalog/go2_public_motion_catalog.json",
        ROOT / "amp_supermimic/scripts/render_go2_public_motion_catalog.py",
        ROOT / "amp_supermimic/scripts/prepare_go2_motion_catalog_bundle.py",
        ROOT / "third_party/go2_flip_TO/LICENSE",
        ROOT / "third_party/go2_flip_TO/datasets/go2/mocap_motions_go2",
        ROOT / "third_party/Latent-Variation-Regularization/LICENSE",
    ]
    raw_root = ROOT / "amp_supermimic/data/raw/kine2go_all40"
    if raw_root.exists():
        include.extend(p for p in raw_root.glob("*") if p.name.lower().startswith("license"))
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in include:
            if not path.exists():
                continue
            paths = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file() and ".git" not in p.parts)
            for p in paths:
                zf.write(p, p.relative_to(ROOT))
    return output


def main():
    records = compact_lvr()
    build_catalog(records)
    bundle = build_bundle()
    print(json.dumps({
        "lvr_files": len(records),
        "catalog": str((ROOT / "amp_supermimic/data/go2/MOTION_CATALOG.md").relative_to(ROOT)),
        "bundle": str(bundle.relative_to(ROOT)),
        "bundle_bytes": bundle.stat().st_size,
    }, indent=2))


if __name__ == "__main__":
    main()
