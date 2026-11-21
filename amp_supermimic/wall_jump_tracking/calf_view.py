"""Render real Warp rollout files from the calf-support tracker side by side."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw

from .brace_experiment import make_model
from .brace_view import font


def choose_trial(entries, requested):
    if requested is not None:
        return requested
    for data, report, _ in reversed(entries):
        if report["stage"] != "second":
            continue
        passed = np.flatnonzero(data["success"])
        if passed.size:
            return int(passed[0])
    return 0


def caption(k, end, passed, readings, trial):
    calf = readings["calf"][trial, k]
    toe = readings["toe"][trial, k, 2:]
    ground = readings["ground"][trial, k]
    peak = float(readings["bad_n"][trial, :k+1].max())
    rear = readings.get("rear_calf")
    if k == end:
        phase = "验收通过" if passed else "未通过整段验收"
    elif readings["clear"][trial, k] and (ground > 5).all():
        phase = "四足已越墙并落地"
    elif readings["air"][trial, k]:
        phase = "腾空"
    elif rear is not None and (rear[trial,k] > 7).all() and (ground[2:] > 5).all():
        phase = "后双小腿接墙 · 前足着地"
    elif (calf > 7).all():
        phase = "双前小腿承重"
    elif (toe > 7).all():
        phase = "双前足承重"
    else:
        phase = "支撑转换"
    display = rear[trial,k] if rear is not None else calf
    label = "后小腿" if rear is not None else "小腿"
    torque = readings.get("torque_200hz")
    summary = f"非预定接触峰值 {peak:.0f} N"
    if torque is not None and k:
        limit = np.abs(torque[trial,:k*4]).max()
        summary += f"  ·  关节峰值 {limit:.1f} N·m"
    return (phase, f"{label} {display[0]:.0f}/{display[1]:.0f} N  地面承重 {(ground>5).sum()}/4", summary)


def render(args):
    model = make_model(args.work / "wbl_tracking.xml")
    data = mj.MjData(model)
    entries = []
    for directory in args.results:
        report = json.loads((directory / "evaluation.json").read_text())
        with np.load(directory / "rollouts.npz",allow_pickle=False) as file:
            readings = {key:file[key] for key in file.files}
        physics = directory / "physics_200hz.npz"
        if physics.exists():
            with np.load(physics,allow_pickle=False) as trace:
                readings["torque_200hz"] = trace["torque"]
        entries.append((readings, report, directory.name))
    index = choose_trial(entries,args.trial)
    if any(index < 0 or index >= report["trials"] for _,report,_ in entries):
        raise ValueError("Trial index is missing from one or more result files")
    count = len(entries)
    width, height, panel = 414, 302, 428
    frame_width, frame_height = count * panel, 476
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation = 103., -17.
    title, body, small = font(25,True), font(17,True), font(14)

    def frame(renderer, tick):
        image = Image.new("RGB",(frame_width,frame_height),"#142030")
        pen = ImageDraw.Draw(image)
        pen.text((12,10),"WBL · 30 cm 墙 · SuperMimic 闭环复验",font=title,fill="#f2f7fc")
        pen.text((13,44),f"原扰动批次第 {index+1} 个起点  |  ¼ 倍速  |  回合结束后定格",font=small,fill="#adbed0")
        for col,(record, report, label) in enumerate(entries):
            end=int(record["stop"][index])
            k=min(tick,end)
            state=record["states"][index,k]
            data.qpos[:]=state[:19]
            data.qvel[:]=state[19:]
            mj.mj_forward(model,data)
            camera.lookat[:]=[np.clip(state[0]*.60-.12,-.27,.8),0.,.23]
            camera.distance=1.48+.84*max(float(state[0])-.04,0.)
            renderer.update_scene(data,camera=camera)
            x=col*panel+7
            image.paste(Image.fromarray(renderer.render()),(x,98))
            pretty = {
                "second-warp-replay":"第二阶段 · 原开环动作",
                "second-trained-eval":"第二阶段 · SuperMimic",
                "second-refined-train-eval":"第二阶段 · 修正后 SuperMimic",
                "full-trained-eval":"完整闭环 · 从墙前起跳",
                "full-refined-eval":"完整闭环 · 修正后",
                "full-refined-strict":"完整闭环 · 修正后",
                "full-balance-eval":"完整闭环 · 落地反馈蒸馏",
                "descent-teacher":"下墙 · 原开环参考",
                "descent-eval":"下墙 · 第一版 Actor",
                "descent-refined-eval":"下墙 · 修正后 Actor",
                "descent-transfer-eval":"下墙 · 改进 Actor",
                "descent-refined-clone":"下墙 · 动作预热",
                "full-teacher":"完整四段 · 原开环参考",
                "full-eval":"完整四段 · 第一版 Actor",
                "full-transfer-eval":"完整四段 · 修正后 Actor",
                "full-transfer-0000200":"完整四段 · 最佳检查点",
                "full-balance-strict":"完整闭环 · 落地反馈蒸馏",
                "full-phase-calibrated":"完整闭环 · 分阶段残差",
                "full-touchdown-eval":"完整闭环 · 落地前双足同步",
                "second-balance-strict":"第二阶段 · 落地反馈蒸馏",
                "second-warp-strict":"第二阶段 · 原开环动作",
                "second-aligned-eval":"第二阶段 · 修正后闭环",
                "second-refined-clone-eval":"第二阶段 · 动作预热",
            }.get(label,label)
            pen.text((x,73),pretty,font=body,fill="#f2f7fc")
            passed=bool(record["success"][index])
            text, forces, impact = caption(k,end,passed,record,index)
            label_color = "#ffb184" if k == end and not passed else "#62e4a7"
            pen.text((x+3,409),f"{k/50:.2f} s    {text}",font=body,fill=label_color)
            pen.text((x+3,434),forces,font=small,fill="#adbed0")
            pen.text((x+3,453),impact,font=small,fill="#adbed0")
        return image

    args.output.parent.mkdir(parents=True,exist_ok=True)
    total = max(int(record["stop"][index]) for record,_,_ in entries)
    ffmpeg = ["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pixel_format","rgb24",
              "-video_size",f"{frame_width}x{frame_height}","-framerate","25","-i","-","-an",
              "-c:v","libx264","-threads","2","-crf","19","-pix_fmt","yuv420p","-movflags","+faststart",
              str(args.output)]
    with mj.Renderer(model,height=height,width=width) as renderer:
        frame(renderer,total).save(args.output.with_suffix(".png"))
        with subprocess.Popen(ffmpeg,stdin=subprocess.PIPE) as process:
            for tick in range(total+12):
                rgb=frame(renderer,min(tick,total)).tobytes()
                process.stdin.write(rgb)
                process.stdin.write(rgb)
            process.stdin.close()
            if process.wait():
                raise RuntimeError("Unable to encode comparison video")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work",type=Path,required=True)
    parser.add_argument("--results",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--trial",type=int)
    render(parser.parse_args())


if __name__=="__main__":
    main()
