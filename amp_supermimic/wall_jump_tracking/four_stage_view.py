"""Render side-by-side MuJoCo replays recorded by four_stage_replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw

from .brace_experiment import CONTROL_DT, make_model
from .brace_view import font


def describe(key, report, frame):
    if key == "direct":
        if frame < report["unload_frame"]:
            return "前小腿支撑与补步"
        return "直接越墙 · 后足下到地面"
    handoff = report["first_handoff_frame"]
    catch = report["rear_shin_catch_frame"]
    if handoff and frame <= handoff:
        return "① 第一次跳跃 · 前腿登墙"
    if frame <= report["unload_frame"]:
        return "② 补步 · 机身靠墙"
    if not catch or frame <= catch:
        return "③ 第二次跳跃 · 后小腿接墙"
    if frame >= report["cut_frame"]:
        return "四足稳定落地" if report["step_down_stable"] else "后足落地 · 未达到稳定门槛"
    return "④ 向前转移 · 后足依次下墙"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    reports = json.loads((args.results/"evaluation.json").read_text())
    model = make_model(args.asset)
    data = mj.MjData(model)
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation = 103., -18.
    clips = []
    titles = {"direct":"原方案 · 直接越墙", "split":"新方案 · 同一起点对照", "continuous":"新方案 · 从墙前连续四段"}
    for key, report in reports.items():
        with np.load(args.results/f"{key}-replay.npz",allow_pickle=False) as file:
            clips.append((key,report,{k:file[k] for k in ("qpos","front_wall","rear_wall","floor")}))
    width,height,panel = 414,296,428
    size = (panel*len(clips),480)
    max_frames = max(report["cut_frame"] for _,report,_ in clips)
    title, body, small = font(24,True), font(17,True), font(13)
    white, muted, green = "#eef4f8", "#a9bccc", "#68e7a7"

    def frame(renderer, tick):
        image=Image.new("RGB",size,"#14202d")
        draw=ImageDraw.Draw(image)
        draw.text((11,9),"WBL · 30 cm 墙 · 分段接墙与直接越墙对照",font=title,fill=white)
        draw.text((12,44),"¼ 倍速   |   左、中从同一支撑姿态开始   |   各列在本回合末帧定格",font=small,fill=muted)
        for column,(key,report,record) in enumerate(clips):
            t=min(tick,report["cut_frame"])
            xy=record["qpos"][t,:3]
            data.qpos[:]=record["qpos"][t]
            mj.mj_forward(model,data)
            camera.lookat[:]=[np.clip(xy[0]*.76-.15,-.32,.79),0.,.23]
            camera.distance=1.51+.57*max(float(xy[0])-.02,0.)
            renderer.update_scene(data,camera=camera)
            x=column*panel+7
            draw.text((x,73),titles[key],font=body,fill=white)
            image.paste(Image.fromarray(renderer.render()),(x,100))
            draw.text((x,405),f"{t*CONTROL_DT:.2f} s   {describe(key,report,t)}",font=body,
                      fill=green if t==report["cut_frame"] and report["step_down_stable"] else white)
            if t:
                shin=record["rear_wall"][t-1]
                floor=record["floor"][t-1]
                draw.text((x,431),f"后小腿墙顶   {shin[0]:.0f} / {shin[1]:.0f} N",font=small,fill=muted)
                draw.text((x,452),f"地面承重   前足 {(floor[2:]>5).sum()}/2   后足 {(floor[:2]>5).sum()}/2",font=small,fill=muted)
        return image

    args.output.parent.mkdir(parents=True,exist_ok=True)
    ffmpeg=["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pixel_format","rgb24",
            "-video_size",f"{size[0]}x{size[1]}","-framerate","25","-i","-","-an",
            "-c:v","libx264","-threads","2","-crf","19","-pix_fmt","yuv420p","-movflags","+faststart",str(args.output)]
    with mj.Renderer(model,height=height,width=width) as renderer:
        snapshots=[report["rear_shin_catch_frame"] for _,report,_ in clips if report["rear_shin_catch_frame"]]
        frame(renderer,max(snapshots,default=max_frames)).save(args.output.with_suffix(".png"))
        with subprocess.Popen(ffmpeg,stdin=subprocess.PIPE) as process:
            for tick in range(max_frames+12):
                pixels=frame(renderer,min(tick,max_frames)).tobytes()
                process.stdin.write(pixels)
                process.stdin.write(pixels)
            process.stdin.close()
            if process.wait():
                raise RuntimeError("FFmpeg failed to encode the video")


if __name__ == "__main__":
    main()
