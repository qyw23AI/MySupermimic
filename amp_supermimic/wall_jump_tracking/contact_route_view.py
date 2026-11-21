"""Render the recorded free-base WBL wall maneuver with load and pitch overlays."""
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
from .contact_route import STEPS

CAPTIONS = ("① 靠近起跳，前小腿直接搭墙", "② 后脚补步，下蹲准备",
            "③ 二跳：前脚着地，后小腿接墙", "④ 前移，四足着地")


def render(asset, run):
    report = json.loads((run/"evaluation.json").read_text())
    with np.load(run/"replay.npz",allow_pickle=False) as f:
        records = {key:f[key] for key in ("qpos","front","rear","floor","pitch")}
    double_belly = report.get("double_belly_route",False)
    crawl = "belly_contact_frame" in report or double_belly
    if crawl:
        with np.load(run/"replay.npz",allow_pickle=False) as f:
            belly = f["belly"]
    first_only = report.get("first_jump_only",False)
    second_preview = report.get("second_launch_preview",False)
    model = make_model(asset)
    data = mj.MjData(model)
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation = 91., -12.
    camera.distance = 1.53
    width, height, scene_height = 800, 534, 396
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = scene_height
    ends = (() if crawl else
            tuple(report["stage_frames"].get(name,len(records["qpos"])) for name in STEPS))
    last = len(records["qpos"])-1
    font_title, font_main, font_small = font(27,True),font(20,True),font(17)
    bg, white, muted, accent = "#14202d","#eef4f8","#a9bccc","#68e7a7"
    args = ["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pixel_format","rgb24",
            "-video_size",f"{width}x{height}","-framerate","25","-i","-","-an",
            "-c:v","libx264","-threads","2","-crf","19","-pix_fmt","yuv420p","-movflags","+faststart",
            str(run/"replay.mp4")]
    with mj.Renderer(model,height=scene_height,width=width) as renderer, subprocess.Popen(args,stdin=subprocess.PIPE) as ff:
        for frame in range(last+1):
            pose = records["qpos"][frame]
            data.qpos[:] = pose
            mj.mj_forward(model,data)
            camera.lookat[:] = [np.clip(.73*pose[0]+.04,-.37,.57),0.,.27]
            renderer.update_scene(data,camera=camera)
            image = Image.new("RGB",(width,height),bg)
            image.paste(Image.fromarray(renderer.render()),(0,48))
            draw = ImageDraw.Draw(image)
            title = "WBL · 第一跳 · 前腿主推" if first_only else "WBL · 30 cm 窄墙 · 四段动作参考"
            if second_preview:
                title = ("二跳 · 前小腿与后脚协同发力" if report["variant"]=="shins-push"
                         else "二跳 · 前腿先伸，后腿接着推")
            if crawl:
                title = "WBL · 两跳换支撑 · 前腿牵引" if double_belly else "WBL · 机身托墙 · 前腿牵引下墙"
            draw.text((13,8),title,font=font_title,fill=white)
            rate = "¼" if first_only else "½"
            draw.text((620,13),f"{frame*CONTROL_DT:5.2f} s  |  {rate} 倍速",font=font_small,fill=muted)
            phase = min(3,sum(frame>end for end in ends))
            label = CAPTIONS[phase]
            if frame == last:
                if second_preview:
                    label = "通过接墙验收" if report["catch_qualified"] else "起跳可行，接墙未稳"
                elif first_only:
                    label = "前双小腿稳定搭墙" if report["first_jump_passed"] else "未达第一跳验收"
                else:
                    label += "  ·  " + ("四足稳定承重" if report["passed"] else "未达本段验收")
            if double_belly:
                events=report["events"]
                if frame<=events.get("front_body_mount",last):
                    label="① 第一跳，机身前部上墙"
                elif frame<=events.get("rear_body_catch",last):
                    label="② 前脚伸出，后脚蹬地，机身后部接墙"
                else:
                    label="③ 前脚牵引，后脚下墙"
                if frame==last:
                    label="四脚稳定落地" if report["passed"] else "候选回放 · 整段动作尚未通过"
            elif crawl:
                if frame<=report["first_shins_frame"]:
                    label = "① 第一跳，前双小腿搭墙"
                elif frame<=report["brace_frame"]:
                    label = "② 后腿补步，准备第二跳"
                elif frame<=(report["belly_contact_frame"] or last):
                    label = "③ 前脚伸过墙沿，后脚推动机身"
                else:
                    label = "④ 机身靠墙，前腿向墙后牵引"
                if frame==last:
                    label = "四脚稳定落地" if report["passed"] else "候选动作 · 下墙尚未通过"
            draw.text((12,451),label,font=font_main,fill=accent if frame in ends else white)
            if frame:
                front,rear,floor,pitch = (records[key][frame-1] for key in ("front","rear","floor","pitch"))
                if crawl:
                    draw.text((12,481),f"机身墙顶载荷 {belly[frame-1]:.0f} N    前小腿墙顶载荷 {front.sum():.0f} N",
                              font=font_small,fill=muted)
                    draw.text((12,504),f"前脚地面载荷 {floor[2:].sum():.0f} N    后脚地面载荷 {floor[:2].sum():.0f} N",
                              font=font_small,fill=muted)
                elif second_preview:
                    draw.text((12,481),f"前小腿墙顶载荷 {front[0]:.0f} / {front[1]:.0f} N    后脚地面载荷 {floor[0]:.0f} / {floor[1]:.0f} N",
                              font=font_small,fill=muted)
                    draw.text((12,504),f"前脚地面载荷 {floor[2]:.0f} / {floor[3]:.0f} N    后小腿墙顶载荷 {rear[0]:.0f} / {rear[1]:.0f} N",
                              font=font_small,fill=muted)
                elif first_only:
                    rear_knees = -np.rad2deg(pose[[9,12]])
                    draw.text((12,481),f"地面竖直载荷    前足合计 {floor[2:].sum():.0f} N    后足合计 {floor[:2].sum():.0f} N",
                              font=font_small,fill=muted)
                    draw.text((12,504),f"后膝弯曲 {rear_knees[0]:.0f}° / {rear_knees[1]:.0f}°    前小腿墙顶载荷 {front[0]:.0f} / {front[1]:.0f} N",
                              font=font_small,fill=muted)
                else:
                    draw.text((12,481),f"墙顶载荷  前小腿 {front[0]:.0f}/{front[1]:.0f} N    后小腿 {rear[0]:.0f}/{rear[1]:.0f} N",
                              font=font_small,fill=muted)
                    draw.text((12,504),f"地面承重   前足 {(floor[2:]>5).sum()}/2   后足 {(floor[:2]>5).sum()}/2      机头下俯 {np.rad2deg(pitch):+.1f}°",
                              font=font_small,fill=muted)
            if frame in report.get("events" if crawl else "stage_frames",{}).values() or frame==last:
                image.save(run/f"frame-{frame:03d}.png")
            ff.stdin.write(image.tobytes())
            if first_only:
                ff.stdin.write(image.tobytes())
        for _ in range(25):
            ff.stdin.write(image.tobytes())
        ff.stdin.close()
        if ff.wait():
            raise RuntimeError("Could not encode the MuJoCo replay")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset",type=Path,required=True)
    parser.add_argument("--run",type=Path,required=True)
    args=parser.parse_args()
    render(args.asset,args.run)
