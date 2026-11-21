"""Render an annotated comparison from independent MuJoCo replays."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .brace_experiment import CONTROL_DT, SUBSTEPS, make_model
from .brace_replay import replay

FONT_ROOT = Path("/usr/share/fonts/opentype/noto")
W, H = 412, 290
PANEL = W + 12
GREEN, WHITE, MUTED, AMBER = "#5fe29d", "#eff3f7", "#a8b7c6", "#ffcf7a"


def font(size, bold=False):
    path = FONT_ROOT / f"NotoSansCJK-{'Bold' if bold else 'Regular'}.ttc"
    return ImageFont.truetype(str(path), size, index=2)


def milestone(report, frame):
    if report.get("second") and frame >= report["second"]["start_frame"]:
        second = report["second"]
        if second["stable_landing"] and frame >= second["stable_landing_frame"]:
            return "越墙并稳定落地", GREEN
        return "再次起跳与落地", WHITE
    if report["unload_frame"] and frame >= report["unload_frame"]:
        return "小腿已卸载，后足仍承重", GREEN
    if report["rear_advance_frame"] and frame >= report["rear_advance_frame"]:
        return "补步与机身前移通过", GREEN
    return "支撑转换与补步", WHITE


def draw_frame(model, data, renderer, camera, clips, tick, fonts):
    heading, body, small = fonts
    image = Image.new("RGB", (PANEL * len(clips), 472), "#131e2a")
    draw = ImageDraw.Draw(image)
    draw.text((12,9), "WBL · 30 cm 墙 · MuJoCo 连续动力学回放", font=heading, fill=WHITE)
    draw.text((12,45), "¼ 倍速  |  前两栏从同一姿态开始  |  定格表示该段验收结束", font=small, fill=MUTED)
    for rank,(title, report, results, initial, first_tick) in enumerate(clips):
        x = rank * PANEL + 6
        frame = min(first_tick + tick, report["accepted_cut_frame"])
        index = frame - 1
        state = initial[1:38] if frame == 0 else results["state"][index]
        data.qpos[:] = state[:model.nq]
        data.qvel[:] = state[model.nq:]
        mj.mj_forward(model,data)
        camera.lookat[:] = [np.clip(state[0] * .60 - .12, -.25, .75), 0., .22]
        camera.distance = 1.5 + .85 * max(float(state[0]) - .05, 0.)
        renderer.update_scene(data, camera=camera)
        image.paste(Image.fromarray(renderer.render()), (x,101))
        draw.text((x+4,75),title,font=body,fill=WHITE)
        label,color=milestone(report,frame)
        if frame and results["raw_bad_force"][:frame*SUBSTEPS].max()>5.:
            label,color="出现非预定碰撞",AMBER
        if frame:
            toe, calf, rear = (results[key][index] for key in ("toe","calf","rear"))
            delta_base = results["base_delta"][index]*100.
            delta_rear = results["rear_delta"][index]*100.
        else:
            toe = calf = rear = np.zeros(2)
            delta_base = delta_rear = 0.
        draw.text((x+4,396),f"{frame*CONTROL_DT:.2f} s    {label}",font=body,fill=color)
        draw.text((x+4,422),f"墙顶前足 {toe[0]:.0f}/{toe[1]:.0f} N  小腿 {calf[0]:.0f}/{calf[1]:.0f} N  后足 {rear[0]:.0f}/{rear[1]:.0f} N",
                  font=small,fill=MUTED)
        draw.text((x+4,444),f"机身前移 {delta_base:+.1f} cm    较少一侧后脚 {delta_rear:+.1f} cm",font=small,fill=MUTED)
    return image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work",type=Path,required=True)
    parser.add_argument("--toe",type=Path,required=True)
    parser.add_argument("--calf",type=Path,required=True)
    parser.add_argument("--second",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    model=make_model(args.work/"input/wbl_tracking.xml")
    inputs=[("足端支撑对照",args.toe,"toe"),("双小腿搭墙、补步到卸载",args.calf,"calf")]
    if args.second:
        inputs.append(("小腿卸载后接第二跳",args.second,"calf"))
    clips=[]
    for label,path,mode in inputs:
        report, values, initial, _=replay(model,path,mode)
        start=report["second"]["start_frame"] if report.get("second") else 0
        clips.append((label,report,values,initial,start))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    camera=mj.MjvCamera()
    camera.type=mj.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance=103.,-17.,1.50
    data=mj.MjData(model)
    fonts=(font(25,True),font(17,True),font(14))
    command=["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pixel_format","rgb24",
             "-video_size",f"{PANEL * len(clips)}x472","-framerate","25","-i","-","-an",
             "-c:v","libx264","-threads","2","-crf","19","-pix_fmt","yuv420p","-movflags","+faststart",str(args.output)]
    total=max(report["accepted_cut_frame"]-start for _,report,_,_,start in clips)
    with mj.Renderer(model,height=H,width=W) as renderer:
        still=draw_frame(model,data,renderer,camera,clips,total,fonts)
        still.save(args.output.with_suffix(".png"))
        with subprocess.Popen(command,stdin=subprocess.PIPE) as process:
            for i in range(total+12):
                pixels=draw_frame(model,data,renderer,camera,clips,min(i,total),fonts).tobytes()
                process.stdin.write(pixels)
                process.stdin.write(pixels)
            process.stdin.close()
            if process.wait():
                raise RuntimeError("FFmpeg did not finish writing the video")


if __name__=="__main__":
    main()
