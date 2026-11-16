"""Render the generated A1 and WBL pose sequences without physics stepping."""
import argparse
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj
import numpy as np

from render_wblwbl_catalog import load_model


def render(folder, a1_scene):
    meta = json.loads((folder/"report.json").read_text())
    ref = np.loadtxt(folder/(folder.name+".csv"),delimiter=",")
    source = folder/"a1_preview.npy"
    models = [load_model()]
    sequences = [ref]
    if source.is_file():
        models.insert(0, mj.MjModel.from_xml_path(str(a1_scene)))
        sequences.insert(0, np.load(source,allow_pickle=False))
    joints = [[m.joint(n).qposadr[0] for n in meta["joint_names"]] for m in models]
    states = [mj.MjData(m) for m in models]
    camera = mj.MjvCamera()
    camera.distance = 1.48
    camera.azimuth = 132
    camera.elevation = -21
    w,h=432,264
    title = f"pad=iw:ih+32:0:32:color=0x151c26,drawtext=text='{folder.name}':x=10:y=8:fontsize=15:fontcolor=white"
    if len(models)==2:
        title+=",drawtext=text='A1 source | 0.5x':x=261:y=8:fontsize=15:fontcolor=0xbcd9ea,drawtext=text='WBL reference':x=444:y=8:fontsize=15:fontcolor=white"
    else:
        title+=",drawtext=text='WBL IK':x=342:y=8:fontsize=15:fontcolor=white"
    video=folder/"reference.mp4"
    fps="25" if len(models)==1 else "12.5"
    command=["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{w*len(models)}x{h}",
             "-r",fps,"-i","pipe:0","-vf",title,"-c:v","libx264","-threads","2","-preset","veryfast",
             "-pix_fmt","yuv420p","-movflags","+faststart",str(video)]
    renders=[mj.Renderer(m,width=w,height=h) for m in models]
    try:
        with subprocess.Popen(command,stdin=subprocess.PIPE) as process:
            for i in range(0,len(ref),2):
                panels=[]
                for model,state,ids,seq,renderer in zip(models,states,joints,sequences,renders):
                    row=seq[i]
                    state.qpos[:3]=row[:3]
                    state.qpos[3:7]=row[[6,3,4,5]]
                    state.qpos[ids]=row[7:]
                    mj.mj_forward(model,state)
                    camera.lookat[:]=[row[0],row[1],row[2]-.13]
                    renderer.update_scene(state,camera=camera)
                    panels.append(renderer.render().copy())
                process.stdin.write(np.concatenate(panels,axis=1).tobytes())
            process.stdin.close()
            if process.wait():
                raise RuntimeError(f"Video encoding failed: {video}")
    finally:
        for obj in renders:
            obj.close()
    print(video,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a1_scene",type=Path)
    parser.add_argument("folders",nargs="+",type=Path)
    args=parser.parse_args()
    for folder in args.folders:
        render(folder,args.a1_scene)
