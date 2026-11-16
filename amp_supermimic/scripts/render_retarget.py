"""Render a physical PD playback of the audited WBLWBL Go2 retarget."""
import argparse,json,os,subprocess
from pathlib import Path
os.environ.setdefault("MUJOCO_GL","egl")
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parents[2]
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--seconds",type=float,default=6)
 ap.add_argument("--output",type=Path,default=ROOT/"amp_supermimic/outputs/go2_retarget_preview.mp4")
 a=ap.parse_args()
 data_dir=ROOT/"amp_supermimic/data/wblwbl"
 report=json.loads((data_dir/"report.json").read_text())
 if not report["playback_pass"]: raise RuntimeError("Physical replay has not passed")
 rows=np.loadtxt(data_dir/"ai4_dog_trot_00.csv",delimiter=",")
 mjcf=ROOT.parent/"robocon2027_sim_control/deploy_cpp/robot/wblwbl_dog_v1/xml/wblwbl_dog_v1.xml"
 m=mujoco.MjModel.from_xml_path(str(mjcf))
 for i in range(m.ngeom):
  if m.geom_bodyid[i]==0 and m.geom(i).name!="floor":
   m.geom_contype[i]=m.geom_conaffinity[i]=0
   m.geom_rgba[i,3]=0
 m.geom("floor").rgba[:]=[.55,.64,.59,1]
 d=mujoco.MjData(m)
 names=report["joint_names"];ids=np.array([m.joint(n).id for n in names])
 q=np.array([m.jnt_qposadr[i] for i in ids]);v=np.array([m.jnt_dofadr[i] for i in ids])
 act=np.array([next(j for j in range(m.nu) if m.actuator_trnid[j,0]==i) for i in ids])
 d.qpos[:3]=rows[0,:3];d.qpos[3:7]=rows[0,[6,3,4,5]];d.qpos[q]=rows[0,7:]
 mujoco.mj_forward(m,d)
 cam=mujoco.MjvCamera()
 cam.type=mujoco.mjtCamera.mjCAMERA_FREE
 cam.distance=1.35;cam.azimuth=135;cam.elevation=-24
 renderer=mujoco.Renderer(m,height=360,width=640)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 cmd=["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pixel_format","rgb24","-video_size","640x360","-framerate","25","-i","-","-an","-c:v","libx264","-preset","veryfast","-crf","23","-pix_fmt","yuv420p",str(a.output)]
 proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
 try:
  for k in range(min(len(rows),int(a.seconds*50))):
   for _ in range(round(.02/m.opt.timestep)):
    d.ctrl[act]=np.clip(40*(rows[k,7:]-d.qpos[q])-4*d.qvel[v],-23.7,23.7)
    mujoco.mj_step(m,d)
   if k%2==0:
    cam.lookat[:]=d.qpos[:3];cam.lookat[2]=.26
    renderer.update_scene(d,camera=cam)
    proc.stdin.write(renderer.render().tobytes())
 finally:
  proc.stdin.close()
  renderer.close()
  if proc.wait()!=0: raise RuntimeError("ffmpeg failed")
 print(a.output, a.output.stat().st_size)
if __name__=="__main__": main()
