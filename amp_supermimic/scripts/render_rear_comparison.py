"""Side-by-side WBLWBL IK reference and physical replay (candidate, not validated)."""
import os,subprocess
from pathlib import Path
os.environ.setdefault("MUJOCO_GL","egl")
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parents[2]
data=ROOT/"amp_supermimic/data/wblwbl"
rows=np.loadtxt(data/"ai4_dog_trot_00.csv",delimiter=",")
m=mujoco.MjModel.from_xml_path(str(ROOT.parent/"robocon2027_sim_control/deploy_cpp/robot/wblwbl_dog_v1/xml/wblwbl_dog_v1.xml"))
for i in range(m.ngeom):
 if m.geom_bodyid[i]==0 and m.geom(i).name!="floor":
  m.geom_contype[i]=m.geom_conaffinity[i]=0
  m.geom_rgba[i,3]=0
m.geom("floor").rgba[:]=[.55,.64,.59,1]
physics=mujoco.MjData(m);reference=mujoco.MjData(m)
for d in (physics,reference):
 d.qpos[:3]=rows[0,:3];d.qpos[3:7]=rows[0,[6,3,4,5]];d.qpos[7:19]=rows[0,7:];mujoco.mj_forward(m,d)
cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.distance=1.6;cam.azimuth=145;cam.elevation=-25
renderer=mujoco.Renderer(m,height=360,width=640)
out=ROOT/"amp_supermimic/outputs/go2_retarget_full_comparison.mp4";out.parent.mkdir(parents=True,exist_ok=True)
cmd=["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pixel_format","rgb24","-video_size","1280x360","-framerate","25","-i","-","-an","-c:v","libx264","-preset","veryfast","-crf","22","-pix_fmt","yuv420p",str(out)]
p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
try:
 for k in range(len(rows)):
  for _ in range(4):
   physics.ctrl[:]=np.clip(60*(rows[k,7:]-physics.qpos[7:19])-4*physics.qvel[6:18],-23.7,23.7)
   mujoco.mj_step(m,physics)
  if k%2:continue
  reference.qpos[:3]=rows[k,:3];reference.qpos[3:7]=rows[k,[6,3,4,5]];reference.qpos[7:19]=rows[k,7:];mujoco.mj_forward(m,reference)
  frames=[]
  for d,label in ((reference,"IK reference (kinematic)"),(physics,"PD physical (60/4)")):
   cam.lookat[:]=d.qpos[:3];cam.lookat[2]=.27
   renderer.update_scene(d,camera=cam)
   frames.append(renderer.render().copy())
  p.stdin.write(np.concatenate(frames,axis=1).tobytes())
finally:
 p.stdin.close();renderer.close()
 if p.wait()!=0:raise RuntimeError("ffmpeg failed")
print(out,out.stat().st_size)
