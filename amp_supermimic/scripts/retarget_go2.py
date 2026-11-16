"""Retarget pinned Kine2Go Go2 trot to WBLWBL; audit FK and PD playback."""
import argparse, ast, hashlib, json, sys
from pathlib import Path
from xml.etree import ElementTree
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
ROOT=Path(__file__).resolve().parents[2]
UP=ROOT/"third_party/trackerlab_supermimic"
sys.path.insert(0,str(UP/"scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()
from trackerTask.supermimic.robots import get_robot
from trackerTask.supermimic.gait import FOOT_BODIES, FOOT_OFFSETS
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def target_init():
 snapshot = ROOT/"him_supermimic/outputs/a3-psv8Y8/train/config.json"
 if snapshot.is_file():
  return json.loads(snapshot.read_text())["legacy"]["init_state"]
 config = ROOT/"third_party/MyHIMLoco/isaaclab_migration/himloco_lab/legacy/wblwbl_dog_v1/wblwbl_dog_v1_config.py"
 init = next(node for node in ast.walk(ast.parse(config.read_text()))
             if isinstance(node, ast.ClassDef) and node.name == "init_state")
 return {node.targets[0].id: ast.literal_eval(node.value)
         for node in init.body if isinstance(node, ast.Assign)
         and isinstance(node.targets[0], ast.Name)
         and node.targets[0].id in {"pos", "default_joint_angles"}}

def a1_neutral_feet(urdf, legs):
 joints = {joint.attrib["name"]: joint for joint in ElementTree.parse(urdf).findall("joint")}
 feet = []
 for leg in legs:
  position = np.zeros(3)
  rotation = Rotation.identity()
  for suffix, angle in zip(("hip_joint", "thigh_joint", "calf_joint", "foot_fixed"),
                           (0.0, 0.9, -1.8, None)):
   joint = joints[f"{leg}_{suffix}"]
   origin = joint.find("origin")
   position += rotation.apply(np.fromstring(origin.attrib["xyz"], sep=" "))
   rotation *= Rotation.from_euler("xyz", np.fromstring(origin.attrib["rpy"], sep=" "))
   if angle is not None:
    axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
    rotation *= Rotation.from_rotvec(axis * angle)
  feet.append(position)
 return np.array(feet)

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--swing-boost",type=float,default=0.12)
 ap.add_argument("--motion-scale",type=float,default=1.0)
 ap.add_argument("--source",type=Path,default=ROOT/"amp_supermimic/data/go2/ai4_dog_trot_00.csv")
 ap.add_argument("--output",type=Path,default=ROOT/"amp_supermimic/data/wblwbl")
 ap.add_argument("--mjcf",type=Path,default=ROOT.parent/"robocon2027_sim_control/deploy_cpp/robot/wblwbl_dog_v1/xml/wblwbl_dog_v1.xml")
 ap.add_argument("--rotate-foot-calibration", action="store_true",
                 help="Rotate the initial foot offsets with the torso during turning.")
 ap.add_argument("--neutral-foot-calibration", action="store_true",
                 help="Map Go2 nominal stance to WBLWBL nominal stance in the torso frame.")
 ap.add_argument("--align-flat-ground", action="store_true",
                 help="Align the lower envelope of the mapped feet to the flat floor with one constant height offset.")
 source_format = ap.add_mutually_exclusive_group()
 source_format.add_argument("--optimized-json", action="store_true", help="Read a go2_flip_TO trajectory.")
 source_format.add_argument("--nju-json", action="store_true", help="Read an NJU quadrupedal-agility clip.")
 source_format.add_argument("--amp-a1-urdf", type=Path,
                            help="Read an AMP A1 JSON using recorded local toes and this A1 URDF for neutral stance.")
 a=ap.parse_args()
 source_manifest_path = None if a.optimized_json or a.nju_json or a.amp_a1_urdf else a.source.parent/"manifest.json"
 if source_manifest_path is not None:
  source_manifest=json.loads(source_manifest_path.read_text())
  rec=next(x for x in source_manifest["files"] if x["file"]==a.source.name)
  assert sha(a.source)==rec["sha256"]
 donor=get_robot("go2")
 go=mujoco.MjModel.from_xml_path(str(Path(donor.mjcf_path).resolve()))
 target=mujoco.MjModel.from_xml_path(str(a.mjcf.resolve()))
 for i in range(target.ngeom):
  if target.geom_bodyid[i]==0 and target.geom(i).name!="floor":
   target.geom_contype[i]=target.geom_conaffinity[i]=0
 gd,td=mujoco.MjData(go),mujoco.MjData(target)
 joint_names=json.loads((ROOT/"amp_supermimic/configs/wblwbl_dog_v1.plan.json").read_text())["contract"]["policy_joint_names"]
 gj=np.array([go.joint(n).qposadr[0] for n in donor.joint_names])
 tj=np.array([target.joint(n).qposadr[0] for n in joint_names])
 tv=np.array([target.joint(n).dofadr[0] for n in joint_names])
 ids=np.array([target.joint(n).id for n in joint_names])
 lo,hi=target.jnt_range[ids].T
 init=target_init()
 neutral=np.array([init["default_joint_angles"][n] for n in joint_names])
 neutral=np.clip(neutral,lo+1e-4,hi-1e-4)
 legs=("RR","RL","FR","FL")
 gbody=np.array([go.body(n).id for n in FOOT_BODIES["go2"]])
 gb_order=[FOOT_BODIES["go2"].index(next(n for n in FOOT_BODIES["go2"] if n.startswith(leg))) for leg in legs]
 tbody=np.array([target.body(leg+"_calf").id for leg in legs])
 foot_geom_ids=np.array([next(i for i in range(target.ngeom) if target.geom_bodyid[i]==b and target.geom_type[i]==mujoco.mjtGeom.mjGEOM_SPHERE) for b in tbody])
 foot_offsets=np.array([target.geom(i).pos.copy() for i in foot_geom_ids])
 def feet(model,data,bodies,offsets):
  return data.xpos[bodies]+np.einsum("nij,nj->ni",data.xmat[bodies].reshape(-1,3,3),offsets)
 g_off=np.tile(np.asarray(FOOT_OFFSETS.get("go2",(0,0,-.213))), (4,1))[gb_order]
 def gfeet(): return feet(go,gd,gbody[gb_order],g_off)
 def tfeet(): return feet(target,td,tbody,foot_offsets)
 local_toes = None
 if a.optimized_json or a.nju_json or a.amp_a1_urdf:
  payload = json.loads(a.source.read_text())
  values = np.asarray(payload["Frames"], dtype=float)
  raw = values[:, :19].copy()
  source_fps = 1 / float(payload["FrameDuration"])
  if a.amp_a1_urdf:
   if values.shape[1] < 31 or not np.isfinite(values[:, 19:31]).all():
    raise ValueError("AMP A1 frames must include four finite local toe positions")
   local_toes = values[:, 19:31].reshape(-1, 4, 3)[:, [2, 3, 0, 1]]
  if a.nju_json:
   # Upstream MotionLoader.reorder maps PyBullet FR/FL/RR/RL to Isaac FL/FR/RL/RR.
   raw = raw[:, [*range(7), 10, 11, 12, 7, 8, 9, 16, 17, 18, 13, 14, 15]]
   raw[:, [7, 10, 13, 16]] *= -1
 else:
  raw=np.loadtxt(a.source,delimiter=",",ndmin=2)
  source_fps = float(source_manifest.get("source_fps", 60))
  if source_manifest.get("joint_names"):
   names = source_manifest["joint_names"]
   raw = raw[:, [*range(7), *(7 + names.index(name) for name in donor.joint_names)]]
  if source_manifest.get("quaternion_order", "xyzw") != "xyzw":
   raise ValueError("CSV source quaternions must be stored as xyzw")
 assert raw.shape[1]==19 and np.isfinite(raw).all()
 gd.qpos[:3]=raw[0,:3];gd.qpos[3:7]=raw[0,[6,3,4,5]];gd.qpos[gj]=raw[0,7:];mujoco.mj_forward(go,gd)
 g_rest=gfeet()-gd.qpos[:3]
 if a.neutral_foot_calibration:
  gd.qpos[3:7]=[1,0,0,0]
  gd.qpos[gj]=donor.default_q
  mujoco.mj_forward(go,gd)
  g_rest=gfeet()-gd.qpos[:3]
 if local_toes is not None:
  g_rest = a1_neutral_feet(a.amp_a1_urdf, legs)
 td.qpos[:3]=init["pos"][:3];td.qpos[3:7]=[1,0,0,0];td.qpos[tj]=neutral;mujoco.mj_forward(target,td)
 t_rest=tfeet()-td.qpos[:3]
 scale=np.median(np.linalg.norm(t_rest,axis=1)/np.maximum(np.linalg.norm(g_rest,axis=1),1e-4))*a.motion_scale
 duration = (len(raw)-1)/source_fps
 t = np.minimum(np.arange(int(np.floor(duration*50+1e-9))+1)/50, duration)
 source_t=np.arange(len(raw))/source_fps
 pos=np.stack([np.interp(t,source_t,raw[:,i]) for i in range(3)],axis=1)
 quat=Slerp(source_t,Rotation.from_quat(raw[:,3:7]))(t).as_quat()
 dq=np.stack([np.interp(t,source_t,raw[:,7+i]) for i in range(12)],axis=1)
 if local_toes is not None:
  toes = np.stack([np.interp(t, source_t, column)
                   for column in local_toes.reshape(-1, 12).T], axis=1).reshape(-1, 4, 3)
 root_path = scale * (pos - pos[0])
 root_path[:, 2] += init["pos"][2]
 desired_feet=[]
 swing_offsets=[]
 for k in range(len(t)):
  if local_toes is None:
   gd.qpos[:3]=pos[k];gd.qpos[3:7]=quat[k,[3,0,1,2]];gd.qpos[gj]=dq[k];mujoco.mj_forward(go,gd)
   donor_feet=gfeet()
  else:
   donor_feet=pos[k]+Rotation.from_quat(quat[k]).apply(toes[k])
  offset = t_rest - scale*g_rest
  if a.neutral_foot_calibration or local_toes is not None:
   offset = Rotation.from_quat(quat[k]).apply(offset)
  elif a.rotate_foot_calibration:
   offset = (Rotation.from_quat(quat[k]) * Rotation.from_quat(quat[0]).inv()).apply(offset)
  desired = scale*(donor_feet-pos[k]) + offset
  desired_feet.append(desired)
  swing_offsets.append(a.swing_boost*np.clip((donor_feet[:,2]-.02)/.05,0,1))
 desired_feet = np.array(desired_feet)
 ground_shift = 0.0
 if a.align_flat_ground:
  foot_radii = target.geom_size[foot_geom_ids, 0]
  sole_heights = root_path[:, 2, None] + desired_feet[:, :, 2] - foot_radii
  ground_shift = -float(np.quantile(sole_heights.min(axis=1), 0.10))
  root_path[:, 2] += ground_shift
 else:
  root_path[:, 2] = np.maximum(root_path[:, 2], .20)
 rows=[];errors=[]
 for k, root in enumerate(root_path):
  td.qpos[:3]=root;td.qpos[3:7]=quat[k,[3,0,1,2]]
  wanted=root+desired_feet[k]
  ground = target.geom_size[foot_geom_ids, 0] if a.align_flat_ground else .024
  wanted[:,2]=np.maximum(wanted[:,2],ground)+swing_offsets[k]
  def residual(q):
   td.qpos[tj]=q;mujoco.mj_forward(target,td)
   return np.r_[((tfeet()-wanted)*8).ravel(),(q-neutral)*.035,(q-(rows[-1][7:] if rows else neutral))*.05]
  guess=rows[-1][7:] if rows else neutral
  fit=least_squares(residual,guess,bounds=(lo+1e-5,hi-1e-5),max_nfev=25,ftol=1e-5)
  q=fit.x;td.qpos[tj]=q;mujoco.mj_forward(target,td)
  errors.append(np.linalg.norm(tfeet()-wanted,axis=1))
  rows.append(np.r_[root,quat[k],q])
 rows=np.asarray(rows);errors=np.asarray(errors)
 out=a.output;out.mkdir(parents=True,exist_ok=True)
 csv=out/a.source.with_suffix(".csv").name
 np.savetxt(csv,rows,delimiter=",",fmt="%.9g")
 report={"source_sha256":sha(a.source),"source_manifest_sha256":sha(source_manifest_path) if source_manifest_path else None,"target_mjcf_sha256":sha(a.mjcf),"frames":len(rows),"fps":50,"joint_names":list(joint_names),"scale":float(scale),"swing_boost_m":a.swing_boost,"foot_error_mean_m":float(errors.mean()),"foot_error_p95_m":float(np.quantile(errors,.95)),"foot_error_max_m":float(errors.max()),"joint_limit_violation_rad":float(np.maximum(lo-rows[:,7:],rows[:,7:]-hi).clip(min=0).max()),"joint_velocity_p99_rad_s":float(np.quantile(np.abs(np.diff(rows[:,7:],axis=0)*50),.99)),"csv_sha256":sha(csv),"playback_torque_limits_nm":None,"playback_kp":40,"playback_kd":4}
 # PD reference tracking in the exact deployment MJCF, with flat floor and initial recorded pose.
 td=mujoco.MjData(target);td.qpos[:3]=rows[0,:3];td.qpos[3:7]=rows[0,[6,3,4,5]];td.qpos[tj]=rows[0,7:];mujoco.mj_forward(target,td)
 actuator=np.array([next(i for i in range(target.nu) if target.actuator_trnid[i,0]==j) for j in ids])
 torque_limits=np.where(np.diff(target.actuator_ctrlrange[actuator],axis=1).ravel()>0,
                        np.max(np.abs(target.actuator_ctrlrange[actuator]),axis=1),23.7)
 report["playback_torque_limits_nm"]=torque_limits.tolist()
 report["flat_ground_height_shift_m"] = ground_shift
 report["foot_error_p95_by_leg_m"] = dict(zip(legs, np.quantile(errors, .95, axis=0).tolist()))
 report["foot_calibration"] = (
  "a1_local_toes" if local_toes is not None else "torso_neutral" if a.neutral_foot_calibration else
  "rotated_first_frame" if a.rotate_foot_calibration else "world_first_frame"
 )
 deviations=[];falls=0;support_frames=0;support_foot_counts=[];actual_roots=[];actual_quats=[];actual_joints=[];actual_controls=[];foot_contacts=[]
 for k in range(len(rows)):
  for _ in range(max(1,round(.02/target.opt.timestep))):
   td.ctrl[actuator]=np.clip(40*(rows[k,7:]-td.qpos[tj])-4*td.qvel[tv],-torque_limits,torque_limits)
   mujoco.mj_step(target,td)
  support=set()
  for contact in td.contact:
   if target.geom(contact.geom1).name=="floor":
    other=contact.geom2
   elif target.geom(contact.geom2).name=="floor":
    other=contact.geom1
   else: continue
   if other in foot_geom_ids: support.add(int(other))
  foot_contacts.append([int(i in support) for i in foot_geom_ids])
  actual_roots.append(td.qpos[:3].copy());actual_quats.append(td.qpos[[4,5,6,3]].copy());actual_joints.append(td.qpos[tj].copy());actual_controls.append(td.ctrl[actuator].copy())
  support_frames+=bool(support)
  support_foot_counts.append(len(support))
  deviations.append(np.linalg.norm(td.qpos[tj]-rows[k,7:])/np.sqrt(12))
  falls+=bool(td.qpos[2]<.15 or not np.isfinite(td.qpos).all())
 actual_roots=np.asarray(actual_roots);actual_joints=np.asarray(actual_joints);actual_controls=np.asarray(actual_controls);foot_contacts=np.asarray(foot_contacts)
 ref_displacement=rows[-1,:2]-rows[0,:2]
 actual_displacement=actual_roots[-1,:2]-actual_roots[0,:2]
 progress=float(np.dot(actual_displacement,ref_displacement)/max(np.dot(ref_displacement,ref_displacement),1e-8))
 report.update({"reference_displacement_xy_m":ref_displacement.tolist(),"playback_displacement_xy_m":actual_displacement.tolist(),"playback_progress_ratio":progress,"playback_joint_rmse_by_leg_rad":np.sqrt(np.mean((actual_joints-rows[:,7:])**2,axis=0)).reshape(4,3).tolist(),"playback_foot_contact_fraction_by_leg":foot_contacts.mean(0).tolist(),"playback_contact_frame_fraction":support_frames/len(rows),"playback_mean_support_feet":float(np.mean(support_foot_counts)),"playback_joint_rmse_rad":float(np.sqrt(np.mean(np.square(deviations)))),"playback_fall_frames":falls,"playback_final_height_m":float(td.qpos[2]),"playback_pass":bool(falls==0 and progress>.5 and support_frames/len(rows)>.5 and np.sqrt(np.mean(np.square(deviations)))<.35 and errors.max()<.08)})
 report["motion_usable_for_amp"] = bool(report["foot_error_p95_m"] < 0.05 and report["joint_limit_violation_rad"] < 0.02 and report["joint_velocity_p99_rad_s"] < 20)
 for key, quats in (("reference", rows[:,3:7]), ("playback", actual_quats)):
  rotation = Rotation.from_quat(quats).as_matrix()
  heading = np.unwrap(np.arctan2(rotation[:,1,0], rotation[:,0,0]))
  report[f"{key}_heading_change_deg"] = float(np.rad2deg(heading[-1] - heading[0]))
 np.savez_compressed(out/"playback.npz",root=actual_roots,root_quat_xyzw=actual_quats,joints=actual_joints,controls=actual_controls,foot_contacts=foot_contacts)
 (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
 if report["motion_usable_for_amp"]:
  (out/"train.json").write_text(json.dumps({"version":1,"root":".","source_fps":50,"quaternion_order":"xyzw","joint_names":list(joint_names),"files":[csv.name]},indent=2)+"\n")
 else:
  (out/"train.json").unlink(missing_ok=True)
 print(json.dumps(report,indent=2))
if __name__=="__main__": main()
