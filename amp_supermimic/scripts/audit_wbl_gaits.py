"""Estimate gait from touchdown timing in flat-ground WBL motion references."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import mujoco
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation

LEGS = ("RR", "RL", "FR", "FL")
PAIRS = {"trot": ((0, 3), (1, 2)), "pace": ((0, 2), (1, 3)), "bound": ((0, 1), (2, 3))}


def measure(rows, fps, model, data, joints, feet):
    data.qvel[:] = 0
    xyz = []
    for row in rows:
        data.qpos[:3] = row[:3]
        data.qpos[3:7] = row[[6, 3, 4, 5]]
        data.qpos[joints] = row[7:]
        mujoco.mj_kinematics(model, data)
        xyz.append(data.geom_xpos[feet].copy())
    xyz = np.asarray(xyz)
    sole = xyz[:,:,2] - model.geom_size[feet,0]
    ground = model.geom("floor").pos[2]
    speed = np.linalg.norm(np.gradient(xyz[:,:,:2], 1/fps, axis=0), axis=-1)
    # Correct a constant vertical offset for phase analysis only. Report actual penetration separately.
    offset = np.quantile(sole.min(axis=1), .15) - ground
    contact = median_filter((sole < ground + offset + .027) & (speed < .65), size=(3,1))
    events = [np.flatnonzero(np.diff(c.astype(int)) == 1) + 1 for c in contact.T]
    periods = [np.diff(e) for e in events]
    valid = np.concatenate(periods)
    valid = valid[(valid >= fps*.18) & (valid <= fps*1.8)]
    phase = np.full((4,4), np.nan)
    label = 'uncertain'
    if len(valid) >= 8 and min(map(len, events)) >= 3:
        period = np.median(valid)
        for i in range(4):
            for j in range(i+1,4):
                distances = np.abs(events[i][:,None]-events[j][None,:]).min(axis=1)
                reverse = np.abs(events[j][:,None]-events[i][None,:]).min(axis=1)
                phase[i,j] = phase[j,i] = np.median(np.r_[distances,reverse])/period
        for gait, pairs in PAIRS.items():
            same = [phase[i,j] for i,j in pairs]
            other = [phase[i,j] for i in pairs[0] for j in pairs[1]]
            if max(same) <= .16 and min(other) >= .30:
                label = gait
                break
        if label == 'uncertain' and np.min(phase[np.triu_indices(4,1)]) >= .15 and np.median(contact.mean(0)) >= .40:
            label = 'walk'
    velocity = Rotation.from_quat(rows[:,3:7]).inv().apply(np.gradient(rows[:,:3],1/fps,axis=0))
    heading = Rotation.from_quat(rows[:,3:7]).as_euler('xyz')[:,2]
    ph = {LEGS[i]+'-'+LEGS[j]: round(float(phase[i,j]),3)
          for i,j in zip(*np.triu_indices(4,1)) if np.isfinite(phase[i,j])}
    return dict(gait=label, touchdown_counts=dict(zip(LEGS,map(len,events))),
                touchdown_separation_cycles=ph, duty_factor=contact.mean(axis=0).round(3).tolist(),
                body_velocity_m_s=np.median(velocity,axis=0).round(3).tolist(),
                yaw_abs_median_rad_s=round(float(np.median(np.abs(np.gradient(np.unwrap(heading),1/fps)))),3),
                penetration_over_1cm_frames=int(np.sum(sole.min(axis=1)<ground-.01)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data",type=Path)
    parser.add_argument("mjcf",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    model=mujoco.MjModel.from_xml_path(str(args.mjcf))
    data=mujoco.MjData(model)
    feet=[next(i for i in range(model.ngeom) if model.geom_bodyid[i]==model.body(leg+"_calf").id
              and model.geom_type[i]==mujoco.mjtGeom.mjGEOM_SPHERE) for leg in LEGS]
    labels={}
    by_hash={}
    for folder, dirs, files in os.walk(args.data):
        dirs[:]=sorted(d for d in dirs if d!='rubbish' and not (Path(folder)/d).is_symlink())
        for name in sorted(files):
            if not name.endswith('.csv'):
                continue
            path=Path(folder)/name
            if path.is_symlink():
                continue
            parent=path.parent
            rel=str(path.relative_to(args.data))
            report=parent/'report.json'
            manifest=parent/'train.json'
            if not (report.exists() or manifest.exists()):
                labels[rel]={'gait':'uncertain','reason':'missing reference metadata'}
                continue
            meta=json.loads((report if report.exists() else manifest).read_text())
            row_hash=hashlib.sha256(path.read_bytes()).hexdigest()
            key=(row_hash,tuple(meta['joint_names']),float(meta.get('fps',meta.get('source_fps',0))))
            source=by_hash.get(key)
            if source is None:
                joints=[model.joint(n).qposadr[0] for n in key[1]]
                rows=np.loadtxt(path,delimiter=',',ndmin=2)
                if rows.shape[1]!=19 or len(rows)<4 or not np.isfinite(rows).all():
                    labels[rel]={'gait':'uncertain','reason':'invalid reference shape or values'}
                    continue
                metrics=measure(rows,key[2],model,data,joints,feet)
                by_hash[key]=rel
            else:
                metrics={k:v for k,v in labels[source].items() if k not in ('ik_pass','pd_pass','duplicate_of')}
                metrics['duplicate_of']=source
            if report.exists():
                metrics.update(ik_pass=meta.get('motion_usable_for_amp'),pd_pass=meta.get('playback_pass'))
            labels[rel]=metrics
    args.output.write_text(json.dumps({'method':'WBL forward kinematics; inferred flat support from sole height and XY speed, not measured forces','foot_order':LEGS,'motions':labels},indent=2)+'\n')


if __name__=='__main__':
    main()
