"""Build and audit WBL kinematic candidates from local A1 AMP references."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from retarget_go2 import ROOT, target_init
from retarget_lvr_reference import Kinematics, LEGS, SIDES, metrics

SOURCE_LEGS = ("FL", "FR", "RL", "RR")
NAMES = [f"{leg}_{kind}_joint" for leg in LEGS for kind in ("hip", "thigh", "calf")]
ORDER = [SOURCE_LEGS.index(leg) for leg in LEGS]
FPS = 50


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resample(values, dt, time_factor):
    time = np.arange(len(values)) * dt * time_factor
    output_time = np.arange(int(time[-1] * FPS) + 1) / FPS
    resampled = np.column_stack([np.interp(output_time, time, value) for value in values.T])
    resampled[:, 3:7] = Slerp(time, Rotation.from_quat(values[:, 3:7]))(output_time).as_quat()
    return resampled


def solve_clip(model, root, target_feet, neutral, minimum_width):
    lo, hi = model.limits.copy()
    hi[2::3] = np.minimum(hi[2::3], -.85)
    prev = neutral
    pose, errors = [], []
    for row, feet in zip(root, target_feet):
        row = row.copy()
        def residual(q):
            local = model.fk(q)
            world = model.fk(q, row[3:7], row[:3])
            inward = np.minimum(0, SIDES * local[:, 1] - minimum_width)
            return np.r_[12 * (world - feet).ravel(), 12 * inward, .055*(q-neutral), .065*(q-prev)]

        prev = least_squares(residual, prev, bounds=(lo+1e-7, hi-1e-7), max_nfev=32).x
        pose.append(np.r_[row, prev])
        errors.append(np.linalg.norm(model.fk(prev, row[3:7], row[:3])-feet,axis=1))
    pose = np.asarray(pose)
    required = np.array([max(0., .001-np.min(model.fk(r[7:],r[3:7],r[:3])[:,2]-model.radius))
                         for r in pose])
    # A smooth envelope avoids per-frame changes at the lowest-foot switch.
    lift = gaussian_filter1d(maximum_filter1d(required, 9, mode="nearest"), 1., mode="nearest")
    pose[:,2] += np.maximum(required,lift)
    return pose, float(np.quantile(errors,.95))


def candidates(a1, wbl, folder, neutral):
    source_stand = np.array([(.1 if leg[1]=="L" else -.1) if kind=="hip" else
                             (.8 if leg[0]=="F" else 1.) if kind=="thigh" else -1.5
                             for leg in LEGS for kind in ("hip","thigh","calf")])
    stand = wbl.fk(neutral)
    a1_feet = a1.fk(source_stand)
    for path in sorted(folder.glob("*.txt")):
        doc = json.loads(path.read_text())
        raw = np.asarray(doc["Frames"], dtype=float)
        if raw.shape[1] != 61 or not np.isfinite(raw).all():
            raise ValueError(f"Invalid A1 AMP frame layout: {path}")
        q = raw[:,7:19].reshape(-1,4,3)[:,ORDER].reshape(-1,12)
        toes = raw[:,19:31].reshape(-1,4,3)[:,ORDER]
        fk_error = float(np.max(np.linalg.norm(np.array([a1.fk(x) for x in q])-toes,axis=-1)))
        if fk_error > .001:
            raise ValueError(f"A1 joint order / toe FK mismatch: {path}: {fk_error}")
        factor = 1.6 if "reverse" in path.stem else 1.
        rows = resample(np.c_[raw[:,:7],q,toes.reshape(-1,12)],doc["FrameDuration"],factor)
        # Anatomical scale and travel scale are independent. Preserve wide, rhythmic swings.
        local = stand + (rows[:,19:31].reshape(-1,4,3)-a1_feet)*[1.05,.95,1.1]
        rotation = Rotation.from_quat(rows[:,3:7])
        root = rows[:,:7].copy()
        root[:,:2] = .85*(root[:,:2]-root[0,:2])
        root[:,2] = -np.min(stand[:,2]-wbl.radius)+1.05*(root[:,2]-np.median(raw[:,2]))
        targets = np.einsum('tij,tkj->tki',rotation.as_matrix(),local)
        targets += root[:,None,:3]
        targets[:,:,2] = np.maximum(targets[:,:,2],wbl.radius+.002)
        result, error = solve_clip(wbl,root,targets,neutral,np.abs(stand[:,1])*.65)
        yield "a1_"+path.stem,result,rows[:,:19],dict(
            source_file=str(path),source_sha256=hash_file(path),source_foot_fk_max_error_m=fk_error,
            source_leg_order=SOURCE_LEGS,source_kind="time_reversed_A1_mocap" if "reverse" in path.stem else "A1_amp",
            source_time_stretch=factor,target_foot_fit_p95_m=error)


def synthetic_side(model, neutral, sign):
    stand = model.fk(neutral)
    t = np.arange(200)/FPS
    cycle, duty, speed = .60,.60,sign*.33
    phase = (t[:,None]/cycle + [0,.5,.5,0]) % 1
    step = np.floor(t[:,None]/cycle + [0,.5,.5,0])
    land = (step-np.array([0,.5,.5,0]))*cycle
    u = np.clip((phase-duty)/(1-duty),0,1)
    transfer = u**3*(10-15*u+6*u*u)
    toes = np.broadcast_to(stand,(len(t),4,3)).copy()
    toes[:,:,1] += speed*(land + duty*cycle/2 + cycle*transfer)
    toes[:,:,2] = model.radius + .003 + .055*np.sin(np.pi*u)**2
    root = np.zeros((len(t),7))
    root[:,1] = speed*t
    root[:,2] = -np.min(stand[:,2]-model.radius)-.008
    root[:,6] = 1
    result,error = solve_clip(model,root,toes,neutral,np.abs(stand[:,1])*.65)
    return result,dict(source_kind="procedural_diagonal_contact_plan",target_foot_fit_p95_m=error,
                       command_y_m_s=speed,cycle_s=cycle,duty_factor=duty)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1-mjcf", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asset = ROOT.parent/"robocon2027_sim_control/deploy_cpp/robot/wblwbl_dog_v1/xml/wblwbl_dog_v1.xml"
    a1, wbl = Kinematics(args.a1_mjcf,NAMES),Kinematics(asset,NAMES)
    neutral = np.array([target_init()["default_joint_angles"][n] for n in NAMES])
    stand = wbl.fk(neutral)
    entries = list(candidates(a1,wbl,args.source,neutral))
    for name,sign in [("left",1),("right",-1)]:
        rows,report=synthetic_side(wbl,neutral,sign)
        entries.append(("procedural_strafe_"+name,rows,None,report))
    for name,rows,source,report in entries:
        folder=args.output/name
        folder.mkdir(parents=True)
        csv=folder/(name+".csv")
        np.savetxt(csv,rows,delimiter=",",fmt="%.10g")
        rows = np.loadtxt(csv,delimiter=",")
        summary = metrics(wbl,rows,stand)
        assert not summary["below_floor_frames"]
        assert np.all((rows[:,7:]>=wbl.limits[0]) & (rows[:,7:]<=wbl.limits[1]))
        report.update(summary,frames=len(rows),fps=FPS,joint_names=NAMES,quaternion_order="xyzw",
                      csv_sha256=hash_file(csv),target_mjcf_sha256=hash_file(asset),status="kinematic_candidate")
        if source is not None:
            np.save(folder/"a1_preview.npy",source)
        (folder/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        print(name,report["frames"],np.round(summary["median_body_velocity_m_s"],3),round(report["target_foot_fit_p95_m"],4),flush=True)


if __name__=="__main__":
    main()
