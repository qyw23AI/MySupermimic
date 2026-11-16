"""Retarget reconstructed LVR trajectories using a bent-knee WBL stance."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from prepare_lvr_directions import mirror
from retarget_go2 import ROOT, get_robot, target_init


LEGS = ("RR", "RL", "FR", "FL")
SIDES = np.array([-1, 1, -1, 1])
FLOOR_CLEARANCE = .002


class Kinematics:
    def __init__(self, path, names):
        self.model = mj.MjModel.from_xml_path(str(path))
        self.data = mj.MjData(self.model)
        self.joints = [self.model.joint(name).qposadr[0] for name in names]
        self.feet = [
            next(i for i in range(self.model.ngeom)
                 if self.model.geom_bodyid[i] == self.model.body(leg + "_calf").id
                 and self.model.geom_type[i] == mj.mjtGeom.mjGEOM_SPHERE)
            for leg in LEGS
        ]
        self.radius = self.model.geom_size[self.feet, 0]
        self.limits = np.array([self.model.joint(name).range for name in names]).T

    def fk(self, joints, quat=(0, 0, 0, 1), root=(0, 0, 0)):
        self.data.qpos[:3] = root
        self.data.qpos[3:7] = np.asarray(quat)[[3, 0, 1, 2]]
        self.data.qpos[self.joints] = joints
        mj.mj_kinematics(self.model, self.data)
        return self.data.geom_xpos[self.feet].copy()


def fit(donor, target, rows, donor_stand, wbl_stand, multiplier):
    stand_src, stand_dst = donor.fk(donor_stand), target.fk(wbl_stand)
    scale = float(np.median(np.linalg.norm(stand_dst, axis=1) / np.linalg.norm(stand_src, axis=1))) * multiplier
    lo, hi = target.limits.copy()
    # Symmetric bounds keep the left-to-right mirror within both sets of limits.
    hip_bounds = [(-.23, .40) if side > 0 else (-.40, .23) for side in SIDES]
    lo[::3], hi[::3] = np.asarray(hip_bounds).T
    hi[2::3] = -1.05
    floor_width = np.abs(stand_dst[:, 1]) * .78
    result = rows.copy()
    result[:, :2] = (rows[:, :2] - rows[0, :2]) * scale
    previous = wbl_stand
    errors = []
    for i, row in enumerate(rows):
        delta = donor.fk(row[7:]) - stand_src
        desired = stand_dst + delta * scale * np.array([.82, .42, .72])
        desired[:, 1] = SIDES * np.maximum(desired[:, 1] * SIDES, floor_width)

        def residual(q):
            feet = target.fk(q)
            inward = np.minimum(feet[:, 1] * SIDES - floor_width, 0)
            return np.r_[8 * (feet - desired).ravel(), 16 * inward,
                         .17 * (q - wbl_stand), .10 * (q - previous)]

        previous = least_squares(residual, previous, bounds=(lo + 1e-6, hi - 1e-6),
                                 max_nfev=45).x
        result[i, 7:] = previous
        errors.append(np.linalg.norm(target.fk(previous) - desired, axis=1))
    return result, scale, float(np.quantile(errors, .95))


def ground(target, rows):
    rows = rows.copy()
    for row in rows:
        feet = target.fk(row[7:], row[3:7])
        row[2] = FLOOR_CLEARANCE - np.min(feet[:, 2] - target.radius)
    return rows


def metrics(target, rows, stand):
    local = np.array([target.fk(r[7:]) for r in rows])
    feet = np.array([target.fk(r[7:], r[3:7], r[:3]) for r in rows])
    sole = feet[:,:,2] - target.radius
    width = (local[:,:,1] * SIDES) / (stand[:,1] * SIDES)
    knees = rows[:,9::3]
    velo = Rotation.from_quat(rows[:,3:7]).inv().apply(np.gradient(rows[:,:3],.02,axis=0))
    return dict(knee_median_rad=float(np.median(knees)), knee_most_extended_rad=float(knees.max()),
                minimum_sole_height_m=float(sole.min()), below_floor_frames=int(np.sum(sole.min(axis=1)<-1e-5)),
                min_lateral_foot_offset_ratio=float(width.min()), min_lateral_foot_offset_m=float((local[:,:,1]*SIDES).min()),
                median_body_velocity_m_s=np.median(velo,axis=0).tolist(),
                joint_velocity_p99_rad_s=float(np.quantile(np.abs(np.diff(rows[:,7:],axis=0))/.02,.99)))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = ROOT / "amp_supermimic/data"
    source = data / "go2/lvr_direction_reconstructed"
    old = data / "wblwbl/direction_candidates"
    meta = json.loads((source/"manifest.json").read_text())
    names = json.loads((data.parent/"configs/wblwbl_dog_v1.plan.json").read_text())["contract"]["policy_joint_names"]
    source_asset = Path(get_robot("go2").mjcf_path)
    target_asset = ROOT.parent/"robocon2027_sim_control/deploy_cpp/robot/wblwbl_dog_v1/xml/wblwbl_dog_v1.xml"
    donor = Kinematics(source_asset, names)
    target = Kinematics(target_asset, names)
    wbl_stand = np.array([target_init()["default_joint_angles"][n] for n in names])
    donor_stand = np.array([(-.1 if n[1]=="R" else .1) if "hip" in n else
                            (.8 if n[0]=="F" else 1.) if "thigh" in n else -1.5 for n in names])
    stand = target.fk(wbl_stand)
    for suffix, multiplier in (("065", .65), ("100", 1.0)):
        for direction in ("backward", "left"):
            csv = source / f"lvr_{direction}.csv"
            raw = np.loadtxt(csv,delimiter=",")
            order = [meta["joint_names"].index(n) for n in names]
            raw = raw[:, [*range(7), *(7 + i for i in order)]]
            fitted, scale, error = fit(donor,target,raw,donor_stand,wbl_stand,multiplier)
            motions = {direction:fitted}
            if direction == "left":
                motions["right"] = mirror(fitted,names)
            for name, rows in motions.items():
                rows = ground(target,rows)
                dest = args.output/f"lvr_{name}_{suffix}"
                dest.mkdir(parents=True)
                path = dest / f"lvr_{name}.csv"
                np.savetxt(path,rows,delimiter=",",fmt="%.10g")
                rows = np.loadtxt(path,delimiter=",")
                checks = metrics(target,rows,stand)
                prior = metrics(target,np.loadtxt(old/dest.name/path.name,delimiter=","),stand)
                assert checks["below_floor_frames"] == 0 and checks["knee_most_extended_rad"] <= -1.05
                assert checks["min_lateral_foot_offset_ratio"] >= .74
                assert np.all((rows[:,7:]>=target.limits[0]) & (rows[:,7:]<=target.limits[1]))
                report = dict(frames=len(rows),fps=50,joint_names=names,quaternion_order="xyzw",
                              csv_sha256=sha(path),target_mjcf_sha256=sha(target_asset),
                              source_csv=str(csv.relative_to(ROOT)),source_sha256=sha(csv),
                              source_joint_order=meta["joint_names"],
                              source_joint_order_evidence=meta["joint_order_provenance"],
                              motion_scale_multiplier=multiplier,scale=scale,
                              fitted_left_or_backward_foot_error_p95_m=error,
                              mirrored_from_wbl_left=name=="right",status="kinematic_reference_only",
                              before=prior,after=checks)
                (dest/"report.json").write_text(json.dumps(report,indent=2)+"\n")
                print(dest.name,prior,checks,flush=True)


if __name__=="__main__":
    main()
