"""Load the PhysX robot contract and share its URDF kinematics with TrackerLab."""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def xyz(element, key):
    return tuple(map(float, element.get(key, "0 0 0").split())) if element is not None else (0., 0., 0.)


@dataclass(frozen=True)
class Robot:
    path: Path
    root: str
    joints: tuple
    feet: tuple
    default_q: tuple
    limits: tuple
    motors: tuple
    foot_centers: tuple
    foot_radii: tuple
    meshes: dict
    sha256: str

    @classmethod
    def load(cls, path, cfg, control):
        tree = ET.parse(path).getroot()
        joints = tuple(f"{leg}_{part}_joint" for leg in cfg.legs for part in ("hip", "thigh", "calf"))
        feet = tuple(f"{leg}_foot" for leg in cfg.legs)
        actuated = {j.attrib["name"]: j for j in tree.findall("joint") if j.attrib["type"] != "fixed"}
        links = {link.attrib["name"]: link for link in tree.findall("link")}
        if set(joints) != actuated.keys() or cfg.root_body not in links:
            raise ValueError("WBLWBL URDF root or 12-joint contract does not match")
        if len(cfg.default_q) != len(joints):
            raise ValueError("Expected one default angle per policy joint")
        limits, motors = [], []
        for name, angle in zip(joints, cfg.default_q):
            joint = actuated[name]
            if joint.attrib["type"] != "revolute":
                raise ValueError(f"Expected a revolute URDF joint: {name}")
            lower, upper, effort, velocity = (
                float(joint.find("limit").attrib[key])
                for key in ("lower", "upper", "effort", "velocity")
            )
            if not all(map(math.isfinite, (lower, upper, effort, velocity, angle))) or not (
                lower < upper and lower <= angle <= upper and effort > 0 and velocity > 0
            ):
                raise ValueError(f"Invalid joint limits or standing angle: {name}")
            limits.append((lower, upper))
            motors.append((effort, velocity, control.kp, control.kd, control.armature))
        centers, radii = [], []
        for foot in feet:
            collisions = links[foot].findall("collision")
            sphere = collisions[0].find("geometry/sphere") if len(collisions) == 1 else None
            if sphere is None:
                raise ValueError(f"Expected a single foot collision sphere on {foot}")
            centers.append(xyz(collisions[0].find("origin"), "xyz"))
            radii.append(float(sphere.attrib["radius"]))
        meshes = {}
        for mesh in tree.iter("mesh"):
            filename = mesh.attrib["filename"]
            target = (path.parent.parent / filename.split("/", 3)[3]
                      if filename.startswith("package://") else path.parent / filename)
            meshes[filename] = target.resolve(strict=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return cls(path, cfg.root_body, joints, feet, tuple(cfg.default_q), tuple(limits), tuple(motors),
                   tuple(centers), tuple(radii), meshes, digest)

    def resolved_urdf(self, output):
        tree = ET.parse(self.path)
        for mesh in tree.iter("mesh"):
            mesh.set("filename", str(self.meshes[mesh.attrib["filename"]]))
        for joint in tree.findall("joint[@type='fixed']"):
            axis = joint.find("axis")
            if axis is not None:
                joint.remove(axis)
        output.parent.mkdir(parents=True, exist_ok=True)
        tree.write(output, encoding="utf-8")
        return output

    def kinematics(self, device, spec, tracked=None, *, base=None):
        import torch
        from scipy.spatial.transform import Rotation
        if base is None:
            from trackerLab.kinematics.supermimic.robot import RobotKinematics
            base = RobotKinematics

        tracked = spec.tracked_bodies if tracked is None else tracked
        kin = base.__new__(base)
        kin.robot, kin.device = spec, device
        kin.joint_names, kin.num_joints = self.joints, len(self.joints)
        by_parent = {}
        for joint in ET.parse(self.path).getroot().findall("joint"):
            by_parent.setdefault(joint.find("parent").attrib["link"], []).append(joint)

        kin.body_names, kin.parent, kin.joint_idx = [], [], []
        offsets, rotations, axes = [], [], []

        def visit(name, parent, joint=None):
            index = len(kin.body_names)
            kin.body_names.append(name)
            kin.parent.append(parent)
            kin.joint_idx.append(self.joints.index(joint.attrib["name"])
                                 if joint is not None and joint.attrib["type"] != "fixed" else -1)
            origin = joint.find("origin") if joint is not None else None
            offsets.append(xyz(origin, "xyz"))
            rotations.append(xyz(origin, "rpy"))
            axes.append(xyz(joint.find("axis"), "xyz")
                        if joint is not None and joint.attrib["type"] != "fixed" else (0., 0., 0.))
            for child in by_parent.get(name, ()):
                visit(child.find("child").attrib["link"], index, child)

        visit(self.root, -1)
        kin.num_bodies = len(kin.body_names)
        kin.tracked_idx = [kin.body_names.index(name) for name in tracked]
        kin.axes = torch.tensor(axes, device=device)
        kin.off_p = torch.tensor(offsets, device=device)
        kin.off_R = torch.as_tensor(Rotation.from_euler("xyz", rotations).as_matrix(),
                                    dtype=torch.float32, device=device)
        kin.anchors = torch.zeros_like(kin.off_p)
        kin.qrefs = torch.zeros(kin.num_bodies, device=device)
        kin.q_lo, kin.q_hi = torch.tensor(self.limits, device=device).T
        return kin
