"""WBL train-time robot and fixed-wall scene for the vendored tracker."""
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "robocon2027_sim_control/deploy_cpp/robot/wblwbl_dog_v1/xml/wblwbl_dog_v1.xml"
REF = ROOT / "amp_supermimic/data/wblwbl/competition_gaits/high_wall_jump_direct"
LEGS = ("RR", "RL", "FR", "FL")
JOINTS = tuple(f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "calf"))
NAME = "wbl_wall_jump"
THICKNESS, WIDTH, HEIGHT = .08, .80, .30


def prepare_robot(asset):
    """Keep WBL inertias and colliders, name them for the contact sensors."""
    local = ET.parse(SOURCE).getroot()
    compiler = local.find("compiler")
    mesh_root = SOURCE.parent / compiler.get("meshdir", "")
    compiler.attrib.pop("meshdir", None)
    for mesh in local.findall("./asset/mesh"):
        mesh.set("file", str((mesh_root / mesh.get("file")).resolve(strict=True)))
    world = local.find("worldbody")
    for item in tuple(world):
        if item.tag != "body":
            world.remove(item)
    for body in tuple(world.iter("body")):
        for geom in body.findall("geom"):
            # Warp MULTICCD cannot handle nonzero margins on convex pairs.
            # Keep the WBL collision surfaces at their modeled dimensions.
            if geom.get("type") in ("box", "sphere"):
                geom.set("margin", "0")
            if geom.get("type") == "sphere" and body.get("name", "").endswith("_calf"):
                leg = body.get("name")[:2]
                geom.set("name", leg + "_foot_ball")
                # A fixed, massless frame marks the foot ball's center for the
                # real and imagined tracking rewards; the collision sphere stays on the calf.
                ET.SubElement(body, "body", name=leg + "_foot_frame", pos=geom.get("pos"))
            elif geom.get("type") == "box":
                geom.set("name", body.get("name") + "_collision")
    for tag in ("actuator", "sensor", "keyframe"):
        element = local.find(tag)
        if element is not None:
            local.remove(element)
    asset.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(local).write(asset, encoding="unicode")


def setup_scene(scene, calf_route=False, physics_history=False):
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    def world(spec):
        spec.worldbody.add_geom(name="jump_ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=(0, 0, 1), friction=(.8, .005, .0001))
        spec.worldbody.add_geom(name="jump_wall", type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=(THICKNESS / 2, WIDTH / 2, HEIGHT / 2), pos=(0, 0, HEIGHT / 2),
                                friction=(.8, .005, .0001), rgba=(.2, .5, .3, 1.))

    def sensor(name, primary, secondary):
        return ContactSensorCfg(name=name, primary=ContactMatch("geom", primary, "robot"),
                                secondary=ContactMatch("geom", secondary),
                                fields=("found", "force", "normal", "pos") if calf_route
                                else ("found", "force", "normal"), reduce="maxforce",
                                history_length=4 if physics_history else 0)

    # Each Warp world owns a separate copy of the scene and uses the same origin.
    scene.terrain = None
    scene.spec_fn = world
    scene.sensors += (
        sensor("wall_feet", ".*_foot_ball", "jump_wall"),
        sensor("floor_feet", ".*_foot_ball", "jump_ground"),
        sensor("wall_links", ".*_collision", "jump_wall"),
        sensor("floor_links", ".*_collision", "jump_ground"),
    )


def register(asset, *, calf_route=False, physics_history=False):
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
    from supermimic import robots
    from supermimic.robots import rover

    def make(stiffness=40., action_scale=.25):
        def entity(_randomization):
            motor = BuiltinPositionActuatorCfg(target_names_expr=(".*_joint",), stiffness=stiffness,
                                               damping=1., effort_limit=23.7)
            home = [-.1, 1., -1.5, .1, 1., -1.5, -.1, .8, -1.5, .1, .8, -1.5]
            return EntityCfg(
                spec_fn=lambda: mujoco.MjSpec.from_file(str(asset)),
                init_state=EntityCfg.InitialStateCfg(pos=(0., 0., .375), joint_pos=dict(zip(JOINTS, home)),
                                                     joint_vel={".*": 0.}),
                articulation=EntityArticulationInfoCfg(actuators=(motor,), soft_joint_pos_limit_factor=1.),
            )

        # Baseline normalization from the upstream quadruped, WBL motor constants and named geometry.
        return replace(rover.ROVER_SPEC, name=NAME, xml_path=asset, joint_names=JOINTS,
                       root_body="base_link", anchor_body="base_link", com_body="base_link",
                       tracked_bodies=("base_link",) + tuple(f"{leg}_{part}" for leg in LEGS
                                                            for part in ("thigh", "calf", "foot_frame")),
                       foot_geom_regex=r"^(RR|RL|FR|FL)_foot_ball$", entity_cfg_fn=entity,
                       action_scale=action_scale, stand_height=.365, root_height_threshold=.26,
                       root_tilt_threshold=.50, torque_model=None, reward_w_torque_headroom=0.,
                       reward_std_root_pos=.10, reward_std_body_pos=.075,
                       reward_w_joint_pos=.25, reward_std_joint_pos=.45,
                       mirror_joint_perm=None, mirror_joint_sign=None, mirror_body_map=None,
                       scene_setup=lambda scene: setup_scene(scene, calf_route, physics_history), world_pose_features=True,
                       launch_velocity_weight=2., placement_weight=3.,
                       placement_bodies=("FR_foot_frame", "FL_foot_frame"),
                       sim=replace(rover.ROVER_SPEC.sim, nconmax=96, njmax=384))

    globals()["wbl_spec"] = make
    robots.ROBOTS[NAME] = f"{__name__}:wbl_spec"
    robots._CACHE.pop(NAME, None)


def prepare_library(folder, asset, first_teacher=None):
    from quadmotion.clip import body_states_from_pose

    model = mujoco.MjModel.from_xml_path(str(asset))
    if model.nq != 19 or model.nv != 18 or tuple(model.joint(i).name for i in range(1, model.njnt)) != JOINTS:
        raise ValueError("Unexpected MJCF generalized coordinates or joint order")
    source = REF / "high_wall_jump_direct.csv"
    rows = np.loadtxt(source, delimiter=",")
    labels = np.loadtxt(REF / "support.csv", delimiter=",", skiprows=1, dtype=int)
    if rows.shape[1] != 19 or len(rows) != len(labels):
        raise ValueError("WBL direct-jump reference contract mismatch")
    # Frame 57 is after re-positioning the rear pair, before the second push.
    brace = 57
    if not np.array_equal(labels[brace], [1, 1, 2, 2]):
        raise ValueError("Second jump must start with rear ground contact and front crest support")
    for stage, clip in (("first", rows[:brace + 1]), ("second", rows[brace:]), ("full", rows)):
        clip = np.concatenate([clip, np.repeat(clip[-1:], 20, axis=0)], axis=0)
        motion = body_states_from_pose(model, clip[:, :3], clip[:, [6, 3, 4, 5]], clip[:, 7:])
        path = folder / stage
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / "direct.npz", **motion.to_npz_dict())
    if first_teacher:
        with np.load(first_teacher, allow_pickle=False) as data:
            states = np.asarray(data["states"])
            controls = np.asarray(data["controls"])
        if (states.shape != (len(controls) + 1, model.nq + model.nv)
                or controls.ndim != 2 or controls.shape[1] != len(JOINTS)
                or len(controls) < 24 or not np.isfinite(states).all() or not np.isfinite(controls).all()):
            raise ValueError("Invalid first-jump state/control trajectory")
        pose = states[:, :model.nq]
        pose = np.concatenate([pose, np.repeat(pose[-1:], 20, axis=0)])
        dynamic = body_states_from_pose(model, pose[:, :3], pose[:, 3:7], pose[:, 7:]).to_npz_dict()
        # The free-base reset must match the successful replay's starting velocity.
        with np.load(folder / "first" / "direct.npz") as original:
            for key in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
                dynamic[key][0] = original[key][0]
        np.savez_compressed(folder / "first" / "direct.npz", **dynamic)
        np.save(folder / "first_joint_targets.npy", controls)


def prepare_followup(folder, asset, first_rollouts):
    """Join an accepted stage-one rollout to the second jump reference."""
    from quadmotion.clip import body_states_from_pose

    model = mujoco.MjModel.from_xml_path(str(asset))
    rows = np.loadtxt(REF / "high_wall_jump_direct.csv", delimiter=",")
    upcoming = rows[57:, [0, 1, 2, 6, 3, 4, 5, *range(7, 19)]]
    with np.load(first_rollouts, allow_pickle=False) as data:
        passed = np.flatnonzero(data["success"])
        if not len(passed):
            raise ValueError("Follow-up requires an accepted closed-loop first jump")
        initial = rows[0, :3]
        nearest = np.linalg.norm(data["states"][passed, 0, :3] - initial, axis=-1)
        trial = passed[nearest.argmin()]
        first = data["states"][trial, :int(data["stop"][trial]) + 1]
    anchor = first[-1, :model.nq].copy()
    if np.dot(anchor[3:7], upcoming[0, 3:7]) < 0.:
        upcoming[:, 3:7] *= -1.
    # Allow the controller a short brace phase to move to the planned second-push pose.
    phase = np.linspace(0., 1., 13)[:-1, None]
    phase = phase**2 * (3. - 2. * phase)
    bridge = anchor + phase * (upcoming[0] - anchor)
    bridge[:, 3:7] /= np.linalg.norm(bridge[:, 3:7], axis=-1, keepdims=True)
    second = np.concatenate((bridge, upcoming, np.repeat(upcoming[-1:], 20, axis=0)))

    def save(stage, pose, start_state):
        clip = body_states_from_pose(model, pose[:, :3], pose[:, 3:7], pose[:, 7:])
        data = mujoco.MjData(model)
        data.qpos[:] = start_state[:model.nq]
        data.qvel[:] = start_state[model.nq:]
        mujoco.mj_forward(model, data)
        clip.joint_vel[0] = data.qvel[6:]
        velocity = np.zeros(6)
        for index, name in enumerate(clip.body_names):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, model.body(name).id, velocity, 0)
            clip.body_ang_vel_w[0, index] = velocity[:3]
            clip.body_lin_vel_w[0, index] = velocity[3:]
        output = folder / stage / "direct.npz"
        pending = output.with_suffix(".pending.npz")
        np.savez_compressed(pending, **clip.to_npz_dict())
        pending.replace(output)

    save("second", second, first[-1])
    save("full", np.concatenate((first[:-1, :model.nq], second)), first[0])
    return int(trial), len(first)
