"""Live MuJoCo passive viewer hook for small-batch multi-agent SuperMimic training.

Runs ``mujoco.viewer`` in an isolated CPU-only child process so OpenGL/GLFW never
collides with CUDA/Warp during training or interpreter shutdown.
"""

from __future__ import annotations

import copy
import io
import os
import pickle
import struct
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch


def _add_arrow(scn, p0, p1, radius, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    if np.linalg.norm(p1 - p0) < 1e-4:
        return
    import mujoco

    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).ravel(),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        float(radius),
        np.asarray(p0, dtype=np.float64),
        np.asarray(p1, dtype=np.float64),
    )
    scn.ngeom += 1


def _add_sphere(scn, pos, radius, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    import mujoco

    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).ravel(),
        np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def _rename_subtree(elem: ET.Element, suffix: str) -> None:
    if elem.tag in ("body", "joint", "geom", "site", "camera") and "name" in elem.attrib:
        elem.set("name", f"{elem.attrib['name']}{suffix}")
    if elem.tag == "geom":
        elem.set("contype", "0")
        elem.set("conaffinity", "0")
    for child in elem:
        _rename_subtree(child, suffix)


def build_batch_viewer_xml(sim, count: int):
    """Build XML + virtual files + terrain hfield for ``count`` cloned robots."""
    path = Path(sim.spec.mjcf_path).resolve()
    document = ET.parse(path).getroot()
    compiler = document.find("compiler")
    directories = {}
    if compiler is not None:
        for key in ("meshdir", "texturedir"):
            directories[key] = (path.parent / compiler.get(key, ".")).resolve()
            compiler.attrib.pop(key, None)

    for section in ("actuator", "sensor", "keyframe", "contact", "equality"):
        for node in document.findall(section):
            document.remove(node)

    world = document.find("worldbody")
    body_templates = []
    for node in list(world):
        if node.tag == "body":
            body_templates.append(node)
        world.remove(node)

    for parent in document.iter():
        for child in list(parent):
            if child.tag in ("light", "camera"):
                parent.remove(child)

    asset = document.find("asset")
    if asset is None:
        asset = ET.SubElement(document, "asset")
    virtual_files = {}
    for index, node in enumerate(asset):
        filename = node.get("file")
        if filename:
            if node.tag == "mesh" and not node.get("name"):
                node.set("name", Path(filename).stem)
            folder = directories.get("meshdir" if node.tag == "mesh" else "texturedir", path.parent)
            original = (folder / filename).resolve()
            safe_name = f"viewer_robot_asset_{index}" + original.suffix
            virtual_files[safe_name] = original.read_bytes()
            node.set("file", safe_name)

    for node in list(asset):
        if node.tag == "texture" and node.get("type") == "skybox":
            asset.remove(node)

    ET.SubElement(asset, "texture", {
        "name": "viewer_sky", "type": "skybox", "builtin": "gradient",
        "rgb1": ".36 .52 .72", "rgb2": ".75 .85 .94", "width": "256", "height": "1536",
    })
    ET.SubElement(asset, "texture", {
        "name": "viewer_grid", "type": "2d", "builtin": "checker",
        "rgb1": ".30 .42 .34", "rgb2": ".45 .56 .47", "width": "512", "height": "512",
    })
    ET.SubElement(asset, "material", {
        "name": "viewer_ground_mat", "texture": "viewer_grid",
        "texrepeat": "40 28", "texuniform": "false", "reflectance": "0",
    })
    ET.SubElement(asset, "hfield", {
        "name": "supermimic_terrain",
        "nrow": str(sim.terrain.heights.shape[0]),
        "ncol": str(sim.terrain.heights.shape[1]),
        "size": " ".join(map(str, sim.terrain.size)),
    })

    visual = document.find("visual")
    if visual is None:
        visual = ET.SubElement(document, "visual")
    v_global = visual.find("global")
    if v_global is None:
        v_global = ET.SubElement(visual, "global")
    v_global.set("offwidth", "1280")
    v_global.set("offheight", "720")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.set("active", "1")
    headlight.set("ambient", "0.06 0.06 0.06")
    headlight.set("diffuse", "0.18 0.18 0.18")
    headlight.set("specular", "0.0 0.0 0.0")

    ET.SubElement(world, "light", {
        "pos": "0 0 18", "dir": "0.2 0.35 -1", "directional": "true",
        "diffuse": "0.55 0.55 0.55", "ambient": "0.18 0.18 0.18", "specular": "0.05 0.05 0.05",
    })
    ET.SubElement(world, "geom", {
        "name": "supermimic_ground",
        "type": "hfield",
        "hfield": "supermimic_terrain",
        "pos": " ".join(map(str, sim.terrain.position)),
        "material": "viewer_ground_mat",
        "group": "0",
        "contype": "0",
        "conaffinity": "0",
    })

    for idx in range(count):
        suffix = f"_v{idx}"
        for tmpl in body_templates:
            cloned = copy.deepcopy(tmpl)
            _rename_subtree(cloned, suffix)
            world.append(cloned)

    xml_str = ET.tostring(document, encoding="unicode")
    return {
        "xml": xml_str,
        "virtual_files": virtual_files,
        "hfield_data": sim.terrain.normalized.ravel().copy(),
        "root_body": sim.spec.root_body,
        "joint_names": list(sim.spec.joint_names),
        "count": count,
    }


def _compile_viewer_model(bundle):
    import mujoco

    viewer_model = mujoco.MjModel.from_xml_string(bundle["xml"], bundle["virtual_files"])
    viewer_model.hfield_data[:] = bundle["hfield_data"]
    root_qpos_adrs = []
    joint_qpos_adrs = []
    for idx in range(bundle["count"]):
        suffix = f"_v{idx}"
        root_body_id = viewer_model.body(f"{bundle['root_body']}{suffix}").id
        free_jnt = next(
            j for j in range(viewer_model.njnt)
            if viewer_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            and viewer_model.jnt_bodyid[j] == root_body_id
        )
        root_qpos_adrs.append(int(viewer_model.jnt_qposadr[free_jnt]))
        j_ids = [viewer_model.joint(f"{name}{suffix}").id for name in bundle["joint_names"]]
        joint_qpos_adrs.append(viewer_model.jnt_qposadr[j_ids].copy())
    return (
        viewer_model,
        np.asarray(root_qpos_adrs, dtype=np.int64),
        np.asarray(joint_qpos_adrs, dtype=np.int64),
    )


def build_batch_viewer_model(sim, count: int):
    """Compatibility helper returning (viewer_model, root_qpos_adrs, joint_qpos_adrs)."""
    return _compile_viewer_model(build_batch_viewer_xml(sim, count))


def _sync_viewer_state(sim):
    """Compatibility helper for direct MjData tests."""
    K = sim._viewer_count
    qpos_batch = sim.qpos[:K].detach().cpu().numpy()
    root_src = qpos_batch[:, sim._root_qpos : sim._root_qpos + 7]
    jnt_src = qpos_batch[:, sim._qpos_ids_np]
    for i in range(K):
        r0 = sim._viewer_root_adrs[i]
        sim._viewer_data.qpos[r0 : r0 + 7] = root_src[i]
        sim._viewer_data.qpos[sim._viewer_joint_adrs[i]] = jnt_src[i]


def _recv_exact(stream, nbytes: int):
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = stream.read(nbytes - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _worker_main(bundle_path: str):
    import mujoco
    import mujoco.viewer

    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    try:
        os.unlink(bundle_path)
    except OSError:
        pass

    model, root_adrs, joint_adrs = _compile_viewer_model(bundle)
    data = mujoco.MjData(model)
    K = bundle["count"]

    latest = {"frame": None, "eof": False}
    lock = threading.Lock()

    def _reader():
        stdin = sys.stdin.buffer
        while True:
            hdr = _recv_exact(stdin, 4)
            if hdr is None:
                with lock:
                    latest["eof"] = True
                return
            size = struct.unpack("<I", hdr)[0]
            payload = _recv_exact(stdin, size)
            if payload is None:
                with lock:
                    latest["eof"] = True
                return
            frame = pickle.loads(payload)
            with lock:
                latest["frame"] = frame

    threading.Thread(target=_reader, daemon=True).start()

    # Wait briefly for initial frame so camera centers nicely on robots
    deadline = time.monotonic() + 10.0
    first_frame = None
    while time.monotonic() < deadline:
        with lock:
            if latest["frame"] is not None or latest["eof"]:
                first_frame = latest["frame"]
                break
        time.sleep(0.01)

    if first_frame is not None:
        root_src, jnt_src = first_frame["root"], first_frame["jnt"]
        for i in range(K):
            r0 = root_adrs[i]
            data.qpos[r0 : r0 + 7] = root_src[i]
            data.qpos[joint_adrs[i]] = jnt_src[i]
        mujoco.mj_kinematics(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[:] = 1
        if K > 1:
            roots = np.array([data.qpos[a : a + 3] for a in root_adrs])
            center = roots.mean(axis=0)
            viewer.cam.lookat[:] = [float(center[0]), float(center[1]), float(center[2]) + 0.15]
            viewer.cam.distance = 14.5
            viewer.cam.elevation = -28.0
            viewer.cam.azimuth = 135.0
        else:
            root = data.qpos[root_adrs[0] : root_adrs[0] + 3]
            viewer.cam.lookat[:] = [float(root[0]), float(root[1]), float(root[2]) + 0.1]
            viewer.cam.distance = 3.5
            viewer.cam.elevation = -24.0
            viewer.cam.azimuth = 135.0
        viewer.sync()

        last_seen_seq = -1
        while viewer.is_running():
            with lock:
                frame = latest["frame"]
                eof = latest["eof"]
            if frame is not None and frame["seq"] != last_seen_seq:
                last_seen_seq = frame["seq"]
                root_src = frame["root"]
                jnt_src = frame["jnt"]
                p_np = frame["p"]
                v_np = frame["v"]
                cmd_vx_w = frame["cmd_vx_w"]
                cmd_vy_w = frame["cmd_vy_w"]
                cmd_wz = frame["cmd_wz"]
                cy_np = frame["cy"]
                sy_np = frame["sy"]
                scan_pts = frame["scan"]
                for i in range(K):
                    r0 = root_adrs[i]
                    data.qpos[r0 : r0 + 7] = root_src[i]
                    data.qpos[joint_adrs[i]] = jnt_src[i]
                with viewer.lock():
                    mujoco.mj_kinematics(model, data)
                    scn = viewer.user_scn
                    scn.ngeom = 0
                    for i in range(K):
                        origin = p_np[i] + np.array([0.0, 0.0, 0.24], dtype=np.float64)
                        _add_arrow(
                            scn,
                            origin,
                            origin + np.array([float(cmd_vx_w[i]), float(cmd_vy_w[i]), 0.0], dtype=np.float64),
                            0.016,
                            [0.1, 0.95, 0.2, 0.9],
                        )
                        _add_arrow(
                            scn,
                            origin + np.array([0.0, 0.0, 0.035], dtype=np.float64),
                            origin + np.array([float(v_np[i, 0]), float(v_np[i, 1]), 0.035], dtype=np.float64),
                            0.013,
                            [1.0, 0.55, 0.1, 0.9],
                        )
                        wz = float(cmd_wz[i])
                        if abs(wz) > 0.05:
                            fwd = np.array([float(cy_np[i]), float(sy_np[i]), 0.0], dtype=np.float64)
                            left = np.array([-float(sy_np[i]), float(cy_np[i]), 0.0], dtype=np.float64)
                            head = origin + 0.26 * fwd + np.array([0.0, 0.0, 0.07], dtype=np.float64)
                            _add_arrow(scn, head, head + 0.32 * wz * left, 0.013, [0.2, 0.65, 1.0, 0.95])
                    if scan_pts is not None:
                        for pt in scan_pts:
                            _add_sphere(scn, pt + np.array([0.0, 0.0, 0.01]), 0.012, [0.1, 0.9, 1.0, 0.6])
                viewer.sync()
            elif eof:
                break
            time.sleep(0.01)


_ACTIVE_PROCS: list[subprocess.Popen] = []


def close_live_viewer():
    """Terminate isolated viewer worker cleanly before Python exits."""
    for proc in list(_ACTIVE_PROCS):
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=3.0)
        except Exception:
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                pass
    _ACTIVE_PROCS.clear()


def _send_frame(sim):
    proc = getattr(sim, "_viewer_proc", None)
    if proc is None or proc.poll() is not None:
        sim._viewer_proc = None
        return

    K = sim._viewer_count
    eid = sim._viewer_env_id
    sim._viewer_seq += 1

    qpos_batch = sim.qpos[:K].detach().cpu().numpy()
    root_src = qpos_batch[:, sim._root_qpos : sim._root_qpos + 7].astype(np.float64)
    jnt_src = qpos_batch[:, sim._qpos_ids_np].astype(np.float64)

    state = sim.state()
    p_all = state[0][:K]
    R_all = state[1][:K]
    v_all = state[2][:K]
    c_all = sim.c[:K]

    yaw_all = torch.atan2(R_all[:, 1, 0], R_all[:, 0, 0])
    cy_all, sy_all = torch.cos(yaw_all), torch.sin(yaw_all)

    scan_pts = None
    if sim._viewer_show_scan and hasattr(sim.mdp, "field"):
        S_single = (state[0][eid : eid + 1], state[1][eid : eid + 1])
        xy = sim.mdp.field.scan_xy(S_single)
        z = sim.mdp.field.sample(xy)
        pts = torch.cat((xy, z.unsqueeze(-1)), dim=-1)[0]
        scan_pts = pts.detach().cpu().numpy().astype(np.float64)

    frame = {
        "seq": sim._viewer_seq,
        "root": root_src,
        "jnt": jnt_src,
        "p": p_all.detach().cpu().numpy().astype(np.float64),
        "v": v_all.detach().cpu().numpy().astype(np.float64),
        "cmd_vx_w": (cy_all * c_all[:, 1] - sy_all * c_all[:, 2]).detach().cpu().numpy().astype(np.float32),
        "cmd_vy_w": (sy_all * c_all[:, 1] + cy_all * c_all[:, 2]).detach().cpu().numpy().astype(np.float32),
        "cmd_wz": c_all[:, 3].detach().cpu().numpy().astype(np.float32),
        "cy": cy_all.detach().cpu().numpy().astype(np.float32),
        "sy": sy_all.detach().cpu().numpy().astype(np.float32),
        "scan": scan_pts,
    }
    payload = pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        proc.stdin.write(struct.pack("<I", len(payload)) + payload)
        proc.stdin.flush()
    except (BrokenPipeError, OSError):
        sim._viewer_proc = None


def install_live_viewer(
    *,
    env_id: int = 0,
    viewer_envs: int = 16,
    max_fps: float = 35.0,
    show_height_scan: bool = True,
):
    """Patch CommandSim to stream up to ``viewer_envs`` robots to a CPU-only MuJoCo viewer subprocess."""
    from trackerLab.tracker_env.supermimic.command import CommandSim

    if getattr(CommandSim, "_live_viewer_installed", False):
        return
    CommandSim._live_viewer_installed = True

    orig_init = CommandSim.__init__
    orig_step = CommandSim.step

    def _wrapped_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        K = max(1, min(int(viewer_envs), self.num_envs))
        eid = max(0, min(int(env_id), K - 1))
        self._viewer_count = K
        self._viewer_env_id = eid
        self._viewer_max_fps = float(max_fps)
        self._viewer_show_scan = bool(show_height_scan)
        self._viewer_last_sync = 0.0
        self._viewer_last_log_step = -999
        self._viewer_total_steps = 0
        self._viewer_seq = 0

        bundle = build_batch_viewer_xml(self, K)
        tmp = tempfile.NamedTemporaryFile(prefix="supermimic_viewer_", suffix=".pkl", delete=False)
        with tmp:
            pickle.dump(bundle, tmp, protocol=pickle.HIGHEST_PROTOCOL)

        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", tmp.name],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=sys.stderr,
        )
        self._viewer_proc = proc
        _ACTIVE_PROCS.append(proc)
        _send_frame(self)
        print(
            f"[live-viewer] MuJoCo 3D viewer opened | showing {K}/{self.num_envs} robots | highlight env={eid} | max_fps={self._viewer_max_fps}",
            flush=True,
        )

    @torch.no_grad()
    def _wrapped_step(self, action):
        transition = orig_step(self, action)
        proc = getattr(self, "_viewer_proc", None)
        if proc is None:
            return transition

        self._viewer_total_steps += 1
        eid = self._viewer_env_id
        _send_frame(self)

        if self._viewer_max_fps > 0.0:
            now = time.monotonic()
            min_dt = 1.0 / self._viewer_max_fps
            rem = min_dt - (now - self._viewer_last_sync)
            if rem > 0.0:
                time.sleep(rem)
        self._viewer_last_sync = time.monotonic()

        ep_step = int(self.ep_step[eid].item())
        done_now = bool(transition["done"][eid].item())
        if done_now or (self._viewer_total_steps - self._viewer_last_log_step >= 50):
            self._viewer_last_log_step = self._viewer_total_steps
            lvl = int(self.levels[eid].item())
            state = self.state()
            R = state[1][eid]
            Rt = R.transpose(-1, -2)
            v_body = (Rt @ state[2][eid].unsqueeze(-1)).squeeze(-1)
            w_body = (Rt @ state[3][eid].unsqueeze(-1)).squeeze(-1)
            c_vec = self.c[eid]
            status = "FALL/RESET" if done_now else "RUN"
            print(
                f"[live env={eid}] step={ep_step:4d} lvl={lvl} {status:10s} | "
                f"cmd=({float(c_vec[1]):+.2f},{float(c_vec[2]):+.2f},{float(c_vec[3]):+.2f}) | "
                f"act=({float(v_body[0]):+.2f},{float(v_body[1]):+.2f},{float(w_body[2]):+.2f})",
                flush=True,
            )

        return transition

    CommandSim.__init__ = _wrapped_init
    CommandSim.step = _wrapped_step


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _worker_main(sys.argv[2])
