"""Isaac Lab articulation and the TrackerLab command transition contract."""

import torch

from trackerLab.kinematics.supermimic.rotations import quat_to_mat
from trackerLab.tracker_env.mdp.supermimic import flatten_state
from trackerLab.tracker_env.mdp.supermimic_command import SCENARIO_DIM
from trackerLab.tracker_env.mdp.supermimic_terrain import FALL_PENALTY, terminal_masks

from .wmp_terrain import CourseEvents


class PhysxCommandSim:
    def __init__(self, cfg, robot, spec, mdp, atlas, run_dir):
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors import ContactSensorCfg

        self.cfg, self.mdp, self.atlas = cfg, mdp, atlas
        self.num_envs, self.device = cfg.env.num_envs, cfg.env.device
        self.dt = cfg.control.physics_dt * cfg.control.decimation
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=cfg.control.physics_dt, device=self.device,
            render_interval=cfg.control.decimation,
        ))
        material = sim_utils.RigidBodyMaterialCfg(
            static_friction=cfg.terrain.friction, dynamic_friction=cfg.terrain.friction)
        if atlas is not None:
            import trimesh
            from isaaclab.terrains.utils import create_prim_from_mesh
            vertices, faces = atlas.mesh()
            create_prim_from_mesh("/World/ground", trimesh.Trimesh(
                vertices=vertices, faces=faces, process=False), physics_material=material)
        scene_cfg = InteractiveSceneCfg(
            num_envs=self.num_envs, env_spacing=cfg.terrain.spacing,
            replicate_physics=True, filter_collisions=True)
        scene_cfg.ground = AssetBaseCfg(
            prim_path="/World/ground", collision_group=-1,
            spawn=sim_utils.GroundPlaneCfg(physics_material=material) if atlas is None else None,
        )
        scene_cfg.light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=900.))
        urdf = robot.resolved_urdf(run_dir / "asset" / "wblwbl_dog.urdf")
        scene_cfg.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UrdfFileCfg(
                asset_path=str(urdf), usd_dir=str(run_dir / "asset"), fix_base=False,
                merge_fixed_joints=False, activate_contact_sensors=True,
                joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
                    gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0., damping=0.)),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=cfg.domain_rand.self_collisions,
                    solver_position_iteration_count=cfg.env.solver_iterations,
                    solver_velocity_iteration_count=1),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0., 0., cfg.robot.standing_height),
                joint_pos=dict(zip(robot.joints, robot.default_q))),
            actuators={"legs": ImplicitActuatorCfg(
                joint_names_expr=list(robot.joints), stiffness=cfg.control.kp, damping=cfg.control.kd,
                effort_limit_sim=dict(zip(robot.joints, (row[0] for row in robot.motors))),
                velocity_limit_sim=dict(zip(robot.joints, (row[1] for row in robot.motors))),
                armature=cfg.control.armature)},
        )
        if atlas is not None:
            scene_cfg.contacts = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=1, update_period=0.)
        self.scene = InteractiveScene(scene_cfg)
        if atlas is not None:
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.sim.reset()
        self.scene.update(cfg.control.physics_dt)
        self.robot = self.scene["robot"]
        self.joint_ids = [self.robot.joint_names.index(name) for name in robot.joints]
        self.foot_ids = [self.robot.body_names.index(name) for name in robot.feet]
        self.foot_offsets = torch.tensor(robot.foot_centers, device=self.device)
        self.foot_radii = torch.tensor(robot.foot_radii, device=self.device)
        self.default_q = torch.tensor(spec.default_q, device=self.device)
        self.qlo, self.qhi = torch.tensor(robot.limits, device=self.device).T
        self.events = (CourseEvents(atlas, self.num_envs, self.device, self.dt)
                       if atlas is not None else None)
        if self.events is not None:
            self.contacts = self.scene["contacts"]
            self.contact_ids = [self.contacts.body_names.index(name) for name in robot.feet]
            self.root_contact_id = self.contacts.body_names.index(robot.root)
            self.levels = self.events.core.terrain_levels
        else:
            self.levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.c = torch.zeros(self.num_envs, SCENARIO_DIM, device=self.device)
        self.xi = torch.zeros(self.num_envs, spec.context_dim, device=self.device)
        self.xi[:, 0] = cfg.terrain.friction
        self.a_prev = self.c.new_zeros((self.num_envs, spec.num_joints))
        self.ep_return = self.c.new_zeros(self.num_envs)
        self.ep_tracking = torch.zeros_like(self.ep_return)
        self.ep_step = torch.zeros_like(self.levels)
        self.command_age = torch.zeros_like(self.levels)
        self.command_interval = torch.zeros_like(self.levels)
        self.twists = self.c.new_tensor(cfg.commands.samples)
        self.contract = {
            "backend": "isaaclab-physx", "robot_sha256": robot.sha256,
            "joint_names": list(robot.joints), "feet": list(robot.feet),
            "motor_limits_and_pd": [list(row) for row in robot.motors],
            "default_q": list(robot.default_q), "action_limit": cfg.env.action_limit,
            "action_scale": cfg.control.action_scale, "physics_dt": cfg.control.physics_dt,
            "decimation": cfg.control.decimation, "control_dt": self.dt,
            "solver_iterations": cfg.env.solver_iterations,
            "reference_resets": {"probability": 0.},
            "terrain": cfg.terrain.profile, "command_samples": [list(row) for row in cfg.commands.samples],
            "wmp_events": "real rollouts only" if atlas is not None else None,
        }

    def state(self):
        data = self.robot.data
        position = data.root_link_pos_w
        if self.atlas is None:
            position = position - self.scene.env_origins
        return (position.clone(), quat_to_mat(data.root_link_quat_w),
                data.root_link_lin_vel_w.clone(), data.root_link_ang_vel_w.clone(),
                data.joint_pos[:, self.joint_ids], data.joint_vel[:, self.joint_ids])

    def observe(self):
        return self.mdp.obs(self.state(), self.a_prev, self.xi, self.c)

    def set_reference_resets(self, motion, probability, grounded_mask=None):
        if probability:
            raise ValueError("The PhysX backend currently supports nominal resets only")

    def sample_commands(self, ids):
        self.c[ids, 1:4] = self.twists[torch.randint(len(self.twists), (len(ids),), device=self.device)]
        self.c[ids, 4] = self.mdp.target_clearance_m
        self.command_age[ids] = 0
        low, high = self.cfg.commands.resample_steps
        self.command_interval[ids] = torch.randint(low, high + 1, (len(ids),), device=self.device)

    @torch.no_grad()
    def reset(self, ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if ids is None else ids
        if not len(ids):
            return
        self.scene.reset(ids)
        root = self.robot.data.default_root_state[ids].clone()
        origins = self.scene.env_origins if self.events is None else self.events.core.env_origins
        root[:, :3] += origins[ids]
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids=ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids=ids)
        pos = self.robot.data.default_joint_pos[ids].clone()
        pos[:, self.joint_ids] = self.default_q
        self.robot.write_joint_state_to_sim(pos, torch.zeros_like(pos), env_ids=ids)
        self.robot.set_joint_position_target(pos, env_ids=ids)
        self.ep_step[ids] = self.ep_return[ids] = self.ep_tracking[ids] = 0
        self.a_prev[ids] = self.c[ids] = 0
        self.c[ids, 0] = origins[ids, 1] if self.events is not None else 0.
        self.sample_commands(ids)

    @torch.no_grad()
    def step(self, action, *, fixed_command=False):
        before = self.state()
        command, previous = self.c.clone(), self.a_prev.clone()
        time_left = self.command_interval - self.command_age
        a = action.clamp(-self.cfg.env.action_limit, self.cfg.env.action_limit)
        target = (self.default_q + a * self.cfg.control.action_scale).clamp(self.qlo, self.qhi)
        self.robot.set_joint_position_target(target, joint_ids=self.joint_ids)
        for _ in range(self.cfg.control.decimation):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.cfg.control.physics_dt)
        after = self.state()
        self.c.copy_(self.mdp.advance_command(command, self.dt))
        self.ep_step += 1
        self.command_age += 1
        timeout = self.ep_step >= self.cfg.env.max_episode_steps
        fallen = self.mdp.termination(after, self.c)
        reached = torch.zeros_like(fallen)
        event_rewards = {}
        if self.events is not None:
            data = self.robot.data
            foot_rotation = quat_to_mat(data.body_link_quat_w[:, self.foot_ids])
            sole = data.body_link_pos_w[:, self.foot_ids] + (
                foot_rotation @ self.foot_offsets[..., None]).squeeze(-1)
            sole[..., 2] -= self.foot_radii
            forces = self.contacts.data.net_forces_w
            fallen |= ((forces[:, self.root_contact_id].norm(dim=-1) > self.cfg.terrain.body_contact_limit)
                       | (after[2][:, 2] < -3.))
            reached, fallen, event_rewards = self.events.update(
                after, sole, forces[:, self.contact_ids], self.c, fallen, timeout,
                self.ep_step, self.cfg.rewards)
        end_command = self.c.clone()
        output = getattr(self, "command_outputs_fn", self.mdp.transition_outputs)
        reward, terms, metrics, _ = output(after, a, previous, end_command)
        for name, value in event_rewards.items():
            terms[f"wmp/{name}"] = value
            reward = reward + value
        regularizer = getattr(self, "locomotion_reward_fn", None)
        if regularizer is not None:
            terms["locomotion_regularization"] = regularizer(before, after, a, self.xi)
            reward = reward + terms["locomotion_regularization"]
        success, done, truncated = terminal_masks(reached, fallen, timeout)
        self.ep_tracking += metrics["command_tracking"]
        if self.events is None:
            success = truncated & (self.ep_tracking / self.ep_step >= .8)
        terms["terminal"] = self.mdp.terminal_reward(success, fallen)
        reward = torch.nan_to_num(reward + terms["terminal"], nan=-FALL_PENALTY,
                                  posinf=-FALL_PENALTY, neginf=-FALL_PENALTY)
        self.ep_return += reward
        resampled = (self.command_age >= self.command_interval) & ~done & (not fixed_command)
        row = {
            "s": flatten_state(before), "s1": flatten_state(after), "a": a, "a_prev": previous,
            "policy_action": action, "xi": self.xi.clone(), "c": command, "c1": end_command,
            "f": self.ep_step - 1, "ep_step": self.ep_step - 1, "motion_id": self.levels.clone(),
            "r": reward, "reward_terms": terms, "term": fallen, "terminated": done & ~truncated,
            "truncated": truncated, "done": done, "episode_success": success,
            "episode_fall": fallen, "episode_timeout": truncated,
            "episode_strict_success": (self.events.course.strict.clone() if self.events else torch.zeros_like(done)),
            "command_metrics": metrics, "command_resampled": resampled,
            "command_time_left_steps": time_left,
            "ep_len": torch.where(done, self.ep_step.float(), 0.),
            "ep_ret_norm": torch.where(done, self.ep_return / self.ep_step, 0.),
        }
        ids = done.nonzero(as_tuple=False).flatten()
        if self.events is not None and len(ids):
            self.events.finish(ids)
        self.reset(ids)
        self.sample_commands(resampled.nonzero(as_tuple=False).flatten())
        return row
