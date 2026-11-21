"""Isaac Lab articulation and the TrackerLab command transition contract."""

from types import SimpleNamespace

import torch

from trackerLab.kinematics.supermimic.rotations import mat_to_quat, quat_to_mat, yaw_mat
from trackerLab.tracker_env.mdp.supermimic import flatten_state
from trackerLab.tracker_env.mdp.supermimic_command import SCENARIO_DIM
from trackerLab.tracker_env.mdp.supermimic_terrain import FALL_PENALTY, terminal_masks

from .wmp_terrain import CourseEvents
from .parkour import collision_mesh
from .domain import CommandDomain
from .config import DEFAULT_FRICTION_RANGE
from .world_model import clamp_joint_targets
from amp_supermimic.legacy import class_to_dict
from amp_supermimic.scripts.open_terrain import sample_open_flat, update_open_flat_curriculum
from trackerLab.tracker_env.mdp.supermimic_terrain import update_terrain_levels


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
        self.parkour = cfg.terrain.profile == "parkour"
        self.open_flat = cfg.terrain.profile == "open_flat"
        self.merge_fixed_joints = bool(getattr(cfg.robot, "merge_fixed_joints", False))
        randomized = cfg.domain_rand.physical_domain_randomization
        # With multiplication the robot coefficient is the effective foot-ground
        # coefficient because the ground is neutral. With averaging both match.
        ground_friction = 1. if randomized else cfg.terrain.friction
        friction_combine = "multiply" if randomized else "average"
        friction_range = getattr(cfg.terrain, "friction_range", DEFAULT_FRICTION_RANGE)
        material = sim_utils.RigidBodyMaterialCfg(
            static_friction=ground_friction, dynamic_friction=ground_friction,
            friction_combine_mode=friction_combine)
        if atlas is not None or self.parkour:
            import trimesh
            from isaaclab.terrains.utils import create_prim_from_mesh
            vertices, faces = collision_mesh(mdp.field.terrain) if self.parkour else atlas.mesh()
            create_prim_from_mesh("/World/ground", trimesh.Trimesh(
                vertices=vertices, faces=faces, process=False), physics_material=material)
        scene_cfg = InteractiveSceneCfg(
            num_envs=self.num_envs, env_spacing=cfg.terrain.spacing,
            replicate_physics=True, filter_collisions=True)
        scene_cfg.ground = AssetBaseCfg(
            prim_path="/World/ground", collision_group=-1,
            spawn=sim_utils.GroundPlaneCfg(physics_material=material) if atlas is None and not self.parkour else None,
        )
        scene_cfg.light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=900.))
        urdf = robot.resolved_urdf(run_dir / "asset" / "wblwbl_dog.urdf")
        scene_cfg.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UrdfFileCfg(
                asset_path=str(urdf), usd_dir=str(run_dir / "asset"), fix_base=False,
                merge_fixed_joints=self.merge_fixed_joints, activate_contact_sensors=True,
                joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
                    gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0., damping=0.)),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=cfg.domain_rand.self_collisions,
                    solver_position_iteration_count=cfg.env.solver_iterations,
                    solver_velocity_iteration_count=getattr(cfg.env, "solver_velocity_iterations", 1)),
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
        if atlas is not None or self.parkour:
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.sim.reset()
        self.scene.update(cfg.control.physics_dt)
        self.robot = self.scene["robot"]
        self.joint_ids = [self.robot.joint_names.index(name) for name in robot.joints]
        # Flat command tasks merge fixed foot links into calf bodies, matching
        # MuJoCo. Named foot bodies are only required by WMP contact events.
        self.foot_ids = ([self.robot.body_names.index(name) for name in robot.feet]
                         if atlas is not None else [])
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
        self.lane_centers = (torch.as_tensor(mdp.field.terrain.centers, device=self.device, dtype=torch.float32)
                             if self.parkour else None)
        if self.parkour:
            if cfg.terrain.level is None:
                self.levels.random_(0, 2)
            else:
                self.levels.fill_(cfg.terrain.level)
        self.c = torch.zeros(self.num_envs, SCENARIO_DIM, device=self.device)
        self.xi = torch.zeros(self.num_envs, spec.context_dim, device=self.device)
        self._robot_material = self.robot.root_physx_view.get_material_properties().clone()
        self._set_ground_contact_friction(
            torch.arange(self.num_envs, device=self.device),
            self.xi.new_full((self.num_envs,), cfg.terrain.friction),
        )
        self.domain = (CommandDomain(self.robot, self.joint_ids, robot.root, self.xi, mdp.field.scale,
                                     cfg.control, robot.motors, friction_range)
                       if randomized else None)
        self.spec = spec
        self.references = None
        self.curriculum = class_to_dict(cfg.commands.curriculum)
        self.reward_config = class_to_dict(cfg.rewards)
        self.max_episode_steps = cfg.env.max_episode_steps
        self._vel_curriculum_alpha = self.curriculum["initial_alpha"]
        self._vel_curriculum_steps = self._vel_curriculum_env_steps = 0
        self._vel_curriculum_vel_sum = self._vel_curriculum_yaw_sum = 0.
        self._vel_curriculum_dones = self._vel_curriculum_falls = 0
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
            "solver_velocity_iterations": getattr(cfg.env, "solver_velocity_iterations", 1),
            "merge_fixed_joints": self.merge_fixed_joints,
            "reference_resets": {"probability": 0.},
            "terrain": cfg.terrain.profile, "command_samples": [list(row) for row in cfg.commands.samples],
            "physical_domain_randomization": randomized,
            "world_action_input": ("joint_target" if getattr(cfg.control, "project_joint_targets", False)
                                   else "policy_action"),
            "contact_friction": {
                "initial": cfg.terrain.friction, "ground_material": ground_friction,
                "combine_mode": friction_combine, "context_index": 0,
                "training_reset_range": list(friction_range) if randomized else None,
                "fixed_command_reset": "retain current coefficient",
            },
            "velocity_command_curriculum": self.curriculum if self.open_flat else None,
            "reward_config": self.reward_config,
            "wmp_events": "real rollouts only" if atlas is not None else None,
        }

    def state(self):
        data = self.robot.data
        position = data.root_link_pos_w
        if self.atlas is None and not self.parkour:
            position = position - self.scene.env_origins
        return (position.clone(), quat_to_mat(data.root_link_quat_w),
                data.root_link_lin_vel_w.clone(), data.root_link_ang_vel_w.clone(),
                data.joint_pos[:, self.joint_ids], data.joint_vel[:, self.joint_ids])

    def observe(self):
        return self.mdp.obs(self.state(), self.a_prev, self.xi, self.c)

    def _set_ground_contact_friction(self, ids, friction):
        cpu_ids = ids.cpu()
        self._robot_material[cpu_ids, :, :2] = friction.cpu()[:, None, None]
        self.robot.root_physx_view.set_material_properties(self._robot_material, cpu_ids)
        self.xi[ids, 0] = friction

    def set_reference_resets(self, motion, probability, grounded_mask=None):
        if not probability:
            self.references = None
            return
        # Reuse TrackerLab's frame eligibility filter. Applying the selected poses
        # uses PhysX articulation writes in reset(), independent of MuJoCo storage.
        proxy = SimpleNamespace(contract={}, spec=self.spec, robot=self.cfg.robot.name, qpos=self.c,
                                self_collisions=False, height=self.cfg.robot.standing_height,
                                target_clearance_m=self.mdp.target_clearance_m, mdp=self.mdp,
                                scale=self.mdp.field.scale, device=self.device, qlo=self.qlo, qhi=self.qhi,
                                action_limit=self.cfg.env.action_limit,
                                action_scale=self.c.new_tensor(self.spec.action_scale), default_q=self.default_q)
        receipt = self.reference_filter(proxy, motion, probability, grounded_mask)
        self.references = proxy._reference_reset_states
        self.reference_probability = probability
        self.contract["reference_resets"] = receipt

    def sample_commands(self, ids, *, fixed=False):
        if fixed or not (self.open_flat or self.parkour):
            commands = self.twists[torch.randint(len(self.twists), (len(ids),), device=self.device)]
        elif self.open_flat:
            commands, _ = sample_open_flat(self.curriculum, self._vel_curriculum_alpha,
                                           len(ids), self.device, self.c.dtype,
                                           getattr(self.cfg.commands, "training_modes", None))
        else:
            commands = self.c.new_zeros((len(ids), 3))
            commands[:, 0] = (.5 + .6 * torch.rand(len(ids), device=self.device)) * self.mdp.field.scale
        self.c[ids, 1:4] = commands
        self.c[ids, 4] = self.mdp.target_clearance_m
        self.c[ids, 6] = 0.
        self.command_age[ids] = 0
        low, high = self.cfg.commands.resample_steps
        self.command_interval[ids] = torch.randint(low, high + 1, (len(ids),), device=self.device)

    def _reference_root_height(self, root_xy, env_origins, reference_z):
        """Place a reference root at its recorded height above the sampled terrain."""
        sample_xy = root_xy
        if self.atlas is None and not self.parkour:
            # Plane/open-flat HeightField coordinates are environment-local.
            sample_xy = root_xy - env_origins
        # WMP atlas and parkour coordinates are already in their field's frame.
        return self.mdp.field.sample(sample_xy) + reference_z

    @torch.no_grad()
    def reset(self, ids=None, *, fixed_command=False):
        ids = torch.arange(self.num_envs, device=self.device) if ids is None else ids
        if not len(ids):
            return
        self.scene.reset(ids)
        domain = self.domain
        if domain is not None and not fixed_command:
            self._set_ground_contact_friction(ids, domain.reset(ids))
        root = self.robot.data.default_root_state[ids].clone()
        origins = self.scene.env_origins if self.events is None else self.events.core.env_origins
        root[:, :3] += origins[ids]
        if self.open_flat or self.parkour:
            scale = self.mdp.field.scale
            if self.open_flat:
                xy = (2 * torch.rand((len(ids), 2), device=self.device) - 1) * (4 * scale)
                yaw = (2 * torch.rand(len(ids), device=self.device) - 1) * torch.pi
            else:
                center = self.lane_centers[self.levels[ids]]
                xy = torch.stack((torch.zeros_like(center), center +
                                  (torch.rand(len(ids), device=self.device) - .5) * .15 * scale), -1)
                yaw = (torch.rand(len(ids), device=self.device) - .5) * .12
            root[:, :2] = xy if self.parkour else xy + origins[ids, :2]
            root[:, 2] = self.mdp.field.sample(xy) + self.cfg.robot.standing_height + .02 * scale
            root[:, 3:7] = torch.stack((torch.cos(yaw / 2), torch.zeros_like(yaw),
                                        torch.zeros_like(yaw), torch.sin(yaw / 2)), -1)
        q = (self.default_q + self.xi[ids, 1:-3]).clamp(self.qlo, self.qhi)
        qd = torch.zeros_like(q)
        self.a_prev[ids] = self.c[ids] = 0
        if self.references is not None and not fixed_command:
            take = torch.rand(len(ids), device=self.device) < self.reference_probability
            frames = torch.randint(len(self.references[0]), (int(take.sum().item()),), device=self.device)
            p, rotation, v, w, ref_q, ref_qd = (value[frames] for value in self.references)
            alignment = yaw_mat(quat_to_mat(root[take, 3:7])) @ yaw_mat(rotation).transpose(-1, -2)
            root[take, 2] = self._reference_root_height(
                root[take, :2], origins[ids[take], :2], p[:, 2])
            root[take, 3:7] = mat_to_quat(alignment @ rotation)
            root[take, 7:10] = (alignment @ v[..., None]).squeeze(-1)
            root[take, 10:13] = (alignment @ w[..., None]).squeeze(-1)
            offsets = self.xi[ids[take], 1:-3]
            q[take] = (ref_q + offsets).clamp(self.qlo, self.qhi)
            qd[take] = ref_qd
            self.a_prev[ids[take]] = ((q[take] - self.default_q - offsets) /
                                     self.cfg.control.action_scale).clamp(-self.cfg.env.action_limit,
                                                                         self.cfg.env.action_limit)
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids=ids)
        self.robot.write_root_link_velocity_to_sim(root[:, 7:], env_ids=ids)
        pos = self.robot.data.default_joint_pos[ids].clone()
        vel = torch.zeros_like(pos)
        pos[:, self.joint_ids], vel[:, self.joint_ids] = q, qd
        self.robot.write_joint_state_to_sim(pos, vel, env_ids=ids)
        self.robot.set_joint_position_target(pos, env_ids=ids)
        self.ep_step[ids] = self.ep_return[ids] = self.ep_tracking[ids] = 0
        if self.parkour:
            self.c[ids, 0] = self.lane_centers[self.levels[ids]]
        elif self.events is not None:
            self.c[ids, 0] = origins[ids, 1]
        self.sample_commands(ids, fixed=fixed_command)

    @torch.no_grad()
    def step(self, action, *, fixed_command=False, auto_reset=True):
        before = self.state()
        command, previous = self.c.clone(), self.a_prev.clone()
        time_left = self.command_interval - self.command_age
        a = action.clamp(-self.cfg.env.action_limit, self.cfg.env.action_limit)
        target = clamp_joint_targets(a, self.default_q + self.xi[:, 1:-3],
                                     self.cfg.control.action_scale, self.qlo, self.qhi)
        self.robot.set_joint_position_target(target, joint_ids=self.joint_ids)
        for _ in range(self.cfg.control.decimation):
            self.scene.write_data_to_sim()
            self.sim.step(render=self.sim.has_gui())
            self.scene.update(self.cfg.control.physics_dt)
        after = self.state()
        self.c.copy_(self.mdp.advance_command(command, self.dt))
        self.ep_step += 1
        self.command_age += 1
        timeout = self.ep_step >= self.cfg.env.max_episode_steps
        fallen = self.mdp.termination(after, self.c)
        reached = (after[0][:, 0] >= self.mdp.field.terrain.goal_x
                   if self.parkour else torch.zeros_like(fallen))
        event_rewards = {}
        if self.events is not None:
            data = self.robot.data
            foot_rotation = quat_to_mat(data.body_link_quat_w[:, self.foot_ids])
            sole = data.body_link_pos_w[:, self.foot_ids] + (
                foot_rotation @ self.foot_offsets[..., None]).squeeze(-1)
            sole[..., 2] -= self.foot_radii
            forces = self.contacts.data.net_forces_w
            base_hit = forces[:, self.root_contact_id].norm(dim=-1) > self.cfg.terrain.body_contact_limit
            if self.cfg.terrain.profile == "high_wall":
                # The wall may support the body during a climb. A grounded
                # trunk at floor height is still a failure.
                base_hit = base_hit & (after[0][:, 2] < .12)
            fallen |= base_hit | (after[2][:, 2] < -3.)
            reached, fallen, event_rewards = self.events.update(
                after, sole, forces[:, self.contact_ids], self.c, fallen, timeout,
                self.ep_step, self.cfg.rewards)
            reached, fallen = reached.clone(), fallen.clone()
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
        if self.events is None and not self.parkour:
            success = truncated & (self.ep_tracking / self.ep_step >= .8)
            if self.open_flat:
                success &= self.ep_step * self.dt >= 10.
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
        # Match MuJoCo CommandSim.step(): the next actor observation must
        # receive the action just executed. Reset below then initializes done envs.
        self.a_prev.copy_(a)
        ids = done.nonzero(as_tuple=False).flatten()
        if self.events is not None and len(ids):
            self.events.finish(ids)
        if self.parkour and self.cfg.terrain.level is None:
            self.levels[ids] = update_terrain_levels(self.levels[ids], success[ids], fallen[ids])
        if auto_reset:
            self.reset(ids, fixed_command=fixed_command)
        self.sample_commands(resampled.nonzero(as_tuple=False).flatten(), fixed=fixed_command)
        if self.open_flat and not fixed_command and self.curriculum["enabled"]:
            update_open_flat_curriculum(self, row, self.curriculum, self.reward_config)
        return row
