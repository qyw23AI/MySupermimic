"""Transition capture for the existing WBLWBL Isaac Lab environment."""

from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

import torch
from himloco_lab import env_cfg
from himloco_lab.config import _urdf_mesh_path, asset_contract
from himloco_lab.env import HIMLabEnv
from himloco_lab.math import quat_rotate_inverse


def make_flat_env_cfg(legacy, device: str, cache: Path):
    """Use the upstream config builder with a run-local URDF conversion directory."""
    source, _, _, _ = asset_contract(legacy)

    def prepare_urdf(path):
        tree = ElementTree.parse(path)
        for mesh in tree.iter("mesh"):
            mesh.set("filename", str(_urdf_mesh_path(path, mesh.attrib["filename"])))
        for joint in tree.findall("joint[@type='fixed']"):
            axis = joint.find("axis")
            if axis is not None:
                joint.remove(axis)
        target = cache / "urdf" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        tree.write(target, encoding="utf-8", xml_declaration=True)
        return target

    # The upstream builder resolves its URDF preparer and USD directory at call time.
    with patch.object(env_cfg, "prepare_urdf", prepare_urdf):
        cfg = env_cfg.make_env_cfg(legacy, stage=3, device=device)
    cfg.robot.spawn.usd_dir = str(cache / "usd")
    return cfg


class RecordingHIMLabEnv(HIMLabEnv):
    """Own the capture hooks. The original Lab step and PPO adapter still manage resets."""

    def __init__(self, cfg, **kwargs):
        self.replay = None
        self._transition = None
        self.command_override = None
        super().__init__(cfg, **kwargs)
        self.episode_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        core = self.core
        self.reward_term_names = tuple(core.reward_names)
        weights = tuple(core.reward_scales[name] for name in self.reward_term_names)
        self._term_values = {}
        core.reward_functions = [
            self._record_reward_term(name, func, weight)
            for name, func, weight in zip(self.reward_term_names, core.reward_functions, weights)
        ]

    def _record_reward_term(self, name, func, weight):
        def evaluate():
            value = func()
            if self._transition is not None:
                self._term_values[name] = value * weight
            return value
        return evaluate

    def physical_state(self):
        core = self.core
        root = self.robot.data.root_state_w
        quat = root[:, [4, 5, 6, 3]]
        return torch.cat((
            root[:, :3] - core.env_origins,
            quat,
            quat_rotate_inverse(quat, root[:, 7:10]),
            quat_rotate_inverse(quat, root[:, 10:13]),
            self.robot.data.joint_pos[:, self.joint_ids],
            self.robot.data.joint_vel[:, self.joint_ids],
        ), dim=-1)

    def randomize_body_properties(self, core, ids=None):
        super().randomize_body_properties(core, ids)
        view = self.robot.root_physx_view
        self.mass = view.get_masses().to(self.device)
        self.com = view.get_coms()[..., :3].to(self.device)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "episode_id"):
            self.episode_id[env_ids] += 1

    def _pre_physics_step(self, actions):
        core = self.core
        if self.replay is not None:
            self._transition = {
                "state": self.physical_state(),
                "actor_history": core.obs_buf.clone(),
                "episode_id": self.episode_id.clone(),
                "episode_step": self.episode_length_buf.clone(),
                "raw_action": actions.clone(),
                "prev_action": core.last_actions.clone(),
                "prev_prev_action": core.last_last_actions.clone(),
                "command": core.commands.clone(),
                "friction": core.friction_coeffs.clone(),
                "restitution": core.restitution_coeffs.clone(),
                "kp_factor": core.Kp_factors.clone(),
                "kd_factor": core.Kd_factors.clone(),
                "body_mass": self.mass.clone(),
                "body_com": self.com.clone(),
                "applied_force_body": self.pending_force[:, core.base_body_id].clone(),
            }
        super()._pre_physics_step(actions)
        if self._transition is not None:
            # The effective substep commands encode delays even when successive actions are identical.
            self._transition["executed_action"] = core.actions.clone()
            self._transition["substep_actions"] = core.delayed_actions.clone()

    def _get_rewards(self):
        if self.command_override is not None:
            self.core.commands[:, :3] = self.command_override
        self._term_values.clear()
        reward = super()._get_rewards()
        if self._transition is not None:
            core = self.core
            period = int(core.cfg.commands.resampling_time / self.step_dt)
            physical_end = self.physical_state()
            self._transition.update(
                next_state_pre_reset=physical_end,
                reward=reward.clone(),
                reward_terms=torch.stack([self._term_values[n] for n in self.reward_term_names], dim=-1),
                terminated=self.reset_terminated.clone(),
                truncated=self.reset_time_outs.clone(),
                terminal_contact=torch.linalg.vector_norm(
                    core.contact_forces[:, core.termination_contact_indices], dim=-1
                ).amax(dim=-1),
                terminal_torque=core.torques.clone(),
                command_pre_reset=core.commands.clone(),
                command_resampled=(self.episode_length_buf % period == 0),
                push_applied=torch.full_like(
                    self.reset_terminated,
                    bool(core.cfg.domain_rand.push_robots and self.common_step_counter % core.cfg.domain_rand.push_interval == 0),
                ),
                pending_force_next_body=self.pending_force[:, core.base_body_id].clone(),
            )
        return reward

    def _get_observations(self):
        if self.command_override is not None:
            self.core.commands[:, :3] = self.command_override
        return super()._get_observations()

    def step(self, action):
        result = super().step(action)
        if self._transition is not None:
            observations = result[0]
            next_critic = observations["critic"].clone()
            next_critic[self.terminal_ids] = self.terminal_obs
            self._transition.update(
                next_critic_obs_pre_reset=next_critic,
                actor_history_after_step=observations["policy"].clone(),
            )
            self.replay.append(self._transition)
            self._transition = None
        return result
