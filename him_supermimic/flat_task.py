"""Differentiable WBLWBL flat-ground task functions over A1 state tensors."""

from collections.abc import Mapping

import torch


CORE_TERMS = (
    "action_rate", "ang_vel_xy", "base_height", "dof_acc", "dof_pos",
    "lin_vel_z", "orientation", "rear_thigh_pos", "smoothness",
    "tracking_ang_vel", "tracking_lin_vel",
)


def rotate_xyzw(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz = quat[..., :3]
    vector = vector.expand_as(xyz)
    twice_cross = 2 * torch.cross(xyz, vector, dim=-1)
    return vector + quat[..., 3:] * twice_cross + torch.cross(xyz, twice_cross, dim=-1)


def projected_gravity(state: torch.Tensor) -> torch.Tensor:
    quat = state[..., 3:7]
    inverse = torch.cat((-quat[..., :3], quat[..., 3:]), dim=-1)
    return rotate_xyzw(inverse, state.new_tensor([0, 0, -1]))


class FlatTask:
    """Pure Torch bridge for Stage 3. State uses A1's 37-dimensional layout."""

    def __init__(self, legacy: Mapping, joint_names: list[str]):
        if legacy["terrain"]["mesh_type"] != "plane" or len(joint_names) != 12:
            raise ValueError("Expected the WBLWBL flat-ground configuration with 12 joints")
        self.dt = legacy["sim"]["dt"] * legacy["control"]["decimation"]
        self.command_period = int(legacy["commands"]["resampling_time"] / self.dt)
        self.heading_command = legacy["commands"]["heading_command"]
        self.normalization = legacy["normalization"]
        self.scales = self.normalization["obs_scales"]
        self.noise = legacy["noise"]
        self.reward_cfg = legacy["rewards"]
        self.default_q = [legacy["init_state"]["default_joint_angles"][name] for name in joint_names]
        self.rear_thigh_indices = [i for i, name in enumerate(joint_names)
                                   if name.lower().startswith(("rr_thigh", "rl_thigh"))]
        self.q_weights = [
            self.reward_cfg.get(f"dof_pos_{name.split('_')[1].lower()}_weight", 1.0)
            for name in joint_names
        ]
        self.height_count = legacy["env"]["num_one_step_privileged_obs"] - 51
        self.history_width = legacy["env"]["num_observations"]

    def command_resampled(self, episode_step: torch.Tensor) -> torch.Tensor:
        """A1 stores the counter before Isaac Lab increments it."""
        return (episode_step + 1).remainder(self.command_period) == 0

    def update_command(self, command: torch.Tensor, next_state: torch.Tensor,
                       episode_step: torch.Tensor, sampled_command: torch.Tensor) -> torch.Tensor:
        """Inject the exogenous sampled command on its scheduled step, then update yaw.

        The sampler supplies physical units after the legacy dead zone. Supply an
        arbitrary placeholder for rows where no resampling is scheduled.
        """
        resampled = self.command_resampled(episode_step)
        command = torch.where(resampled.unsqueeze(-1), sampled_command, command)
        if not self.heading_command:
            return command
        forward = rotate_xyzw(next_state[..., 3:7], next_state.new_tensor([1, 0, 0]))
        heading = torch.atan2(forward[..., 1], forward[..., 0])
        error = (command[..., 3] - heading).remainder(2 * torch.pi)
        error = torch.where(error > torch.pi, error - 2 * torch.pi, error)
        yaw_rate = (0.5 * error).clamp(-2, 2)
        return torch.cat((command[..., :2], yaw_rate.unsqueeze(-1), command[..., 3:4]), dim=-1)

    def clip_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        limit = self.normalization["clip_actions"]
        return raw_action.clamp(-limit, limit)

    def actor_frame(self, state: torch.Tensor, command: torch.Tensor,
                    executed_action: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        scales = self.scales
        command_scale = state.new_tensor([scales["lin_vel"], scales["lin_vel"], scales["ang_vel"]])
        frame = torch.cat((
            command[..., :3] * command_scale,
            state[..., 10:13] * scales["ang_vel"],
            projected_gravity(state),
            (state[..., 13:25] - state.new_tensor(self.default_q)) * scales["dof_pos"],
            state[..., 25:37] * scales["dof_vel"],
            executed_action,
        ), dim=-1)
        return frame if noise is None else frame + noise

    def update_history(self, prior_history: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        limit = self.normalization["clip_observations"]
        return torch.cat((frame, prior_history[..., :self.history_width - 45]), dim=-1).clamp(-limit, limit)

    def critic_frame(self, state: torch.Tensor, frame: torch.Tensor, disturbance: torch.Tensor,
                     height_noise: torch.Tensor | None = None) -> torch.Tensor:
        # Flat ground height is zero. The A1 state uses plane origins with z = 0.
        heights = (state[..., 2:3] - 0.5).clamp(-1, 1) * self.scales["height_measurements"]
        heights = heights.expand(*state.shape[:-1], self.height_count)
        if height_noise is not None:
            heights = heights + height_noise
        return torch.cat((frame, state[..., 7:10] * self.scales["lin_vel"], disturbance, heights), dim=-1)

    def observation_noise_bounds(self, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.noise["noise_scales"]
        scale = self.noise["noise_level"]
        frame = like.new_tensor(
            [0] * 3 + [n["ang_vel"] * self.scales["ang_vel"]] * 3 + [n["gravity"]] * 3
            + [n["dof_pos"] * self.scales["dof_pos"]] * 12
            + [n["dof_vel"] * self.scales["dof_vel"]] * 12 + [0] * 12
        ) * (scale if self.noise["add_noise"] else 0)
        height = like.new_full((self.height_count,), n["height_measurements"]
                               * self.scales["height_measurements"] * scale)
        return frame, height

    def reward_terms(self, state: torch.Tensor, next_state: torch.Tensor, command: torch.Tensor,
                     action: torch.Tensor, prev_action: torch.Tensor, prev_prev_action: torch.Tensor,
                     *, terminal_torque: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Weighted terms, matching legacy dt scaling and post-physics timing."""
        velocity = next_state[..., 7:10]
        angular = next_state[..., 10:13]
        q_error = (next_state[..., 13:25] - next_state.new_tensor(self.default_q)).square()
        sigma = self.reward_cfg["tracking_sigma"]
        terms = {
            "action_rate": (prev_action - action).square().sum(-1),
            "ang_vel_xy": angular[..., :2].square().sum(-1),
            "base_height": (next_state[..., 2] - self.reward_cfg["base_height_target"]).square(),
            "dof_acc": ((state[..., 25:37] - next_state[..., 25:37]) / self.dt).square().sum(-1),
            "dof_pos": (q_error * next_state.new_tensor(self.q_weights)).sum(-1),
            "lin_vel_z": velocity[..., 2].square(),
            "orientation": projected_gravity(next_state)[..., :2].square().sum(-1),
            "rear_thigh_pos": q_error[..., self.rear_thigh_indices].sum(-1),
            "smoothness": (action - prev_action - prev_action + prev_prev_action).square().sum(-1),
            "tracking_ang_vel": (-(command[..., 2] - angular[..., 2]).square() / sigma).exp(),
            "tracking_lin_vel": (-(command[..., :2] - velocity[..., :2]).square().sum(-1) / sigma).exp(),
        }
        if terminal_torque is not None:
            terms["joint_power"] = (next_state[..., 25:37].abs() * terminal_torque.abs()).sum(-1)
        scales = self.reward_cfg["scales"]
        return {name: value * (scales[name] * self.dt) for name, value in terms.items()}
