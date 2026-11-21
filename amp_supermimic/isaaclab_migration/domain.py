"""Apply the TrackerLab command randomization distribution to Isaac Lab."""

import torch

from trackerLab.algorithms.supermimic.command_domain_randomization import command_domain_scales


class CommandDomain:
    def __init__(self, robot, joint_ids, root_name, xi, scale, control, motors, friction_range):
        self.xi = xi
        self.friction_range = friction_range
        self.view = robot.root_physx_view
        self.root = robot.body_names.index(root_name)
        n = len(xi)
        ids = torch.arange(n, device=xi.device)
        cpu_ids = ids.cpu()
        xi[:, 1:5].uniform_(-.01, .01)
        xi[:, -3:].uniform_(-.025 * scale, .025 * scale)
        com = self.view.get_coms()
        com[:, self.root, :3] += xi[:, -3:].cpu()
        self.view.set_coms(com, cpu_ids)
        m, kp, kd, effort = command_domain_scales(xi, len(joint_ids)).unbind(-1)
        self.view.set_masses(robot.data.default_mass.cpu() * m.cpu()[:, None], cpu_ids)
        self.view.set_inertias(robot.data.default_inertia.cpu() * m.cpu()[:, None, None], cpu_ids)
        robot.write_joint_stiffness_to_sim((control.kp * kp[:, None]).expand(-1, len(joint_ids)), joint_ids, ids)
        robot.write_joint_damping_to_sim((control.kd * kd[:, None]).expand(-1, len(joint_ids)), joint_ids, ids)
        robot.write_joint_effort_limit_to_sim(
            xi.new_tensor([row[0] for row in motors]) * effort[:, None], joint_ids, ids)

    def reset(self, ids):
        xi = self.xi
        low, high = self.friction_range
        friction = low + (high - low) * torch.rand(len(ids), device=xi.device)
        xi[ids, 5:-3] = (2 * torch.rand((len(ids), xi.shape[1] - 8), device=xi.device) - 1) * .01
        return friction
