"""Use WMP collision meshes and course bookkeeping in the PhysX backend."""

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch


OBSTACLE_LEVELS = {"narrow_bridge": 9, "high_wall": 6}


def load_wmp(source):
    source = Path(source).resolve(strict=True)
    package = source / "wmp_lab"
    for name in ("competition.py", "competition_tasks.py"):
        if not (package / name).is_file():
            raise FileNotFoundError(package / name)
    loaded = sys.modules.get("wmp_lab")
    if loaded is not None and Path(loaded.__file__).resolve().parent != package:
        raise RuntimeError("A different wmp_lab installation is already imported")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from wmp_lab import competition, competition_tasks
    return competition, competition_tasks


class CourseAtlas:
    """Positions are in shared mesh world coordinates, including in imagined states."""

    tile_size = 8.

    def __init__(self, cfg, device):
        from amp_supermimic.legacy import resolve_repo_path
        self.cfg = cfg
        self.geometry, self.tasks = load_wmp(resolve_repo_path(cfg.source))
        self.column_types = (self.geometry.COMBO_TYPES if cfg.profile == "gap_combos"
                             else ("platform",) if cfg.profile == "narrow_bridge" else (cfg.profile,))
        target = OBSTACLE_LEVELS.get(cfg.profile)
        self.rows = target + 1 if target is not None else max(
            self.geometry.level_count(name) for name in self.column_types)
        self.cols = len(self.column_types)
        self.cells = {
            (row, col): self.geometry.make_course(name, row)
            for col, name in enumerate(self.column_types)
            for row in range(min(self.rows, self.geometry.level_count(name)))
        }
        if cfg.profile == "narrow_bridge":
            self.cells = {(row, 0): self.bridge(row) for row in range(self.rows)}
        self.query = self.geometry.HeightQuery(self.cells, self.rows, self.cols, device)
        spawns = np.zeros((self.rows, self.cols, 3), dtype=np.float32)
        for (row, col), course in self.cells.items():
            spawns[row, col] = np.array(course.spawn) + self.center(row, col)
        self.spawns = torch.as_tensor(spawns, device=device)

    def bridge(self, row):
        fraction = row / OBSTACLE_LEVELS["narrow_bridge"]
        width, height = .70 - .30 * fraction, .05 + .05 * fraction
        template = self.geometry.make_course("platform", row)
        return replace(template, boxes=[[0., 0., 3., width, height]], spawn=(-2.25, 0., 0.),
                       entry=-1.5, exit=1.5, half_width=width / 2, height=height,
                       parameters={"length_m": 3., "width_m": width, "height_m": height})

    def center(self, row, col):
        return ((row + .5) * self.tile_size, (col + .5) * self.tile_size, 0.)

    def mesh(self):
        pieces = []
        for (row, col), spec in self.cells.items():
            vertices, faces = self.geometry.course_mesh(spec, ground=True)
            pieces.append((vertices + np.asarray(self.center(row, col)), faces))
        return self.geometry.join_meshes(pieces)

    def sample(self, xy):
        flat = xy.reshape(-1, 2)
        cells = (flat / self.tile_size).floor()
        row = cells[:, 0].long().clamp(0, self.rows - 1)
        col = cells[:, 1].long().clamp(0, self.cols - 1)
        center = torch.stack((row, col), -1).to(flat).add(.5).mul(self.tile_size)
        return self.query((flat - center)[:, None], row, col)[:, 0].reshape(xy.shape[:-1])


class CourseEvents:
    """Adapt PhysX contacts to WMP course rules and obstacle-specific checks."""

    def __init__(self, atlas, count, device, dt):
        self.atlas = atlas
        cfg = atlas.cfg
        cols = torch.arange(count, device=device) % atlas.cols
        levels = torch.full_like(cols, cfg.level)
        origins = atlas.spawns[levels, cols].clone()
        self.core = SimpleNamespace(
            cfg=SimpleNamespace(terrain=SimpleNamespace(
                num_rows=atlas.rows, num_cols=atlas.cols,
                competition_stage=cfg.competition_stage,
                competition_window=cfg.competition_window,
                competition_promote=cfg.competition_promote, competition_demote=cfg.competition_demote,
            )),
            terrain=SimpleNamespace(column_types=atlas.column_types, competition_cells=atlas.cells,
                                    env_length=atlas.tile_size, env_width=atlas.tile_size),
            num_envs=count, device=device, dt=dt,
            terrain_levels=levels, terrain_types=cols,
            env_origins=origins, terrain_origins=atlas.spawns,
            feet_indices=torch.arange(4, device=device), foot_names=("RR_foot", "RL_foot", "FR_foot", "FL_foot"),
        )
        self.course = atlas.tasks.CourseState(self.core)
        self.profile = cfg.profile
        self.deck_steps = torch.zeros(count, dtype=torch.long, device=device)
        self.diagonal_steps = torch.zeros((count, 2), dtype=torch.long, device=device)
        self.last_diagonal = torch.zeros_like(self.deck_steps)
        self.diagonal_switches = torch.zeros_like(self.deck_steps)
        self.front_support_frames = torch.zeros_like(self.deck_steps)
        self.min_wall_support_frames = round(.08 / dt)
        self.top_support = torch.zeros(count, dtype=torch.bool, device=device)

    def obstacle_rewards(self, root, sole, forces, rewards):
        course, core = self.course, self.core
        if self.profile not in OBSTACLE_LEVELS:
            return {}
        local = sole[:, :, :2] - course.centers[:, None]
        body_xy = root[:, :2] - course.centers
        bounds = self.atlas.query.bounds[core.terrain_levels, core.terrain_types]
        height = self.atlas.query.boxes[core.terrain_levels, core.terrain_types, 0, 4]
        grounded = forces[..., 2] > 1.
        on_top = grounded & ((sole[..., 2] - height[:, None]).abs() < .018)
        on_top &= ((local[..., 0] >= bounds[:, :1]) & (local[..., 0] <= bounds[:, 1:2])
                   & (local[..., 1].abs() < bounds[:, 2:3]))
        if self.profile == "high_wall":
            front = on_top[:, 2:].any(-1)
            self.front_support_frames = torch.where(front, self.front_support_frames + 1, 0)
            supported = self.front_support_frames >= self.min_wall_support_frames
            gained = supported & ~self.top_support
            self.top_support |= supported
            course.strict &= self.top_support
            return {"wall_front_support": rewards.top_support * gained}

        # The robot is fully on the obstacle when its trunk is at least one
        # stance length from either end. Entry/exit can contain floor contacts.
        midspan = body_xy[:, 0].abs() < 1.10
        ground_off_deck = grounded & (sole[..., 2] < height[:, None] * .5)
        invalid = (midspan & ((ground_off_deck.sum(-1) >= 2)
                             | (body_xy[:, 1].abs() > bounds[:, 2])))
        invalid |= (body_xy[:, 0] > .45) & ~course.entered
        course.violated |= invalid
        course.failed |= invalid
        course.success &= ~invalid
        diag_a = on_top[:, 0] & on_top[:, 3] & ~grounded[:, 1] & ~grounded[:, 2]
        diag_b = on_top[:, 1] & on_top[:, 2] & ~grounded[:, 0] & ~grounded[:, 3]
        self.deck_steps += midspan
        self.diagonal_steps += torch.stack((diag_a, diag_b), -1) & midspan[:, None]
        diagonal = (diag_a.long() + 2 * diag_b.long()) * midspan
        valid_pair = diagonal > 0
        self.diagonal_switches += valid_pair & (self.last_diagonal > 0) & (diagonal != self.last_diagonal)
        self.last_diagonal = torch.where(valid_pair, diagonal, self.last_diagonal)
        gait_ok = ((self.diagonal_steps.amin(-1) >= .15 * self.deck_steps)
                   & (self.deck_steps >= 30) & (self.diagonal_switches >= 4))
        course.strict &= course.success & gait_ok
        return {}

    def update(self, state, sole, forces, command, fallen, timeout, episode_steps, rewards):
        core, course = self.core, self.course
        pos, rotation, linear, _, _, _ = state
        core.root_states = torch.cat((pos, pos.new_zeros((len(pos), 4)), linear), -1)
        core.projected_gravity = -rotation[:, 2, :]
        # CourseState uses A1 foot-center z minus .02. Supply a virtual center
        # so its existing tests see the actual WBLWBL collision sphere sole.
        core.rigid_body_pos = sole.clone()
        core.rigid_body_pos[..., 2] += .02
        core.contact_forces = forces
        core.commands = command[:, 1:4]
        core.reset_buf = (fallen | timeout).clone()
        core.time_out_buf = (timeout & ~fallen).clone()
        core.episode_length_buf = episode_steps
        previous = course.coverage.clone()
        previous_strict = course.strict.clone()
        course.update()
        obstacle_rewards = self.obstacle_rewards(pos, sole, forces, rewards)
        if self.profile in OBSTACLE_LEVELS:
            core.time_out_buf &= ~(course.failed | course.success)
            return course.success, course.failed, obstacle_rewards | {
                "strict_success": rewards.strict_success * (course.strict & ~previous_strict),
            }
        new = course.coverage & ~previous
        ramp = course.kind == self.atlas.geometry.COMPETITION_TYPES.index("stair_gap_ramp")
        event_rewards = {
            "stairs": rewards.stair_support * new[:, :2].sum(-1),
            "landing": rewards.landing * new[:, 3],
            "downhill": rewards.downhill * (new[:, 4] & ramp),
            "strict_success": rewards.strict_success * (course.strict & ~previous_strict),
        }
        return course.success, course.failed, event_rewards

    def finish(self, ids):
        self.course.finish(ids)
        target = OBSTACLE_LEVELS.get(self.profile)
        if target is not None:
            self.course.frontier.clamp_(max=target)
        self.course.select_levels(ids)
        self.course.reset(ids)
        self.deck_steps[ids] = 0
        self.diagonal_steps[ids] = 0
        self.last_diagonal[ids] = 0
        self.diagonal_switches[ids] = 0
        self.front_support_frames[ids] = 0
        self.top_support[ids] = False
