"""Use WMP combo collision meshes and course bookkeeping in the PhysX backend."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch


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
                             else (cfg.profile,))
        self.rows = max(self.geometry.level_count(name) for name in self.column_types)
        self.cols = len(self.column_types)
        self.cells = {
            (row, col): self.geometry.make_course(name, row)
            for col, name in enumerate(self.column_types)
            for row in range(self.geometry.level_count(name))
        }
        self.query = self.geometry.HeightQuery(self.cells, self.rows, self.cols, device)
        spawns = np.zeros((self.rows, self.cols, 3), dtype=np.float32)
        for (row, col), course in self.cells.items():
            spawns[row, col] = np.array(course.spawn) + self.center(row, col)
        self.spawns = torch.as_tensor(spawns, device=device)

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
    """Expose only the state WMP's combo rules consume."""

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
        self.course.select_levels(ids)
        self.course.reset(ids)
