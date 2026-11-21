"""Triangulate the same seeded heightfield used by TrackerLab's terrain MDP."""

import numpy as np


def collision_mesh(terrain):
    heights = terrain.heights
    rows, cols = heights.shape
    x, y = np.meshgrid(np.linspace(terrain.x_min, terrain.x_max, cols),
                       np.linspace(terrain.y_min, terrain.y_max, rows))
    vertices = np.stack((x, y, heights), axis=-1).reshape(-1, 3)
    ids = np.arange(rows * cols).reshape(rows, cols)
    lo, right, hi, left = ids[:-1, :-1], ids[:-1, 1:], ids[1:, 1:], ids[1:, :-1]
    faces = np.stack((np.stack((lo, right, hi), -1), np.stack((lo, hi, left), -1)), -2)
    return vertices, faces.reshape(-1, 3)
