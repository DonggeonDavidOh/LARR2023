import sys

import numpy as np

from astar import *


def plot_global_map(path, obstacles):
    fig, ax = plt.subplots()
    for o in obstacles:
        patch = o.get_plot_patch()
        ax.add_patch(patch)
    ax.plot(path[:, 0], path[:, 1])
    plt.xlim([-1 * 0.15, 11 * 0.15])
    plt.ylim([0 * 0.15, 8 * 0.15])
    plt.show()


class AstarPathGenerator:
    def __init__(self, grid, quad, margin):
        self._global_path = None
        self._grid = GridMap(bounds=grid[0], cell_size=grid[1], quad=quad)
        self._margin = margin

    def generate_path(self, sys, obstacles, goal_pos):
        graph = GraphSearch(graph=self._grid, obstacles=obstacles, margin=self._margin)
        path = graph.a_star(sys.get_state()[:2], goal_pos)
        self._global_path = np.array([p.pos for p in path])
        print(self._global_path)
        if self._global_path == []:
            print("Global Path not found.")
            sys.exit(1)
        if True:
            plot_global_map(self._global_path, obstacles)
        return self._global_path

    def logging(self, logger):
        logger._paths.append(self._global_path)


class AstarLoSPathGenerator:
    # TODO: design a global planner for 3D environment
    # def __init__(self, grid, quad, margin):
    def __init__(self, start_pos, goal_pos):
        self._global_path = None
        self._start_pos = start_pos
        self._goal_pos = goal_pos
        # self._grid = GridMap(bounds=grid[0], cell_size=grid[1], quad=quad)
        # self._margin = margin

    def generate_path(self):
        # graph = GraphSearch(graph=self._grid, obstacles=obstacles, margin=self._margin)
        # path = graph.a_star(sys.get_state()[:2], goal_pos)
        # path = graph.reduce_path(path)
        # self._global_path = np.array([p.pos for p in path])
        # print(self._global_path)
        # if self._global_path == []:
        #     print("Global Path not found.")
        #     sys.exit(1)
        # if False:
        #     plot_global_map(self._global_path, obstacles)
        start_pos = self._start_pos
        goal_pos = self._goal_pos
        radius = 0.5 * np.linalg.norm(goal_pos - start_pos)
        center = 0.5 * (start_pos + goal_pos)
        init_angle = np.math.atan2(start_pos[1] - center[1], start_pos[0] - center[0])
        self._global_path = []
        for i in range(1800):
            angle_i = init_angle + (2 * math.pi / 60) * i
            next_waypoint = np.array([radius * math.cos(angle_i) + center[0], radius * math.sin(angle_i) + center[1], center[2]])
            self._global_path.append(next_waypoint)
        self._global_path = np.array(self._global_path)
        # self._global_path = np.block([[self._start_pos], [self._goal_pos], [self._start_pos], [self._goal_pos], [self._start_pos], [self._goal_pos], [self._start_pos], [self._goal_pos], [self._start_pos], [self._goal_pos], [self._start_pos], [self._goal_pos], [self._start_pos], [self._goal_pos], [self._start_pos], [self._goal_pos], [self._start_pos], [self._goal_pos], [self._start_pos], [self._goal_pos], [self._start_pos], [self._goal_pos], [self._start_pos], [self._goal_pos], [self._start_pos], [self._goal_pos], [self._start_pos], [self._goal_pos], [self._start_pos], [self._goal_pos], [self._start_pos], [self._goal_pos], [self._start_pos], [self._goal_pos], [self._start_pos], [self._goal_pos], [self._start_pos], [self._goal_pos], [self._start_pos], [self._goal_pos]])
        return self._global_path

    def logging(self, logger):
        logger._paths.append(self._global_path)


class ThetaStarPathGenerator:
    def __init__(self, grid, quad, margin):
        self._global_path = None
        self._grid = GridMap(bounds=grid[0], cell_size=grid[1], quad=False)
        self._margin = margin

    def generate_path(self, sys, obstacles, goal_pos):
        graph = GraphSearch(graph=self._grid, obstacles=obstacles, margin=self._margin)
        path = graph.theta_star(sys.get_state()[:2], goal_pos)
        self._global_path = np.array([p.pos for p in path])
        print(self._global_path)
        if self._global_path == []:
            print("Global Path not found.")
            sys.exit(1)
        if True:
            plot_global_map(self._global_path, obstacles)
        return self._global_path

    def logging(self, logger):
        logger._paths.append(self._global_path)