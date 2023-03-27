import math

import casadi as ca
import numpy as np

from geometry_utils import RectangleRegion
from logger import (
    ControllerLogger,
    GlobalPlannerLogger,
    LocalPlannerLogger,
    SystemLogger,
)


class System:
    def __init__(self, time=0.0, state=None, geometry=None, dynamics=None):
        self._time = time
        self._state = state
        self._geometry = geometry
        self._dynamics = dynamics


class Robot:
    def __init__(self, system):
        self._system = system
        self._system_logger = SystemLogger()

    def set_global_planner(self, global_planner):
        self._global_planner = global_planner
        self._global_planner_logger = GlobalPlannerLogger()

    def set_local_planner(self, local_planner):
        self._local_planner = local_planner
        self._local_planner_logger = LocalPlannerLogger()

    def set_controller(self, controller):
        self._controller = controller
        self._controller_logger = ControllerLogger()

    def run_global_planner(self, sys, static_obstacles, goal_pos):
        # TODO: global path shall be generated with `system` and `obstacles`.
        self._global_path = self._global_planner.generate_path() # for planning a global path with static obstacles
        # print(self._global_path)
        self._global_planner.logging(self._global_planner_logger)

    def run_local_planner(self):
        # TODO: local path shall be generated with `obstacles`.
        self._local_trajectory = self._local_planner.generate_trajectory(self, self._global_path)
        self._local_planner.logging(self._local_planner_logger)

    def run_controller(self, static_obstacles, dynamic_obstacles):
        self._control_action = self._controller.generate_control_input(
            self._system, self._global_path, self._local_trajectory, static_obstacles, dynamic_obstacles, self._controller_logger
        )
        # self._controller.logging(self._controller_logger)

    def run_system(self):
        self._system.update(self._control_action)
        self._system.logging(self._system_logger)

    def obstacle_controller(self):
        max_acc = self._system._geometry._max_acc_real
        rand = np.random.rand(3)
        bunmo = np.sqrt(rand[0]**2 + rand[1]**2 + rand[2]**2)
        control_input = np.zeros(3)
        control_input[0] = (rand[0] / bunmo) * max_acc
        control_input[1] = (rand[1] / bunmo) * max_acc
        control_input[2] = (rand[2] / bunmo) * max_acc
        rand_sgn = np.random.rand(3)
        if rand_sgn[0] > 0.5:
            control_input[0] = -control_input[0]
        if rand_sgn[1] > 0.5:
            control_input[1] = -control_input[1]
        if rand_sgn[2] > 0.5:
            control_input[2] = -control_input[2]
        self._control_action = control_input

    def stationary_obstacle_controller(self):
        self._control_action = np.zeros(3)


class SingleAgentSimulation:
    def __init__(self, robot, static_obstacles, dynamic_obstacles, goal_position):
        self._robot = robot
        self._static_obstacles = static_obstacles
        self._dynamic_obstacles = dynamic_obstacles
        self._goal_position = goal_position

    def run_navigation(self, navigation_time):
        # self._robot.run_global_planner(self._robot._system, self._obstacles, self._goal_position) # for static obstacles, global path generated before the simulation
        self._robot.run_global_planner(self._robot._system, [], self._goal_position)
        while self._robot._system._time < navigation_time:
            self._robot.run_local_planner()
            self._robot.run_controller(self._static_obstacles, self._dynamic_obstacles)
            self._robot.run_system()
            print("time: ", self._robot._system._time)
            # for dynamic_obstacle in self._dynamic_obstacles[:2]:
            #     dynamic_obstacle.stationary_obstacle_controller()
            #     dynamic_obstacle.run_system()
            for dynamic_obstacle in self._dynamic_obstacles:
                dynamic_obstacle.obstacle_controller()
                dynamic_obstacle.run_system()
                obs_state = dynamic_obstacle._system._state._x
                obs_max_vel = dynamic_obstacle._system._geometry._max_vel
                obs_start_vel = dynamic_obstacle._system._geometry._start_vel
                obs_vel = obs_state[3:6]
                if np.linalg.norm(obs_vel) > obs_max_vel:
                    rand = np.random.rand(3)
                    bunmo = np.sqrt(rand[0]**2 + rand[1]**2 + rand[2]**2)
                    dynamic_obstacle_vel = np.zeros(3)
                    dynamic_obstacle_vel[0] = (rand[0] / bunmo) * obs_start_vel
                    dynamic_obstacle_vel[1] = (rand[1] / bunmo) * obs_start_vel
                    dynamic_obstacle_vel[2] = (rand[2] / bunmo) * obs_start_vel
                    rand_sgn = np.random.rand(3)
                    if rand_sgn[0] > 0.5:
                        dynamic_obstacle_vel[0] = - dynamic_obstacle_vel[0]
                    if rand_sgn[1] > 0.5:
                        dynamic_obstacle_vel[1] = - dynamic_obstacle_vel[1]
                    if rand_sgn[2] > 0.5:
                        dynamic_obstacle_vel[2] = - dynamic_obstacle_vel[2]
                    dynamic_obstacle._system._state._x[3:6] = dynamic_obstacle_vel
