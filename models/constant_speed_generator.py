import numpy as np


class ConstantSpeedTrajectoryGenerator:
    # TODO: Refactor this class to make it light-weight.
    # TODO: Create base class for this local planner
    def __init__(self, DCBF_horizon, center):
        # TODO: wrap params
        self._global_path_index = 0
        # TODO: number of waypoints shall equal to length of global path
        self._num_waypoint = None
        # local path
        self._reference_speed = 10.0
        self._num_horizon = DCBF_horizon # 11 originally
        self._center = center
        self._local_path_timestep = 0.1
        self._local_trajectory = None
        self._proj_dist_buffer = 0.05
        self._num_of_rotations = 0

    def generate_trajectory(self, robot, global_path):
        # TODO: move initialization of _num_waypoint and _global_path to constructor
        if self._num_waypoint is None:
            self._global_path = global_path
            self._num_waypoint = global_path.shape[0]
        # TODO: pass _global_path as a reference
        # print(self.generate_trajectory_internal(pos, self._global_path)) # for debugging purposes
        return self.generate_trajectory_internal(robot, self._global_path)

    def generate_trajectory_internal(self, robot, global_path):
        pos = robot._system._state._x[3:6]
        local_index = self._global_path_index
        trunc_path = np.vstack([global_path[local_index:, :], global_path[-1, :]])
        curv_vec = trunc_path[1:, :] - trunc_path[:-1, :]
        curv_length = np.linalg.norm(curv_vec, axis=1)

        if curv_length[0] == 0.0:
            curv_direct = np.zeros((3,))
        else:
            curv_direct = curv_vec[0, :] / curv_length[0]
        proj_dist = np.dot(pos - trunc_path[0, :], curv_direct)

        if proj_dist >= curv_length[0] - self._proj_dist_buffer and local_index < self._num_waypoint - 1:
            self._global_path_index += 1
            return self.generate_trajectory_internal(robot, global_path)

        # TODO: make the if statement optional
        if proj_dist <= 0.0:
            proj_dist = 0.0

        t_c = (proj_dist + self._proj_dist_buffer) / self._reference_speed
        t_s = t_c + self._local_path_timestep * np.linspace(0, self._num_horizon - 1, self._num_horizon)

        curv_time = np.cumsum(np.hstack([0.0, curv_length / self._reference_speed]))
        curv_time[-1] += (
            t_c + 2 * self._local_path_timestep * self._num_horizon + self._proj_dist_buffer / self._reference_speed
        )

        path_idx = np.searchsorted(curv_time, t_s, side="right") - 1
        path = np.vstack(
            [
                np.interp(t_s, curv_time, trunc_path[:, 0]),
                np.interp(t_s, curv_time, trunc_path[:, 1]),
                np.interp(t_s, curv_time, trunc_path[:, 2]),
            ]
        ).T
        # TODO: how to implement the cost for the bank angle and the flight path angle?
        path_vel = self._reference_speed * np.ones((self._num_horizon, 1))
        center = self._center
        path_psi = np.arctan2(path[:,1] - center[1], path[:,0] - center[0]).reshape(self._num_horizon, 1) + np.math.pi / 2
        for index in range(self._num_horizon - 1):
            if path_psi[index] > path_psi[index + 1]:
                path_psi[index + 1] = path_psi[index + 1] + 2 * np.math.pi
        # if len(robot._local_planner_logger._trajs) > 0:
        #     print(robot._local_planner_logger._trajs[-1][0, 3], path_psi[0])
        while len(robot._local_planner_logger._trajs) > 0 and robot._local_planner_logger._trajs[-1][0, 2] > path_psi[0]:
            for index in range(self._num_horizon):
                path_psi[index] = path_psi[index] + 2 * np.math.pi
        path_V = np.zeros_like(path_psi)
        for index in range(self._num_horizon):
            path_V[index] = 0.0 # or 0.525, if the velocity error has to converge to 0
        path_gamma = np.zeros_like(path_psi)
        self._local_trajectory = np.hstack([path_V, path_gamma, path_psi, path])
        # print(self._local_trajectory)
        return self._local_trajectory

    def logging(self, logger):
        logger._trajs.append(self._local_trajectory)

class ConstantSpeedCircularTrajectoryGenerator:
    # TODO: Refactor this class to make it light-weight.
    # TODO: Create base class for this local planner
    def __init__(self, center, radius):
        self._reference_speed = 0.1
        self._num_horizon = 7 # 11 originally
        self._local_path_timestep = 0.1
        self._local_trajectory = None
        self._proj_dist_buffer = 0.05
        self._center = center
        self._radius = radius

    def generate_trajectory(self, system, global_path):
        state = system._state._x
        pos = state[0:3]
        center = self._center
        radius = self._radius
        init_angle = np.math.atan2(pos[1] - center[1], pos[0] - center[0])
        if init_angle < 0:
            init_angle = init_angle + 2 * np.math.pi
        self._local_trajectory = np.zeros((self._num_horizon, 4))
        for t in range(self._num_horizon):
            i = t + 1
            g = 9.81
            angle_i = init_angle + self._reference_speed * i
            x_i = center[0] + radius * np.math.cos(angle_i)
            y_i = center[1] + radius * np.math.sin(angle_i)
            z_i = center[2]
            psi_i = angle_i + np.math.pi / 2
            if psi_i < 0:
                psi_i = psi_i + 2 * np.math.pi
            elif psi_i > 2 * np.math.pi:
                psi_i = psi_i - 2 * np.math.pi
            self._local_trajectory[t, 0] = x_i
            self._local_trajectory[t, 1] = y_i
            self._local_trajectory[t, 2] = z_i
            self._local_trajectory[t, 3] = psi_i
        return self._local_trajectory

    def logging(self, logger):
        logger._trajs.append(self._local_trajectory)