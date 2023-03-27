import datetime

import matplotlib.patches as patches

from geometry_utils import *
from simulation import *


class KinematicObstacleDynamics:
    @staticmethod
    def forward_dynamics(x, u, timestep): # double integrator dynamics
        """Return updated state in a form of `np.ndnumpy`"""
        x_next = np.ndarray(shape=(6,), dtype=float)
        x_next[0] = x[0] + x[3] * timestep + 0.5 * u[0] * timestep**2
        x_next[1] = x[1] + x[4] * timestep + 0.5 * u[1] * timestep**2
        x_next[2] = x[2] + x[5] * timestep + 0.5 * u[2] * timestep**2
        x_next[3] = x[3] + u[0] * timestep
        x_next[4] = x[4] + u[1] * timestep
        x_next[5] = x[5] + u[2] * timestep
        return x_next

    @staticmethod
    def forward_dynamics_opt(timestep): # double integrator dynamics
        """Return updated state in a form of `ca.SX`"""
        x_symbol = ca.SX.sym("x", 6)
        u_symbol = ca.SX.sym("u", 3)
        x_symbol_next = x_symbol[0] + x_symbol[3] * timestep + 0.5 * u_symbol[0] * timestep**2
        y_symbol_next = x_symbol[1] + x_symbol[4] * timestep + 0.5 * u_symbol[1] * timestep**2
        z_symbol_next = x_symbol[2] + x_symbol[5] * timestep + 0.5 * u_symbol[2] * timestep**2
        xdot_symbol_next = x_symbol[3] + u_symbol[0] * timestep
        ydot_symbol_next = x_symbol[4] + u_symbol[1] * timestep
        zdot_symbol_next = x_symbol[5] + u_symbol[2] * timestep
        state_symbol_next = ca.vertcat(x_symbol_next, y_symbol_next, z_symbol_next, xdot_symbol_next, ydot_symbol_next, zdot_symbol_next)
        return ca.Function("dubin_car_dynamics", [x_symbol, u_symbol], [state_symbol_next]) # double integrator dynamics, not dubins car dynamics

    # @staticmethod
    # def nominal_safe_controller(x, timestep, amin, amax):
    #     """Return updated state using nominal safe controller in a form of `np.ndnumpy`"""
    #     u_nom = np.zeros(shape=(2,))
    #     u_nom[0] = np.clip(-x[2] / timestep, amin, amax)
    #     return KinematicObstacleDynamics.forward_dynamics(x, u_nom, timestep), u_nom

    # @staticmethod
    # def safe_dist(x, timestep, amin, amax, dist_margin):
    #     """Return a safe distance outside which to ignore obstacles"""
    #     # TODO: wrap params
    #     safe_ratio = 4 # originally 1.25
    #     brake_min_dist = (abs(x[2]) + amax * timestep) ** 2 / (2 * amax) + dist_margin
    #     return safe_ratio * brake_min_dist + abs(x[2]) * timestep + 0.5 * amax * timestep ** 2


class KinematicObstacleStates:
    def __init__(self, x, u=np.array([0.0, 0.0, 0.0])):
        self._x = x
        self._u = u

    def translation(self):
        return np.array([[self._x[0]], [self._x[1]], [self._x[2]]])

    # def rotation(self): # no rotation since double integrator dynamics
    #     return np.array(
    #         [
    #             [math.cos(self._x[3]), -math.sin(self._x[3])],
    #             [math.sin(self._x[3]), math.cos(self._x[3])],
    #         ]
    #     )


class KinematicObstaclePointGeometry:
    def __init__(self, max_acc_real, max_acc_cbf, max_vel, start_vel, real_radius, bound):
        self._num_geometry = 6 # depicts the reachable set for 5 timesteps
        self._max_acc_real = max_acc_real
        self._max_acc_cbf = max_acc_cbf
        self._max_vel = max_vel
        self._start_vel = start_vel
        self._real_radius = real_radius
        self._bound = bound

    def equiv_rep(self):
        return []

    def get_plot_patch(self, state, i): # the reachable set after i-th timestep 
        obs_pos = state[0:2]
        obs_vel = state[2:4]
        obs_center = obs_pos + (0.1 * i) * obs_vel
        if i == 0 or i == 1:
            obs_radius = self._real_radius
        else:
            obs_radius = self._real_radius
            for j in range(i - 1):
                obs_radius += (i - j - 1) * self._max_acc_cbf * 0.1**2
        return patches.Circle(obs_center, obs_radius, alpha=1, fc="tab:brown", ec="None", linewidth=0.5)


class KinematicObstacleSystem(System):
    def get_state(self):
        return self._state._x

    def update(self, unew):
        curr_state = self.get_state()
        lower_bound = self._geometry._bound[0]
        upper_bound = self._geometry._bound[1]
        radius = self._geometry._real_radius
        if curr_state[0] < lower_bound[0] + radius or curr_state[0] > upper_bound[0] - radius:
            curr_state[3] = -curr_state[3]
        if curr_state[1] < lower_bound[1] + radius or curr_state[1] > upper_bound[1] - radius:
            curr_state[4] = -curr_state[4]
        if curr_state[2] < lower_bound[2] + radius or curr_state[2] > upper_bound[2] - radius:
            curr_state[5] = -curr_state[5]
        xnew = self._dynamics.forward_dynamics(curr_state, unew, 0.1)
        self._state._x = xnew
        self._state._u = unew
        self._time += 0.1

    def logging(self, logger):
        logger._xs.append(self._state._x)
        logger._us.append(self._state._u)
