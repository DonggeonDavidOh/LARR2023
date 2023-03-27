import datetime

import matplotlib.patches as patches

from geometry_utils import *
from simulation import *


class KinematicCarDynamics:
    @staticmethod
    def forward_dynamics(x, u, timestep):
        # "Obstacle Avoidance Using Image-based Visual Servoing Integrated with Nonlinear Model Predictive Control", H. Jin. Kim
        # x = [V, gamma, psi, P_x, P_y, P_z].T, u = [u_V, u_gamma, u_psi].T
        x_next = np.ndarray(shape=(6,), dtype=float)
        def state_derivative(x, u):
            g = 9.81 # gravitational acceleration, m/s^2
            V_d = 10.0
            xdot = np.ndarray(shape=(6,), dtype=float)
            xdot[0] = u[0]
            xdot[1] = (u[1] - g * math.cos(x[1])) / (x[0] + V_d)
            xdot[2] = u[2] / ((x[0] + V_d) * math.cos(x[1]))
            xdot[3] = (x[0] + V_d) * math.cos(x[1]) * math.cos(x[2])
            xdot[4] = (x[0] + V_d) * math.cos(x[1]) * math.sin(x[2])
            xdot[5] = (x[0] + V_d) * math.sin(x[1])
            return xdot
        x_next = x + state_derivative(x, u) * timestep
        return x_next

    @staticmethod
    def forward_dynamics_opt(timestep):
        # "Obstacle Avoidance Using Image-based Visual Servoing Integrated with Nonlinear Model Predictive Control", H. Jin. Kim
        # x = [V, gamma, psi, P_x, P_y, P_z].T, u = [u_V, u_gamma, u_psi].T
        x_symbol = ca.SX.sym("x", 6)
        u_symbol = ca.SX.sym("u", 3)
        def state_derivative(x, u):
            g = 9.81 # gravitational acceleration, m/s^2
            V_d = 10.0
            Vdot = u[0]
            gammadot = (u[1] - g * ca.cos(x[1])) / (x[0] + V_d)
            psidot = u[2] / ((x[0] + V_d) * ca.cos(x[1]))
            P_xdot = (x[0] + V_d) * ca.cos(x[1]) * ca.cos(x[2])
            P_ydot = (x[0] + V_d) * ca.cos(x[1]) * ca.sin(x[2])
            P_zdot = (x[0] + V_d) * ca.sin(x[1])
            return ca.vertcat(Vdot, gammadot, psidot, P_xdot, P_ydot, P_zdot)
        state_symbol_next = x_symbol + state_derivative(x_symbol, u_symbol) * timestep
        return ca.Function("dubins_plane_dynamics", [x_symbol, u_symbol], [state_symbol_next])

    @staticmethod
    def nominal_safe_controller(x, timestep, amin, amax):
        """Return updated state using nominal safe controller in a form of `np.ndnumpy`"""
        u_nom = np.zeros(shape=(2,))
        u_nom[0] = np.clip(-x[2] / timestep, amin, amax)
        return KinematicCarDynamics.forward_dynamics(x, u_nom, timestep), u_nom

    @staticmethod
    def safe_dist(x, timestep, amin, amax, dist_margin):
        """Return a safe distance outside which to ignore obstacles"""
        # TODO: wrap params
        safe_ratio = 4 # originally 1.25
        brake_min_dist = (abs(x[2]) + amax * timestep) ** 2 / (2 * amax) + dist_margin
        return safe_ratio * brake_min_dist + abs(x[2]) * timestep + 0.5 * amax * timestep ** 2


class KinematicCarStates:
    def __init__(self, x, u=np.array([0.0, 0.0, 0.0])):
        self._x = x
        self._u = u

    def translation(self):
        return np.array([[self._x[3]], [self._x[4]], [self._x[5]]])

    # TODO : define the rotation matrix using Euler angles
    # def rotation(self):
    #     return np.array(
    #         [
    #             [math.cos(self._x[3]), -math.sin(self._x[3])],
    #             [math.sin(self._x[3]), math.cos(self._x[3])],
    #         ]
    #     )


class KinematicCarRectangleGeometry:
    def __init__(self, length, width, rear_dist):
        self._length = length
        self._width = width
        self._rear_dist = rear_dist
        self._region = RectangleRegion((-length + rear_dist) / 2, (length + rear_dist) / 2, -width / 2, width / 2)

    def equiv_rep(self):
        return [self._region]

    def get_plot_patch(self, state, i=0, alpha=0.5):
        length, width, rear_dist = self._length, self._width, self._rear_dist
        x, y, theta = state[0], state[1], state[3]
        xc = x + (rear_dist / 2) * math.cos(theta)
        yc = y + (rear_dist / 2) * math.sin(theta)
        vertices = np.array(
            [
                [
                    xc + length / 2 * np.cos(theta) - width / 2 * np.sin(theta),
                    yc + length / 2 * np.sin(theta) + width / 2 * np.cos(theta),
                ],
                [
                    xc + length / 2 * np.cos(theta) + width / 2 * np.sin(theta),
                    yc + length / 2 * np.sin(theta) - width / 2 * np.cos(theta),
                ],
                [
                    xc - length / 2 * np.cos(theta) + width / 2 * np.sin(theta),
                    yc - length / 2 * np.sin(theta) - width / 2 * np.cos(theta),
                ],
                [
                    xc - length / 2 * np.cos(theta) - width / 2 * np.sin(theta),
                    yc - length / 2 * np.sin(theta) + width / 2 * np.cos(theta),
                ],
            ]
        )
        return patches.Polygon(vertices, alpha=alpha, closed=True, fc="tab:brown", ec="None", linewidth=0.5)


class KinematicCarMultipleGeometry:
    def __init__(self):
        self._num_geometry = 0
        self._geometries = []
        self._regions = []

    def equiv_rep(self):
        return self._regions

    def add_geometry(self, geometry):
        self._geometries.append(geometry)
        self._regions.append(geometry._region)
        self._num_geometry += 1

    def get_plot_patch(self, state, region_idx, alpha=0.5):
        return self._geometries[region_idx].get_plot_patch(state, alpha)


class KinematicCarPointGeometry:
    def __init__(self, radius):
        self._radius = radius
        self._num_geometry = 2

    def equiv_rep(self):
        return []

    def get_plot_patch(self, state, i, alpha=0.5):
        if i == 0:
            x, y, theta = state[0], state[1], state[3]
            vertex_points = np.array([[self._radius, 0], [-0.5 * self._radius, (0.75)**0.5 * self._radius], [-0.5 * self._radius, -(0.75)**0.5 * self._radius]])
            mat_rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
            vertices_points_curr = (mat_rotation @ vertex_points.T).T + np.array([x, y])
            return patches.Polygon(vertices_points_curr, alpha=alpha, closed=True, fc="tab:brown", ec="None", linewidth=0.5)
        elif i == 1:
            x, y, theta = state[0], state[1], state[3]
            return patches.Circle(np.array([x, y]), radius=self._radius, alpha=alpha, fc="tab:brown", ec="None", linewidth=0.5)


class KinematicCarTriangleGeometry:
    def __init__(self, vertex_points):
        self._vertex_points = vertex_points
        self._region = PolytopeRegion.convex_hull(vertex_points)

    def equiv_rep(self):
        return [self._region]

    def get_plot_patch(self, state, i=0, alpha=0.5):
        x, y, theta = state[0], state[1], state[3]
        mat_rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        vertices_points_curr = (mat_rotation @ self._vertex_points.T).T + np.array([x, y])
        return patches.Polygon(vertices_points_curr, alpha=alpha, closed=True, fc="tab:brown", ec="None", linewidth=0.5)


class KinematicCarPentagonGeometry:
    def __init__(self, vertex_points):
        self._vertex_points = vertex_points
        self._region = PolytopeRegion.convex_hull(vertex_points)

    def equiv_rep(self):
        return [self._region]

    def get_plot_patch(self, state, i=0, alpha=0.5):
        x, y, theta = state[0], state[1], state[3]
        mat_rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        vertices_points_curr = (mat_rotation @ self._vertex_points.T).T + np.array([x, y])
        return patches.Polygon(vertices_points_curr, alpha=alpha, closed=True, fc="tab:brown", ec="None", linewidth=0.5)


class KinematicCarSystem(System):
    def get_state(self):
        return self._state._x

    def update(self, unew):
        self._state._u = unew
        xnew = self._dynamics.forward_dynamics(self.get_state(), unew, 0.1)
        self._state._x = xnew
        # print(xnew)
        # print(unew)
        self._time += 0.1

    def logging(self, logger):
        logger._xs.append(self._state._x)
        logger._us.append(self._state._u)
