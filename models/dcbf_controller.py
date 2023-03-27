import datetime
import numpy as np
from sympy import *
import casadi as ca

from dcbf_optimizer import NmpcDbcfOptimizer, NmpcDcbfOptimizerParam


class NmpcDcbfController:
    # TODO: Refactor this class to inheritate from a general optimizer
    def __init__(self, dynamics=None, opt_param=None, constraint_type=None):
        # TODO: Rename self._param into self._opt_param to avoid ambiguity
        self._param = opt_param
        self._constraint_type = constraint_type
        self._optimizer = NmpcDbcfOptimizer({}, {}, dynamics.forward_dynamics_opt(0.1))
        self._dynamics = dynamics.forward_dynamics_opt(0.1)
        self._solver_times = []
        self._nominal_input = np.zeros((3,))
        self._prop_time = 60
        self._control_gains = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0] # k_V_1, k_V_2, k_V_3, k_gamma_1, k_gamma_2, k_gamma_3, k_psi
        self._input_constraints = [-1.0, 1.0, 6.81, 12.81, -3.0, 3.0] # u_V_min, u_V_max, u_gamma_min, u_gamma_max, u_psi_min, u_psi_max
        self._state_constraints = [-5.0, 10.0, -0.5, 0.5] # V_min, V_max, gamma_min, gamma_max
        self.margin_dist = 1.0
        self.g = 9.81
        self.V_d = 10.0
        self.epsilon = 0.01

    def evading_maneuver_comp(self, robot_state, static_obstacle):
        # For numerical Stability, min/max of the states are modified
        V, gamma, psi, P_x, P_y, P_z = robot_state[0], robot_state[1], robot_state[2], robot_state[3], robot_state[4], robot_state[5]
        obs_center = static_obstacle[0]
        P_x_obs, P_y_obs, P_z_obs = obs_center[0], obs_center[1], obs_center[2]
        obs_radius = static_obstacle[1]
        k_V_1, k_V_2, k_V_3, k_gamma_1, k_gamma_2, k_gamma_3, k_psi = self._control_gains[0], self._control_gains[1], self._control_gains[2], self._control_gains[3], self._control_gains[4], self._control_gains[5], self._control_gains[6]
        u_V_min, u_V_max, u_gamma_min, u_gamma_max, u_psi_min, u_psi_max = self._input_constraints[0], self._input_constraints[1], self._input_constraints[2], self._input_constraints[3], self._input_constraints[4], self._input_constraints[5]
        V_min, V_max, gamma_min, gamma_max = self._state_constraints[0], self._state_constraints[1], self._state_constraints[2], self._state_constraints[3]
        d_V = - ((P_x - P_x_obs) * np.math.cos(gamma) * np.math.cos(psi) + (P_y - P_y_obs) * np.math.cos(gamma) * np.math.sin(psi) + (P_z - P_z_obs) * np.math.sin(gamma))
        d_gamma = - (- (P_x - P_x_obs) * np.math.sin(gamma) * np.math.cos(psi) - (P_y - P_y_obs) * np.math.sin(gamma) * np.math.sin(psi) + (P_z - P_z_obs) * np.math.cos(gamma))
        d_psi = - (- (P_x - P_x_obs) * np.math.sin(psi) + (P_y - P_y_obs) * np.math.cos(psi))
        f_V, g_V, f_gamma, g_gamma, f_psi, g_psi = 0, 1, - self.g * np.math.cos(gamma) / (V + self.V_d), 1 / (V + self.V_d), 0, 1 / ((V + self.V_d) * np.math.cos(gamma))
        P_V_1, P_V_2 = - u_V_min - f_V / g_V, u_V_max + f_V / g_V
        P_gamma_1, P_gamma_2 = - u_gamma_min - f_gamma / g_gamma, u_gamma_max + f_gamma / g_gamma
        P_psi_1, P_psi_2 = - u_psi_min - f_psi / g_psi, u_psi_max + f_psi / g_psi
        u_tilde_V_max = ((P_V_1 + P_V_2) - np.math.sqrt((P_V_1 - P_V_2)**2 + self.epsilon)) / 2
        u_tilde_gamma_max = ((P_gamma_1 + P_gamma_2) - np.math.sqrt((P_gamma_1 - P_gamma_2)**2 + self.epsilon)) / 2
        u_tilde_psi_max = ((P_psi_1 + P_psi_2) - np.math.sqrt((P_psi_1 - P_psi_2)**2 + self.epsilon)) / 2
        u_tilde_V_star = u_tilde_V_max * np.math.tanh(- k_V_1 * g_V) * np.math.tanh(k_V_2 * (V - (((V_max - V_min) / 2) * np.math.tanh(- k_V_3 * g_V * d_V) + ((V_max + V_min) / 2))))
        u_tilde_gamma_star = u_tilde_gamma_max * np.math.tanh(- k_gamma_1 * g_gamma) * np.math.tanh(k_gamma_2 * (gamma - (((gamma_max - gamma_min)/2) * np.math.tanh(- k_gamma_3 * g_gamma * d_gamma) * ((gamma_max + gamma_min) / 2))))
        u_tilde_psi_star = u_tilde_psi_max * np.math.tanh(- k_psi * d_psi)
        u_V_star = u_tilde_V_star - f_V / g_V
        u_gamma_star = u_tilde_gamma_star - f_gamma / g_gamma
        u_psi_star = u_tilde_psi_star - f_psi / g_psi
        u_star = np.array([u_V_star, u_gamma_star, u_psi_star])
        return u_star
    
    def evading_maneuver_comp_no_RD1(self, robot_state, static_obstacle):
        # For numerical Stability, min/max of the states are modified
        V, gamma, psi, P_x, P_y, P_z = robot_state[0], robot_state[1], robot_state[2], robot_state[3], robot_state[4], robot_state[5]
        obs_center = static_obstacle[0]
        P_x_obs, P_y_obs, P_z_obs = obs_center[0], obs_center[1], obs_center[2]
        obs_radius = static_obstacle[1]
        k_V_1, k_V_2, k_V_3, k_gamma_1, k_gamma_2, k_gamma_3, k_psi = self._control_gains[0], self._control_gains[1], self._control_gains[2], self._control_gains[3], self._control_gains[4], self._control_gains[5], self._control_gains[6]
        u_V_min, u_V_max, u_gamma_min, u_gamma_max, u_psi_min, u_psi_max = self._input_constraints[0], self._input_constraints[1], self._input_constraints[2], self._input_constraints[3], self._input_constraints[4], self._input_constraints[5]
        V_min, V_max, gamma_min, gamma_max = self._state_constraints[0], self._state_constraints[1], self._state_constraints[2], self._state_constraints[3]
        d_V = - ((P_x - P_x_obs) * np.math.cos(gamma) * np.math.cos(psi) + (P_y - P_y_obs) * np.math.cos(gamma) * np.math.sin(psi) + (P_z - P_z_obs) * np.math.sin(gamma))
        d_gamma = - (- (P_x - P_x_obs) * np.math.sin(gamma) * np.math.cos(psi) - (P_y - P_y_obs) * np.math.sin(gamma) * np.math.sin(psi) + (P_z - P_z_obs) * np.math.cos(gamma))
        d_psi = - (- (P_x - P_x_obs) * np.math.sin(psi) + (P_y - P_y_obs) * np.math.cos(psi))
        f_V, g_V, f_gamma, g_gamma, f_psi, g_psi = 0, 1, - self.g * np.math.cos(gamma) / (V + self.V_d), 1 / (V + self.V_d), 0, 1 / ((V + self.V_d) * np.math.cos(gamma))
        P_V_1, P_V_2 = - u_V_min - f_V / g_V, u_V_max + f_V / g_V
        P_gamma_1, P_gamma_2 = - u_gamma_min - f_gamma / g_gamma, u_gamma_max + f_gamma / g_gamma
        P_psi_1, P_psi_2 = - u_psi_min - f_psi / g_psi, u_psi_max + f_psi / g_psi
        u_tilde_V_max = ((P_V_1 + P_V_2) - np.math.sqrt((P_V_1 - P_V_2)**2 + self.epsilon)) / 2
        u_tilde_gamma_max = ((P_gamma_1 + P_gamma_2) - np.math.sqrt((P_gamma_1 - P_gamma_2)**2 + self.epsilon)) / 2
        u_tilde_psi_max = ((P_psi_1 + P_psi_2) - np.math.sqrt((P_psi_1 - P_psi_2)**2 + self.epsilon)) / 2
        u_tilde_V_star = u_tilde_V_max * np.math.tanh(- k_V_1 * d_V)
        u_tilde_gamma_star = u_tilde_gamma_max * np.math.tanh(- k_gamma_1 * d_gamma)
        u_tilde_psi_star = u_tilde_psi_max * np.math.tanh(- k_psi * d_psi)
        u_V_star = u_tilde_V_star - f_V / g_V
        u_gamma_star = u_tilde_gamma_star - f_gamma / g_gamma
        u_psi_star = u_tilde_psi_star - f_psi / g_psi
        u_star = np.array([u_V_star, u_gamma_star, u_psi_star])
        return u_star

    def h_obs_comp(self, robot_state, robot_radius, static_obstacle):
        V, gamma, psi, P_x, P_y, P_z = robot_state[0], robot_state[1], robot_state[2], robot_state[3], robot_state[4], robot_state[5]
        obs_center = static_obstacle[0]
        P_x_obs, P_y_obs, P_z_obs = obs_center[0], obs_center[1], obs_center[2]
        obs_radius = static_obstacle[1]
        h_obs = (robot_radius + obs_radius)**2 - ((P_x - P_x_obs)**2 + (P_y - P_y_obs)**2 + (P_z - P_z_obs)**2)
        return h_obs

    def theta_dot_expr(self):
        # for computing the expression for theta_dot
        V, gamma, psi, P_x, P_y, P_z = symbols('V gamma psi P_x P_y P_z')
        P_x_obs, P_y_obs, P_z_obs = symbols('P_x_obs P_y_obs P_z_obs')
        k_V_1, k_V_2, k_V_3, k_gamma_1, k_gamma_2, k_gamma_3, k_psi = symbols('k_V_1 k_V_2 k_V_3 k_gamma_1 k_gamma_2 k_gamma_3 k_psi')
        u_V_min, u_V_max, u_gamma_min, u_gamma_max, u_psi_min, u_psi_max = symbols('u_V_min u_V_max u_gamma_min u_gamma_max u_psi_min u_psi_max')
        V_min, V_max, gamma_min, gamma_max = symbols('V_min V_max gamma_min gamma_max')
        V_d, epsilon = symbols('V_d epsilon')
        d_V = - ((P_x - P_x_obs) * cos(gamma) * cos(psi) + (P_y - P_y_obs) * cos(gamma) * sin(psi) + (P_z - P_z_obs) * sin(gamma))
        d_gamma = - (- (P_x - P_x_obs) * sin(gamma) * cos(psi) - (P_y - P_y_obs) * sin(gamma) * sin(psi) + (P_z - P_z_obs) * cos(gamma))
        d_psi = - (- (P_x - P_x_obs) * sin(psi) + (P_y - P_y_obs) * cos(psi))
        f_V, g_V, f_gamma, g_gamma, f_psi, g_psi = 0, 1, - self.g * cos(gamma) / (V + V_d), 1 / (V + V_d), 0, 1 / ((V + V_d) * cos(gamma))
        P_V_1, P_V_2 = - u_V_min - f_V / g_V, u_V_max + f_V / g_V
        P_gamma_1, P_gamma_2 = - u_gamma_min - f_gamma / g_gamma, u_gamma_max + f_gamma / g_gamma
        P_psi_1, P_psi_2 = - u_psi_min - f_psi / g_psi, u_psi_max + f_psi / g_psi
        u_tilde_V_max = (P_V_1 + P_V_2 - sqrt((P_V_1 - P_V_2)**2 + epsilon)) / 2
        u_tilde_gamma_max = (P_gamma_1 + P_gamma_2 - sqrt((P_gamma_1 - P_gamma_2)**2 + epsilon)) / 2
        u_tilde_psi_max = (P_psi_1 + P_psi_2 - sqrt((P_psi_1 - P_psi_2)**2 + epsilon)) / 2
        u_tilde_V_star = u_tilde_V_max * tanh(- k_V_1 * g_V) * tanh(k_V_2 * (V - (((V_max - V_min) / 2) * tanh(- k_V_3 * g_V * d_V) + ((V_max + V_min) / 2))))
        u_tilde_gamma_star = u_tilde_gamma_max * tanh(- k_gamma_1 * g_gamma) * tanh(k_gamma_2 * (gamma - (((gamma_max - gamma_min) / 2) * tanh(- k_gamma_3 * g_gamma * d_gamma) + ((gamma_max + gamma_min) / 2))))
        u_tilde_psi_star = u_tilde_psi_max * tanh(- k_psi * d_psi)
        u_V_star = u_tilde_V_star - f_V / g_V
        u_gamma_star = u_tilde_gamma_star - f_gamma / g_gamma
        u_psi_star = u_tilde_psi_star - f_psi / g_psi
        fygyustary = Matrix([[u_V_star], [(u_gamma_star - self.g * cos(gamma)) / (V + V_d)], [u_psi_star / ((V + V_d) * cos(gamma))], [(V + V_d) * cos(gamma) * cos(psi)], [(V + V_d) * cos(gamma) * sin(psi)], [(V + V_d) * sin(gamma)]])
        state_vec = [V, gamma, psi, P_x, P_y, P_z]
        theta_dot = zeros(6, 6)
        for row in range(6):
            for col in range(6):
                theta_dot[row, col] = diff(fygyustary[row], state_vec[col])
        print(theta_dot)

    def theta_dot_expr_no_RD1(self):
        # for computing the expression for theta_dot
        V, gamma, psi, P_x, P_y, P_z = symbols('V gamma psi P_x P_y P_z')
        P_x_obs, P_y_obs, P_z_obs = symbols('P_x_obs P_y_obs P_z_obs')
        k_V_1, k_V_2, k_V_3, k_gamma_1, k_gamma_2, k_gamma_3, k_psi = symbols('k_V_1 k_V_2 k_V_3 k_gamma_1 k_gamma_2 k_gamma_3 k_psi')
        u_V_min, u_V_max, u_gamma_min, u_gamma_max, u_psi_min, u_psi_max = symbols('u_V_min u_V_max u_gamma_min u_gamma_max u_psi_min u_psi_max')
        V_min, V_max, gamma_min, gamma_max = symbols('V_min V_max gamma_min gamma_max')
        V_d, epsilon = symbols('V_d epsilon')
        d_V = - ((P_x - P_x_obs) * cos(gamma) * cos(psi) + (P_y - P_y_obs) * cos(gamma) * sin(psi) + (P_z - P_z_obs) * sin(gamma))
        d_gamma = - (- (P_x - P_x_obs) * sin(gamma) * cos(psi) - (P_y - P_y_obs) * sin(gamma) * sin(psi) + (P_z - P_z_obs) * cos(gamma))
        d_psi = - (- (P_x - P_x_obs) * sin(psi) + (P_y - P_y_obs) * cos(psi))
        f_V, g_V, f_gamma, g_gamma, f_psi, g_psi = 0, 1, - self.g * cos(gamma) / (V + V_d), 1 / (V + V_d), 0, 1 / ((V + V_d) * cos(gamma))
        P_V_1, P_V_2 = - u_V_min - f_V / g_V, u_V_max + f_V / g_V
        P_gamma_1, P_gamma_2 = - u_gamma_min - f_gamma / g_gamma, u_gamma_max + f_gamma / g_gamma
        P_psi_1, P_psi_2 = - u_psi_min - f_psi / g_psi, u_psi_max + f_psi / g_psi
        u_tilde_V_max = (P_V_1 + P_V_2 - sqrt((P_V_1 - P_V_2)**2 + epsilon)) / 2
        u_tilde_gamma_max = (P_gamma_1 + P_gamma_2 - sqrt((P_gamma_1 - P_gamma_2)**2 + epsilon)) / 2
        u_tilde_psi_max = (P_psi_1 + P_psi_2 - sqrt((P_psi_1 - P_psi_2)**2 + epsilon)) / 2
        u_tilde_V_star = u_tilde_V_max * tanh(- k_V_1 * d_V)
        u_tilde_gamma_star = u_tilde_gamma_max * tanh(- k_gamma_1 * d_gamma)
        u_tilde_psi_star = u_tilde_psi_max * tanh(- k_psi * d_psi)
        u_V_star = u_tilde_V_star - f_V / g_V
        u_gamma_star = u_tilde_gamma_star - f_gamma / g_gamma
        u_psi_star = u_tilde_psi_star - f_psi / g_psi
        fygyustary = Matrix([[u_V_star], [(u_gamma_star - self.g * cos(gamma)) / (V + V_d)], [u_psi_star / ((V + V_d) * cos(gamma))], [(V + V_d) * cos(gamma) * cos(psi)], [(V + V_d) * cos(gamma) * sin(psi)], [(V + V_d) * sin(gamma)]])
        state_vec = [V, gamma, psi, P_x, P_y, P_z]
        theta_dot = zeros(6, 6)
        for row in range(6):
            for col in range(6):
                theta_dot[row, col] = diff(fygyustary[row], state_vec[col])
        print(theta_dot)

    def generate_control_input(self, system, global_path, local_trajectory, static_obstacles, dynamic_obstacles, logger):
        if self._constraint_type == "MPC_form" or self._constraint_type == "CBF_form":
            self._optimizer.setup(self._param, system, local_trajectory)
            start_timer = datetime.datetime.now()
            self._opt_sol = self._optimizer.solve_nlp()
            end_timer = datetime.datetime.now()
            delta_timer = end_timer - start_timer
            self._solver_times.append(delta_timer.total_seconds())
            logger._utrajs.append(self._opt_sol.value(self._optimizer.variables["u"][:, 0]).T)
            print("solver time: ", delta_timer.total_seconds())
            print("nominal input from MPC: ", self._opt_sol.value(self._optimizer.variables["u"][:, 0]).T)
            print("state: ", system._dynamics.forward_dynamics(system._state._x, self._opt_sol.value(self._optimizer.variables["u"][:, 0]).T, 0.1))
            return self._opt_sol.value(self._optimizer.variables["u"][:, 0]).T

        if self._constraint_type == "nominal_evading_maneuver":
            # self.theta_dot_expr()
            self._optimizer.setup(self._param, system, local_trajectory)
            start_timer = datetime.datetime.now()
            self._opt_sol = self._optimizer.solve_nlp()
            # print(self._opt_sol.value(self._optimizer.variables["x"][:, 0]))
            # print(self._opt_sol.value(self._optimizer.variables["u"][:, 0]))

            # Assuming single static obstacle!!
            static_obstacle = static_obstacles[0]
            self._nominal_input = self._opt_sol.value(self._optimizer.variables["u"][:, 0])
            h_obs_evolution = []
            x_evolution = []
            t_c_obs = [] # list of indices(not time!!) that maximize h_obs
            robot_state = system._state._x
            robot_radius = system._geometry._radius + self.margin_dist
            h_obs = self.h_obs_comp(robot_state, robot_radius, static_obstacle)
            h_obs_evolution.append(h_obs)
            x_evolution.append(robot_state)
            N_prop = self._prop_time * 10 # propagation discretization time 0.1s
            for i in range(N_prop):
                evading_maneuver = self.evading_maneuver_comp(robot_state, static_obstacle)
                robot_state = system._dynamics.forward_dynamics(robot_state, evading_maneuver, 0.1)
                h_obs = self.h_obs_comp(robot_state, robot_radius, static_obstacle)
                h_obs_evolution.append(h_obs)
                x_evolution.append(robot_state)
            h_obs_evolution = np.array(h_obs_evolution)
            x_evolution = np.array(x_evolution)

            for index in range(len(h_obs_evolution)):
                if h_obs_evolution[index] >= np.max(h_obs_evolution):
                    t_c_obs.append(index)
            
            theta = np.identity(6)
            theta_evolution = []
            theta_evolution.append(theta)
            P_x_obs, P_y_obs, P_z_obs = static_obstacle[0][0], static_obstacle[0][1], static_obstacle[0][2]
            k_V_1, k_V_2, k_V_3, k_gamma_1, k_gamma_2, k_gamma_3, k_psi = self._control_gains[0], self._control_gains[1], self._control_gains[2], self._control_gains[3], self._control_gains[4], self._control_gains[5], self._control_gains[6]
            u_V_min, u_V_max, u_gamma_min, u_gamma_max, u_psi_min, u_psi_max = self._input_constraints[0], self._input_constraints[1], self._input_constraints[2], self._input_constraints[3], self._input_constraints[4], self._input_constraints[5]
            V_min, V_max, gamma_min, gamma_max = self._state_constraints[0], self._state_constraints[1], self._state_constraints[2], self._state_constraints[3]
            V_d, epsilon = self.V_d, self.epsilon
            for i in range(N_prop):
                V, gamma, psi, P_x, P_y, P_z = x_evolution[i][0], x_evolution[i][1], x_evolution[i][2], x_evolution[i][3], x_evolution[i][4], x_evolution[i][5]
                theta_dot = np.array([[k_V_2*(1 - np.math.tanh(k_V_2*(V - V_max/2 - V_min/2 + (V_max/2 - V_min/2)*np.math.tanh(k_V_3*(-(P_x - P_x_obs)*np.math.cos(gamma)*np.math.cos(psi) - (P_y - P_y_obs)*np.math.sin(psi)*np.math.cos(gamma) - (P_z - P_z_obs)*np.math.sin(gamma)))))**2)*(-u_V_max/2 + u_V_min/2 + np.math.sqrt(epsilon + (-u_V_max - u_V_min)**2)/2)*np.math.tanh(k_V_1), k_V_2*k_V_3*(1 - np.math.tanh(k_V_2*(V - V_max/2 - V_min/2 + (V_max/2 - V_min/2)*np.math.tanh(k_V_3*(-(P_x - P_x_obs)*np.math.cos(gamma)*np.math.cos(psi) - (P_y - P_y_obs)*np.math.sin(psi)*np.math.cos(gamma) - (P_z - P_z_obs)*np.math.sin(gamma)))))**2)*(1 - np.math.tanh(k_V_3*(-(P_x - P_x_obs)*np.math.cos(gamma)*np.math.cos(psi) - (P_y - P_y_obs)*np.math.sin(psi)*np.math.cos(gamma) - (P_z - P_z_obs)*np.math.sin(gamma)))**2)*(V_max/2 - V_min/2)*(-u_V_max/2 + u_V_min/2 + np.math.sqrt(epsilon + (-u_V_max - u_V_min)**2)/2)*(-(-P_x + P_x_obs)*np.math.sin(gamma)*np.math.cos(psi) - (-P_y + P_y_obs)*np.math.sin(gamma)*np.math.sin(psi) + (-P_z + P_z_obs)*np.math.cos(gamma))*np.math.tanh(k_V_1), k_V_2*k_V_3*(1 - np.math.tanh(k_V_2*(V - V_max/2 - V_min/2 + (V_max/2 - V_min/2)*np.math.tanh(k_V_3*(-(P_x - P_x_obs)*np.math.cos(gamma)*np.math.cos(psi) - (P_y - P_y_obs)*np.math.sin(psi)*np.math.cos(gamma) - (P_z - P_z_obs)*np.math.sin(gamma)))))**2)*(1 - np.math.tanh(k_V_3*(-(P_x - P_x_obs)*np.math.cos(gamma)*np.math.cos(psi) - (P_y - P_y_obs)*np.math.sin(psi)*np.math.cos(gamma) - (P_z - P_z_obs)*np.math.sin(gamma)))**2)*(V_max/2 - V_min/2)*(-(-P_x + P_x_obs)*np.math.sin(psi)*np.math.cos(gamma) + (-P_y + P_y_obs)*np.math.cos(gamma)*np.math.cos(psi))*(-u_V_max/2 + u_V_min/2 + np.math.sqrt(epsilon + (-u_V_max - u_V_min)**2)/2)*np.math.tanh(k_V_1), -k_V_2*k_V_3*(1 - np.math.tanh(k_V_2*(V - V_max/2 - V_min/2 + (V_max/2 - V_min/2)*np.math.tanh(k_V_3*(-(P_x - P_x_obs)*np.math.cos(gamma)*np.math.cos(psi) - (P_y - P_y_obs)*np.math.sin(psi)*np.math.cos(gamma) - (P_z - P_z_obs)*np.math.sin(gamma)))))**2)*(1 - np.math.tanh(k_V_3*(-(P_x - P_x_obs)*np.math.cos(gamma)*np.math.cos(psi) - (P_y - P_y_obs)*np.math.sin(psi)*np.math.cos(gamma) - (P_z - P_z_obs)*np.math.sin(gamma)))**2)*(V_max/2 - V_min/2)*(-u_V_max/2 + u_V_min/2 + np.math.sqrt(epsilon + (-u_V_max - u_V_min)**2)/2)*np.math.cos(gamma)*np.math.cos(psi)*np.math.tanh(k_V_1), -k_V_2*k_V_3*(1 - np.math.tanh(k_V_2*(V - V_max/2 - V_min/2 + (V_max/2 - V_min/2)*np.math.tanh(k_V_3*(-(P_x - P_x_obs)*np.math.cos(gamma)*np.math.cos(psi) - (P_y - P_y_obs)*np.math.sin(psi)*np.math.cos(gamma) - (P_z - P_z_obs)*np.math.sin(gamma)))))**2)*(1 - np.math.tanh(k_V_3*(-(P_x - P_x_obs)*np.math.cos(gamma)*np.math.cos(psi) - (P_y - P_y_obs)*np.math.sin(psi)*np.math.cos(gamma) - (P_z - P_z_obs)*np.math.sin(gamma)))**2)*(V_max/2 - V_min/2)*(-u_V_max/2 + u_V_min/2 + np.math.sqrt(epsilon + (-u_V_max - u_V_min)**2)/2)*np.math.sin(psi)*np.math.cos(gamma)*np.math.tanh(k_V_1), -k_V_2*k_V_3*(1 - np.math.tanh(k_V_2*(V - V_max/2 - V_min/2 + (V_max/2 - V_min/2)*np.math.tanh(k_V_3*(-(P_x - P_x_obs)*np.math.cos(gamma)*np.math.cos(psi) - (P_y - P_y_obs)*np.math.sin(psi)*np.math.cos(gamma) - (P_z - P_z_obs)*np.math.sin(gamma)))))**2)*(1 - np.math.tanh(k_V_3*(-(P_x - P_x_obs)*np.math.cos(gamma)*np.math.cos(psi) - (P_y - P_y_obs)*np.math.sin(psi)*np.math.cos(gamma) - (P_z - P_z_obs)*np.math.sin(gamma)))**2)*(V_max/2 - V_min/2)*(-u_V_max/2 + u_V_min/2 + np.math.sqrt(epsilon + (-u_V_max - u_V_min)**2)/2)*np.math.sin(gamma)*np.math.tanh(k_V_1)], [k_gamma_1*(1 - np.math.tanh(k_gamma_1/(V + V_d))**2)*(u_gamma_max/2 - u_gamma_min/2 - 9.81*np.math.sqrt(0.00259777775699556*epsilon + (-0.0509683995922528*u_gamma_max - 0.0509683995922528*u_gamma_min + np.math.cos(gamma))**2))*np.math.tanh(k_gamma_2*(gamma - gamma_max/2 - gamma_min/2 + (gamma_max/2 - gamma_min/2)*np.math.tanh(k_gamma_3*(-(-P_x + P_x_obs)*np.math.sin(gamma)*np.math.cos(psi) + (P_y - P_y_obs)*np.math.sin(gamma)*np.math.sin(psi) - (P_z - P_z_obs)*np.math.cos(gamma))/(V + V_d))))/(V + V_d)**3 + k_gamma_2*k_gamma_3*(1 - np.math.tanh(k_gamma_2*(gamma - gamma_max/2 - gamma_min/2 + (gamma_max/2 - gamma_min/2)*np.math.tanh(k_gamma_3*(-(-P_x + P_x_obs)*np.math.sin(gamma)*np.math.cos(psi) + (P_y - P_y_obs)*np.math.sin(gamma)*np.math.sin(psi) - (P_z - P_z_obs)*np.math.cos(gamma))/(V + V_d))))**2)*(1 - np.math.tanh(k_gamma_3*(-(-P_x + P_x_obs)*np.math.sin(gamma)*np.math.cos(psi) + (P_y - P_y_obs)*np.math.sin(gamma)*np.math.sin(psi) - (P_z - P_z_obs)*np.math.cos(gamma))/(V + V_d))**2)*(gamma_max/2 - gamma_min/2)*(u_gamma_max/2 - u_gamma_min/2 - 9.81*np.math.sqrt(0.00259777775699556*epsilon + (-0.0509683995922528*u_gamma_max - 0.0509683995922528*u_gamma_min + np.math.cos(gamma))**2))*(-(-P_x + P_x_obs)*np.math.sin(gamma)*np.math.cos(psi) + (P_y - P_y_obs)*np.math.sin(gamma)*np.math.sin(psi) - (P_z - P_z_obs)*np.math.cos(gamma))*np.math.tanh(k_gamma_1/(V + V_d))/(V + V_d)**3 + (u_gamma_max/2 - u_gamma_min/2 - 9.81*np.math.sqrt(0.00259777775699556*epsilon + (-0.0509683995922528*u_gamma_max - 0.0509683995922528*u_gamma_min + np.math.cos(gamma))**2))*np.math.tanh(k_gamma_1/(V + V_d))*np.math.tanh(k_gamma_2*(gamma - gamma_max/2 - gamma_min/2 + (gamma_max/2 - gamma_min/2)*np.math.tanh(k_gamma_3*(-(-P_x + P_x_obs)*np.math.sin(gamma)*np.math.cos(psi) + (P_y - P_y_obs)*np.math.sin(gamma)*np.math.sin(psi) - (P_z - P_z_obs)*np.math.cos(gamma))/(V + V_d))))/(V + V_d)**2, -k_gamma_2*(1 - np.math.tanh(k_gamma_2*(gamma - gamma_max/2 - gamma_min/2 + (gamma_max/2 - gamma_min/2)*np.math.tanh(k_gamma_3*(-(-P_x + P_x_obs)*np.math.sin(gamma)*np.math.cos(psi) + (P_y - P_y_obs)*np.math.sin(gamma)*np.math.sin(psi) - (P_z - P_z_obs)*np.math.cos(gamma))/(V + V_d))))**2)*(k_gamma_3*(1 - np.math.tanh(k_gamma_3*(-(-P_x + P_x_obs)*np.math.sin(gamma)*np.math.cos(psi) + (P_y - P_y_obs)*np.math.sin(gamma)*np.math.sin(psi) - (P_z - P_z_obs)*np.math.cos(gamma))/(V + V_d))**2)*(gamma_max/2 - gamma_min/2)*((P_x - P_x_obs)*np.math.cos(gamma)*np.math.cos(psi) + (P_y - P_y_obs)*np.math.sin(psi)*np.math.cos(gamma) - (-P_z + P_z_obs)*np.math.sin(gamma))/(V + V_d) + 1)*(u_gamma_max/2 - u_gamma_min/2 - 9.81*np.math.sqrt(0.00259777775699556*epsilon + (-0.0509683995922528*u_gamma_max - 0.0509683995922528*u_gamma_min + np.math.cos(gamma))**2))*np.math.tanh(k_gamma_1/(V + V_d))/(V + V_d) - 9.81*(-0.0509683995922528*u_gamma_max - 0.0509683995922528*u_gamma_min + np.math.cos(gamma))*np.math.sin(gamma)*np.math.tanh(k_gamma_1/(V + V_d))*np.math.tanh(k_gamma_2*(gamma - gamma_max/2 - gamma_min/2 + (gamma_max/2 - gamma_min/2)*np.math.tanh(k_gamma_3*(-(-P_x + P_x_obs)*np.math.sin(gamma)*np.math.cos(psi) + (P_y - P_y_obs)*np.math.sin(gamma)*np.math.sin(psi) - (P_z - P_z_obs)*np.math.cos(gamma))/(V + V_d))))/((V + V_d)*np.math.sqrt(0.00259777775699556*epsilon + (-0.0509683995922528*u_gamma_max - 0.0509683995922528*u_gamma_min + np.math.cos(gamma))**2)), -k_gamma_2*k_gamma_3*(1 - np.math.tanh(k_gamma_2*(gamma - gamma_max/2 - gamma_min/2 + (gamma_max/2 - gamma_min/2)*np.math.tanh(k_gamma_3*(-(-P_x + P_x_obs)*np.math.sin(gamma)*np.math.cos(psi) + (P_y - P_y_obs)*np.math.sin(gamma)*np.math.sin(psi) - (P_z - P_z_obs)*np.math.cos(gamma))/(V + V_d))))**2)*(1 - np.math.tanh(k_gamma_3*(-(-P_x + P_x_obs)*np.math.sin(gamma)*np.math.cos(psi) + (P_y - P_y_obs)*np.math.sin(gamma)*np.math.sin(psi) - (P_z - P_z_obs)*np.math.cos(gamma))/(V + V_d))**2)*(gamma_max/2 - gamma_min/2)*(-(P_x - P_x_obs)*np.math.sin(gamma)*np.math.sin(psi) + (P_y - P_y_obs)*np.math.sin(gamma)*np.math.cos(psi))*(u_gamma_max/2 - u_gamma_min/2 - 9.81*np.math.sqrt(0.00259777775699556*epsilon + (-0.0509683995922528*u_gamma_max - 0.0509683995922528*u_gamma_min + np.math.cos(gamma))**2))*np.math.tanh(k_gamma_1/(V + V_d))/(V + V_d)**2, -k_gamma_2*k_gamma_3*(1 - np.math.tanh(k_gamma_2*(gamma - gamma_max/2 - gamma_min/2 + (gamma_max/2 - gamma_min/2)*np.math.tanh(k_gamma_3*(-(-P_x + P_x_obs)*np.math.sin(gamma)*np.math.cos(psi) + (P_y - P_y_obs)*np.math.sin(gamma)*np.math.sin(psi) - (P_z - P_z_obs)*np.math.cos(gamma))/(V + V_d))))**2)*(1 - np.math.tanh(k_gamma_3*(-(-P_x + P_x_obs)*np.math.sin(gamma)*np.math.cos(psi) + (P_y - P_y_obs)*np.math.sin(gamma)*np.math.sin(psi) - (P_z - P_z_obs)*np.math.cos(gamma))/(V + V_d))**2)*(gamma_max/2 - gamma_min/2)*(u_gamma_max/2 - u_gamma_min/2 - 9.81*np.math.sqrt(0.00259777775699556*epsilon + (-0.0509683995922528*u_gamma_max - 0.0509683995922528*u_gamma_min + np.math.cos(gamma))**2))*np.math.sin(gamma)*np.math.cos(psi)*np.math.tanh(k_gamma_1/(V + V_d))/(V + V_d)**2, -k_gamma_2*k_gamma_3*(1 - np.math.tanh(k_gamma_2*(gamma - gamma_max/2 - gamma_min/2 + (gamma_max/2 - gamma_min/2)*np.math.tanh(k_gamma_3*(-(-P_x + P_x_obs)*np.math.sin(gamma)*np.math.cos(psi) + (P_y - P_y_obs)*np.math.sin(gamma)*np.math.sin(psi) - (P_z - P_z_obs)*np.math.cos(gamma))/(V + V_d))))**2)*(1 - np.math.tanh(k_gamma_3*(-(-P_x + P_x_obs)*np.math.sin(gamma)*np.math.cos(psi) + (P_y - P_y_obs)*np.math.sin(gamma)*np.math.sin(psi) - (P_z - P_z_obs)*np.math.cos(gamma))/(V + V_d))**2)*(gamma_max/2 - gamma_min/2)*(u_gamma_max/2 - u_gamma_min/2 - 9.81*np.math.sqrt(0.00259777775699556*epsilon + (-0.0509683995922528*u_gamma_max - 0.0509683995922528*u_gamma_min + np.math.cos(gamma))**2))*np.math.sin(gamma)*np.math.sin(psi)*np.math.tanh(k_gamma_1/(V + V_d))/(V + V_d)**2, k_gamma_2*k_gamma_3*(1 - np.math.tanh(k_gamma_2*(gamma - gamma_max/2 - gamma_min/2 + (gamma_max/2 - gamma_min/2)*np.math.tanh(k_gamma_3*(-(-P_x + P_x_obs)*np.math.sin(gamma)*np.math.cos(psi) + (P_y - P_y_obs)*np.math.sin(gamma)*np.math.sin(psi) - (P_z - P_z_obs)*np.math.cos(gamma))/(V + V_d))))**2)*(1 - np.math.tanh(k_gamma_3*(-(-P_x + P_x_obs)*np.math.sin(gamma)*np.math.cos(psi) + (P_y - P_y_obs)*np.math.sin(gamma)*np.math.sin(psi) - (P_z - P_z_obs)*np.math.cos(gamma))/(V + V_d))**2)*(gamma_max/2 - gamma_min/2)*(u_gamma_max/2 - u_gamma_min/2 - 9.81*np.math.sqrt(0.00259777775699556*epsilon + (-0.0509683995922528*u_gamma_max - 0.0509683995922528*u_gamma_min + np.math.cos(gamma))**2))*np.math.cos(gamma)*np.math.tanh(k_gamma_1/(V + V_d))/(V + V_d)**2], [(u_psi_max/2 - u_psi_min/2 - np.math.sqrt(epsilon + (-u_psi_max - u_psi_min)**2)/2)*np.math.tanh(k_psi*(-(-P_x + P_x_obs)*np.math.sin(psi) - (P_y - P_y_obs)*np.math.cos(psi)))/((V + V_d)**2*np.math.cos(gamma)), -(u_psi_max/2 - u_psi_min/2 - np.math.sqrt(epsilon + (-u_psi_max - u_psi_min)**2)/2)*np.math.sin(gamma)*np.math.tanh(k_psi*(-(-P_x + P_x_obs)*np.math.sin(psi) - (P_y - P_y_obs)*np.math.cos(psi)))/((V + V_d)*np.math.cos(gamma)**2), -k_psi*(1 - np.math.tanh(k_psi*(-(-P_x + P_x_obs)*np.math.sin(psi) - (P_y - P_y_obs)*np.math.cos(psi)))**2)*((P_x - P_x_obs)*np.math.cos(psi) - (-P_y + P_y_obs)*np.math.sin(psi))*(u_psi_max/2 - u_psi_min/2 - np.math.sqrt(epsilon + (-u_psi_max - u_psi_min)**2)/2)/((V + V_d)*np.math.cos(gamma)), -k_psi*(1 - np.math.tanh(k_psi*(-(-P_x + P_x_obs)*np.math.sin(psi) - (P_y - P_y_obs)*np.math.cos(psi)))**2)*(u_psi_max/2 - u_psi_min/2 - np.math.sqrt(epsilon + (-u_psi_max - u_psi_min)**2)/2)*np.math.sin(psi)/((V + V_d)*np.math.cos(gamma)), k_psi*(1 - np.math.tanh(k_psi*(-(-P_x + P_x_obs)*np.math.sin(psi) - (P_y - P_y_obs)*np.math.cos(psi)))**2)*(u_psi_max/2 - u_psi_min/2 - np.math.sqrt(epsilon + (-u_psi_max - u_psi_min)**2)/2)*np.math.cos(psi)/((V + V_d)*np.math.cos(gamma)), 0], [np.math.cos(gamma)*np.math.cos(psi), -(V + V_d)*np.math.sin(gamma)*np.math.cos(psi), -(V + V_d)*np.math.sin(psi)*np.math.cos(gamma), 0, 0, 0], [np.math.sin(psi)*np.math.cos(gamma), -(V + V_d)*np.math.sin(gamma)*np.math.sin(psi), (V + V_d)*np.math.cos(gamma)*np.math.cos(psi), 0, 0, 0], [np.math.sin(gamma), (V + V_d)*np.math.cos(gamma), 0, 0, 0, 0]])
                theta = theta + theta_dot * 0.1
                theta_evolution.append(theta)
            round_psi_h_obs_round_x = []
            for t_c_0_obs in t_c_obs:
                theta_t_c_0 = theta_evolution[t_c_0_obs]
                V, gamma, psi, P_x, P_y, P_z = x_evolution[t_c_0_obs][0], x_evolution[t_c_0_obs][1], x_evolution[t_c_0_obs][2], x_evolution[t_c_0_obs][3], x_evolution[t_c_0_obs][4], x_evolution[t_c_0_obs][5]
                round_h_round_x = np.array([[0], [0], [0], [-2*(P_x - P_x_obs)], [-2*(P_y - P_y_obs)], [-2*(P_z - P_z_obs)]]).T
                round_psi_h_obs_round_x.append(round_h_round_x @ theta_t_c_0)
            round_psi_h_obs_round_x = np.array(round_psi_h_obs_round_x)
            H_obs = np.max(h_obs_evolution)

            u_V_smoothness, u_gamma_smoothness, u_psi_smoothness = 0.0, 0.1, 2.5 # 0.0, 0.1, 2.5
            alpha = 0.1 # if alpha>1.0, then the optimization problem becomes infeasible!! (alpha=1.01 -> infeasible tested)
            prev_input = system._state._u
            prev_state = system._state._x
            opti = ca.Opti()
            u_cbf = opti.variable(3, 1)
            x_next = opti.variable(6, 1)
            opti.subject_to(x_next == self._dynamics(prev_state, u_cbf))
            cost = (ca.norm_2(u_cbf - self._nominal_input))**2 + u_V_smoothness * (u_cbf[0] - prev_input[0])**2 + u_gamma_smoothness * (u_cbf[1] - prev_input[1])**2 + u_psi_smoothness * (u_cbf[2] - prev_input[2])**2
            V, gamma, psi, P_x, P_y, P_z = x_evolution[0][0], x_evolution[0][1], x_evolution[0][2], x_evolution[0][3], x_evolution[0][4], x_evolution[0][5]
            H_dot_obs = round_psi_h_obs_round_x[0][0][0] * u_cbf[0] + round_psi_h_obs_round_x[0][0][1] * (u_cbf[1] - self.g * np.math.cos(gamma)) / (V + V_d) + round_psi_h_obs_round_x[0][0][2] * u_cbf[2] / ((V + V_d) * ca.cos(gamma)) + round_psi_h_obs_round_x[0][0][3] * (V + V_d) * ca.cos(gamma) * ca.cos(psi) + round_psi_h_obs_round_x[0][0][4] * (V + V_d) * ca.cos(gamma) * ca.sin(psi) + round_psi_h_obs_round_x[0][0][5] * (V + V_d) * ca.sin(gamma)

            # state constraints
            opti.subject_to(((x_next[0] - (V_min + V_max) / 2)**2 - ((V_max - V_min) / 2)**2) <= 0)
            opti.subject_to(((x_next[1] - (gamma_min + gamma_max) / 2)**2 - ((gamma_max - gamma_min) / 2)**2) <= 0)

            # obstacle avoidance CBF constraint
            if (round_psi_h_obs_round_x[0][0][0] != 0 or (round_psi_h_obs_round_x[0][0][1] / (V + V_d)) != 0 or round_psi_h_obs_round_x[0][0][2] / ((V + V_d) * ca.cos(gamma)) != 0):
                # opti.subject_to(H_dot_obs <= 0.8 * (- H_obs)) # originally 0.8
                opti.subject_to(H_dot_obs * 0.1 <= alpha * (- H_obs)) 

            # input constraints
            opti.subject_to(u_cbf[0] <= u_V_max)
            opti.subject_to(u_cbf[0] >= u_V_min)
            opti.subject_to(u_cbf[1] <= u_gamma_max)
            opti.subject_to(u_cbf[1] >= u_gamma_min)
            opti.subject_to(u_cbf[2] <= u_psi_max)
            opti.subject_to(u_cbf[2] >= u_psi_min)
            opti.minimize(cost)
            option_ipopt = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
            s_opts = {
                "max_iter": 1e10,
            }
            opti.solver("ipopt", option_ipopt, s_opts)
            opt_sol = opti.solve()
            end_timer = datetime.datetime.now()
            delta_timer = end_timer - start_timer
            self._solver_times.append(delta_timer.total_seconds())
            self._cbf_opt_sol = opt_sol
            logger._utrajs.append(self._cbf_opt_sol.value(u_cbf).T)
            print("solver time: ", delta_timer.total_seconds())
            print("nominal input from MPC: ", self._nominal_input)
            print("safe input from CBF: ", opt_sol.value(u_cbf).T)
            print("state: ", system._dynamics.forward_dynamics(system._state._x, opt_sol.value(u_cbf).T, 0.1))
            return opt_sol.value(u_cbf).T
        
        if self._constraint_type == "nominal_evading_maneuver_no_RD1":
            # self.theta_dot_expr_no_RD1()
            self._optimizer.setup(self._param, system, local_trajectory)
            start_timer = datetime.datetime.now()
            self._opt_sol = self._optimizer.solve_nlp()
            # print(self._opt_sol.value(self._optimizer.variables["x"][:, 0]))
            # print(self._opt_sol.value(self._optimizer.variables["u"][:, 0]))

            # Assuming single static obstacle!!
            static_obstacle = static_obstacles[0]
            self._nominal_input = self._opt_sol.value(self._optimizer.variables["u"][:, 0])
            h_obs_evolution = []
            x_evolution = []
            t_c_obs = [] # list of indices(not time!!) that maximize h_obs
            robot_state = system._state._x
            robot_radius = system._geometry._radius + self.margin_dist
            h_obs = self.h_obs_comp(robot_state, robot_radius, static_obstacle)
            h_obs_evolution.append(h_obs)
            x_evolution.append(robot_state)
            N_prop = self._prop_time * 10 # propagation discretization time 0.1s
            for i in range(N_prop):
                evading_maneuver = self.evading_maneuver_comp_no_RD1(robot_state, static_obstacle)
                robot_state = system._dynamics.forward_dynamics(robot_state, evading_maneuver, 0.1)
                h_obs = self.h_obs_comp(robot_state, robot_radius, static_obstacle)
                h_obs_evolution.append(h_obs)
                x_evolution.append(robot_state)
            h_obs_evolution = np.array(h_obs_evolution)
            x_evolution = np.array(x_evolution)

            for index in range(len(h_obs_evolution)):
                if h_obs_evolution[index] >= np.max(h_obs_evolution):
                    t_c_obs.append(index)
            
            theta = np.identity(6)
            theta_evolution = []
            theta_evolution.append(theta)
            P_x_obs, P_y_obs, P_z_obs = static_obstacle[0][0], static_obstacle[0][1], static_obstacle[0][2]
            k_V_1, k_V_2, k_V_3, k_gamma_1, k_gamma_2, k_gamma_3, k_psi = self._control_gains[0], self._control_gains[1], self._control_gains[2], self._control_gains[3], self._control_gains[4], self._control_gains[5], self._control_gains[6]
            u_V_min, u_V_max, u_gamma_min, u_gamma_max, u_psi_min, u_psi_max = self._input_constraints[0], self._input_constraints[1], self._input_constraints[2], self._input_constraints[3], self._input_constraints[4], self._input_constraints[5]
            V_min, V_max, gamma_min, gamma_max = self._state_constraints[0], self._state_constraints[1], self._state_constraints[2], self._state_constraints[3]
            V_d, epsilon = self.V_d, self.epsilon
            for i in range(N_prop):
                V, gamma, psi, P_x, P_y, P_z = x_evolution[i][0], x_evolution[i][1], x_evolution[i][2], x_evolution[i][3], x_evolution[i][4], x_evolution[i][5]
                theta_dot = np.array([[0, k_V_1*(1 - np.math.tanh(k_V_1*(-(P_x - P_x_obs)*np.math.cos(gamma)*np.math.cos(psi) - (P_y - P_y_obs)*np.math.sin(psi)*np.math.cos(gamma) - (P_z - P_z_obs)*np.math.sin(gamma)))**2)*(-u_V_max/2 + u_V_min/2 + np.math.sqrt(epsilon + (-u_V_max - u_V_min)**2)/2)*(-(-P_x + P_x_obs)*np.math.sin(gamma)*np.math.cos(psi) - (-P_y + P_y_obs)*np.math.sin(gamma)*np.math.sin(psi) + (-P_z + P_z_obs)*np.math.cos(gamma)), k_V_1*(1 - np.math.tanh(k_V_1*(-(P_x - P_x_obs)*np.math.cos(gamma)*np.math.cos(psi) - (P_y - P_y_obs)*np.math.sin(psi)*np.math.cos(gamma) - (P_z - P_z_obs)*np.math.sin(gamma)))**2)*(-(-P_x + P_x_obs)*np.math.sin(psi)*np.math.cos(gamma) + (-P_y + P_y_obs)*np.math.cos(gamma)*np.math.cos(psi))*(-u_V_max/2 + u_V_min/2 + np.math.sqrt(epsilon + (-u_V_max - u_V_min)**2)/2), -k_V_1*(1 - np.math.tanh(k_V_1*(-(P_x - P_x_obs)*np.math.cos(gamma)*np.math.cos(psi) - (P_y - P_y_obs)*np.math.sin(psi)*np.math.cos(gamma) - (P_z - P_z_obs)*np.math.sin(gamma)))**2)*(-u_V_max/2 + u_V_min/2 + np.math.sqrt(epsilon + (-u_V_max - u_V_min)**2)/2)*np.math.cos(gamma)*np.math.cos(psi), -k_V_1*(1 - np.math.tanh(k_V_1*(-(P_x - P_x_obs)*np.math.cos(gamma)*np.math.cos(psi) - (P_y - P_y_obs)*np.math.sin(psi)*np.math.cos(gamma) - (P_z - P_z_obs)*np.math.sin(gamma)))**2)*(-u_V_max/2 + u_V_min/2 + np.math.sqrt(epsilon + (-u_V_max - u_V_min)**2)/2)*np.math.sin(psi)*np.math.cos(gamma), -k_V_1*(1 - np.math.tanh(k_V_1*(-(P_x - P_x_obs)*np.math.cos(gamma)*np.math.cos(psi) - (P_y - P_y_obs)*np.math.sin(psi)*np.math.cos(gamma) - (P_z - P_z_obs)*np.math.sin(gamma)))**2)*(-u_V_max/2 + u_V_min/2 + np.math.sqrt(epsilon + (-u_V_max - u_V_min)**2)/2)*np.math.sin(gamma)], [(u_gamma_max/2 - u_gamma_min/2 - 9.81*np.math.sqrt(0.00259777775699556*epsilon + (-0.0509683995922528*u_gamma_max - 0.0509683995922528*u_gamma_min + np.math.cos(gamma))**2))*np.math.tanh(k_gamma_1*(-(-P_x + P_x_obs)*np.math.sin(gamma)*np.math.cos(psi) + (P_y - P_y_obs)*np.math.sin(gamma)*np.math.sin(psi) - (P_z - P_z_obs)*np.math.cos(gamma)))/(V + V_d)**2, -k_gamma_1*(1 - np.math.tanh(k_gamma_1*(-(-P_x + P_x_obs)*np.math.sin(gamma)*np.math.cos(psi) + (P_y - P_y_obs)*np.math.sin(gamma)*np.math.sin(psi) - (P_z - P_z_obs)*np.math.cos(gamma)))**2)*(u_gamma_max/2 - u_gamma_min/2 - 9.81*np.math.sqrt(0.00259777775699556*epsilon + (-0.0509683995922528*u_gamma_max - 0.0509683995922528*u_gamma_min + np.math.cos(gamma))**2))*((P_x - P_x_obs)*np.math.cos(gamma)*np.math.cos(psi) + (P_y - P_y_obs)*np.math.sin(psi)*np.math.cos(gamma) - (-P_z + P_z_obs)*np.math.sin(gamma))/(V + V_d) - 9.81*(-0.0509683995922528*u_gamma_max - 0.0509683995922528*u_gamma_min + np.math.cos(gamma))*np.math.sin(gamma)*np.math.tanh(k_gamma_1*(-(-P_x + P_x_obs)*np.math.sin(gamma)*np.math.cos(psi) + (P_y - P_y_obs)*np.math.sin(gamma)*np.math.sin(psi) - (P_z - P_z_obs)*np.math.cos(gamma)))/((V + V_d)*np.math.sqrt(0.00259777775699556*epsilon + (-0.0509683995922528*u_gamma_max - 0.0509683995922528*u_gamma_min + np.math.cos(gamma))**2)), -k_gamma_1*(1 - np.math.tanh(k_gamma_1*(-(-P_x + P_x_obs)*np.math.sin(gamma)*np.math.cos(psi) + (P_y - P_y_obs)*np.math.sin(gamma)*np.math.sin(psi) - (P_z - P_z_obs)*np.math.cos(gamma)))**2)*(-(P_x - P_x_obs)*np.math.sin(gamma)*np.math.sin(psi) + (P_y - P_y_obs)*np.math.sin(gamma)*np.math.cos(psi))*(u_gamma_max/2 - u_gamma_min/2 - 9.81*np.math.sqrt(0.00259777775699556*epsilon + (-0.0509683995922528*u_gamma_max - 0.0509683995922528*u_gamma_min + np.math.cos(gamma))**2))/(V + V_d), -k_gamma_1*(1 - np.math.tanh(k_gamma_1*(-(-P_x + P_x_obs)*np.math.sin(gamma)*np.math.cos(psi) + (P_y - P_y_obs)*np.math.sin(gamma)*np.math.sin(psi) - (P_z - P_z_obs)*np.math.cos(gamma)))**2)*(u_gamma_max/2 - u_gamma_min/2 - 9.81*np.math.sqrt(0.00259777775699556*epsilon + (-0.0509683995922528*u_gamma_max - 0.0509683995922528*u_gamma_min + np.math.cos(gamma))**2))*np.math.sin(gamma)*np.math.cos(psi)/(V + V_d), -k_gamma_1*(1 - np.math.tanh(k_gamma_1*(-(-P_x + P_x_obs)*np.math.sin(gamma)*np.math.cos(psi) + (P_y - P_y_obs)*np.math.sin(gamma)*np.math.sin(psi) - (P_z - P_z_obs)*np.math.cos(gamma)))**2)*(u_gamma_max/2 - u_gamma_min/2 - 9.81*np.math.sqrt(0.00259777775699556*epsilon + (-0.0509683995922528*u_gamma_max - 0.0509683995922528*u_gamma_min + np.math.cos(gamma))**2))*np.math.sin(gamma)*np.math.sin(psi)/(V + V_d), k_gamma_1*(1 - np.math.tanh(k_gamma_1*(-(-P_x + P_x_obs)*np.math.sin(gamma)*np.math.cos(psi) + (P_y - P_y_obs)*np.math.sin(gamma)*np.math.sin(psi) - (P_z - P_z_obs)*np.math.cos(gamma)))**2)*(u_gamma_max/2 - u_gamma_min/2 - 9.81*np.math.sqrt(0.00259777775699556*epsilon + (-0.0509683995922528*u_gamma_max - 0.0509683995922528*u_gamma_min + np.math.cos(gamma))**2))*np.math.cos(gamma)/(V + V_d)], [(u_psi_max/2 - u_psi_min/2 - np.math.sqrt(epsilon + (-u_psi_max - u_psi_min)**2)/2)*np.math.tanh(k_psi*(-(-P_x + P_x_obs)*np.math.sin(psi) - (P_y - P_y_obs)*np.math.cos(psi)))/((V + V_d)**2*np.math.cos(gamma)), -(u_psi_max/2 - u_psi_min/2 - np.math.sqrt(epsilon + (-u_psi_max - u_psi_min)**2)/2)*np.math.sin(gamma)*np.math.tanh(k_psi*(-(-P_x + P_x_obs)*np.math.sin(psi) - (P_y - P_y_obs)*np.math.cos(psi)))/((V + V_d)*np.math.cos(gamma)**2), -k_psi*(1 - np.math.tanh(k_psi*(-(-P_x + P_x_obs)*np.math.sin(psi) - (P_y - P_y_obs)*np.math.cos(psi)))**2)*((P_x - P_x_obs)*np.math.cos(psi) - (-P_y + P_y_obs)*np.math.sin(psi))*(u_psi_max/2 - u_psi_min/2 - np.math.sqrt(epsilon + (-u_psi_max - u_psi_min)**2)/2)/((V + V_d)*np.math.cos(gamma)), -k_psi*(1 - np.math.tanh(k_psi*(-(-P_x + P_x_obs)*np.math.sin(psi) - (P_y - P_y_obs)*np.math.cos(psi)))**2)*(u_psi_max/2 - u_psi_min/2 - np.math.sqrt(epsilon + (-u_psi_max - u_psi_min)**2)/2)*np.math.sin(psi)/((V + V_d)*np.math.cos(gamma)), k_psi*(1 - np.math.tanh(k_psi*(-(-P_x + P_x_obs)*np.math.sin(psi) - (P_y - P_y_obs)*np.math.cos(psi)))**2)*(u_psi_max/2 - u_psi_min/2 - np.math.sqrt(epsilon + (-u_psi_max - u_psi_min)**2)/2)*np.math.cos(psi)/((V + V_d)*np.math.cos(gamma)), 0], [np.math.cos(gamma)*np.math.cos(psi), -(V + V_d)*np.math.sin(gamma)*np.math.cos(psi), -(V + V_d)*np.math.sin(psi)*np.math.cos(gamma), 0, 0, 0], [np.math.sin(psi)*np.math.cos(gamma), -(V + V_d)*np.math.sin(gamma)*np.math.sin(psi), (V + V_d)*np.math.cos(gamma)*np.math.cos(psi), 0, 0, 0], [np.math.sin(gamma), (V + V_d)*np.math.cos(gamma), 0, 0, 0, 0]])
                theta = theta + theta_dot * 0.1
                theta_evolution.append(theta)
            round_psi_h_obs_round_x = []
            for t_c_0_obs in t_c_obs:
                theta_t_c_0 = theta_evolution[t_c_0_obs]
                V, gamma, psi, P_x, P_y, P_z = x_evolution[t_c_0_obs][0], x_evolution[t_c_0_obs][1], x_evolution[t_c_0_obs][2], x_evolution[t_c_0_obs][3], x_evolution[t_c_0_obs][4], x_evolution[t_c_0_obs][5]
                round_h_round_x = np.array([[0], [0], [0], [-2*(P_x - P_x_obs)], [-2*(P_y - P_y_obs)], [-2*(P_z - P_z_obs)]]).T
                round_psi_h_obs_round_x.append(round_h_round_x @ theta_t_c_0)
            round_psi_h_obs_round_x = np.array(round_psi_h_obs_round_x)
            H_obs = np.max(h_obs_evolution)
            h_V = (V - ((V_min + V_max) / 2))**2 - ((V_max - V_min) / 2)**2
            h_gamma = (gamma - ((gamma_min + gamma_max) / 2))**2 - ((gamma_max - gamma_min) / 2)**2

            alpha_obs = 0.1 # if alpha>1.0, then the optimization problem becomes infeasible!! (alpha=1.01 -> infeasible tested)
            alpha_V = 0.1
            alpha_gamma = 0.1
            opti = ca.Opti()
            u_cbf = opti.variable(3, 1)
            cost = (ca.norm_2(u_cbf - self._nominal_input))**2
            V, gamma, psi, P_x, P_y, P_z = x_evolution[0][0], x_evolution[0][1], x_evolution[0][2], x_evolution[0][3], x_evolution[0][4], x_evolution[0][5]
            H_dot_obs = round_psi_h_obs_round_x[0][0][0] * u_cbf[0] + round_psi_h_obs_round_x[0][0][1] * (u_cbf[1] - self.g * np.math.cos(gamma)) / (V + V_d) + round_psi_h_obs_round_x[0][0][2] * u_cbf[2] / ((V + V_d) * ca.cos(gamma)) + round_psi_h_obs_round_x[0][0][3] * (V + V_d) * ca.cos(gamma) * ca.cos(psi) + round_psi_h_obs_round_x[0][0][4] * (V + V_d) * ca.cos(gamma) * ca.sin(psi) + round_psi_h_obs_round_x[0][0][5] * (V + V_d) * ca.sin(gamma)
            h_dot_V = 2 * (V - ((V_min + V_max) / 2)) * u_cbf[0]
            h_dot_gamma = 2 * (gamma - ((gamma_min + gamma_max) / 2)) * (u_cbf[1] - self.g * np.math.cos(gamma)) / (V + V_d)

            # obstacle avoidance and state constraints as CBF constraints
            if (round_psi_h_obs_round_x[0][0][0] != 0 or (round_psi_h_obs_round_x[0][0][1] / (V + V_d)) != 0 or round_psi_h_obs_round_x[0][0][2] / ((V + V_d) * ca.cos(gamma)) != 0):
                opti.subject_to(H_dot_obs * 0.1 <= alpha_obs * (- H_obs))
                print(H_dot_obs)
            # if (2 * (V - ((V_min + V_max) / 2)) != 0):
            #     opti.subject_to(h_dot_V * 0.1 <= alpha_V * (-h_V))
            #     print(h_dot_V)
            # if (2 * (gamma - ((gamma_min + gamma_max) / 2)) != 0):
            #     opti.subject_to(h_dot_gamma * 0.1 <= alpha_gamma * (-h_gamma))
            #     print(h_dot_gamma)

            # input constraints
            opti.subject_to(u_cbf[0] <= u_V_max)
            opti.subject_to(u_cbf[0] >= u_V_min)
            opti.subject_to(u_cbf[1] <= u_gamma_max)
            opti.subject_to(u_cbf[1] >= u_gamma_min)
            opti.subject_to(u_cbf[2] <= u_psi_max)
            opti.subject_to(u_cbf[2] >= u_psi_min)
            opti.minimize(cost)
            option_ipopt = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
            s_opts = {
                "max_iter": 1e10,
            }
            opti.solver("ipopt", option_ipopt, s_opts)
            opt_sol = opti.solve()
            end_timer = datetime.datetime.now()
            delta_timer = end_timer - start_timer
            self._solver_times.append(delta_timer.total_seconds())
            self._cbf_opt_sol = opt_sol
            logger._utrajs.append(self._cbf_opt_sol.value(u_cbf).T)
            print("solver time: ", delta_timer.total_seconds())
            print("nominal input from MPC: ", self._nominal_input)
            print("safe input from CBF: ", opt_sol.value(u_cbf).T)
            print("state: ", system._dynamics.forward_dynamics(system._state._x, opt_sol.value(u_cbf).T, 0.1))
            return opt_sol.value(u_cbf).T


    # def logging(self, logger):
    #     logger._xtrajs.append(self._opt_sol.value(self._optimizer.variables["x"]).T)
    #     logger._utrajs.append(self._cbf_opt_sol.value(self._optimizer.variables["u"]).T)
