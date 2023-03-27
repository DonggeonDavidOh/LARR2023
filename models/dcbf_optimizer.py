import datetime

import casadi as ca
import numpy as np

from geometry_utils import *

from simulation import Robot, SingleAgentSimulation

class NmpcDcbfOptimizerParam:
    def __init__(self, DCBF_horizon, constraint_type, static_obstacles):
        self.horizon = DCBF_horizon
        self.constraint_type = constraint_type
        self.mat_Q = np.diag([1.0, 20.0, 20.0, 1.0, 1.0, 1.0])
        self.mat_R = np.diag([0.1, 0.1, 0.1])
        self.mat_Rold = np.diag([0.1, 0.1, 0.1])
        self.mat_dR = np.diag([0.1, 0.1, 0.1])
        self.terminal_weight = 10.0
        self.obs_center = static_obstacles[0][0]
        self.obs_radius = static_obstacles[0][1]
        self.margin_dist = 1.0
        self.V_d = 10.0
        self.path_radius = 50 * np.math.sqrt(2)

class NmpcDbcfOptimizer:
    def __init__(self, variables: dict, costs: dict, dynamics_opt):
        self.opti = None
        self.variables = variables
        self.costs = costs
        self.dynamics_opt = dynamics_opt
        self.solver_times = []

    def set_state(self, param, system):
        self.state = system._state
        self.robot_radius = system._geometry._radius + param.margin_dist

    def initialize_variables(self, param):
        self.variables["x"] = self.opti.variable(6, param.horizon + 1)
        self.variables["u"] = self.opti.variable(3, param.horizon)

    def add_initial_condition_constraint(self):
        self.opti.subject_to(self.variables["x"][:, 0] == self.state._x)

    def add_input_constraint(self, param):
        # "Obstacle Avoidance Using Image-based Visual Servoing Integrated with Nonlinear Model Predictive Control", H. Jin. Kim
        # x = [V, gamma, psi, P_x, P_y, P_z].T, u = [u_gamma, u_psi].T
        u_V_min, u_V_max = -1.0, 1.0
        u_gamma_min, u_gamma_max = 6.81, 12.81
        u_psi_min, u_psi_max = -3.0, 3.0
        for i in range(param.horizon):
            # input constraints
            self.opti.subject_to(self.variables["u"][0, i] <= u_V_max)
            self.opti.subject_to(u_V_min <= self.variables["u"][0, i])
            self.opti.subject_to(self.variables["u"][1, i] <= u_gamma_max)
            self.opti.subject_to(u_gamma_min <= self.variables["u"][1, i])
            self.opti.subject_to(self.variables["u"][2, i] <= u_psi_max)
            self.opti.subject_to(u_psi_min <= self.variables["u"][2, i])
    
    def add_state_constraints_MPC_form(self, param):
        V_min, V_max = -5.0, 10.0
        gamma_min, gamma_max = -0.5, 0.5
        for i in range(param.horizon):
            # print(param.obs_center)
            self.opti.subject_to((param.obs_radius + self.robot_radius)**2 - (ca.norm_2(self.variables["x"][3:6, i+1] - param.obs_center))**2 <= 0)
            self.opti.subject_to(self.variables["x"][0, i + 1] <= V_max)
            self.opti.subject_to(self.variables["x"][0, i + 1] >= V_min)
            self.opti.subject_to(self.variables["x"][1, i + 1] <= gamma_max)
            self.opti.subject_to(self.variables["x"][1, i + 1] >= gamma_min)

    def add_state_constraints_CBF_form(self, param):
        V_min, V_max = -5.0, 10.0
        gamma_min, gamma_max = -0.5, 0.5
        for i in range(param.horizon):
            # print("CBF")
            self.opti.subject_to((param.obs_radius + self.robot_radius)**2 - (ca.norm_2(self.variables["x"][3:6, i+1] - param.obs_center))**2 <= 0.2 * ((param.obs_radius + self.robot_radius)**2 - (ca.norm_2(self.variables["x"][3:6, i] - param.obs_center))**2))
            self.opti.subject_to(self.variables["x"][0, i + 1] <= V_max)
            self.opti.subject_to(self.variables["x"][0, i + 1] >= V_min)
            self.opti.subject_to(self.variables["x"][1, i + 1] <= gamma_max)
            self.opti.subject_to(self.variables["x"][1, i + 1] >= gamma_min)

    def add_dynamics_constraint(self, param):
        for i in range(param.horizon):
            self.opti.subject_to(
                self.variables["x"][:, i + 1] == self.dynamics_opt(self.variables["x"][:, i], self.variables["u"][:, i])
            )

    def add_reference_trajectory_tracking_cost(self, param, reference_trajectory):
        self.costs["reference_trajectory_tracking"] = 0
        for i in range(param.horizon - 1):
            x_diff = self.variables["x"][:, i + 1] - reference_trajectory[i, :]
            self.costs["reference_trajectory_tracking"] += ca.mtimes(x_diff.T, ca.mtimes(param.mat_Q, x_diff))
        x_diff = self.variables["x"][:, -1] - reference_trajectory[-1, :]
        self.costs["reference_trajectory_tracking"] += param.terminal_weight * ca.mtimes(
            x_diff.T, ca.mtimes(param.mat_Q, x_diff)
        )

    def add_input_stage_cost(self, param):
        self.costs["input_stage"] = 0
        radius = param.path_radius
        u_V_nominal = 0.0
        u_gamma_nominal = 9.81
        u_psi_nominal = param.V_d**2 / radius
        u_nominal = np.array([u_V_nominal, u_gamma_nominal, u_psi_nominal])
        for i in range(param.horizon):
            u_diff = self.variables["u"][:, i] - u_nominal
            self.costs["input_stage"] += ca.mtimes(
                u_diff.T, ca.mtimes(param.mat_R, u_diff)
            )

    def add_prev_input_cost(self, param):
        self.costs["prev_input"] = 0
        self.costs["prev_input"] += ca.mtimes(
            (self.variables["u"][:, 0] - self.state._u).T,
            ca.mtimes(param.mat_Rold, (self.variables["u"][:, 0] - self.state._u)),
        )

    def add_input_smoothness_cost(self, param):
        self.costs["input_smoothness"] = 0
        for i in range(param.horizon - 1):
            self.costs["input_smoothness"] += ca.mtimes(
                (self.variables["u"][:, i + 1] - self.variables["u"][:, i]).T,
                ca.mtimes(param.mat_dR, (self.variables["u"][:, i + 1] - self.variables["u"][:, i])),
            )

    def add_warm_start(self, param, system, reference_trajectory):
        # TODO: wrap params
        # x_ws, u_ws = system._dynamics.nominal_safe_controller(self.state._x, 0.1, -1.0, 1.0)
        # for i in range(param.horizon):
        #     self.opti.set_initial(self.variables["x"][:, i + 1], x_ws)
        #     self.opti.set_initial(self.variables["u"][:, i], u_ws)
        for i in range(param.horizon):
            radius = param.path_radius
            u_V_nominal = 0.0
            u_gamma_nominal = 9.81
            u_psi_nominal = param.V_d**2 / radius
            u_nominal = np.array([u_V_nominal, u_gamma_nominal, u_psi_nominal])
            self.opti.set_initial(self.variables["x"][:, i + 1], reference_trajectory[i, :])
            self.opti.set_initial(self.variables["u"][:, i], u_nominal)

    def setup(self, param, system, reference_trajectory):
        self.set_state(param, system)
        self.opti = ca.Opti()
        self.initialize_variables(param)
        self.add_initial_condition_constraint()
        self.add_input_constraint(param)
        if param.constraint_type == "MPC_form":
            self.add_state_constraints_MPC_form(param)
        if param.constraint_type == "CBF_form":
            self.add_state_constraints_CBF_form(param)
        self.add_dynamics_constraint(param)
        self.add_reference_trajectory_tracking_cost(param, reference_trajectory)
        self.add_input_stage_cost(param)
        self.add_prev_input_cost(param)
        self.add_input_smoothness_cost(param)
        self.add_warm_start(param, system, reference_trajectory)

    def solve_nlp(self):
        cost = 0
        for cost_name in self.costs:
            cost += self.costs[cost_name]
        self.opti.minimize(cost)
        # option_ipopt = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        option_ipopt = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        # s_opts = {"tol": 1e-1,
        # "max_iter": 1e10,
        # "dual_inf_tol": 1e3,
        # "constr_viol_tol": 1e-1,
        # "compl_inf_tol": 1e-1,
        # "acceptable_tol": 1e-2,
        # "acceptable_iter": 5,
        # "acceptable_dual_inf_tol": 1e15,
        # "acceptable_constr_viol_tol": 1e2,
        # "acceptable_compl_inf_tol": 1e2,
        # "acceptable_obj_change_tol": 1e30,
        # "diverging_iterates_tol": 1e30,
        # "mu_target": 1e1
        # }
        # s_opts = {"jacobian_approximation": "finite-difference-values",
        # "gradient_approximation": "finite-difference-values",
        # "hessian_approximation": "limited-memory",
        # }
        s_opts = {
            "max_iter": 1e10,
        }
        self.opti.solver("ipopt", option_ipopt, s_opts)
        # self.opti.solver("bonmin", {"verbose": False, "bonmin.print_level": False, "print_time": True})
        # print(self.opti.debug.value) # for debugging purposes
        opt_sol = self.opti.solve()
        return opt_sol
