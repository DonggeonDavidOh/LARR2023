import math
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl
from mpl_toolkits import mplot3d
import statistics as st
from matplotlib.gridspec import GridSpec

import datetime

from dcbf_optimizer import NmpcDcbfOptimizerParam
from dcbf_controller import NmpcDcbfController

from geometry_utils import *
from kinematic_car import (
    KinematicCarDynamics,
    KinematicCarRectangleGeometry,
    KinematicCarMultipleGeometry,
    KinematicCarTriangleGeometry,
    KinematicCarPentagonGeometry,
    KinematicCarPointGeometry,
    KinematicCarStates,
    KinematicCarSystem,
)
from kinematic_obstacle import (
    KinematicObstacleDynamics,
    KinematicObstacleStates,
    KinematicObstaclePointGeometry,
    KinematicObstacleSystem,
)
from search_path_generator import (
    AstarLoSPathGenerator,
    AstarPathGenerator,
    ThetaStarPathGenerator,
)
from constant_speed_generator import (
    ConstantSpeedTrajectoryGenerator,
    ConstantSpeedCircularTrajectoryGenerator,
)
from simulation import Robot, SingleAgentSimulation

def render_3D_video_with_computational_time(simulation, animation_name="world"):
    # TODO: make this plotting function general applicable to different systems
    # if maze_type == "maze":
    #     fig, ax = plt.subplots(figsize=(8.3, 5.0))
    # elif maze_type == "oblique_maze":
    #     fig, ax = plt.subplots(figsize=(6.7, 5.0))
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1, 1], height_ratios=[3, 1, 1])
    ax = fig.add_subplot(gs[0, 0], projection = '3d')
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 250)
    ax.set_zlim(-50, 100)
    ax.set_box_aspect((5, 5, 3))
    ax.view_init(elev=80, azim=215) # originally, (30, 300)
    ax_2 = fig.add_subplot(gs[0, 1], projection = '3d')
    ax_2.set_xlim(0, 250)
    ax_2.set_ylim(0, 250)
    ax_2.set_zlim(-50, 100)
    ax_2.set_box_aspect((5, 5, 3))
    ax_2.view_init(elev=20, azim=125) # originally, (30, 300)
    ax_2.grid(False)
    tmp_planes = ax_2.axes.zaxis._PLANES
    ax_2.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                     tmp_planes[0], tmp_planes[1], 
                     tmp_planes[4], tmp_planes[5])

    ax_dist = fig.add_subplot(gs[1, 0])
    ax_dist.grid(True)
    ax_vel = fig.add_subplot(gs[1, 1])
    ax_vel.grid(True)
    ax_gamma = fig.add_subplot(gs[2, 0])
    ax_gamma.grid(True)
    ax_input = fig.add_subplot(gs[2, 1])
    ax_input.grid(True)

    opti_solver_time = simulation._robot._controller._solver_times
    closedloop_traj = np.vstack(simulation._robot._system_logger._xs)
    num_dynamic_obs = len(simulation._dynamic_obstacles)
    dynamic_obs_traj = []
    for i in range(num_dynamic_obs):
        dynamic_obs_traj.append(np.vstack(simulation._dynamic_obstacles[i]._system_logger._xs))
                
    time, distance, comp_time = [], [], []
    V_his, gamma_his, u_gamma_his, u_psi_his, u_V_his = [], [], [], [], []
    distance_bound, V_upper_bound, V_lower_bound, gamma_upper_bound, gamma_lower_bound, u_V_upper_bound, u_V_lower_bound, u_gamma_upper_bound, u_gamma_lower_bound, u_psi_upper_bound, u_psi_lower_bound = [], [], [], [], [], [], [], [], [], [], []
    u_V_min, u_V_max, u_gamma_min, u_gamma_max, u_psi_min, u_psi_max = -1.0, 1.0, 6.81, 12.81, -3.0, 3.0
    V_min, V_max, gamma_min, gamma_max = -5.0, 10.0, -0.5, 0.5
    no_of_collisions = np.array([0])
    def update(index):
        ax.collections.clear()
        ax.clear()
        ax.set_xlim(0, 250)
        ax.set_ylim(0, 250)
        ax.set_zlim(-50, 100)
        ax.plot([250, 250], [0, 250], [-50, -50], "k-", linewidth=1.0)
        ax.plot([0, 250], [250, 250], [-50, -50], "k-", linewidth=1.0)
        ax.plot([250, 250], [250, 250], [-50, 100], "k-", linewidth=1.0)
        ax.set_box_aspect((5, 5, 3))
        ax.view_init(elev=80, azim=215)
        ax.set_xlabel(r'$P_x(m)$', fontsize=7)
        ax.xaxis.set_tick_params(labelsize=7)
        ax.set_ylabel(r'$P_y(m)$', fontsize=7)
        ax.yaxis.set_tick_params(labelsize=7)
        ax.axes.zaxis.set_ticklabels([])
        ax.grid(False)

        tmp_planes = ax_2.axes.zaxis._PLANES
        ax_2.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                     tmp_planes[0], tmp_planes[1], 
                     tmp_planes[4], tmp_planes[5])

        ax_2.collections.clear()
        ax_2.clear()
        ax_2.set_xlim(0, 250)
        ax_2.set_ylim(0, 250)
        ax_2.set_zlim(-50, 100)
        ax_2.plot([250, 250], [0, 250], [-50, -50], "k-", linewidth=1.0)
        ax_2.plot([0, 250], [0, 0], [-50, -50], "k-", linewidth=1.0)
        ax_2.plot([250, 250], [0, 0], [-50, 100], "k-", linewidth=1.0)
        ax_2.set_box_aspect((5, 5, 3))
        ax_2.view_init(elev=20, azim=125)
        # ax_2.axes.xaxis.set_ticklabels([])
        # ax_2.axes.yaxis.set_ticklabels([])
        ax_2.set_xlabel(r'$P_x(m)$', fontsize=7)
        ax_2.xaxis.set_tick_params(labelsize=7)
        ax_2.set_ylabel(r'$P_y(m)$', fontsize=7)
        ax_2.yaxis.set_tick_params(labelsize=7)
        ax_2.set_zlabel(r'$P_z(m)$', fontsize=7, labelpad=0)
        # ax_2.axes.zaxis.set_label(r'$P_z(m)$')
        ax_2.axes.zaxis.set_tick_params(labelsize=7)    

        tmp_planes = ax_2.axes.zaxis._PLANES
        ax_2.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                     tmp_planes[0], tmp_planes[1], 
                     tmp_planes[4], tmp_planes[5])
        ax_2.grid(False)    

        print(index)

        ax_dist.collections.clear()
        ax_dist.clear()
        ax_vel.collections.clear()
        ax_vel.clear()
        ax_gamma.collections.clear()
        ax_gamma.clear()
        ax_input.collections.clear()
        ax_input.clear()

        # plot global waypoints
        global_paths = simulation._robot._global_planner_logger._paths
        global_path = global_paths[0]
        ax.plot(global_path[:, 0], global_path[:, 1], global_path[:, 2], "bo", linewidth=0.5, markersize=0.5)
        ax_2.plot(global_path[:, 0], global_path[:, 1], global_path[:, 2], "bo", linewidth=0.5, markersize=0.5)

        # plot constant speed trajectory following the global path
        local_paths = simulation._robot._local_planner_logger._trajs
        local_path = local_paths[index]
        (reference_traj_line,) = ax.plot(local_path[:, 3], local_path[:, 4], local_path[:, 5], "-", color="blue", linewidth=1, markersize=1)
        (reference_traj_line,) = ax_2.plot(local_path[:, 3], local_path[:, 4], local_path[:, 5], "-", color="blue", linewidth=1, markersize=1)

        # plot optimized trajectory from MPC-DCBF
        # optimized_trajs = simulation._robot._controller_logger._xtrajs
        optimized_inputs = simulation._robot._controller_logger._utrajs
        # optimized_traj = optimized_trajs[index]
        optimized_input = optimized_inputs[index]
        # (optimized_traj_line,) = ax.plot(
        #     optimized_traj[:, 3],
        #     optimized_traj[:, 4],
        #     optimized_traj[:, 5],
        #     "-",
        #     color="gold",
        #     linewidth=1,
        #     markersize=1,
        # )
        
        # draw the robot
        radius_robot = simulation._robot._system._geometry._radius
        margin_dist = 1.0
        position_robot = closedloop_traj[index, 3:6]
        V = closedloop_traj[index, 0]
        gamma = closedloop_traj[index, 1]
        u_V = optimized_input[0]
        u_gamma = optimized_input[1]
        u_psi = optimized_input[2]
        x_robot_0 = radius_robot * np.outer(np.cos(u), np.sin(v)) + position_robot[0]
        y_robot_0 = radius_robot * np.outer(np.sin(u), np.sin(v)) + position_robot[1]
        z_robot_0 = radius_robot * np.outer(np.ones(np.size(u)), np.cos(v)) + position_robot[2]
        ax.plot_surface(x_robot_0, y_robot_0, z_robot_0, color="tab:brown", alpha=1.0)
        ax_2.plot_surface(x_robot_0, y_robot_0, z_robot_0, color="tab:brown", alpha=1.0)

        # draw the static obstacles
        dist_to_nearest_obs = 10000
        for static_obs in simulation._static_obstacles:
            obs_center = static_obs[0]
            obs_radius = static_obs[1]
            dist_to_obs_i = np.linalg.norm(position_robot - obs_center) - radius_robot - obs_radius - margin_dist
            if dist_to_obs_i < dist_to_nearest_obs:
                dist_to_nearest_obs = dist_to_obs_i
            x_obs_0 = obs_radius * np.outer(np.cos(u), np.sin(v)) + obs_center[0]
            y_obs_0 = obs_radius * np.outer(np.sin(u), np.sin(v)) + obs_center[1]
            z_obs_0 = obs_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + obs_center[2]
            ax.plot_surface(x_obs_0, y_obs_0, z_obs_0, color="tab:red", alpha = 0.7)
            ax_2.plot_surface(x_obs_0, y_obs_0, z_obs_0, color="tab:red", alpha = 0.7)
        # # draw the dynamic obstacles
        # for i in range(num_dynamic_obs):
        #     position_obstacle = dynamic_obs_traj[i][index, 0:3]
        #     velocity_obstacle = dynamic_obs_traj[i][index, 3:6]
        #     real_radius = simulation._dynamic_obstacles[i]._system._geometry._real_radius
        #     max_acc_cbf = simulation._dynamic_obstacles[i]._system._geometry._max_acc_cbf
        #     dist_to_obs_i = np.linalg.norm(position_robot - position_obstacle) - radius_robot - real_radius
        #     if dist_to_obs_i < dist_to_nearest_obs:
        #         dist_to_nearest_obs = dist_to_obs_i
        #     # to draw the reachable set for the near future
        #     # for j in range(simulation._robot._controller._param.horizon_dcbf + 1):
        #     #     center_obstacle = position_obstacle + (0.1 * j) * velocity_obstacle
        #     #     if j == 0 or j == 1:
        #     #         radius_obstacle = real_radius
        #     #     else: 
        #     #         radius_obstacle = real_radius
        #     #         for k in range(j - 1):
        #     #             radius_obstacle += (j - k - 1) * max_acc_cbf * 0.1**2
        #     #     x_obs_0 = radius_obstacle * np.outer(np.cos(u), np.sin(v)) + center_obstacle[0]
        #     #     y_obs_0 = radius_obstacle * np.outer(np.sin(u), np.sin(v)) + center_obstacle[1]
        #     #     z_obs_0 = radius_obstacle * np.outer(np.ones(np.size(u)), np.cos(v)) + center_obstacle[2]
        #     #     ax.plot_surface(x_obs_0, y_obs_0, z_obs_0, color="tab:red", alpha=1 - j * (1.0 / (simulation._robot._controller._param.horizon_dcbf + 1)))
        #     # to draw only the obstacles themselves
        #     x_obs_0 = real_radius * np.outer(np.cos(u), np.sin(v)) + position_obstacle[0]
        #     y_obs_0 = real_radius * np.outer(np.sin(u), np.sin(v)) + position_obstacle[1]
        #     z_obs_0 = real_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + position_obstacle[2]
        #     ax.plot_surface(x_obs_0, y_obs_0, z_obs_0, color="tab:red", alpha=1)

        # show the distance to the nearest obstacle
        time.append(index * 0.1)
        distance.append(dist_to_nearest_obs)
        distance_bound.append(0)
        if distance[index - 1] > 0 and distance[index] < 0:
            no_of_collisions[0] = no_of_collisions[0] + 1
        ax_dist.plot(time, distance, "b", linewidth=1, markersize=1, label=r'$\sqrt{-h_{obs}}\hspace{0.2}(m)$')
        # ax_dist.plot(time, distance_bound, "b--", linewidth=0.5, markersize=0.5)
        ax_dist.plot([0, 70], [0, 0], "b--", linewidth=0.5)
        legend_dist = ax_dist.legend(loc='upper right', fontsize='xx-small')
        ax_dist.set_ylim(-50, 200)
        ax_dist.set_xlim(0, 70)
        ax_dist.xaxis.set_tick_params(labelsize=7)
        ax_dist.yaxis.set_tick_params(labelsize=7)
        ax_dist.grid(True)
        # ax_dist.text(0, 0, 'collisions: %i' %no_of_collisions[0], verticalalignment='bottom', horizontalalignment='left', fontsize=10)
        # plot airspeed V
        comp_time.append(opti_solver_time[index])
        V_his.append(V)
        V_lower_bound.append(V_min)
        V_upper_bound.append(V_max)
        ax_vel.plot(time, V_his, "b", linewidth=1, markersize=1, label=r'$V\hspace{0.2}(m/s)$')
        # ax_vel.plot(time, V_lower_bound, "b--", linewidth=0.5, markersize=0.5)
        # ax_vel.plot(time, V_upper_bound, "b--", linewidth=0.5, markersize=0.5)
        ax_vel.plot([0, 70], [V_min, V_min], "b--", [0, 70], [V_max, V_max], "b--", linewidth=0.5)
        ax_vel.set_ylim(-10.0, 15.0)
        ax_vel.set_xlim(0, 70)
        ax_vel.xaxis.set_tick_params(labelsize=7)
        ax_vel.yaxis.set_tick_params(labelsize=7)
        ax_vel.grid(True)
        legend_vel = ax_vel.legend(loc='upper right', fontsize='xx-small')
        # plot flight path angle gamma
        gamma_his.append(gamma)
        gamma_lower_bound.append(gamma_min)
        gamma_upper_bound.append(gamma_max)
        ax_gamma.plot(time, gamma_his, "b", linewidth=1, markersize=1, label=r'$\gamma\hspace{0.2}(rad)$')
        # ax_gamma.plot(time, gamma_lower_bound, "b--", linewidth=0.5, markersize=0.5)
        # ax_gamma.plot(time, gamma_upper_bound, "b--", linewidth=0.5, markersize=0.5)
        ax_gamma.plot([0, 70], [gamma_min, gamma_min], "b--", [0, 70], [gamma_max, gamma_max], "b--", linewidth=0.5)
        ax_gamma.set_ylim(-1.0, 1.0)
        ax_gamma.set_xlim(0, 70)
        ax_gamma.xaxis.set_tick_params(labelsize=7)
        ax_gamma.yaxis.set_tick_params(labelsize=7)
        ax_gamma.set_xlabel(r'time$(s)$', fontsize=7, labelpad=0)
        ax_gamma.grid(True)
        legend_gamma = ax_gamma.legend(loc='upper right', fontsize='xx-small')
        # plot input
        u_gamma_his.append(u_gamma)
        u_gamma_lower_bound.append(u_gamma_min)
        u_gamma_upper_bound.append(u_gamma_max)
        u_psi_his.append(u_psi)
        u_psi_lower_bound.append(u_psi_min)
        u_psi_upper_bound.append(u_psi_max)
        u_V_his.append(u_V)
        u_V_lower_bound.append(u_V_min)
        u_V_upper_bound.append(u_V_max)
        ax_input.plot(time, u_V_his, "r", linewidth=1, markersize=1, label=r'$u_V\hspace{0.2}(m/s^2)$')
        # ax_input.plot(time, u_V_lower_bound, "r--", linewidth=0.5, markersize=0.5)
        # ax_input.plot(time, u_V_upper_bound, "r--", linewidth=0.5, markersize=0.5)
        ax_input.plot([0, 70], [u_V_min, u_V_min], "r--", [0, 70], [u_V_max, u_V_max], "r--", linewidth=0.5)
        ax_input.plot(time, u_gamma_his, "g", linewidth=1, markersize=1, label=r'$u_\gamma\hspace{0.2}(m/s^2)$')
        # ax_input.plot(time, u_gamma_lower_bound, "g--", linewidth=0.5, markersize=0.5)
        # ax_input.plot(time, u_gamma_upper_bound, "g--", linewidth=0.5, markersize=0.5)
        ax_input.plot([0, 70], [u_gamma_min, u_gamma_min], "g--", [0, 70], [u_gamma_max, u_gamma_max], "g--", linewidth=0.5)
        ax_input.plot(time, u_psi_his, "b", linewidth=1, markersize=1, label=r'$u_\psi\hspace{0.2}(m/s^2)$')
        # ax_input.plot(time, u_psi_lower_bound, "b--", linewidth=0.5, markersize=0.5)
        # ax_input.plot(time, u_psi_upper_bound, "b--", linewidth=0.5, markersize=0.5)
        ax_input.plot([0, 70], [u_psi_min, u_psi_min], "b--", [0, 70], [u_psi_max, u_psi_max], "b--", linewidth=0.5)
        ax_input.set_ylim(-5.0, 15.0)
        ax_input.set_xlim(0, 70)
        ax_input.xaxis.set_tick_params(labelsize=7)
        ax_input.yaxis.set_tick_params(labelsize=7)
        ax_input.set_xlabel(r'time$(s)$', fontsize=7, labelpad=0)
        ax_input.grid(True)
        legend_input = ax_input.legend(loc='upper right', fontsize='xx-small')
        # draw the actual trajectory
        ax.plot(closedloop_traj[:index, 3], closedloop_traj[:index, 4], closedloop_traj[:index, 5], "k-", linewidth=1, markersize=1)
        ax_2.plot(closedloop_traj[:index, 3], closedloop_traj[:index, 4], closedloop_traj[:index, 5], "k-", linewidth=1, markersize=1)
        return fig,

    anim = animation.FuncAnimation(fig, update, frames=len(closedloop_traj), interval=1000 * 0.1)
    anim.save(animation_name + ".mp4", dpi=300, writer=animation.writers["ffmpeg"](fps=10))

def render_3D_image(simulation, animation_name="world"):
    # TODO: make this plotting function general applicable to different systems
    # if maze_type == "maze":
    #     fig, ax = plt.subplots(figsize=(8.3, 5.0))
    # elif maze_type == "oblique_maze":
    #     fig, ax = plt.subplots(figsize=(6.7, 5.0))
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1, 1], height_ratios=[3, 1, 1])
    ax = fig.add_subplot(gs[0, 0], projection = '3d')
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 250)
    ax.set_zlim(-50, 100)
    ax.set_box_aspect((5, 5, 3))
    ax.view_init(elev=80, azim=215) # originally, (30, 300)
    ax_2 = fig.add_subplot(gs[0, 1], projection = '3d')
    ax_2.set_xlim(0, 250)
    ax_2.set_ylim(0, 250)
    ax_2.set_zlim(-50, 100)
    ax_2.set_box_aspect((5, 5, 3))
    ax_2.view_init(elev=20, azim=125) # originally, (30, 300)
    ax_2.grid(False)
    tmp_planes = ax_2.axes.zaxis._PLANES
    ax_2.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                     tmp_planes[0], tmp_planes[1], 
                     tmp_planes[4], tmp_planes[5])

    ax_dist = fig.add_subplot(gs[1, 0])
    ax_dist.grid(True)
    ax_vel = fig.add_subplot(gs[1, 1])
    ax_vel.grid(True)
    ax_gamma = fig.add_subplot(gs[2, 0])
    ax_gamma.grid(True)
    ax_input = fig.add_subplot(gs[2, 1])
    ax_input.grid(True)

    opti_solver_time = simulation._robot._controller._solver_times
    closedloop_traj = np.vstack(simulation._robot._system_logger._xs)
    num_dynamic_obs = len(simulation._dynamic_obstacles)
    dynamic_obs_traj = []
    for i in range(num_dynamic_obs):
        dynamic_obs_traj.append(np.vstack(simulation._dynamic_obstacles[i]._system_logger._xs))
                
    time, distance, comp_time = [], [], []
    V_his, gamma_his, u_gamma_his, u_psi_his, u_V_his = [], [], [], [], []
    distance_bound, V_upper_bound, V_lower_bound, gamma_upper_bound, gamma_lower_bound, u_V_upper_bound, u_V_lower_bound, u_gamma_upper_bound, u_gamma_lower_bound, u_psi_upper_bound, u_psi_lower_bound = [], [], [], [], [], [], [], [], [], [], []
    u_V_min, u_V_max, u_gamma_min, u_gamma_max, u_psi_min, u_psi_max = -1.0, 1.0, 6.81, 12.81, -3.0, 3.0
    V_min, V_max, gamma_min, gamma_max = -5.0, 10.0, -0.5, 0.5
    no_of_collisions = np.array([0])
    for index in range(700):
        ax.collections.clear()
        ax.clear()
        ax.set_xlim(0, 250)
        ax.set_ylim(0, 250)
        ax.set_zlim(-50, 100)
        ax.plot([250, 250], [0, 250], [-50, -50], "k-", linewidth=1.0)
        ax.plot([0, 250], [250, 250], [-50, -50], "k-", linewidth=1.0)
        ax.plot([250, 250], [250, 250], [-50, 100], "k-", linewidth=1.0)
        ax.set_box_aspect((5, 5, 3))
        ax.view_init(elev=80, azim=215)
        ax.set_xlabel(r'$P_x(m)$', fontsize=7)
        ax.xaxis.set_tick_params(labelsize=7)
        ax.set_ylabel(r'$P_y(m)$', fontsize=7)
        ax.yaxis.set_tick_params(labelsize=7)
        ax.axes.zaxis.set_ticklabels([])
        ax.grid(False)

        tmp_planes = ax_2.axes.zaxis._PLANES
        ax_2.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                     tmp_planes[0], tmp_planes[1], 
                     tmp_planes[4], tmp_planes[5])

        ax_2.collections.clear()
        ax_2.clear()
        ax_2.set_xlim(0, 250)
        ax_2.set_ylim(0, 250)
        ax_2.set_zlim(-50, 100)
        ax_2.plot([250, 250], [0, 250], [-50, -50], "k-", linewidth=1.0)
        ax_2.plot([0, 250], [0, 0], [-50, -50], "k-", linewidth=1.0)
        ax_2.plot([250, 250], [0, 0], [-50, 100], "k-", linewidth=1.0)
        ax_2.set_box_aspect((5, 5, 3))
        ax_2.view_init(elev=20, azim=125)
        # ax_2.axes.xaxis.set_ticklabels([])
        # ax_2.axes.yaxis.set_ticklabels([])
        ax_2.set_xlabel(r'$P_x(m)$', fontsize=7)
        ax_2.xaxis.set_tick_params(labelsize=7)
        ax_2.set_ylabel(r'$P_y(m)$', fontsize=7)
        ax_2.yaxis.set_tick_params(labelsize=7)
        ax_2.set_zlabel(r'$P_z(m)$', fontsize=7, labelpad=0)
        # ax_2.axes.zaxis.set_label(r'$P_z(m)$')
        ax_2.axes.zaxis.set_tick_params(labelsize=7)    

        tmp_planes = ax_2.axes.zaxis._PLANES
        ax_2.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                     tmp_planes[0], tmp_planes[1], 
                     tmp_planes[4], tmp_planes[5])
        ax_2.grid(False)    

        print(index)

        ax_dist.collections.clear()
        ax_dist.clear()
        ax_vel.collections.clear()
        ax_vel.clear()
        ax_gamma.collections.clear()
        ax_gamma.clear()
        ax_input.collections.clear()
        ax_input.clear()

        # plot global waypoints
        global_paths = simulation._robot._global_planner_logger._paths
        global_path = global_paths[0]
        ax.plot(global_path[:, 0], global_path[:, 1], global_path[:, 2], "bo", linewidth=0.5, markersize=0.5)
        ax_2.plot(global_path[:, 0], global_path[:, 1], global_path[:, 2], "bo", linewidth=0.5, markersize=0.5)

        # plot constant speed trajectory following the global path
        # local_paths = simulation._robot._local_planner_logger._trajs
        # local_path = local_paths[index]
        # (reference_traj_line,) = ax.plot(local_path[:, 3], local_path[:, 4], local_path[:, 5], "-", color="blue", linewidth=1, markersize=1)
        # (reference_traj_line,) = ax_2.plot(local_path[:, 3], local_path[:, 4], local_path[:, 5], "-", color="blue", linewidth=1, markersize=1)

        # plot optimized trajectory from MPC-DCBF
        # optimized_trajs = simulation._robot._controller_logger._xtrajs
        optimized_inputs = simulation._robot._controller_logger._utrajs
        # optimized_traj = optimized_trajs[index]
        optimized_input = optimized_inputs[index]
        # (optimized_traj_line,) = ax.plot(
        #     optimized_traj[:, 3],
        #     optimized_traj[:, 4],
        #     optimized_traj[:, 5],
        #     "-",
        #     color="gold",
        #     linewidth=1,
        #     markersize=1,
        # )
        
        # draw the robot
        radius_robot = simulation._robot._system._geometry._radius
        margin_dist = 1.0
        position_robot = closedloop_traj[index, 3:6]
        V = closedloop_traj[index, 0]
        gamma = closedloop_traj[index, 1]
        u_V = optimized_input[0]
        u_gamma = optimized_input[1]
        u_psi = optimized_input[2]
        x_robot_0 = radius_robot * np.outer(np.cos(u), np.sin(v)) + position_robot[0]
        y_robot_0 = radius_robot * np.outer(np.sin(u), np.sin(v)) + position_robot[1]
        z_robot_0 = radius_robot * np.outer(np.ones(np.size(u)), np.cos(v)) + position_robot[2]
        ax.plot_surface(x_robot_0, y_robot_0, z_robot_0, color="tab:brown", alpha=1.0)
        ax_2.plot_surface(x_robot_0, y_robot_0, z_robot_0, color="tab:brown", alpha=1.0)

        # draw the static obstacles
        dist_to_nearest_obs = 10000
        for static_obs in simulation._static_obstacles:
            obs_center = static_obs[0]
            obs_radius = static_obs[1]
            dist_to_obs_i = np.linalg.norm(position_robot - obs_center) - radius_robot - obs_radius - margin_dist
            if dist_to_obs_i < dist_to_nearest_obs:
                dist_to_nearest_obs = dist_to_obs_i
            x_obs_0 = obs_radius * np.outer(np.cos(u), np.sin(v)) + obs_center[0]
            y_obs_0 = obs_radius * np.outer(np.sin(u), np.sin(v)) + obs_center[1]
            z_obs_0 = obs_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + obs_center[2]
            ax.plot_surface(x_obs_0, y_obs_0, z_obs_0, color="tab:red", alpha = 0.7)
            ax_2.plot_surface(x_obs_0, y_obs_0, z_obs_0, color="tab:red", alpha = 0.7)
        # # draw the dynamic obstacles
        # for i in range(num_dynamic_obs):
        #     position_obstacle = dynamic_obs_traj[i][index, 0:3]
        #     velocity_obstacle = dynamic_obs_traj[i][index, 3:6]
        #     real_radius = simulation._dynamic_obstacles[i]._system._geometry._real_radius
        #     max_acc_cbf = simulation._dynamic_obstacles[i]._system._geometry._max_acc_cbf
        #     dist_to_obs_i = np.linalg.norm(position_robot - position_obstacle) - radius_robot - real_radius
        #     if dist_to_obs_i < dist_to_nearest_obs:
        #         dist_to_nearest_obs = dist_to_obs_i
        #     # to draw the reachable set for the near future
        #     # for j in range(simulation._robot._controller._param.horizon_dcbf + 1):
        #     #     center_obstacle = position_obstacle + (0.1 * j) * velocity_obstacle
        #     #     if j == 0 or j == 1:
        #     #         radius_obstacle = real_radius
        #     #     else: 
        #     #         radius_obstacle = real_radius
        #     #         for k in range(j - 1):
        #     #             radius_obstacle += (j - k - 1) * max_acc_cbf * 0.1**2
        #     #     x_obs_0 = radius_obstacle * np.outer(np.cos(u), np.sin(v)) + center_obstacle[0]
        #     #     y_obs_0 = radius_obstacle * np.outer(np.sin(u), np.sin(v)) + center_obstacle[1]
        #     #     z_obs_0 = radius_obstacle * np.outer(np.ones(np.size(u)), np.cos(v)) + center_obstacle[2]
        #     #     ax.plot_surface(x_obs_0, y_obs_0, z_obs_0, color="tab:red", alpha=1 - j * (1.0 / (simulation._robot._controller._param.horizon_dcbf + 1)))
        #     # to draw only the obstacles themselves
        #     x_obs_0 = real_radius * np.outer(np.cos(u), np.sin(v)) + position_obstacle[0]
        #     y_obs_0 = real_radius * np.outer(np.sin(u), np.sin(v)) + position_obstacle[1]
        #     z_obs_0 = real_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + position_obstacle[2]
        #     ax.plot_surface(x_obs_0, y_obs_0, z_obs_0, color="tab:red", alpha=1)

        # show the distance to the nearest obstacle
        time.append(index * 0.1)
        distance.append(dist_to_nearest_obs)
        distance_bound.append(0)
        if distance[index - 1] > 0 and distance[index] < 0:
            no_of_collisions[0] = no_of_collisions[0] + 1
        ax_dist.plot(time, distance, "b", linewidth=1, markersize=1, label=r'$\sqrt{-h_{obs}}\hspace{0.2}(m)$')
        # ax_dist.plot(time, distance_bound, "b--", linewidth=0.5, markersize=0.5)
        ax_dist.plot([0, 70], [0, 0], "b--", linewidth=0.5)
        legend_dist = ax_dist.legend(loc='upper right', fontsize='xx-small')
        ax_dist.set_ylim(-50, 200)
        ax_dist.set_xlim(0, 70)
        ax_dist.xaxis.set_tick_params(labelsize=7)
        ax_dist.yaxis.set_tick_params(labelsize=7)
        ax_dist.grid(True)
        # ax_dist.text(0, 0, 'collisions: %i' %no_of_collisions[0], verticalalignment='bottom', horizontalalignment='left', fontsize=10)
        # plot airspeed V
        comp_time.append(opti_solver_time[index])
        V_his.append(V)
        V_lower_bound.append(V_min)
        V_upper_bound.append(V_max)
        ax_vel.plot(time, V_his, "b", linewidth=1, markersize=1, label=r'$V\hspace{0.2}(m/s)$')
        # ax_vel.plot(time, V_lower_bound, "b--", linewidth=0.5, markersize=0.5)
        # ax_vel.plot(time, V_upper_bound, "b--", linewidth=0.5, markersize=0.5)
        ax_vel.plot([0, 70], [V_min, V_min], "b--", [0, 70], [V_max, V_max], "b--", linewidth=0.5)
        ax_vel.set_ylim(-10.0, 15.0)
        ax_vel.set_xlim(0, 70)
        ax_vel.xaxis.set_tick_params(labelsize=7)
        ax_vel.yaxis.set_tick_params(labelsize=7)
        ax_vel.grid(True)
        legend_vel = ax_vel.legend(loc='upper right', fontsize='xx-small')
        # plot flight path angle gamma
        gamma_his.append(gamma)
        gamma_lower_bound.append(gamma_min)
        gamma_upper_bound.append(gamma_max)
        ax_gamma.plot(time, gamma_his, "b", linewidth=1, markersize=1, label=r'$\gamma\hspace{0.2}(rad)$')
        # ax_gamma.plot(time, gamma_lower_bound, "b--", linewidth=0.5, markersize=0.5)
        # ax_gamma.plot(time, gamma_upper_bound, "b--", linewidth=0.5, markersize=0.5)
        ax_gamma.plot([0, 70], [gamma_min, gamma_min], "b--", [0, 70], [gamma_max, gamma_max], "b--", linewidth=0.5)
        ax_gamma.set_ylim(-1.0, 1.0)
        ax_gamma.set_xlim(0, 70)
        ax_gamma.xaxis.set_tick_params(labelsize=7)
        ax_gamma.yaxis.set_tick_params(labelsize=7)
        ax_gamma.set_xlabel(r'time$(s)$', fontsize=7, labelpad=0)
        ax_gamma.grid(True)
        legend_gamma = ax_gamma.legend(loc='upper right', fontsize='xx-small')
        # plot input
        u_gamma_his.append(u_gamma)
        u_gamma_lower_bound.append(u_gamma_min)
        u_gamma_upper_bound.append(u_gamma_max)
        u_psi_his.append(u_psi)
        u_psi_lower_bound.append(u_psi_min)
        u_psi_upper_bound.append(u_psi_max)
        u_V_his.append(u_V)
        u_V_lower_bound.append(u_V_min)
        u_V_upper_bound.append(u_V_max)
        ax_input.plot(time, u_V_his, "r", linewidth=1, markersize=1, label=r'$u_V\hspace{0.2}(m/s^2)$')
        # ax_input.plot(time, u_V_lower_bound, "r--", linewidth=0.5, markersize=0.5)
        # ax_input.plot(time, u_V_upper_bound, "r--", linewidth=0.5, markersize=0.5)
        ax_input.plot([0, 70], [u_V_min, u_V_min], "r--", [0, 70], [u_V_max, u_V_max], "r--", linewidth=0.5)
        ax_input.plot(time, u_gamma_his, "g", linewidth=1, markersize=1, label=r'$u_\gamma\hspace{0.2}(m/s^2)$')
        # ax_input.plot(time, u_gamma_lower_bound, "g--", linewidth=0.5, markersize=0.5)
        # ax_input.plot(time, u_gamma_upper_bound, "g--", linewidth=0.5, markersize=0.5)
        ax_input.plot([0, 70], [u_gamma_min, u_gamma_min], "g--", [0, 70], [u_gamma_max, u_gamma_max], "g--", linewidth=0.5)
        ax_input.plot(time, u_psi_his, "b", linewidth=1, markersize=1, label=r'$u_\psi\hspace{0.2}(m/s^2)$')
        # ax_input.plot(time, u_psi_lower_bound, "b--", linewidth=0.5, markersize=0.5)
        # ax_input.plot(time, u_psi_upper_bound, "b--", linewidth=0.5, markersize=0.5)
        ax_input.plot([0, 70], [u_psi_min, u_psi_min], "b--", [0, 70], [u_psi_max, u_psi_max], "b--", linewidth=0.5)
        ax_input.set_ylim(-5.0, 15.0)
        ax_input.set_xlim(0, 70)
        ax_input.xaxis.set_tick_params(labelsize=7)
        ax_input.yaxis.set_tick_params(labelsize=7)
        ax_input.set_xlabel(r'time$(s)$', fontsize=7, labelpad=0)
        ax_input.grid(True)
        legend_input = ax_input.legend(loc='upper right', fontsize='xx-small')
        # draw the actual trajectory
        ax.plot(closedloop_traj[:index, 3], closedloop_traj[:index, 4], closedloop_traj[:index, 5], "k-", linewidth=1, markersize=1)
        ax_2.plot(closedloop_traj[:index, 3], closedloop_traj[:index, 4], closedloop_traj[:index, 5], "k-", linewidth=1, markersize=1)
    
    fig.savefig(animation_name, dpi=300, format="eps")


def count_collisions_without_visualization(simulation):

    closedloop_traj = np.vstack(simulation._robot._system_logger._xs)
    num_dynamic_obs = len(simulation._dynamic_obstacles)
    dynamic_obs_traj = []
    for i in range(num_dynamic_obs):
        dynamic_obs_traj.append(np.vstack(simulation._dynamic_obstacles[i]._system_logger._xs))
                
    time, distance = [], []
    no_of_collisions = np.array([0])
    for index in range(len(closedloop_traj)):
        radius_robot = simulation._robot._system._geometry._radius
        position_robot = closedloop_traj[index, 3:6]
        dist_to_nearest_obs = 10000
        for static_obs in simulation._static_obstacles:
            obs_center = static_obs[0]
            obs_radius = static_obs[1]
            dist_to_obs_i = np.linalg.norm(position_robot - obs_center) - radius_robot - obs_radius
            if dist_to_obs_i < dist_to_nearest_obs:
                dist_to_nearest_obs = dist_to_obs_i
        for i in range(num_dynamic_obs):
            position_obstacle = dynamic_obs_traj[i][index, 0:3]
            real_radius = simulation._dynamic_obstacles[i]._system._geometry._real_radius
            dist_to_obs_i = np.linalg.norm(position_robot - position_obstacle) - radius_robot - real_radius
            if dist_to_obs_i < dist_to_nearest_obs:
                dist_to_nearest_obs = dist_to_obs_i
        time.append(index * 0.1)
        distance.append(dist_to_nearest_obs)
        if distance[index - 1] > 0 and distance[index] < 0:
            no_of_collisions[0] = no_of_collisions[0] + 1
    return no_of_collisions[0]

def kinematic_car_all_shapes_simulation_test(DCBF_horizon, constraint_type):
    start_pos, goal_pos, grid, static_obstacles, center, radius = create_env()

    car_radius = 1.5
    V_d = 10.0
    sim_time = 70.0 # originally 70
    
    u_V_nominal = 0.0
    u_gamma_nominal = 9.81
    u_psi_nominal = V_d**2 / radius
    u_nominal = np.array([u_V_nominal, u_gamma_nominal, u_psi_nominal])
    geometry_regions = KinematicCarPointGeometry(car_radius)
    robot = Robot(
        KinematicCarSystem(
            state=KinematicCarStates(x=start_pos, u=u_nominal),
            geometry=geometry_regions,
            dynamics=KinematicCarDynamics(),
        )
    )
    dynamic_obstacles = [] 

    # TODO: design a global planner for 3D environment
    robot.set_global_planner(AstarLoSPathGenerator(start_pos[3:6], goal_pos))
    robot.set_local_planner(ConstantSpeedTrajectoryGenerator(DCBF_horizon, center))
    # robot.set_local_planner(ConstantSpeedCircularTrajectoryGenerator(center, radius))
    opt_param = NmpcDcbfOptimizerParam(DCBF_horizon, constraint_type, static_obstacles)
    robot.set_controller(NmpcDcbfController(dynamics=KinematicCarDynamics(), opt_param=opt_param, constraint_type=constraint_type))
    sim = SingleAgentSimulation(robot, static_obstacles, dynamic_obstacles, goal_pos)
    sim.run_navigation(sim_time)
    # print(robot._controller._optimizer.solver_times)
    print("median: ", st.median(robot._controller._solver_times))
    print("std: ", st.stdev(robot._controller._solver_times))
    print("min: ", min(robot._controller._solver_times))
    print("max: ", max(robot._controller._solver_times))
    print("Simulation finished.")
    num_of_collision = count_collisions_without_visualization(sim)
    time = datetime.datetime.now()
    name = "alpha_0.1_" + constraint_type + "_" + str(DCBF_horizon) + "MPC_horizon" + "_" + str(num_of_collision) + "collision_" + str(time)
    print("number of collisions: ", num_of_collision)
    # render a 3D video
    render_3D_image(sim, animation_name=name)
    return robot._controller._solver_times, num_of_collision


def create_env():
    s = 1  # scale of environment
    start = np.array([0.0, 0.0, - math.pi/4, 50.0 * s, 50.0 * s, 25.0 * s])
    goal = np.array([150.0 * s, 150.0 * s, 25.0 * s])
    center = np.array([100.0 * s, 100.0 *s, 25.0 * s])
    radius = np.linalg.norm(start[3:6] - center)
    bounds = ((0.0 * s, 0.0 * s, 0.0 * s), (200.0 * s, 200.0 * s, 50.0 * s))
    cell_size = 0.25 * s
    grid = (bounds, cell_size)
    static_obstacles = [] # list of static obstacles
    static_obs_1_center = np.array([160.0, 140.0, 20.0])
    static_obs_1_radius = 35.0 # originally 35.0
    static_obs_1 = (static_obs_1_center, static_obs_1_radius)
    static_obstacles.append(static_obs_1)
    return start, goal, grid, static_obstacles, center, radius


if __name__ == "__main__":
    # constraint_types = ["nominal_evading_maneuver", "CBF_form", "MPC_form", "nominal_evading_maneuver_no_RD1"]
    constraint_type = "nominal_evading_maneuver_no_RD1"
    if constraint_type == "nominal_evading_maneuver": # always feasible, sim_time 70.0s
        MPC_horizon = 5
    if constraint_type == "nominal_evading_maneuver_no_RD1": # max. feasible sim_time 9.8s
        MPC_horizon = 5
    if constraint_type == "CBF_form": # max. feasible sim_time 17.7s
        MPC_horizon = 20
    if constraint_type == "MPC_form": # max. feasible sim_time 22.8s
        MPC_horizon = 20
    solver_time, coll = kinematic_car_all_shapes_simulation_test(MPC_horizon, constraint_type)
    print("horizon length: ", MPC_horizon, ", constraint type: ", constraint_type, ", no. of collision: ", coll)
    print("mean: ", np.mean(solver_time), "median: ", np.median(solver_time), "std. dev.: ", np.std(solver_time), "min: ", np.min(solver_time), "max: ", np.max(solver_time))