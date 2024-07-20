from __future__ import annotations  # Python 3.10 type hints

import numpy as np
import yaml
import pybullet as p
from scipy import interpolate
import math
from itertools import permutations
from lsy_drone_racing.command import Command
from lsy_drone_racing.controller import BaseController
from lsy_drone_racing.utils import draw_trajectory
from scipy.interpolate import CubicSpline
import os

class Controller(BaseController):
    """PD controller class."""

    def distance(self, p1, p2):
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)
    
    def find_shortest_path(self, waypoints):
        shortest_path = None
        shortest_distance = float('inf')
        for order in permutations(range(len(waypoints))):
            dist = self.total_path_distance(waypoints, order)
            if dist < shortest_distance:
                shortest_distance = dist
                shortest_path = order
        return shortest_path, shortest_distance

    def total_path_distance(self, waypoints, order):
        total_dist = 0
        for i in range(len(waypoints) - 1):
            start_point = waypoints[order[i]]
            end_point = waypoints[order[i + 1]]
            total_dist += self.distance(start_point, end_point)
        return total_dist

    def __init__(self, initial_obs: np.ndarray, initial_info: dict, buffer_size: int = 100, verbose: bool = False):
        super().__init__(initial_obs, initial_info, buffer_size, verbose)
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]

        #########################
        # REPLACE THIS (START) ##
        #########################

        # Define the path to the YAML file
        print("Current working directory:", os.getcwd())

        yaml_path = '/home/julian/repos/lsy_drone_racing/safe-control-gym/competition/level2.yaml'

        # Check if the file exists
        if not os.path.exists(yaml_path):
            print("File does not exist.")
        else:
            # Parse the YAML file
            with open(yaml_path, 'r') as file:
                level_data = yaml.safe_load(file)
            print("Successfully loaded YAML file.")
        
        # Access the gates data
        gates = level_data['quadrotor_config']['gates']

        # Get the z values from initial_info
        z_low = initial_info["gate_dimensions"]["low"]["height"]
        z_high = initial_info["gate_dimensions"]["tall"]["height"]

        waypoints = []
        for gate in gates:
            x, y, _, _, _, _, gate_type = gate
            z = z_high if gate_type == 1 else z_low
            waypoints.append([x, y, z])

        # Find the shortest path through the waypoints
        order, total_dist = self.find_shortest_path(waypoints)
        waypoints = [waypoints[i] for i in order]
        waypoints.insert(0, [initial_obs[0], initial_obs[1], 0.3])
        waypoints.append([initial_info["x_reference"][0], initial_info["x_reference"][2], initial_info["x_reference"][4]])

        waypoints = np.array(waypoints)
        self.waypoints = waypoints

        # Create smooth trajectory using cubic splines
        num_waypoints = len(waypoints)
        t = np.linspace(0, 1, num_waypoints)
        cs_x = CubicSpline(t, waypoints[:, 0])
        cs_y = CubicSpline(t, waypoints[:, 1])
        cs_z = CubicSpline(t, waypoints[:, 2])
        dense_t = np.linspace(0, 1, num_waypoints * 10)
        x_reference = cs_x(dense_t)
        y_reference = cs_y(dense_t)
        z_reference = cs_z(dense_t)
        reference_trajectory = np.vstack((x_reference, y_reference, z_reference)).T
        self.reference_trajectory = reference_trajectory

        for i in range(len(reference_trajectory) - 1):
            point1 = reference_trajectory[i]
            point2 = reference_trajectory[i + 1]
            p.addUserDebugLine(point1, point2, [1, 0, 0], 3)

        # PD controller gains
        self.kp = np.array([1.0, 1.0, 1.0])  # Proportional gains
        self.kd = np.array([0.1, 0.1, 0.1])  # Derivative gains

        self._take_off = False
        self._setpoint_land = False
        self._land = False

        #########################
        # REPLACE THIS (END) ####
        #########################


    def compute_control(self, ep_time: float, obs: np.ndarray, reward: float | None = None, done: bool | None = None, info: dict | None = None) -> tuple[Command, list]:
        iteration = int(ep_time * self.CTRL_FREQ)

        #########################
        # REPLACE THIS (START) ##
        #########################

        if not self._take_off:
            command_type = Command.TAKEOFF
            args = [0.3, 2]  # Height, duration
            self._take_off = True
        else:
            if ep_time - 20 > 0:
                # Compute desired state from the reference trajectory
                ref_index = min(iteration, len(self.reference_trajectory) - 1)
                target_pos = self.reference_trajectory[ref_index]
                target_vel = np.zeros(3)
                target_acc = np.zeros(3)
                target_yaw = 0.0
                target_rpy_rates = np.zeros(3)

                # PD control
                pos_error = target_pos - obs[:3]
                vel_error = target_vel - obs[3:6]
                control_output = self.kp * pos_error + self.kd * vel_error

                command_type = Command.FULLSTATE
                args = [target_pos, control_output, target_acc, target_yaw, target_rpy_rates, ep_time]
            else:
                command_type = Command.NONE
                args = []

        #########################
        # REPLACE THIS (END) ####
        #########################

        return command_type, args

    def step_learn(self, action: list, obs: np.ndarray, reward: float | None = None, done: bool | None = None, info: dict | None = None):
        #########################
        # REPLACE THIS (START) ##
        #########################

        self.action_buffer.append(action)
        self.obs_buffer.append(obs)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.info_buffer.append(info)

        #########################
        # REPLACE THIS (END) ####
        #########################

    def episode_learn(self):
        #########################
        # REPLACE THIS (START) ##
        #########################

        _ = self.action_buffer
        _ = self.obs_buffer
        _ = self.reward_buffer
        _ = self.done_buffer
        _ = self.info_buffer

        #########################
        # REPLACE THIS (END) ####
        #########################