"""Write your control strategy.

Then run:

    $ python scripts/sim --config config/getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS:` and `REPLACE THIS (START)` in this file.

    Change the code between the 5 blocks starting with
        #########################
        # REPLACE THIS (START) ##
        #########################
    and ending with
        #########################
        # REPLACE THIS (END) ####
        #########################
    with your own code.

    They are in methods:
        1) __init__
        2) compute_control
        3) step_learn (optional)
        4) episode_learn (optional)

"""

from __future__ import annotations  # Python 3.10 type hints

import numpy as np
from scipy import interpolate
import pybullet as p
import copy

from lsy_drone_racing.command import Command
from lsy_drone_racing.controller import BaseController
from lsy_drone_racing.utils import draw_trajectory


class Controller(BaseController):
    """Template controller class."""

    def rotation_matrix(self, yaw):
        """Used for handling Gate rotation"""
        return np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

    def get_prior_post_points(self, waypoint, yaw, offset):
        """This function produces a point before and after the gate"""
        R = self.rotation_matrix(yaw)
        forward_offset = R @ np.array([0, offset, 0])
        backward_offset = R @ np.array([0, -offset, 0])

        waypoint_prior = waypoint + backward_offset
        waypoint_post = waypoint + forward_offset

        return waypoint_prior, waypoint_post

    def is_obstacle_on_path(self, start, end):
        """The path between start and end is checked for obstacles"""
        for obstacle in self.NOMINAL_OBSTACLES:
            if self.distance_point_to_line([obstacle[0], obstacle[1], (start[-1] + (end[-1] - start[-1])/2)], start, end) < 0.15:  # Checking for obstacle within 1 unit distance
                return True, obstacle[:3]
        return False, None

    def distance_point_to_line(self, point, line_start, line_end):
        """Used to find the proximity of the obstacle to the currently checked path"""
        line_vec = np.array(line_end) - np.array(line_start)
        point_vec = np.array(point) - np.array(line_start)
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        point_vec_scaled = point_vec / line_len
        t = np.dot(line_unitvec, point_vec_scaled)
        nearest = line_start + t * line_vec
        return np.linalg.norm(nearest - point)

    def avoid_obstacle_waypoints(self, start, end, obstacle):
        """Waypoints to avoid obstacle are produced in this function."""
        obstacle = np.array(obstacle)
        start = np.array(start)
        end = np.array(end)

        medium_height = start[-1] + (end[-1] - start[-1]) / 2
        direction_vec = end - start
        direction_vec = direction_vec / np.linalg.norm(direction_vec)

        # Waypoint 0.35 units before the obstacle
        before_obstacle = [(obstacle - direction_vec * 0.35)[0], (obstacle - direction_vec * 0.35)[1], medium_height]

        # Calculate offsets
        right_offset = np.array([-direction_vec[1], direction_vec[0], 0]) * 0.35
        left_offset = np.array([direction_vec[1], -direction_vec[0], 0]) * 0.35

        # Waypoints for right of obstacle
        right_of_obstacle = [(obstacle + right_offset)[0], (obstacle + right_offset)[1], medium_height]

        # Waypoints for left of obstacle
        left_of_obstacle = [(obstacle + left_offset)[0], (obstacle + left_offset)[1], medium_height]

        # Waypoint 0.35 units after the obstacle
        after_obstacle = [(obstacle + direction_vec * 0.35)[0], (obstacle + direction_vec * 0.35)[1], medium_height]

        # Determine if the obstacle is more to the right or left of the path
        if np.cross(direction_vec[:2], (obstacle[:2] - start[:2])) > 0:
            # Obstacle is to the left, so go right
            return before_obstacle, right_of_obstacle, after_obstacle
        else:
            # Obstacle is to the right, so go left
            return before_obstacle, left_of_obstacle, after_obstacle

    def make_smoother(self):
        """Used to create more waypoints to make the trajectory smoother for the Controller to follow"""
        new_waypoints = []
        for i in range(len(self.waypoints) - 1):
            # Get the current waypoint and the next waypoint
            current_wp = self.waypoints[i]
            next_wp = self.waypoints[i + 1]

            # Calculate the midpoint
            midpoint = (current_wp + next_wp) / 2

            # Add the current waypoint and the midpoint to the new waypoints list
            new_waypoints.append(current_wp)
            new_waypoints.append(midpoint)

        # Add the last waypoint
        new_waypoints.append(self.waypoints[-1])

        # Convert the new waypoints list to a numpy array
        self.waypoints = np.array(new_waypoints)

    def __init__(
        self,
        initial_obs: np.ndarray,
        initial_info: dict,
        buffer_size: int = 100,
        verbose: bool = False,
    ):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori
            infromation contained in dictionary `initial_info`. Use this method to initialize
            constants, counters, pre-plan trajectories, etc.

        Args:
            initial_obs: The initial observation of the quadrotor's state
                [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            initial_info: The a priori information as a dictionary with keys 'symbolic_model',
                'nominal_physical_parameters', 'nominal_gates_pos_and_type', etc.
            buffer_size: Size of the data buffers used in method `learn()`.
            verbose: Turn on and off additional printouts and plots.
        """
        super().__init__(initial_obs, initial_info, buffer_size, verbose)
        # Save environment and control parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        # Store a priori scenario information.
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]
        self.initial_info = initial_info

        # Reset counters and buffers.
        self.reset()
        self.episode_reset()

        #########################
        # REPLACE THIS (START) ##
        #########################

        # Landing Parameter
        self.parcours_completion = False

        # Preparation of Gates for trajectory planning
        gates = copy.deepcopy(self.NOMINAL_GATES)
        self.z_low = initial_info["gate_dimensions"]["low"]["height"]
        self.z_high = initial_info["gate_dimensions"]["tall"]["height"]

        # z - Coordinate filled with gate height and bool parameter removed
        for i in range((len(gates))):
            z = self.z_high if gates[i][-1] == 0 else self.z_low
            gates[i].pop(2)
            gates[i].pop()
            gates[i].insert(2, z)

        self.gates = gates

        # Initializations for Controller
        self.past_pos_error = 0
        self.target_waypoint_index = 1
        self.threshold = 0.1

        self._take_off = False
        self._setpoint_land = False
        self._land = False
        #########################
        # REPLACE THIS (END) ####
        #########################

    def Update_Trajectory(self, startpos, gates):
        """Trajectory Planning function, plans trajectory from startpos through remaining gates"""
        self.replanning = True
        initial_info = self.initial_info
        
        better_waypoints = [startpos]

        # Waypoints before and after gate are calculated
        for i in range(0,(len(gates))):
            offset = 0.25
            waypoint_prior, waypoint_post = self.get_prior_post_points(gates[i][:3], gates[i][5], offset)
            for obstacle in self.NOMINAL_OBSTACLES:
                if np.linalg.norm(waypoint_prior[:2] - obstacle[:2]) < 0.3:
                    offset /= 4
                    waypoint_prior, irrelevant = self.get_prior_post_points(gates[i][:3], gates[i][5], offset)
                elif np.linalg.norm(waypoint_post[:2] - obstacle[:2]) < 0.3:
                    offset /= 4
                    irrelevant, waypoint_post = self.get_prior_post_points(gates[i][:3], gates[i][5], offset)
            better_waypoints.append(waypoint_prior)
            better_waypoints.append(waypoint_post)

        # Add end point to trajectory
        better_waypoints.append(np.array([initial_info["x_reference"][0], initial_info["x_reference"][2], initial_info["x_reference"][4]]))
        self.better_waypoints = []

        # Handle Obstacle Avoidance
        for i in range(len(better_waypoints) - 1):
            blocked, obs = self.is_obstacle_on_path(better_waypoints[i], better_waypoints[i+1])
            self.better_waypoints.append(better_waypoints[i])
            if blocked and (i % 2 == 0):
                extrapoint1, extrapoint2, extrapoint3 = self.avoid_obstacle_waypoints(better_waypoints[i], better_waypoints[i+1], obs)
                self.better_waypoints.append(extrapoint1)
                #p.addUserDebugText('.', extrapoint1, textColorRGB=[1, 1, 1], textSize=5)  # Blue dots for waypoints
                self.better_waypoints.append(extrapoint2)
                #p.addUserDebugText('.', extrapoint2, textColorRGB=[0, 1, 0], textSize=5)  # Blue dots for waypoints
                self.better_waypoints.append(extrapoint3)
                #p.addUserDebugText('.', extrapoint3, textColorRGB=[1, 1, 1], textSize=5)  # Blue dots for waypoints

        self.waypoints = np.array(self.better_waypoints)

        # Smoothen Trajectory
        self.make_smoother()
        self.make_smoother()
        self.make_smoother()

        # Ensure that waypoints close to the gate do not generate in the gate borders
        for i in range(len(self.waypoints)):
            for g in range(len(gates)):
                if np.linalg.norm(self.waypoints[i][:2] - np.array(gates[g][:2])) < 0.25:
                    waypoint_prior, waypoint_post = self.get_prior_post_points(gates[g][:3], gates[g][5], 0.25)
                    self.waypoints[i] = self.project_point_onto_vector(self.waypoints[i], waypoint_prior, waypoint_post)
                    # p.addUserDebugText('.', self.waypoints[i], textColorRGB=[0, 1, 0], textSize=5)

        self.replanning = False

    def project_point_onto_vector(self, point, vector_start, vector_end):
        """
        Projects a point onto a vector defined by two points (vector_start, vector_end).
        
        Parameters:
        point (list or np.array): The coordinates of the point, e.g., [px, py, pz].
        vector_start (list or np.array): The start point of the vector, e.g., [vx_start, vy_start, vz_start].
        vector_end (list or np.array): The end point of the vector, e.g., [vx_end, vy_end, vz_end].
        
        Returns:
        np.array: The coordinates of the projected point on the vector.
        """
        point = np.array(point)
        vector_start = np.array(vector_start)
        vector_end = np.array(vector_end)
        
        # Vector from start to end
        vector = vector_end - vector_start
        
        # Vector from start to point
        vector_to_point = point - vector_start
        
        # Normalize the vector
        unit_vector = vector / np.linalg.norm(vector)
        
        # Compute the projection of the point vector onto the unit vector
        projection_length = np.dot(vector_to_point, unit_vector)
        projection = vector_start + projection_length * unit_vector
        
        return projection     

    def find_closest_point(self, points, reference_point):
        """
        Finds the index of the closest point in an array to a given reference point.
        
        Parameters:
        points (np.array): An array of points with shape (n_points, n_dimensions).
        reference_point (list or np.array): The reference point with shape (n_dimensions,).
        
        Returns:
        int: The index of the closest point.
        """
        points = np.array(points)
        reference_point = np.array(reference_point)
        
        # Calculate the Euclidean distance from each point to the reference point
        distances = np.linalg.norm(points - reference_point, axis=1)
        
        # Find the index of the minimum distance
        closest_index = np.argmin(distances)
        
        return closest_index 

    def compute_control(
        self,
        ep_time: float,
        obs: np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
    ) -> tuple[Command, list]:
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this method to return the target position, velocity, acceleration,
            attitude, and attitude rates to be sent from Crazyswarm to the Crazyflie using, e.g., a
            `cmdFullState` call.

        Args:
            ep_time: Episode's elapsed time, in seconds.
            obs: The quadrotor's Vicon data [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].
            reward: The reward signal.
            done: Wether the episode has terminated.
            info: Current step information as a dictionary with keys 'constraint_violation',
                'current_target_gate_pos', etc.

        Returns:
            The command type and arguments to be sent to the quadrotor. See `Command`.
        """
        iteration = int(ep_time * self.CTRL_FREQ)

        #########################
        # REPLACE THIS (START) ##
        #########################

        # Handcrafted solution for getting_stated scenario.

        if not self._take_off:
            command_type = Command.TAKEOFF
            args = [0.1, 2]  # Height, duration
            self._take_off = True  # Only send takeoff command once
            # Trajectory planned using the first obeservation of the drone
            self.Update_Trajectory([obs[0], obs[2], obs[4]], self.gates)
            self.High_Level_Target = self.gates[0]
            self.current_target_index = 0
            self.replan = False
            self.replanning = False
        else:
            if ep_time - 2 > 0 and not self.parcours_completion:
                target_acc = np.zeros(3)
                target_yaw = 0.0
                target_rpy_rates = np.zeros(3)

                # Adaptive Controller to achieve higher speeds on longer straights and lower speeds in near obstacles and gates
                for obstacle in self.NOMINAL_OBSTACLES:
                    if np.linalg.norm(np.array([obs[0], obs[2]]) - obstacle[:2]) < 0.7 or np.linalg.norm(np.array([obs[0], obs[2], obs[4]]) - self.High_Level_Target[:3]) < 0.7:
                        Kp = 1.45   # Proportional Gain
                        Kd = 0.3   # Derivative Gain
                        Vel_Kp = 0.3    # Velocity Proportional Gain
                        self.threshold = 0.1 # Distance to current waypoint which needs to be reached for the next waypoint to become the target
                    else:
                        Kp = 1.8  # Proportional Gain
                        Kd = 0.3   # Derivative Gain
                        Vel_Kp = 0.3    # Velocity Proportional Gain
                        self.threshold = 0.1 # Distance to current waypoint which needs to be reached for the next waypoint to become the target

                observed_target = info["current_target_gate_pos"]
                current_target_index = info["current_target_gate_id"]
                
                # Wait for the waypoint reached after a gate to replan the trajectory
                if current_target_index != self.current_target_index:
                    self.save_index = self.target_waypoint_index
                    if self.save_index < self.target_waypoint_index:
                        self.current_target_index = current_target_index
                        self.replan = True

                # Initialize replanning if information changes
                if (self.High_Level_Target[:3] != observed_target[:3] and observed_target[2] != 0) or self.replan:
                    # Define gates new and pass them into Update trajectory
                    current_target_index = info["current_target_gate_id"]

                    # gates preparation for updating the Trajectory
                    gates = copy.deepcopy(self.NOMINAL_GATES)
                    self.z_low = self.initial_info["gate_dimensions"]["low"]["height"]
                    self.z_high = self.initial_info["gate_dimensions"]["tall"]["height"]

                    for i in range((len(gates))):
                        z = self.z_high if gates[i][-1] == 0 else self.z_low
                        gates[i].pop(2)
                        gates[i].pop()
                        gates[i].insert(2, z)

                    self.gates = gates
                    self.gates.remove(self.gates[current_target_index])
                    self.gates.insert(current_target_index, observed_target)
                    self.gates = self.gates[current_target_index:4]
                    self.Update_Trajectory([obs[0], obs[2], obs[4]], self.gates)
                    self.target_waypoint_index = 0
                    self.High_Level_Target = observed_target[:3]
                    self.replan = False

                # Define high level targets
                distance = np.linalg.norm(([obs[0], obs[2], obs[4]]) - (self.waypoints[self.target_waypoint_index]))
                if distance < self.threshold:
                    self.target_waypoint_index += 1

                # Define when parcours is finished
                if self.target_waypoint_index == len(self.waypoints) - 1:
                    self.parcours_completion = True

                # Mark currently aimed at waypoint
                p.addUserDebugText('.', self.waypoints[self.target_waypoint_index], textColorRGB=[1, 0, 1], textSize=5)
                pos_error = self.waypoints[self.target_waypoint_index] - [obs[0], obs[2], obs[4]]
                if self.target_waypoint_index < (len(self.waypoints) - 1):
                    derivative_pos_error = self.waypoints[self.target_waypoint_index + 1] - [obs[0], obs[2], obs[4]]
                else:
                    derivative_pos_error = pos_error
                pos_input = [obs[0], obs[2], obs[4]] + Kp * pos_error + Kd * derivative_pos_error
                self.past_pos_error = pos_error

                # Compute Target Velocity
                target_vel = pos_error
                target_vel = Vel_Kp * pos_error

                # Make drone wait for new trajectory before moving along
                if self.replanning:
                    pos_input = [obs[0], obs[2], obs[4]]
                    target_vel = [0,0,0]

                command_type = Command.FULLSTATE
                args = [pos_input, target_vel, target_acc, target_yaw, target_rpy_rates, ep_time]
            # Notify set point stop has to be called every time we transition from low-level
            # commands to high-level ones. Prepares for landing
            elif self.parcours_completion and not self._land:
                command_type = Command.LAND
                args = [0.0, 2.0]  # Height, duration
                self._land = True  # Send landing command only once
            elif self._land:
                command_type = Command.FINISHED
                args = []
            else:
                command_type = Command.NONE
                args = []

        #########################
        # REPLACE THIS (END) ####
        #########################

        return command_type, args

    def step_learn(
        self,
        action: list,
        obs: np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
    ):
        """Learning and controller updates called between control steps.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions,
            observations, rewards, done flags, and information dictionaries to learn, adapt, and/or
            re-plan.

        Args:
            action: Most recent applied action.
            obs: Most recent observation of the quadrotor state.
            reward: Most recent reward.
            done: Most recent done flag.
            info: Most recent information dictionary.

        """
        #########################
        # REPLACE THIS (START) ##
        #########################

        # Store the last step's events.
        self.action_buffer.append(action)
        self.obs_buffer.append(obs)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.info_buffer.append(info)

        # Implement some learning algorithm here if needed

        #########################
        # REPLACE THIS (END) ####
        #########################

    def episode_learn(self):
        """Learning and controller updates called between episodes.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions,
            observations, rewards, done flags, and information dictionaries to learn, adapt, and/or
            re-plan.

        """
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