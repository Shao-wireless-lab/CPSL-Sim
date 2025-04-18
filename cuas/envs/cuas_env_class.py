from __future__ import division
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import numpy as np
import time
import ray
import statistics as stats
import shapely
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import math
import cmath
import gym
import os
import pathlib
import sys

from cuas import util
from cuas.agents.cuas_agents import (
    Agent,
    AgentType,
    CuasAgent,
    CuasAgent2D,
    Entity,
    Obstacle,
    ObsType,
)
from cuas.envs.base_cuas_env import BaseCuasEnv
from cuas.safety_layer.safety_layer import SafetyLayer
from gym import spaces
from gym.error import DependencyNotInstalled
from gym.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.spaces.repeated import Repeated
from qpsolvers import solve_qp
from gym import Env
from gym import spaces, logger
from itertools import product
from gym.utils import seeding

from numpy import array
from numpy.linalg import norm
import torch as T
import random  # test out random environment
from random import randint
from operator import itemgetter

import scipy.io
import sys
import json
from pathlib import Path
from cuas.utils.root_finder import find_root


# === Load the pclearlume_scenarios.json and sim_config.cfg (actually JSON) for plume scenarios ===
project_root = find_root()
config_dir = project_root / "configs"
sim_config_path = config_dir / "sim_config.cfg"
scenario_json_path = config_dir / "plume_scenarios.json"

# Load scenario level
with open(sim_config_path, "r") as f:
    sim_config = json.load(f)
scenario_level = sim_config.get("env_config", {}).get("scenario_level")
if not scenario_level:
    raise ValueError("❌ scenario_level not found in sim_config.")

# Load plume scenarios config
with open(scenario_json_path, "r") as f:
    plume_scenarios = json.load(f)
scenario_config = plume_scenarios.get(scenario_level)
if not scenario_config:
    raise ValueError(f"❌ scenario_level '{scenario_level}' not found in plume_scenarios.")

# Get data_dir
data_dir = scenario_config.get("data_dir")
if not data_dir:
    raise ValueError(f"❌ 'data_dir' missing in scenario config for '{scenario_level}'.")

# Load .mat files
#wind_speed_2 = scipy.io.loadmat(os.path.join(data_dir, "Plume-Wind-Data-Height-2.mat"))
#Conc_2 = scipy.io.loadmat(os.path.join(data_dir, "Plume-C-Data-Height-2.mat"))

#TODO: to save room, have to delect data at height-2. Since they are not used, import data of height 5 instead.
wind_speed_2 = scipy.io.loadmat(os.path.join(data_dir, "Plume-Wind-Data-Height-5.mat"))
Conc_2 = scipy.io.loadmat(os.path.join(data_dir, "Plume-C-Data-Height-5.mat"))

wind_speed_5 = scipy.io.loadmat(os.path.join(data_dir, "Plume-Wind-Data-Height-5.mat"))
Conc_5 = scipy.io.loadmat(os.path.join(data_dir, "Plume-C-Data-Height-5.mat"))

# fil_locs = scipy.io.loadmat(base_path / "fil_locs.mat")
fil_locs = []


class cuas_env_class(BaseCuasEnv, MultiAgentEnv):
    def __init__(self, env_config={}):
        super().__init__()
        self.dd = []
        self.config = env_config
        self._parse_config()
        self.seed(seed=self.sim_seed)
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        self.iter = 0
        self._past_obs = self.reset()
        self.num_constraints = (self.num_agents - 1) + self.num_obstacles
        self.safety_layer = None

    def _load_plume_target_info(self):
        scenario_config = plume_scenarios[self.scenario_level]

        # Optional: extract G value from path if needed
        # G_value = data_dir.split("G=")[-1].split("_")[0]

        # Assign target from JSON
        target_info = scenario_config["target"]
        self.target = Entity(
            x=target_info["x"],
            y=target_info["y"],
            r=target_info["radius"],
            type2=AgentType.T,
        )

    def _parse_config(self):

        self.scenario_level = self.config.get("scenario_level", "easy_80_60")
        self._load_plume_target_info()
        #print(f"[INFO] Target: (x={self.target.x}, y={self.target.y}, r={self.target.radius}).")
        # Environment Boundary Settings
        self.max_time_step = self.config.get("max_time_step", 3200)

        self.env_width_x = self.config.get("env_width_x")
        self.env_height_y = self.config.get("env_height_y")
        self.max_distance = util.distance(point_2=(self.env_width_x, self.env_height_y))
        self.env_xgrid_res = self.config.get("env_xgrid_res")
        self.env_ygrid_res = self.config.get("env_ygrid_res")
        # Animation Settings
        self.screen_width = self.config.get("screen_width")
        self.screen_height = self.config.get("screen_height")

        # Resolution of Plume Data in x,y coordinates
        self.xgrid_points = np.arange(self.env_xgrid_res / 2, self.env_width_x, self.env_xgrid_res)
        self.ygrid_points = np.arange(self.env_ygrid_res / 2, self.env_height_y, self.env_ygrid_res)

        self.num_grid_pts = len(self.xgrid_points) * len(self.ygrid_points)

        self.grid_coord = np.array([[x, y] for x in self.xgrid_points for y in self.ygrid_points])
        self.grid_coord_points = [Point(x, y) for x in self.xgrid_points for y in self.ygrid_points]

        self.num_avg_pts = self.config.get("num_avg_pts")

        # Penalty Coefficients
        self.k1 = self.config.get("k1")
        self.k2 = self.config.get("k2")
        self.k_theta1 = self.config.get("k_theta1")
        self.k_theta2 = self.config.get("k_theta2")
        self.k_step = self.config.get("k_step")

        self.time_step_penalty = self.config.get("time_step_penalty")

        self.obstacle_penalty = self.config.get("obstacle_penalty")
        self.obstacle_penalty_weight = self.config.get("obstacle_penalty_weight")

        self.agent_collision_penalty = self.config.get("agent_collision_penalty")
        self.agent_penalty_weight = self.config.get("agent_penalty_weight")

        self._constraint_slack = self.config.get("constraint_slack")
        self._constraint_k = self.config.get("constraint_k")
        self.sim_seed = self.config.get("seed")
        self.norm_env = self.config.get("normalize")
        self.prior_policy_mixin = self.config.get("prior_policy_mixin")

        # Agent Details
        self.num_agents = self.config.get("num_agents")
        self.agent_radius = self.config.get("agent_radius")
        self.agent_v_min = self.config.get("agent_v_min")  # m/s
        self.agent_v_max = self.config.get("agent_v_max")
        self.agent_w_min = -np.pi * 2  # rad/s
        self.agent_w_max = np.pi * 2

        # options are local or global
        self.agent_observation_type = self.config.get("agent_observation_type")
        # self.agent_observation_radius = True
        # when observation type is global, we set the observation radius to max size of environment
        if self.agent_observation_type == "global":
            self.agent_observation_radius = self.max_distance

            self.agent_show_observation_radius = False
        else:
            # observation radius should be less than max_distance
            self.agent_observation_radius = self.config.get("agent_observation_radius")
            self.agent_show_observation_radius = True
            if self.agent_observation_radius >= self.max_distance:
                print('observation radius error')
                quit()

        self.agent_observation_fov = self.config.get("agent_obs_fov") * math.pi / 180
        self.agent_start_type = self.config.get("agent_start_type")
        self.agent_move_type = self.config.get("agent_move_type")
        self.agent_drive_type = getattr(
            sys.modules[__name__], self.config.get("agent_drive_type", "CuasAgent")
        )

        self.norm_low = np.array([-1.0, -1.0])
        self.norm_high = np.array([1.0, 1.0])
        self.success_count = 0
        self.low = np.array([self.agent_v_min, self.agent_w_min])
        self.high = np.array([self.agent_v_max, self.agent_w_max])

        # Plume Details
        self.num_plumes = 1
        '''
        self.target = Entity(
            x=self.config.get("target_x"),
            y=self.config.get("target_y"),
            r=self.config.get("target_radius"),
            type2=AgentType.T,
        )
        '''
        self.flux_threshold = self.config.get("flux_threshold")
        self.lat_flux_threshold_5 = self.config.get("lat_flux_threshold_5")
        self.lat_flux_threshold_2 = self.config.get("lat_flux_threshold_2")

        # Load the wind profile and concentration info
        # self.ws_Height_2 = wind_speed_2["ws"]
        # self.rho_Height_2 = Conc_2["C"]

        # self.ws_Height_5 = wind_speed_5["ws"]
        # self.rho_Height_5 = Conc_5["C"]

        self.ws_Height_2 = wind_speed_2["ws"][-320100:, :]
        self.rho_Height_2 = Conc_2["C"][-320100:, :]

        self.ws_Height_5 = wind_speed_5["ws"][-320100:, :]
        self.rho_Height_5 = Conc_5["C"][-320100:, :]

        # self.fil_locs = fil_locs["fillament_locs"]
        # self.fil_locs = np.reshape(self.fil_locs[0][2000:6000],(1,4000))

        self.start_x_locs = np.arange(self.env_width_x / (self.num_agents + 1), self.env_width_x,
                                      self.env_width_x / (self.num_agents + 1))
        '''
        self.start_x_locs = np.linspace(self.env_width_x / 2 + self.env_width_x / (2 * (self.num_agents + 1)),
                                        self.env_width_x - self.env_width_x / (2 * (self.num_agents + 1)),
                                        self.num_agents)
        '''


        # Obstacle Details
        self.num_obstacles = self.config.get("num_obstacles")
        self.obstacle_radius = self.config.get("obstacle_radius")
        self.obstacle_v = self.config.get("obstacle_v")

        self.max_num_obstacles = self.num_obstacles
        self.max_num_agents = self.num_agents

        # Sim Timing Resolution
        self.time_step = self.config.get("time_step")
        self.max_time = self.config.get("max_time")

        self.alpha = self.config.get("alpha")
        self.beta = self.config.get("beta")
        self.use_safety_layer = self.config.get("use_safety_layer")
        self.use_safe_action = self.config.get("use_safe_action")
        self.safety_layer_type = self.config.get("safety_layer_type")

        self.add_com_delay = self.config.get("add_com_delay")
        self.com_delay = self.config.get("com_delay")

        self.beta_min = self.config.get("beta_min")

        # self.bufferTable = np.zeros([1,self.num_agents * 2])
        # self.indx = np.arange(self.num_agents * 2)
        # self.indx = np.reshape(self.indx,[self.num_agents,2])

        self.bufferTable_data_5 = np.zeros([1, self.num_agents * 3])
        self.bufferTable_data_2 = np.zeros([1, self.num_agents * 3])
        self.indx_data = np.arange(self.num_agents * 3)
        self.indx_data = np.reshape(self.indx_data, [self.num_agents, 3])

        self.agents_list = ["a0", "a1", "a2", "a4"]
        self.obstacles_list = ["o0", "o1", "o2", "o3", "o4"]

        self.dim = 2

        # self.id_dist2cent_5 = math.sqrt(((2*self.agent_radius + self._constraint_slack)**2)/(2*(1 - math.cos(2*math.pi/self.num_agents))))
        self.id_dist2cent_2 = 2
        self.id_dist2cent = self.config.get("id_dist2cent")
        self.tolerance = self.config.get("tolerance")
        self.offset = self.config.get("offset")

        self.centroid_motion_window = self.config.get("centroid_motion_window", 100)
        self.centroid_motion_threshold = self.config.get("centroid_motion_threshold", 0.2)

    def _get_route(self, points):
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        def polygon_area(points):
            """Return the area of the polygon whose vertices are given by the
            sequence points.

            """
            area = 0
            xx = points[:, 0]
            yy = points[:, 1]
            for p in range(len(points) - 1):
                # area+=(yy[p] + yy[p+1])*(xx[p] + xx[p+1])
                area += np.linalg.det([[xx[p], xx[p + 1]], [yy[p], yy[p + 1]]])
            area += np.linalg.det([[xx[-1], xx[0]], [yy[-1], yy[0]]])
            return area / 2

        z = np.array([[complex(*c) for c in points]])  # notice the [[ ... ]]
        distance_matrix_pre = abs(z.T - z)

        distance_matrix = np.floor(distance_matrix_pre * 10000).astype(int).tolist()

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(self.num_agents, 1, 0)

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            index = routing.Start(0)
            route = [manager.IndexToNode(index)]
            while not routing.IsEnd(index):
                index = solution.Value(routing.NextVar(index))
                route.append(manager.IndexToNode(index))

        route.reverse()
        self.route = route
        xpoints = points[route][:-1, 0]
        ypoints = points[route][:-1, 1]
        xc = stats.mean(xpoints)
        yc = stats.mean(ypoints)
        coor = np.column_stack((xpoints, ypoints))
        area = abs(polygon_area(coor))

        return route[0:-1], coor, np.array([xc, yc]), area

    def _add_obstacles(self):
        if self.agent_start_type == "Fluxotaxis":
            obstacle_positions = [(200, 200), (200, 200), (200, 200), (200, 200), (200, 200)]

            for x, y in obstacle_positions:
                theta = np.random.random() * np.pi  # Keep random orientation if needed
                r = self.obstacle_radius
                obs_type = ObsType.M

                temp_obstacle = Obstacle(x, y, theta=theta, r=r, obs_type=obs_type)
                temp_obstacle.v = self.obstacle_v
                self.obstacles.append(temp_obstacle)

        else:
            def get_random_obstacle():
                x = np.random.random() * self.env_width_x
                y = np.random.random() * self.env_height_y
                theta = np.random.random() * np.pi
                r = self.obstacle_radius
                obs_type = ObsType.M

                return Obstacle(x, y, theta=theta, r=r, obs_type=obs_type)

            for _ in range(self.num_obstacles):
                temp_obstacle = get_random_obstacle()
                temp_obstacle.v = self.obstacle_v
                self.obstacles.append(temp_obstacle)

    def _create_agents(self, agent_type, num_agents):

        def is_collision(agent):

            for obs in self.obstacles:
                if agent.collision_with(obs):
                    self.obstacles = []
                    self._add_obstacles()
                    return True

            for _agent in self.agents:

                if agent.collision_with(_agent):
                    return True

            return False

        def get_random_agent(agent_id):
            x = np.random.random() * self.env_width_x
            y = np.random.random(1) * self.env_height_y
            theta = np.random.random() * np.pi
            r = self.agent_radius
            move_type = self.agent_move_type
            return self.agent_drive_type(agent_id,
                                         agent_type,
                                         x,
                                         y,
                                         theta,
                                         r=r,
                                         obs_r=self.agent_observation_radius,
                                         move_type=move_type)

        def get_det_agent_x(agent_id):
            # x = np.random.uniform(low=self.env_width_x/2 + 20,high=self.env_width_x,size=1).item()
            x = np.array([100]).item()
            y = np.array(self.start_y_locs[agent_id])
            # np.linspace(1,self.env_height_y/2 - self.env_ygrid_resolution/2 ,self.num_agents)[agent_id]
            theta = math.pi
            r = self.agent_radius
            move_type = self.agent_move_type
            return self.agent_drive_type(agent_id,
                                         agent_type,
                                         x,
                                         y,
                                         theta,
                                         r=r,
                                         obs_r=self.agent_observation_radius,
                                         move_type=move_type)

        '''
        def get_det_agent_y(agent_id):
            # y = np.random.uniform(low=self.env_height_y/2 + 20,high=self.env_height_y,size=1)
            y = np.array([75])
            # y = np.array([25])
            # Randomly choose initial y position from 25 to 75 with 5 as interval
            # y_choices = np.arange(75, 24, -5)  # [75, 70, ..., 25]
            # y = np.array([np.random.choice(y_choices)])

            x = self.start_x_locs[agent_id]
            # np.linspace(1,self.env_height_y/2 - self.env_ygrid_resolution/2 ,self.num_agents)[agent_id]
            theta = -math.pi / 2
            r = self.agent_radius
            move_type = self.agent_move_type
            return self.agent_drive_type(agent_id,
                                         agent_type,
                                         x,
                                         y,
                                         theta,
                                         r=r,
                                         obs_r=self.agent_observation_radius,
                                         move_type=move_type)
        '''

        # For "YScan": choose one shared y-position randomly from 90 to 75 (step -1)
        if self.agent_start_type == "YScan":
            #y_choices = np.arange(90, 79, -1)  # [90, 89, ..., 75]
            #y_choices = np.arange(180, 140, -1)  # [90, 89, ..., 75]
            scenario_config = plume_scenarios.get(self.scenario_level)
            y_range = scenario_config.get("y_range")
            y_choices = np.arange(y_range[0], y_range[1], y_range[2])
            self.shared_y = np.random.choice(y_choices)  # pick one y value for all agents

        # Modified get_det_agent_y function
        def get_det_agent_y(agent_id):
            y = np.array([self.shared_y])  # All agents use the same y value

            x = self.start_x_locs[agent_id]  # Each agent gets its own x location
            theta = -math.pi / 2  # Facing downward
            r = self.agent_radius
            move_type = self.agent_move_type

            return self.agent_drive_type(agent_id,
                                         agent_type,
                                         x,
                                         y,
                                         theta,
                                         r=r,
                                         obs_r=self.agent_observation_radius,
                                         move_type=move_type)

        def get_fluxotaxis_agent(agent_id):
            # Fixed x positions
            x_positions = [127, 130, 133]

            if agent_id >= len(x_positions):
                raise ValueError(f"No predefined x position for agent_id {agent_id}")

            x = x_positions[agent_id]
            y = np.array(random.uniform(58, 62)).astype(np.float32)
            theta = -math.pi / 2  # Facing downward
            r = self.agent_radius
            move_type = self.agent_move_type

            return self.agent_drive_type(agent_id,
                                         agent_type,
                                         x,
                                         y,
                                         theta,
                                         r=r,
                                         obs_r=self.agent_observation_radius,
                                         move_type=move_type)

        for agent_id in range(self.num_agents):
            in_collision = True
            while in_collision:
                if self.agent_start_type == "XScan":
                    agent = get_det_agent_x(agent_id)
                elif self.agent_start_type == "YScan":
                    agent = get_det_agent_y(agent_id)
                elif self.agent_start_type == "Fluxotaxis":
                    agent = get_fluxotaxis_agent(agent_id)
                else:
                    agent = get_random_agent(agent_id)

                in_collision = is_collision(agent)

            self.agents.append(agent)

    def _get_observation_space(self):
        self.num_agent_states = 2 + 3
        self.num_obs_agent_states = 1 + 2 + 1 + 1
        self.num_obs_form_states = 2
        self.num_obs_other_agent_states = 3
        self.num_obs_other_agent_mea_states = 1 + 2 + 1 + 1
        self.num_obs_to_loc_with_max_sensed_plume = 1 + 1 + 1
        self.num_obs_obstacle_agent_states = 2

        # OBSERVATION LOWER LIMITS
        # (1) agent_state: x, y, theta, v, omega
        self.norm_low_obs = [-1] * self.num_agent_states

        # (2) plume_measurements: wx,wy,wz, and c
        self.norm_low_obs.extend(
            ([-1] * self.num_obs_agent_states)
        )

        # (3) formation: dist to centroid, angle to centroid, div_lattice (t-1), div_lattice(t)
        self.norm_low_obs.extend(
            ([-1] * self.num_obs_form_states)
        )

        # (4) other_agents: type, sensed, dist_agent_j, theta_agent_j, agent_j_rel_bearing,
        # subtract 1 to not include the current agent
        self.norm_low_obs.extend(
            ([-1] * self.num_obs_other_agent_states) * (self.num_agents - 1)
        )

        # (5) other_agent_measurements
        self.norm_low_obs.extend(
            ([-1] * self.num_obs_other_agent_mea_states) * (self.num_agents - 1)
        )

        # (6) measurements to agent in plume with max flux reading
        self.norm_low_obs.extend(
            ([-1] * self.num_obs_to_loc_with_max_sensed_plume)
        )

        # (7) obstacle: type, sensed, distance_obstacle_k, theta_obstacle_k
        self.norm_low_obs.extend(
            ([-1] * self.num_obs_obstacle_agent_states) * (self.num_obstacles)
        )
        self.norm_low_obs = np.array(self.norm_low_obs)

        # OBSERVATION LOWER LIMITS
        # (1) the main agent
        self.low_obs = [0, 0, -np.pi, self.agent_v_min, self.agent_w_min]

        # (2) plume states
        # type, sensed, c, wx,wy
        self.low_obs.extend([0, -100, -100, 0, 0])  #

        # (3) formation states
        self.low_obs.extend([0, -np.pi])

        # (4) other agent states
        # type, sensed, rel_dist, rel_bearing, rel_bearing_self
        self.low_obs.extend([0, -np.pi, -np.pi] * (self.max_num_agents - 1))

        # (5) other agent measurements
        # c,wx,wy
        self.low_obs.extend([0, -100, -100, 0, 0] * (self.max_num_agents - 1))

        # (6) location with latest max. flux measurements
        # c,wx,wy
        self.low_obs.extend([0, -np.pi, 0])

        # (7) obstacle states
        # type, sensed, rel_dist, rel_bearing
        self.low_obs.extend([0, -np.pi] * (self.max_num_obstacles))

        self.low_obs = np.array(self.low_obs)

        #################################################################
        # OBSERVATION UPPER LIMITS
        self.norm_high_obs = np.array([1] * self.norm_low_obs.size)

        # (1) the main agent
        self.high_obs = [
            self.env_width_x,
            self.env_height_y,
            np.pi,
            self.agent_v_max,
            self.agent_w_max
        ]

        # (2) plume states
        # type, sensed, c,wx,wy,I
        self.high_obs.extend([100, 100, 100, 1, 10])

        # (3) formation states
        self.high_obs.extend([self.max_distance, np.pi])

        # (4) other_agent states
        self.high_obs.extend([self.max_distance, np.pi, np.pi] * (self.max_num_agents - 1))

        # (5) other_agent_measurements
        self.high_obs.extend([100, 100, 100, 1, 10] * (self.max_num_agents - 1))

        # (6) location measurements to max flux reading
        self.high_obs.extend([self.max_distance, np.pi, 10])

        # (7) obstacles
        self.high_obs.extend([self.max_distance, np.pi] * (self.max_num_obstacles))

        self.high_obs = np.array(self.high_obs)

        return spaces.Dict(
            {
                i: spaces.Dict(
                    {
                        "observations": spaces.Box(
                            low=self.norm_low_obs,
                            high=self.norm_high_obs,
                            dtype=np.float32,
                        ),
                        "raw_obs": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            dtype=np.float32,
                            shape=(self.norm_low_obs.shape[0],),
                        ),
                        "constraints": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            # shape=(1 + self.num_agents - 1 + self.num_obstacles,),
                            shape=(self.num_agents - 1 + self.num_obstacles,),
                            dtype=np.float32,
                        ),
                        "action_g": spaces.Box(
                            -np.inf,
                            np.inf,
                            # shape=(1 + self.num_agents - 1 + self.num_obstacles, 2),
                            shape=(self.num_agents - 1 + self.num_obstacles, 2),
                            dtype=np.float32,
                        ),
                        "action_h": spaces.Box(
                            -np.inf,
                            np.inf,
                            # shape=(1 + self.num_agents - 1 + self.num_obstacles,),
                            shape=(self.num_agents - 1 + self.num_obstacles,),
                            dtype=np.float32,
                        ),
                        "action_r": spaces.Box(
                            -1.0,
                            1.0,
                            shape=(2, 2),
                            dtype=np.float32,
                        ),
                        #     "action_a": spaces.Box(
                        #         low=self.norm_low, high=self.norm_high, dtype=np.float32
                        #     ),
                    }
                )
                for i in range(self.num_agents)
            }
        )

    def _get_action_space(self):
        """
        The number of action will be a fix size
        """
        return spaces.Dict(
            {
                i: spaces.Box(low=self.norm_low, high=self.norm_high, dtype=np.float32)
                for i in range(self.num_agents)
            }
        )

    def reset(self):
        """
        Reset environment
        """
        # print('check')

        self.timecond1 = 0
        self.timecond2 = 0

        self.cond1met = False
        self.cond2met = False

        self.detection = False

        self.time_elapse = 0
        self.time_t = 0

        self.hh = 5

        self.drop = False
        self.time_tdrop = 0
        self.at_drop_cent_dist_to_emitter = 0
        self.drop_found = False

        self.centroid_stable = False

        self.final_cent_dist_to_emitter = 0
        self.final_cent_dist_to_emitter_offset = 0

        self.bufferTable_data_5 = np.zeros([1, self.num_agents * 3])
        self.bufferTable_data_2 = np.zeros([1, self.num_agents * 3])

        self.sim_done = False
        self.plume_found = False

        self.dones = set()
        self.obstacles = []
        self._add_obstacles()
        self.agents = []
        self.plumes = []
        self.current_cent_dist_to_emitter = 0
        self._create_agents(AgentType.P, self.num_agents)
        self.cum_a_bar = {agent.id: np.array([0.0, 0.0]) for agent in self.agents}

        self.other_agent_data_recorded = np.zeros(self.num_agents, dtype='bool')

        idp = np.arange(100 * (self.time_t), 100 * (self.time_t + 1))

        self.step_x = 0
        self.step_y = 0

        self.rho2 = self.rho_Height_2[idp]
        self.ws2 = self.ws_Height_2[idp]

        self.rho5 = self.rho_Height_5[idp]
        self.ws5 = self.ws_Height_5[idp]

        # self.fillament_locations = self.fil_locs[0][0]

        self._store_plume_data()

        self.bufferTable_data_5 = np.delete(self.bufferTable_data_5, 0, axis=0)
        self.bufferTable_data_2 = np.delete(self.bufferTable_data_2, 0, axis=0)

        self.total_flux_VS_5 = []
        self.check_total_flux_VS_5 = []
        self.total_flux_VS_2 = []
        self.check_total_flux_VS_2 = []

        self.centroid = []
        self.offset_centroid = np.array([0.0, 0.0])

        self.centroid_motion = []  # Reset at the start of each episode

        self.lat_area = []
        self.coor = []
        self.route2 = []

        self.d_ic = np.zeros(self.num_agents)

        self.agent_locs = np.zeros([self.num_agents, 2])
        for agent_id in range(self.num_agents):
            agent = self.agents[agent_id]
            self.agent_locs[agent_id, 0] = agent.x
            self.agent_locs[agent_id, 1] = agent.y
            # self.agent_locs_matrix["a%d" % (agent.id)].append((self.agent_locs[agent.id]))

        self.obstacle_locs = np.zeros([self.num_obstacles, 2])


        for obstacle_id in range(self.num_obstacles):
            obstacle = self.obstacles[obstacle_id]
            self.obstacle_locs[obstacle_id, 0] = obstacle.x
            self.obstacle_locs[obstacle_id, 1] = obstacle.y
            # self.obstacle_locs_matrix["o%d" % (obstacle_id)].append((self.obstacle_locs[obstacle_id]))

        [self.route2, self.coor, centroid, self.lat_area] = self._get_route(self.agent_locs)

        self.VS = Polygon(self.coor)

        self.centroid = centroid

        self.current_cent_dist_to_emitter = np.linalg.norm(
            np.array([self.centroid[0], self.centroid[1]]) - np.array([self.target.x, self.target.y]))

        self.dta = np.zeros([self.num_agents, self.num_obs_agent_states])

        self.dist2anchor_locF = np.zeros(self.num_agents)

        self.agent_detects = np.zeros(self.num_agents)
        self.agent_fluxes = np.zeros(self.num_agents)
        self.agent_rhos = np.zeros(self.num_agents)
        self.agent_winds = np.zeros([self.num_agents, 2])
        self.agent_rhos_tm1 = np.zeros(self.num_agents)
        self.anchor_conc = 0
        self.anchor_loc = np.zeros(2)
        self.h_max = self.hh
        self.agent_states = np.zeros([self.num_agents, 3])

        self.mean_tabledata_all_agents = np.mean(self.bufferTable_data_5[-self.num_avg_pts:], axis=0)

        for agent in self.agents:
            self.agent_states[agent.id, :] = self.mean_tabledata_all_agents[self.indx_data[agent.id]].tolist()
            [det, ff, pp, ww] = agent.get_detect(self.agent_states[agent.id, :], self.flux_threshold, self.centroid)
            self.agent_detects[agent.id] = det
            self.agent_fluxes[agent.id] = ff
            self.agent_rhos[agent.id] = pp
            self.agent_winds[agent.id] = ww
            # self.agent_states[agent.id,0] = self.agent_states[agent.id,0] - 1.98

        if np.any(self.agent_detects):
            which_agents_det = np.where(self.agent_detects)[0]
            # FF = self.agent_fluxes[which_agents_det]
            # aa = which_agents_det[np.where(np.array(abs(FF)) == max(abs(FF)))[0]]
            PP = self.agent_rhos[which_agents_det]
            aa = which_agents_det[np.where(np.array(PP) == max(PP))[0]]

            agent_max_conc = aa[0]
            if len(aa) > 1:
                agent_max_conc = aa[0]  # np.random.choice(aa)

            if self.agent_rhos[agent_max_conc] > self.anchor_conc:
                loc = np.array([self.agents[agent_max_conc].x, self.agents[agent_max_conc].y])
                self.anchor_conc = self.agent_rhos[agent_max_conc]
                self.anchor_loc = loc
                self.h_max = self.hh

        obs = {agent.id: self._calc_obs(agent) for agent in self.agents}

        self.total_flux_VS_5.extend([self.get_total_VS_flux()])
        # sum_flux = np.array(self.total_flux_VS_5[-3:])
        sum_flux = np.array(self.total_flux_VS_5[-1])
        # if all(sum_flux >= self.lat_flux_threshold_5):
        if sum_flux >= self.lat_flux_threshold_5:
            self.check_total_flux_VS_5.extend([True])
        else:
            self.check_total_flux_VS_5.extend([False])

        self._agent_ids = {agent.id for agent in self.agents}

        self.agent_rhos_tm1 = self.agent_rhos
        self.agent_locs_tm1 = self.agent_locs
        self.Vbar_tm1 = np.mean(self.agent_winds, axis=0)
        self.centroid_tm1 = self.centroid
        self.anchor_conc_tm1 = self.anchor_conc
        # self.Vbar_tm1 = self.Vbar_tm1/np.linalg.norm(self.Vbar_tm1)

        if any(self.agent_detects):
            self.detection = True

        return obs

    def get_total_VS_flux(self):

        total_flux_VS = np.sum(self.agent_fluxes)

        return total_flux_VS

    def get_seg_VS_flux(self):

        locs = self.agent_locs[self.route]
        mid_pts = np.zeros([self.num_agents, 2])
        mid_pts[:, 0] = np.convolve(locs[:, 0], [0.5, 0.5], "valid")
        mid_pts[:, 1] = np.convolve(locs[:, 1], [0.5, 0.5], "valid")

        n_midpts = mid_pts - self.centroid

        n_mag = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            n_mag[i] = np.linalg.norm(n_midpts[i, :])
            if n_mag[i] == 0:
                n_midpts[i] = np.array([0, 0])
            else:
                n_midpts[i] = n_midpts[i, :] / n_mag[i]

        # n_midpts_mag = np.linalg.norm(n_midpts,axis=1)

        # n_midpts = n_midpts/n_midpts_mag

        data = np.zeros([self.num_agents, 3])
        for agent_id in range(self.num_agents):
            data[agent_id] = self.dta[agent_id][0:3]

        rhos = data[self.route, 0]
        mid_rhos = np.convolve(rhos, [0.5, 0.5], "valid")

        vels = np.zeros([self.num_agents + 1, 2])
        vels[:, 0] = data[self.route, 1]
        vels[:, 1] = data[self.route, 2]

        mid_vels = np.zeros([self.num_agents, 2])
        mid_vels[:, 0] = np.convolve(vels[:, 0], [0.5, 0.5], "valid")
        mid_vels[:, 1] = np.convolve(vels[:, 1], [0.5, 0.5], "valid")

        self.seg_fluxes = np.zeros(self.num_agents)
        total_seg_flux = 0
        for agent in range(self.num_agents):
            self.seg_fluxes[agent] = (mid_rhos[agent] - 1.98) * (1e-6) * (101325 * 16.04 / (8.3145 * 273)) * np.dot(
                mid_vels[agent], n_midpts[agent]) * math.pi * 1 ** 2
            total_seg_flux += self.seg_fluxes[agent]

        return total_seg_flux

    def is_centroid_stable(self):
        if len(self.centroid_motion) < self.centroid_motion_window:
            return False

        recent_positions = np.array(self.centroid_motion[-self.centroid_motion_window:])
        range_x = recent_positions[:, 0].max() - recent_positions[:, 0].min()
        range_y = recent_positions[:, 1].max() - recent_positions[:, 1].min()

        return range_x < self.centroid_motion_threshold and range_y < self.centroid_motion_threshold

    # TODO: caught agents shouldn't move
    def step(self, actions):
        """[summary]

        action is of type dictionary
        Args:
            action ([type], optional): [description]. Defaults to None.
        """

        # self.counter += 1
        self.time_elapse += self.time_step
        # print(self.time_elapse)
        self.plume_found = False
        # self.d_ic = np.zeros(self.num_agents)

        self.sim_done = False

        obs, rew, done, info = {}, {}, {}, {}

        # agents move
        for i, action in actions.items():
            action = self._unscale_action(action)

            if not self.detection:
                if self.agent_start_type == "Fluxotaxis":
                    action = np.array([0, 0], dtype=np.float64)
                else:
                    action = np.array([self.agent_v_max / 6, 0], dtype=np.float64)
            # if ((self.agents[i].x == self.env_width_x) or (self.agents[i].x == 0)):
            # action = np.array([self.agent_v_max/6,math.pi],dtype=np.float64)

            # if ((self.agents[i].y == self.env_height_y) or (self.agents[i].y == 0)):
            # action = np.array([self.agent_v_max/6,math.pi],dtype=np.float64)

            if (self.agents[i].x == self.env_width_x) or (self.agents[i].x == 0):
                self.agents[i].theta = (self.agents[i].theta + math.pi) % (2 * math.pi)
                action = np.array([self.agent_v_max / 6, 0], dtype=np.float64)

            if (self.agents[i].y == self.env_height_y) or (self.agents[i].y == 0):
                self.agents[i].theta = (self.agents[i].theta + math.pi) % (2 * math.pi)
                action = np.array([self.agent_v_max / 6, 0], dtype=np.float64)

            if self.use_safe_action:
                action_qp = self.proj_safe_action(self.agents[i], action)
                self.cum_a_bar[i] = np.clip(
                    self.cum_a_bar[i] + action_qp, self.low, self.high
                )
                action += action_qp

                # temp safety_action_layer
                # action = self.safety_action_layer(self.agents[i], action)
                action = np.clip(action, self.low, self.high)
            self.agents[i].step(action)

        # obstacle moves
        for obstacle in self.obstacles:
            obstacle.step()

        self.time_t = self.time_t + 1

        if self.time_t >= self.max_time_step:
            self.sim_done = True

        idp = np.arange(100 * (self.time_t), 100 * (self.time_t + 1))
        # idpy = np.arange(2,100*3,3)
        self.rho2 = self.rho_Height_2[idp]
        self.ws2 = self.ws_Height_2[idp]

        self.rho5 = self.rho_Height_5[idp]
        self.ws5 = self.ws_Height_5[idp]

        # self.ws3 = self.ws2[:,idpy]

        self.agent_locs = np.zeros([self.num_agents, 2])
        for agent_id in range(self.num_agents):
            agent = self.agents[agent_id]
            self.agent_locs[agent_id, 0] = agent.x
            self.agent_locs[agent_id, 1] = agent.y
            # self.agent_locs_matrix["a%d" % (agent.id)].append((self.agent_locs[agent.id]))

        self.obstacle_locs = np.zeros([self.num_obstacles, 2])
        for obstacle_id in range(self.num_obstacles):
            obstacle = self.obstacles[obstacle_id]
            self.obstacle_locs[obstacle_id, 0] = obstacle.x
            self.obstacle_locs[obstacle_id, 1] = obstacle.y
            # self.obstacle_locs_matrix["o%d" % (obstacle_id)].append((self.obstacle_locs[obstacle_id]))

        [self.route2, self.coor, self.centroid, self.lat_area] = self._get_route(self.agent_locs)

        self.VS = Polygon(self.coor)

        # self._store_plume_det()
        self._store_plume_data()

        self.dist2anchor_locF = np.zeros(self.num_agents)

        self.d_ic = np.zeros(self.num_agents)
        self.agent_detects = np.zeros(self.num_agents)
        self.agent_fluxes = np.zeros(self.num_agents)
        self.agent_rhos = np.zeros(self.num_agents)
        self.agent_winds = np.zeros([self.num_agents, 2])
        self.agent_states = np.zeros([self.num_agents, 3])

        self.dta = np.zeros([self.num_agents, self.num_obs_agent_states])

        # self.centroid_motion.append(self.centroid - self.centroid_tm1)
        self.centroid_motion.append(self.centroid.copy())

        if self.hh == 5:
            self.mean_tabledata_all_agents = np.mean(self.bufferTable_data_5[-self.num_avg_pts:], axis=0)
        else:
            self.mean_tabledata_all_agents = np.mean(self.bufferTable_data_2[-self.num_avg_pts:], axis=0)

        for agent in self.agents:
            self.agent_states[agent.id, :] = self.mean_tabledata_all_agents[self.indx_data[agent.id]].tolist()
            [det, ff, pp, ww] = agent.get_detect(self.agent_states[agent.id, :], self.flux_threshold, self.centroid)
            self.agent_detects[agent.id] = det
            self.agent_fluxes[agent.id] = ff
            self.agent_rhos[agent.id] = pp
            self.agent_winds[agent.id] = ww
            # self.agent_states[agent.id,0] = self.agent_states[agent.id,0] - 1.98

        self.Vbar_t = np.mean(self.agent_winds, axis=0)
        # self.Vbar_t = self.Vbar_t/np.linalg.norm(self.Vbar_t)
        '''
        if np.any(self.agent_detects):

            which_agents_det = np.where(self.agent_detects)[0]
            # FF = self.agent_fluxes[which_agents_det]
            # aa = which_agents_det[np.where(np.array(abs(FF)) == max(abs(FF)))[0]]
            PP = self.agent_rhos[which_agents_det]
            aa = which_agents_det[np.where(np.array(PP) == max(PP))[0]]

            agent_max_conc = aa[0]
            if len(aa) > 1:
                agent_max_conc = aa[0]  # np.random.choice(aa)

            if (self.agent_rhos[agent_max_conc] > self.anchor_conc):
                loc = np.array([self.agents[agent_max_conc].x, self.agents[agent_max_conc].y])
                if self.anchor_conc != 0:
                    delta_anchor_loc = loc - self.anchor_loc
                    delta_anchor_loc_mag = np.linalg.norm(delta_anchor_loc)
                    Vbar_mag = np.linalg.norm(-self.Vbar_t)

                    if ((delta_anchor_loc_mag == 0) or (Vbar_mag == 0)):
                        angle = 180
                    else:
                        inner = np.dot(delta_anchor_loc, -self.Vbar_t) / (delta_anchor_loc_mag * Vbar_mag)
                        angle = math.acos(min(1, max(inner, -1))) * 180 / math.pi

                    if ((angle >= 0) and (angle <= 75)):
                        self.anchor_conc = self.agent_rhos[agent_max_conc]
                        self.anchor_loc = loc
                        self.h_max = self.hh

                elif (self.anchor_conc == 0):
                    self.anchor_conc = self.agent_rhos[agent_max_conc]
                    self.anchor_loc = loc
                    self.h_max = self.hh
        '''
        if np.any(self.agent_detects):
            which_agents_det = np.where(self.agent_detects)[0]

            best_agent_id = None
            best_angle = 180  # start with worst possible

            for agent_id in which_agents_det:
                if self.agent_rhos[agent_id] >= self.flux_threshold:
                    loc = np.array([self.agents[agent_id].x, self.agents[agent_id].y])

                    if self.anchor_conc == 0:
                        # First anchor — take immediately
                        self.anchor_conc = self.agent_rhos[agent_id]
                        self.anchor_loc = loc
                        self.h_max = self.hh

                    delta = loc - self.anchor_loc
                    delta_mag = np.linalg.norm(delta)
                    wind_mag = np.linalg.norm(-self.Vbar_t)

                    if delta_mag == 0 or wind_mag == 0:
                        angle = 180
                    else:
                        cos_angle = np.dot(delta, -self.Vbar_t) / (delta_mag * wind_mag)
                        angle = math.acos(np.clip(cos_angle, -1, 1)) * 180 / math.pi

                    # Only consider if within angle threshold and best so far
                    if angle <= 75 and angle < best_angle:
                        best_angle = angle
                        best_agent_id = agent_id

            if best_agent_id is not None:
                loc = np.array([self.agents[best_agent_id].x, self.agents[best_agent_id].y])
                self.anchor_conc = self.agent_rhos[best_agent_id]
                self.anchor_loc = loc
                self.h_max = self.hh

        obs = {agent.id: self._calc_obs(agent) for agent in self.agents}

        ##################################################################
        # (4) Task Reward
        self.r_task = 0
        self.agents_done = False
        self.plume_found = False
        self.t_final = 0

        self.current_cent_dist_to_emitter = np.linalg.norm(
            np.array([self.centroid[0], self.centroid[1]]) - np.array([self.target.x, self.target.y]))

        if self.drop == False:
            self.total_flux_VS_5.extend([self.get_total_VS_flux()])
            # sum_flux = np.array(self.total_flux_VS_5[-3:])
            sum_flux = self.total_flux_VS_5[-1]

            # if all(sum_flux >= self.lat_flux_threshold_5):
            if sum_flux >= self.lat_flux_threshold_5:
                self.check_total_flux_VS_5.extend([True])
            else:
                self.check_total_flux_VS_5.extend([False])

            if self.check_total_flux_VS_5[self.time_t]:
                self.drop = True

                self.time_tdrop = self.time_t
                self.drop_found = True
                self.at_drop_cent_dist_to_emitter = self.current_cent_dist_to_emitter

                self.cond1met = True
                self.timecond1 = self.time_t

        if self.detection and self.time_t >= 1000:
            self.centroid_stable = self.is_centroid_stable()
        else:
            self.centroid_stable = False

        # TODO: check no matter if it triggers.
        #if  self.sim_done:
        if self.centroid_stable or self.sim_done:
        #if self.drop or self.sim_done:
            # if self.drop or self.centroid_stable or self.sim_done:
            self.final_cent_dist_to_emitter = self.current_cent_dist_to_emitter
            self.agents_done = True

            self.xc = self.centroid[0]
            self.yc = self.centroid[1]

            self.offset_centroid = self.centroid - self.Vbar_t * self.offset

            self.final_cent_dist_to_emitter_offset = np.linalg.norm(
                np.array([self.offset_centroid[0], self.offset_centroid[1]]) - np.array([self.target.x, self.target.y]))

            self.d_mag = 0

            self.dc = 0
            circle = Point(self.xc, self.yc).buffer(self.dc + self.tolerance)
            self.circ_boun_dist2emitter = 0
            self.cond2met = True
            self.timecond2 = self.time_t
            self.t_final = self.timecond2

            # TODO: Remove the rewards for the final target check since this information should not be sent to model.
            if self.final_cent_dist_to_emitter_offset <= self.tolerance:
            #if circle.contains(Point([self.target.x, self.target.y])):
                self.plume_found = True
                #self.r_task += 10000

                self.success_count += 1


            else:
                self.plume_found = False
                #self.r_task += -2500

        self.circ_boun_dist2emitter = 0
        self.final_agent2centdist = 0
        self.dist_closestUAV2emitter = 0
        self.avg_UAV2emitter = 0

        if (self.sim_done or self.agents_done):
            point = Point(self.target.x, self.target.y)
            self.final_agent2centdist = np.mean(self.d_ic)
            agentmaxcontri2flux = np.argmax(self.agent_fluxes / self.get_total_VS_flux())
            self.dist_closestUAV2emitter = np.linalg.norm(
                self.agent_locs[agentmaxcontri2flux] - np.array([self.target.x, self.target.y]))
            self.avg_UAV2emitter = np.mean(
                np.linalg.norm(self.agent_locs - np.array([self.target.x, self.target.y]), axis=1))
            if not self.plume_found:
                # self.r_task += -5000
                self.final_cent_dist_to_emitter = self.current_cent_dist_to_emitter
                self.t_final = self.time_t

                self.xc = self.centroid[0]
                self.yc = self.centroid[1]
                self.dc = 0

                circle = Point(self.xc, self.yc).buffer(self.dc + self.tolerance)
                self.circ_boun_dist2emitter = point.distance(circle)

                if not self.drop:
                    self.at_drop_cent_dist_to_emitter = self.current_cent_dist_to_emitter
                    self.time_tdrop = self.time_t

        ##################################################################

        self.r_plume = []
        self.r_d = []
        self.r_theta = []
        self.r_col = []
        self.r_total = []
        self.r_task2 = []
        self.r_upwind = []
        self.r_time = []

        rew = {agent.id: self._calc_reward(agent) for agent in self.agents}

        # TODO: Add a custom RLlib callback (on_postprocess_trajectory) to delete those transitions before they even hit the training loop.
        # Force all rewards to zero if no detection has happened yet
        if not self.detection:
            rew = {agent_id: 0.0 for agent_id in rew}

        ##################################################################
        done = self._get_done()
        ##################################################################

        info = self._calc_info()

        self._past_obs = obs

        self.agent_rhos_tm1 = self.agent_rhos
        self.agent_locs_tm1 = self.agent_locs
        self.Vbar_tm1 = self.Vbar_t

        self.anchor_conc_tm1 = self.anchor_conc

        self.centroid_tm1 = self.centroid

        if any(self.agent_detects):
            self.detection = True

        if self.drop == True and self.hh == 5:
            #    quit()
            self.anchor_conc_5 = self.anchor_conc
            self.anchor_loc_5 = self.anchor_loc
            # self.anchor_conc = 0

        #    self.anchor_loc = np.zeros([0,0])
        #    self.h_max = 0

        return obs, rew, done, info

    def close(self):
        # Clean up resources if needed
        pass

    def cond1_check(self):
        return self.cond1met, self.time_t, self.timecond1

    def cond2_check(self):
        return self.cond2met, self.time_t, self.timecond2

    def _calc_info(self):
        """Provides info to calculate metric for scenario

        Returns:
            _type_: _description_
        """
        info = {}

        for agent in self.agents:
            obstacle_collision = 0
            agent_collision = 0

            if self.detection:
                for other_agent in self.agents:
                    if agent.id == other_agent.id:
                        continue
                    agent_collision += 1 if agent.collision_with(other_agent) else 0

                for obstacle in self.obstacles:
                    obstacle_collision += 1 if agent.collision_with(obstacle) else 0

            info[agent.id] = {
                "target_found": self.plume_found * 1,
                "final_cent_dist_to_emitter": self.final_cent_dist_to_emitter,
                "final_cent_dist_to_emitter_offset": self.final_cent_dist_to_emitter_offset,

                "time_tfinal": self.t_final,

                "drop_found": self.drop_found * 1,
                "time_tdrop": self.time_tdrop,

                "centroid_stable": int(self.centroid_stable),

                "at_drop_cent_dist_to_emitter": self.at_drop_cent_dist_to_emitter,

                "avg_agent_to_cent_dist": self.final_agent2centdist,

                "current_cent_dist_to_emitter": self.current_cent_dist_to_emitter,

                "dist_closestUAV2emitter": self.dist_closestUAV2emitter,

                "circ_boun_dist2emitter": self.circ_boun_dist2emitter,

                "avg_UAV2emitter": self.avg_UAV2emitter,

                "R_task": self.r_task2[agent.id],
                "R_plume": self.r_plume[agent.id],
                "R_upwind": self.r_upwind[agent.id],
                "R_time": self.r_time[agent.id],
                "R_col": self.r_col[agent.id],
                "R_d": self.r_d[agent.id],
                "R_theta": self.r_theta[agent.id],

                "R_total": self.r_total[agent.id],

                "agent_collision": agent_collision,
                "obstacle_collision": obstacle_collision,

                "agent_action": self._scale_action(np.array([agent.v, agent.w])),

                "detection": int(self.detection),
                "anchor_loc": self.anchor_loc.tolist(),
                "centroid_loc": self.centroid,
                "target_loc": [self.target.x, self.target.y],
                "offset_centroid_loc": self.offset_centroid.tolist(),
                "sum_flux": self.total_flux_VS_5[-1],
                "shift": [self.step_x, self.step_y],
            }

        return info

    def _calc_reward(self, agent):
        """Calculate rewards for each agent

        Args:
            agent (_type_): _description_

        Returns:
            _type_: _description_
        """
        reward = 0

        agent.done = False
        if (self.agents_done == True or self.sim_done):
            agent.done = True

        ################################################
        # (1) Distance Reward
        r_d = 0

        # d_ic = agent.rel_dist2(self.centroid[0],self.centroid[1])

        dic = self.d_ic[agent.id]

        # dbar = 0
        # diprime_c = np.zeros(self.num_agents)
        # for other_agent in self.agents:
        #    diprime_c[other_agent.id] = other_agent.rel_dist2(self.centroid[0],self.centroid[1])

        # dbar = np.mean(diprime_c)

        dbar = np.mean(abs(self.d_ic - self.id_dist2cent))

        ##sigma = 0
        # for other_agents in self.agents:
        #    sigma += np.sqrt(((diprime_c[other_agents.id] - dbar)**2)/self.num_agents)

        sigma = np.sqrt(np.sum((self.d_ic - self.id_dist2cent) ** 2) / self.num_agents)
        # sigma = np.sqrt(np.sum((abs(self.d_ic - self.id_dist2cent) - dbar)**2)/self.num_agents)

        # r_d = -(self.k1*(dic - self.id_dist2cent) + self.k2 * np.exp(((dic - self.id_dist2cent) - dbar)/sigma) - 1)

        # r_d = -(0.1*math.exp(-(dic - self.id_dist2cent)) + 0.1*math.exp(dic - self.id_dist2cent))

        r_d += -(self.k1 * abs(dic - self.id_dist2cent) + sigma)

        # r_d += -self.k1*abs(dic - self.id_dist2cent)

        ################################################
        # (2) Phase Angle Reward
        r_theta = 0

        # Skip if the polygon is a line (very small area)
        if self.lat_area < 1e-3:
            r_theta = 0

        else:
            a1 = self.route2.index(agent.id)

            if a1 == self.num_agents - 1:
                n1 = self.route[a1 + 1]
            else:
                n1 = self.route2[a1 + 1]

            neighbors = np.array([n1, self.route2[a1 - 1]])

            dij = agent.rel_dist(self.agents[neighbors[0]])
            dik = agent.rel_dist(self.agents[neighbors[1]])

            # TODO: 3 agents may build the polygon as a line so distance may return 0.
            '''
            ##########################
            denom1 = -2 * dic * self.d_ic[neighbors[0]]
            denom2 = -2 * dic * self.d_ic[neighbors[1]]

            # Check before division
            if dic == 0 or self.d_ic[neighbors[0]] == 0 or self.d_ic[neighbors[1]] == 0 or denom1 == 0 or denom2 == 0:
                print(f"\n[DEBUG] ❌ Invalid division detected for agent {agent.id}")
                print(f"[DEBUG] dic: {dic}")
                print(f"[DEBUG] d_ic[neighbor[0]]: {self.d_ic[neighbors[0]]}, d_ic[neighbor[1]]: {self.d_ic[neighbors[1]]}")
                print(f"[DEBUG] denom1: {denom1}, denom2: {denom2}")
                print(f"[DEBUG] neighbors: {neighbors}")
                print(f"[DEBUG] dij: {dij}, dik: {dik}")
            ##########################
            '''

            val1 = (dij ** 2 - dic ** 2 - (self.d_ic[neighbors[0]]) ** 2) / (-2 * dic * (self.d_ic[neighbors[0]]))
            val2 = (dik ** 2 - dic ** 2 - (self.d_ic[neighbors[1]]) ** 2) / (-2 * dic * (self.d_ic[neighbors[1]]))

            gamma_ij = math.acos(min(1, max(val1, -1)))
            gamma_ik = math.acos(min(1, max(val2, -1)))

            r_theta_1 = math.exp(-abs(gamma_ij - 2 * math.pi / self.num_agents)) + math.exp(
                -abs(gamma_ik - 2 * math.pi / self.num_agents)) - 2
            r_theta_2 = math.exp(-(abs(gamma_ik - gamma_ij))) - 1

            r_theta += self.k_theta1 * r_theta_1 + self.k_theta2 * r_theta_2

        ################################################
        # (3) penalty for collision with other agents
        r_col = 0
        if len(self.agents) > 2 and self.agent_collision_penalty:
            for a in self.agents:
                if (a.id != agent.id) and (agent.collision_with(a)):
                    r_col += -self.agent_penalty_weight

        #      penalty for collision with obstacles
        if self.obstacles and self.obstacle_penalty:
            for obstacle in self.obstacles:
                if agent.collision_with(obstacle):
                    r_col += -self.obstacle_penalty_weight

        #      penalty for agents selecting actions that will push them out of bounds
        if ((agent.x < 0) or (agent.x > self.env_width_x)) or ((agent.y < 0) or (agent.y > self.env_height_y)):
            r_col += -0.2

        r_plume = 0
        if self.anchor_conc != 0:

            r_plume += -self.beta_min * self.dist2anchor_locF[
                agent.id]  # /(2*self.agent_radius + self._constraint_slack)
        else:
            r_plume += -self.beta_min * self.max_distance  # /(2*self.agent_radius + self._constraint_slack)

        r_upwind = 0
        # if np.mean(self.dist2anchor) <= 2:
        # delta_p = self.agent_locs[agent.id] - self.agent_locs_tm1[agent.id]

        delta_p = self.agent_locs[agent.id] - self.anchor_loc

        # delta_c = self.centroid - self.centroid_tm1
        delta_p_mag = np.linalg.norm(delta_p)
        # delta_c_mag = np.linalg.norm(delta_c)
        Vbar_mag = np.linalg.norm(-self.Vbar_t)

        if ((delta_p_mag == 0) or (Vbar_mag == 0)):
            angle = 180
        else:
            inner = np.dot(delta_p, -self.Vbar_t) / (delta_p_mag * Vbar_mag)
            angle = math.acos(min(1, max(inner, -1))) * 180 / math.pi

        # if angle < 0:
        #    print('angle error for upwind')
        #    quit()
        if ((angle >= 0) and (angle <= 75) and (self.anchor_conc != 0)):
            # r_upwind += 2*self.beta_min*np.linalg.norm(delta_p) #abs(r_plume)/4  #1.05*np.linalg.norm(delta_p)
            r_upwind += -abs(r_plume) / 4  # 1.05*np.linalg.norm(delta_p)
        elif ((angle > 75) and (self.anchor_conc != 0)):
            r_upwind += -abs(r_plume) / 2  # abs(r_plume)/2 #-np.linalg.norm(delta_p)
            # r_upwind += -self.beta_min*np.linalg.norm(delta_p) #-abs(r_plume)/4  #-np.linalg.norm(delta_p)
        else:
            r_upwind += -abs(r_plume)

            # (4) add penalty to task reward if simulation time runs out and there is no identification

            # self.r_task += -20
            # self.r_task += -700
        # (5) penalty for timestep
        r_time = self.time_step_penalty

        # Total reward
        # weights = [0.01,0.5,0.025,100,0.1]
        # weights = [0.0005,0.0005,1,50,1,1]
        # weights = [0.zzzzz01,0.0001,1,10.5,0]

        # weights = [0.01,0.5,0.005,10,0.25]

        # weights = [0.01,0.01,0.00025,1,0.2*0.55]

        # weights = [0.01,0.5,0.08,5,0.02] - good for formation
        # weights = [0.0008,0.025,0.025,5,0.2,1] # rtask = 1

        # weights = [0.008,0.25,0.025,5,0.2,1]
        # weights = [0.008,
        #           0.25,
        #           0.025,
        #           5,
        #           0.2,
        #           0.005]

        # weights = [0.025,
        #           0.25,
        #           0.05,
        #           1,
        #           0.2,
        #           0.2]

        # weights = [0.025,
        #           0.25,
        #           0.025,
        #           1,
        #           0.2,
        #           0.2]

        # weights = [0.025*0.2,
        #           0.25*0.5,
        #           0.0001,
        #           1,
        #          0.2,
        #           0.2]
        ####################
        # weights = [0.01,
        #           0.25,
        #           0.01,
        #           1,
        #           0.2,
        #           0.2]
        ####################
        weights = [0.015,
                   0.3,
                   0.01,
                   1,
                   0.2,
                   0.2,
                   0]

        # weights = [0.025,
        #           0.5,
        #           0.01,
        #           1,
        #           0.2,
        #           0.2]

        # weights = [0.025,
        #           0.25,
        #           0.1,
        #           0.01,
        #           0.5,
        #           0.5]

        reward = weights[0] * r_d + weights[1] * r_theta + weights[2] * self.r_task + weights[3] * r_col + weights[
            4] * r_plume + weights[5] * r_upwind + weights[6] * r_time
        # reward = 0.01*r_d + 0.01*r_theta + 0.025*self.r_task + 0.01*(r_col) + 0.02*(r_plume) collision weights = 1.5 for obs and 1 for agents
        # print([reward,self.time_t])

        self.r_d.append(weights[0] * r_d)
        self.r_theta.append(weights[1] * r_theta)
        self.r_task2.append(weights[2] * self.r_task)
        self.r_col.append(weights[3] * r_col)
        self.r_plume.append(weights[4] * r_plume)
        self.r_upwind.append(weights[5] * r_upwind)
        self.r_time.append(weights[6] * r_time)
        self.r_total.append(reward)

        # reward = 0.02*(r_d + r_theta) + 0.05*self.r_task + 0.02*(r_col) + 0.04*(r_plume)

        # self.Rdist["a%d" % (agent.id)].append(r_d)
        # self.Rangle["a%d" % (agent.id)].append(r_theta)
        # self.Rcol["a%d" % (agent.id)].append(r_col)
        # self.Rtask["a%d" % (agent.id)].append(r_task)
        # self.Rplume["a%d" % (agent.id)].append(r_plume)
        # self.Rtotal["a%d" % (agent.id)].append(reward)

        # if self.anchor_conc == 0:
        #    reward = -200

        # small reward for time
        # TODO: fix this if want to enable
        # reward += -self.time_step_penalty * self.time_step

        return reward

    '''
    def _get_done(self):
        """The simulation is done if the agents return a positive flux over some time tau or if the simulation time ends 


        # https://github.com/ray-project/ray/issues/10761
        # reporting done multiple times for agent will cause error.
        Returns:
                [type]: [description]
        """
        # Done when:
        #   simulation time ends 
        #   the mass flux divergence 
        done = {agent.id: agent.done for agent in self.agents}

        agent_done = [agent.done for agent in self.agents]

        done["__all__"] = (
            #self.plume_found
            self.time_elapse >= self.max_time - self.time_step
            or all(agent_done)
        )

        return done
    '''

    def _get_done(self):
        """The simulation is done if:
        - drop_found (self.drop) is True
        - or centroid is stable (self.centroid_stable)
        - or simulation time has ended (self.sim_done)
        """
        # Set per-agent done flag
        # done = {agent.id: (self.drop or self.centroid_stable or self.sim_done) for agent in self.agents}

        # Global done flag
        # done["__all__"] = self.drop or self.centroid_stable or self.sim_done

        # Set per-agent done flag
        done = {agent.id: (self.centroid_stable or self.sim_done) for agent in self.agents}

        # Global done flag
        # done["__all__"] = self.drop or self.centroid_stable or self.sim_done
        done["__all__"] = self.centroid_stable or self.sim_done

        # Set per-agent done flag
        #done = {agent.id: self.sim_done for agent in self.agents}

        # Global done flag
        # done["__all__"] = self.drop or self.centroid_stable or self.sim_done
        #done["__all__"] = self.sim_done


        return done

    # def _store_plume_det(self):
    #    b_t = []
    #    x = []
    #    F = []
    # for agent in self.agents:
    #    [det,ff ] = agent.sensed_plume(agent,self.rho2,self.ws2,self.flux_threshold)
    #    x.append(det)
    #    F.append(ff)
    # x = np.array(x)
    # self.detection.append(np.any(x))
    # if not self.sim_done and np.any(x):
    #    J = np.argmax(F)
    #    other_agent = self.agents[J]
    #    for agent in self.agents:
    #        b_t.extend([agent.rel_dist(other_agent)])
    #        b_t.extend([agent.rel_bearing_error(other_agent)])
    #        b_t.extend([agent.rel_bearing_entity_error(other_agent)])

    #    self.bufferTable = np.concatenate((self.bufferTable,np.array([b_t])),axis=0)
    # else:
    #    b_t = np.zeros(self.num_agents * 2)
    #    self.bufferTable = np.concatenate((self.bufferTable,np.array([b_t])),axis=0)

    def _store_plume_data(self):
        b_t = []

        if self.drop == False:
            self.hh == 5
            rho = self.rho5
            wind = self.ws5
        else:
            self.hh = 2
            rho = self.rho2
            wind = self.ws2

        for agent in self.agents:
            rho_val = agent.concentration(rho)
            wind_val = agent.wind_coor(wind)

            b_t.extend([rho_val])
            b_t.extend(wind_val)

        if self.hh == 5:
            self.bufferTable_data_5 = np.concatenate((self.bufferTable_data_5, np.array([b_t])), axis=0)
        else:
            self.bufferTable_data_2 = np.concatenate((self.bufferTable_data_2, np.array([b_t])), axis=0)
        # else:
        #    b_t = np.zeros(self.num_agents * self.num_obs_other_agent_mea_states)
        #    self.bufferTable_data = np.concatenate((self.bufferTable_data,np.array([b_t])),axis=0)

    # normalize the obs space
    def _calc_obs(self, agent, norm=True):
        ##########################################################################################################
        self.other_agent_data_recorded[agent.id] = False

        ##########################################################################################################
        # (1) Plume Measurement States for agent 'i'
        plume_measurement_states = []
        main_agent_meas_states = self.agent_states[agent.id]
        # self.mean_tabledata_all_agents[self.indx_data[agent.id]].tolist()
        # [det, ff, pp] = agent.get_detect(main_agent_meas_states, self.flux_threshold,self.centroid)
        # self.det[agent.id].extend([det])
        # self.F[agent.id].extend([ff])

        # self.agent_detects[agent.id] = det
        # self.agent_fluxes[agent.id] = ff
        # self.agent_rhos[agent.id] = pp

        plume_measurement_states.extend(main_agent_meas_states)
        plume_measurement_states.extend([self.agent_detects[agent.id] * 1])
        plume_measurement_states.extend([self.hh])

        self.dta[agent.id] = plume_measurement_states

        ##########################################################################################################
        # (2) Formation States relative to agent 'i'
        form_obs_states = []

        self.d_ic[agent.id] = agent.rel_dist2(self.centroid[0], self.centroid[1])

        form_obs_states.append(self.d_ic[agent.id].tolist())
        form_obs_states.append(agent.rel_bearing_error2(self.centroid[0], self.centroid[1]))

        ##########################################################################################################
        # (3) Distance/Angular relative information of other agents
        other_agent_states = []
        for other_agent in self.agents:

            # skip the agent were evaluating observation for
            if agent.id != other_agent.id and agent.sensed(other_agent):
                other_agent_rel_state = []

                # distance to other agent
                other_agent_rel_state.append(agent.rel_dist(other_agent))

                # bearing to other agent
                other_agent_rel_state.append(agent.rel_bearing_error(other_agent))

                # relative bearing of other agent to current agent
                other_agent_rel_state.append(
                    agent.rel_bearing_entity_error(other_agent)
                )

                other_agent_states.append(other_agent_rel_state)

        # sort the list by dist closest to agent by this method:
        # https://www.geeksforgeeks.org/python-program-to-sort-a-list-of-tuples-by-second-item/
        other_agent_states = sorted(other_agent_states, key=lambda x: x[0])

        # don't include the main agent
        num_inactive_other_agents = self.num_agents - len(other_agent_states) - 1

        last_other_agent_state = [
            # int(AgentType.P),
            # int(False),  # can't sense this agent
            0,  # relative distance
            0,  # relative bearing
            0,  # relative bearing of the other agent
        ]

        # flatten the list of other agent states
        other_agent_states = [
            state for sublist in other_agent_states for state in sublist
        ]

        ##########################################################################################################
        # (4) Measurement information of other agents
        self.other_agent_data_recorded[agent.id] = True

        # if (not self.sim_done) and (self.add_com_delay) and (self.time_t - self.com_delay >= 0):

        other_agent_measurement_states = []
        # other_agent_measurement_states = self.bufferTable_data[self.time_t - self.com_delay][np.delete(self.indx_data,agent.id,0)].tolist()
        # other_agent_plume_measurement_states = mean_tabledata_all_agents[np.delete(self.indx_data,agent.id,0)].tolist()

        for other_agent in self.agents:
            if other_agent.id != agent.id:
                other_agent_plume_measurement_states = []
                data = self.agent_states[other_agent.id]
                # self.mean_tabledata_all_agents[self.indx_data[other_agent.id]].tolist()
                # [det, ff, pp] = other_agent.get_detect(data, self.flux_threshold,self.centroid)
                # self.agent_detects[other_agent.id] = det
                # self.agent_fluxes[other_agent.id] = ff
                # self.agent_rhos[other_agent.id] = pp

                other_agent_plume_measurement_states.extend(data)
                other_agent_plume_measurement_states.extend([self.agent_detects[other_agent.id] * 1])
                other_agent_plume_measurement_states.extend([self.hh])

                other_agent_measurement_states.append(other_agent_plume_measurement_states)

        # print(other_agent_measurement_states)
        # print(self.agent_detects)
        # print(self.agent_fluxes)

        other_agent_measurement_states = sorted(other_agent_measurement_states, key=lambda x: x[0])

        num_inactive_other_agents2 = self.num_agents - len(other_agent_measurement_states) - 1

        last_other_agent_state2 = [
            # int(AgentType.P),
            # int(False),  # can't sense this agent
            0,  #
            0,  #
            0,  #
            0,
            0,
        ]

        # flatten the list of other agent states
        other_agent_measurement_states = [
            state for sublist in other_agent_measurement_states for state in sublist
        ]

        ##########################################################################################################
        last_loc_plume_is_det_states = []
        # self.dist2anchor_locF[agent.id] = self.max_distance
        if self.anchor_conc != 0:
            # if np.any(self.agent_detects):
            last_loc_plume_states = []
            # which_agents_det = np.where(self.agent_detects)[0]
            # FF = self.agent_fluxes[which_agents_det]
            # aa = which_agents_det[np.where(np.array(abs(FF)) == max(abs(FF)))[0]]
            # PP = self.agent_rhos[which_agents_det]
            # aa = which_agents_det[np.where(np.array(PP) == max(PP))[0]]

            # agentj_max_flux = aa[0]
            # if len(aa) > 1:
            #    agentj_max_flux = aa[0] #np.random.choice(aa)

            # if self.agent_rhos[self.agent_max_conc] > self.anchor_conc:

            #    self.anchor_conc = self.agent_rhos[agent_max_conc]
            #    self.anchor_loc = np.array([self.agents[self.agentj_max_flux].x,self.agents[self.agentj_max_flux].y])

            #    if agentj_max_flux == agent.id and self.add_com_delay == False:
            #        dist2loc_with_max_flux = 0
            #        bearing2loc_with_max_flux = 0
            #    else:
            #        dist2loc_with_max_flux = agent.rel_dist(self.agents[agentj_max_flux])
            #        bearing2loc_with_max_flux = agent.rel_bearing_error(self.agents[agentj_max_flux])
            # else:
            #    dist2loc_with_max_flux = agent.rel_dist2(self.anchor_loc[0],self.anchor_loc[1])
            #    bearing2loc_with_max_flux = agent.rel_bearing_error2(self.anchor_loc[0],self.anchor_loc[1])

            # self.dist2anchor_locF[agent.id] = dist2loc_with_max_flux

            # last_loc_plume_states.append(dist2loc_with_max_flux)
            # last_loc_plume_states.append(bearing2loc_with_max_flux)
            # last_loc_plume_states.append(self.hh)

            # last_loc_plume_is_det_states.append(last_loc_plume_states)

            # elif self.anchor_conc != 0:
            dist2loc_with_max_flux = agent.rel_dist2(self.anchor_loc[0], self.anchor_loc[1])
            bearing2loc_with_max_flux = agent.rel_bearing_error2(self.anchor_loc[0], self.anchor_loc[1])
            h_max = self.h_max

            self.dist2anchor_locF[agent.id] = dist2loc_with_max_flux

            last_loc_plume_states.append(dist2loc_with_max_flux)
            last_loc_plume_states.append(bearing2loc_with_max_flux)
            last_loc_plume_states.append(h_max)

            last_loc_plume_is_det_states.append(last_loc_plume_states)

        last_loc_plume_is_det_states = sorted(last_loc_plume_is_det_states, key=lambda x: x[0])

        num_inactive_plume_states = 1 - len(last_loc_plume_is_det_states)

        last_meas_plume_state = [
            0,  # relative distance to location with max plume reading
            0,  # relative bearing to location with max plume reading
            0,
        ]

        # flatten list of obstacle states
        last_loc_plume_is_det_states = [state for sublist in last_loc_plume_is_det_states for state in sublist]

        ##########################################################################################################
        obstacle_states = []
        for obstacle in self.obstacles:
            if agent.sensed(obstacle):
                obstacle_rel_state = []
                ob_dist = agent.rel_dist(obstacle)
                ob_dist = np.clip(ob_dist, 0, self.max_distance)
                obstacle_rel_state.append(ob_dist)
                obstacle_rel_state.append(agent.rel_bearing_error(obstacle))

                # add the obstacle to the obstacle list
                obstacle_states.append(obstacle_rel_state)

        obstacle_states = sorted(obstacle_states, key=lambda x: x[0])

        num_inactive_obstacles = self.max_num_obstacles - len(obstacle_states)

        last_obstacle_state = [
            # int(AgentType.O),  # obstacle type
            # int(False),  # obstacle sensed or not
            0,  # relative distance to agent
            0,  # relative bearing to agent
        ]

        # flatten list of obstacle states
        obstacle_states = [state for sublist in obstacle_states for state in sublist]

        ##########################################################################################################
        obs = np.array(
            [
                *agent.state,

                *plume_measurement_states,

                *form_obs_states,

                *other_agent_states,
                *(last_other_agent_state * num_inactive_other_agents),

                *other_agent_measurement_states,
                *(last_other_agent_state2 * num_inactive_other_agents2),

                *last_loc_plume_is_det_states,
                *(last_meas_plume_state * num_inactive_plume_states),

                *obstacle_states,
                *(last_obstacle_state * num_inactive_obstacles),
            ],
            dtype=np.float32,
        )

        raw_obs = np.copy(obs)
        if norm:
            obs = self.norm_obs_space(obs)

        agent_constraint = self._get_agent_constraint(agent)
        action_g, action_h, action_r = self.calc_projected_states(agent)
        action_a = self._scale_action(self.cum_a_bar[agent.id])

        obs = obs.astype(np.float32)

        obs_dict = {
            "observations": obs.astype(np.float32),
            "raw_obs": raw_obs.astype(np.float32),
            "constraints": agent_constraint.astype(np.float32),
            "action_g": action_g.astype(np.float32),
            "action_h": action_h.astype(np.float32),
            "action_r": action_r.astype(np.float32),

        }

        return obs_dict

    ######################################################################################################################################################################
    def norm_obs_space(self, obs):

        """"""

        # https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
        # x_{normalized} = (b-a)\frac{x - min(x)}{max(x) - min(x)} + a

        # print(obs)
        # print(len(obs))
        # quit()

        # xx1 = len(self.norm_high_obs)
        # xx2 = len(self.norm_low_obs)
        # xx3 = len(self.low_obs)
        # xx4 = len(self.high_obs)
        # xx5 = len(obs)

        # if xx1 != 44 or xx2 != 44 or xx3 != 44 or xx4 != 44 or xx5 != 44:
        #    print(xx1)
        #    print(xx2)
        #    print(xx3)
        #    print(xx4)
        #    print(xx5)
        #    print(obs)
        #    quit()

        norm_orbs = (self.norm_high_obs - self.norm_low_obs) * (
                (obs - self.low_obs) / (self.high_obs - self.low_obs)
        ) + self.norm_low_obs

        # if (self.high_obs - self.low_obs).any():
        #    print(self.high_obs)
        #    print(self.low_obs)
        #    print(obs)
        #    quit()

        return norm_orbs

    def _scale_action(self, action):
        """Scale agent action between default norm action values"""
        # assert np.all(np.greater_equal(action, self.low)), (action, self.low)
        # assert np.all(np.less_equal(action, self.high)), (action, self.high)
        action = (self.norm_high - self.norm_low) * (
                (action - self.low) / (self.high - self.low)
        ) + self.norm_low

        return action

    def load_safety_layer(self):
        print("using safety layer")
        self.safety_layer = SafetyLayer(self)
        self.safety_layer.load_layer(self.config.get("sl_checkpoint_dir", None))
        self.corrective_action = self.safety_layer.get_safe_actions
        # if self.safety_layer_type == "hard":
        #     self.corrective_action = self.safety_layer.get_hard_safe_action
        # elif self.safety_layer_type == "soft":
        #     self.corrective_action = self.safety_layer.get_soft_safe_action
        # else:
        #     raise ValueError("Unknown safety layer type")

    def seed(self, seed=None):
        """Random value to seed"""
        np.random.seed(seed)

        seed = seeding.np_random(seed)
        return [seed]

    def repulsive_action(self, agent):
        """https://link.springer.com/article/10.1007/s10514-020-09945-6"""
        # kr = 10
        # # ka should be 10
        # ka = 10 * kr
        kr = 220
        ka = 200

        # target = self.evaders[0] if agent.type == AgentType.P else self.target

        # dist_to_target = agent.rel_dist(target) + 0.001
        # agent_q = np.array([agent.x, agent.y])
        # target_q = np.array([target.x, target.y])

        # target_q_star = 1 * (target.radius + agent.radius)
        # if dist_to_target <= target_q_star:
        #    des_v = -ka * (agent_q - target_q)
        # else:
        #    des_v = (
        #        -ka
        #        * (1 / dist_to_target**self.evader_alpha)
        #        * ((agent_q - target_q) / dist_to_target)
        #    )

        # agent agents potential
        # only use when close to obstacle
        for other_agent in self.agents:
            if agent.type2 == AgentType.P and agent.id == other_agent.id:
                continue
            other_agent_q_star = 5 * agent.radius

            dist_to_other_agent = agent.rel_dist(other_agent)
            other_agent_q = np.array([other_agent.x, other_agent.y])

            if dist_to_other_agent <= other_agent_q_star:
                des_v += (
                        kr
                        * ((1 / dist_to_other_agent) - (1 / other_agent_q_star))
                        * (1 / dist_to_other_agent ** self.evader_alpha)
                        * ((agent_q - other_agent_q) / dist_to_other_agent)
                )

        # if agent.type == AgentType.E:
        #    for other_evader in self.evaders:
        #        other_evader_q_star = 5 * agent.radius
        #        if other_evader.id == agent.id:
        #            continue

        #        dist_to_other_evader = agent.rel_dist(other_evader)
        #        other_evader_q = np.array([other_evader.x, other_evader.y])

        #        if dist_to_other_evader <= other_evader_q_star:
        #            des_v += (
        #                kr
        #                * ((1 / dist_to_other_evader) - (1 / other_evader_q_star))
        #                * (1 / dist_to_other_evader**self.evader_alpha)
        #                * ((agent_q - other_evader_q) / dist_to_other_evader)
        #            )

        for obstacle in self.obstacles:
            dist_to_obstacle = agent.rel_dist(obstacle) + 0.001
            obstacle_q = np.array([obstacle.x, obstacle.y])
            obstacle_q_star = 5 * (agent.radius + obstacle.radius)

            if dist_to_obstacle <= obstacle_q_star:
                des_v += (
                        kr
                        * ((1 / dist_to_obstacle) - (1 / obstacle_q_star))
                        * (1 / dist_to_obstacle ** self.evader_alpha)
                        * ((agent_q - obstacle_q) / dist_to_obstacle)
                )

        des_v = self.agent_v_max * des_v
        dxu = self.si_uni_dyn(agent, des_v)

        return dxu

    @staticmethod
    def uni_to_si_dyn(agent, dxu, projection_distance=0.05):
        """
        See:
        https://github.com/robotarium/robotarium_python_simulator/blob/master/rps/utilities/transformations.py

        """
        cs = np.cos(agent.theta)
        ss = np.sin(agent.theta)

        dxi = np.zeros(
            2,
        )
        dxi[0] = cs * dxu[0] - projection_distance * ss * dxu[1]
        dxi[1] = ss * dxu[0] + projection_distance * cs * dxu[1]

        return dxi

    # TODO: projection_distance=.01
    def si_uni_dyn(self, agent, si_v, projection_distance=0.05):
        """
        see:
        https://github.com/robotarium/robotarium_python_simulator/blob/master/rps/utilities/transformations.py
        also:
            https://arxiv.org/pdf/1802.07199.pdf

        Args:
            agent ([type]): [description]
            si_v ([type]): [description]
            projection_distance (float, optional): [description]. Defaults to 0.05.

        Returns:
            [type]: [description]
        """
        cs = np.cos(agent.theta)
        ss = np.sin(agent.theta)

        dxu = np.zeros(
            2,
        )
        dxu[0] = cs * si_v[0] + ss * si_v[1]
        dxu[1] = (1 / projection_distance) * (-ss * si_v[0] + cs * si_v[1])

        dxu = np.clip(
            dxu,
            [self.agent_v_min, self.agent_w_min],
            [self.agent_v_max, self.agent_w_max],
        )

        return dxu

    def get_rl(self, theta, projection_distance=0.05):
        cs = np.cos(theta)
        ss = np.sin(theta)
        rl_array = np.array(
            [[cs, -projection_distance * ss], [ss, projection_distance * cs]]
        )

        return rl_array

    def _unscale_obs(self, obs):
        """[summary]

        Args:
            action ([type]): [description]

        Returns:
            [type]: [description]
        """
        # print("action: ", action)
        # print("action type:", type(action))
        # unnormalize the action
        assert np.all(np.greater_equal(obs, self.norm_low_obs)), (
            obs,
            self.norm_low_obs,
        )
        assert np.all(np.less_equal(obs, self.norm_high_obs)), (obs, self.norm_high_obs)
        obs = self.low_obs + (self.high_obs - self.low_obs) * (
                (obs - self.norm_low_obs) / (self.norm_high_obs - self.norm_low_obs)
        )
        # obs = np.clip(action, self.low, self.high)

        return obs

    def _unscale_action(self, action):
        """[summary]

        Args:
            action ([type]): [description]

        Returns:
            [type]: [description]
        """
        assert np.all(np.greater_equal(action, self.norm_low)), (action, self.norm_low)
        assert np.all(np.less_equal(action, self.norm_high)), (action, self.norm_high)
        action = self.low + (self.high - self.low) * (
                (action - self.norm_low) / (self.norm_high - self.norm_low)
        )
        # TODO: this is not needed
        action = np.clip(action, self.low, self.high)

        return action

    def calc_projected_states(self, agent):

        A = np.eye(2)
        G = []
        h = []

        # target
        # G.append(-np.dot(2 * (agent.pos - self.target.pos), A))
        # h.append(
        #    self._constraint_k
        #    * (
        #        np.linalg.norm(agent.pos - self.target.pos) ** 2
        #        - (agent.radius + self.target.radius + self._constraint_slack) ** 2
        #    )
        # )

        # other agents
        for other_agent in self.agents:
            if agent.id != other_agent.id:
                # z_other_agent = np.array([other_agent.x, other_agent.y])
                G.append(-np.dot(2 * (agent.pos - other_agent.pos), A))
                h.append(
                    self._constraint_k
                    * (
                            np.linalg.norm(agent.pos - other_agent.pos) ** 2
                            - (agent.radius + other_agent.radius + self._constraint_slack)
                            ** 2
                    )
                )

        # obstacles
        for obstacle in self.obstacles:
            # z_obstacle = np.array([obstacle.x, obstacle.y])
            G.append(-np.dot(2 * (agent.pos - obstacle.pos), A))
            h.append(
                self._constraint_k
                * (
                        np.linalg.norm(agent.pos - obstacle.pos) ** 2
                        - (agent.radius + obstacle.radius + self._constraint_slack) ** 2
                )
            )

        G = np.array(G)
        h = np.array(h)

        return G, h, self.get_rl(agent.theta)

    def safety_action_layer(self, agent, a_uni_rl):
        a_si_rl = agent.uni_to_si_dyn(a_uni_rl)

        A = np.dot(np.array([1, 1]), np.eye(2))
        P = np.eye(2)
        q = -np.dot(P.T, a_si_rl)

        G = []
        h = []

        # if agent.sensed(self.target):
        #    G.append(A)
        #    h.append(
        #        -(agent.radius + self.target.radius + self._constraint_slack)
        #        + np.linalg.norm(agent.pos - self.target.pos)
        #    )

        # other agents
        for other_agent in self.agents:
            if agent.id != other_agent.id and agent.sensed(other_agent):
                G.append(A)
                h.append(
                    -(agent.radius + other_agent.radius + self._constraint_slack)
                    + np.linalg.norm(agent.pos - other_agent.pos)
                )

        # obstacles
        for obstacle in self.obstacles:
            if agent.sensed(obstacle):
                G.append(A)
                h.append(
                    -(agent.radius + obstacle.radius + self._constraint_slack)
                    + np.linalg.norm(agent.pos - obstacle.pos)
                )
        G = np.array(G)
        h = np.array(h)
        if G.any() and h.any():
            try:
                a_si_qp = solve_qp(
                    P.astype(np.float64),
                    q.astype(np.float64),
                    G.astype(np.float64),
                    h.astype(np.float64),
                    None,
                    None,
                    None,
                    None,
                    solver="quadprog",
                )
            except Exception as e:
                print(f"error running solver: {e}")
                # just return 0 if infeasible action
                return a_uni_rl
        else:
            return a_uni_rl
            # a_si_qp = q

        # just return 0
        if a_si_qp is None:
            print("infeasible solver")
            return a_uni_rl
            # a_si_qp = q

        # convert to unicycle
        a_uni_qp = agent.si_to_uni_dyn(a_si_qp)
        # a_uni_qp = a_si_qp

        # a_uni_qp = np.clip(a_uni_rl + a_uni_qp, self.low, self.high)

        return a_uni_qp

    def proj_safe_action(self, agent, a_uni_rl):
        """_summary_

        Args:
            agent (_type_): _description_

        Returns:
            _type_: _description_
        """
        # a_si_rl = a_uni_rl
        a_si_rl = agent.uni_to_si_dyn(a_uni_rl)

        # A = np.dot(np.eye(2), self.get_rl(agent.theta))
        A = np.eye(2)

        P = np.eye(2)
        q = np.zeros(2)
        G = []
        h = []

        ## target
        # if agent.sensed(self.target):
        #    G.append(-np.dot(2 * (agent.pos - self.target.pos), A))
        #    h.append(
        #        self._constraint_k
        #        * (
        #            np.linalg.norm(agent.pos - self.target.pos) ** 2
        #            - (agent.radius + self.target.radius + self._constraint_slack) ** 2
        #        )
        #        + np.dot(2 * (agent.pos - self.target.pos), np.dot(A, a_si_rl))
        #    )

        # other agents
        for other_agent in self.agents:
            if agent.id != other_agent.id and agent.sensed(other_agent):
                # z_other_agent = np.array([other_agent.x, other_agent.y])
                G.append(-np.dot(2 * (agent.pos - other_agent.pos), A))
                h.append(
                    self._constraint_k
                    * (
                            np.linalg.norm(agent.pos - other_agent.pos) ** 2
                            - (agent.radius + other_agent.radius + self._constraint_slack)
                            ** 2
                    )
                    + np.dot(2 * (agent.pos - other_agent.pos), np.dot(A, a_si_rl))
                )

        # obstacles
        for obstacle in self.obstacles:
            # z_obstacle = np.array([obstacle.x, obstacle.y])
            if agent.sensed(obstacle):
                G.append(-np.dot(2 * (agent.pos - obstacle.pos), A))
                h.append(
                    self._constraint_k
                    * (
                            np.linalg.norm(agent.pos - obstacle.pos) ** 2
                            - (agent.radius + obstacle.radius + self._constraint_slack) ** 2
                    )
                    + np.dot(2 * (agent.pos - obstacle.pos), np.dot(A, a_si_rl))
                )

        G = np.array(G)
        h = np.array(h)

        if G.any() and h.any():
            try:
                a_si_qp = solve_qp(
                    P.astype(np.float64),
                    q.astype(np.float64),
                    G.astype(np.float64),
                    h.astype(np.float64),
                    None,
                    None,
                    None,
                    None,
                    solver="quadprog",
                )
            except Exception as e:
                print(f"error running solver: {e}")
                # just return 0 if infeasible action
                a_si_qp = q
        else:
            return q
            # a_si_qp = q

        # just return 0
        if a_si_qp is None:
            print("infeasible solver")
            return q
            # a_si_qp = q

        # convert to unicycle
        a_uni_qp = agent.si_to_uni_dyn(a_si_qp)
        # a_uni_qp = a_si_qp

        # a_uni_qp = np.clip(a_uni_rl + a_uni_qp, self.low, self.high)

        return a_uni_qp

    def _agent_step(self, agent, action=None, unscale_action=True):
        """
        Handles how any agent moves in the environment. Agent can be agent or evader.

        Args:
            agent (_type_): _description_
            action (_type_, optional): _description_. Defaults to None.
        """
        if agent.done or agent.move_type == "static":
            action = np.array([0, 0])

        elif agent.move_type == "rl":
            # action = self._unscale_action(action)
            pass

        elif agent.move_type == "repulsive":
            unscale_action = False
            action = self.repulsive_action(agent)

        elif agent.move_type == "random":
            action = self.action_space[0].sample()

        # elif agent.move_type == "go_to_goal":
        #    unscale_action = False
        #    action = self.go_to_goal(agent)

        if unscale_action:
            action = self._unscale_action(action)
        agent.step(action)

    def get_constraints(self):
        return {agent.id: self._get_agent_constraint(agent) for agent in self.agents}

    def get_constraint_margin(self):
        return {
            agent.id: self._get_agent_constraint_margin(agent) for agent in self.agents
        }

    def _get_agent_constraint_margin(self, agent):
        constraint_margin = []

        constraint_margin.extend(
            [
                agent.radius + other_agent.radius
                for other_agent in self.agents
                if other_agent.id != agent.id
            ]
        )

        constraint_margin.extend([agent.radius + ob.radius for ob in self.obstacles])

        constraint_margin = np.array(constraint_margin)

        norm_high_c = np.array([1] * self.num_constraints)
        norm_low_c = np.array([-1] * self.num_constraints)

        high_c = np.array([2 * self.obstacles.radius] * self.num_constraints)
        low_c = np.array([0] * self.num_constraints)

        norm_c = (norm_high_c - norm_low_c) * (
                (constraint_margin - low_c) / (high_c - low_c)
        ) + norm_low_c

        return constraint_margin
        # return norm_c

    def _get_agent_constraint(self, agent):
        """Return simulation constraints"""
        constraints = []

        # agent collision with other agents
        constraints.extend(
            [
                (agent.radius + other_agent.radius + self._constraint_slack)
                - agent.rel_dist(other_agent)
                for other_agent in self.agents
                if other_agent.id != agent.id
            ]
        )

        # agent collision with obstacles
        constraints.extend(
            [
                (agent.radius + ob.radius + self._constraint_slack) - agent.rel_dist(ob)
                for ob in self.obstacles
            ]
        )

        return np.array(constraints)
        # return self._norm_constraints(np.array(constraints))

    def _norm_constraints(self, constraints):
        """Normalize constraints between -1 and 1
        # https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
        # x_{normalized} = (b-a)\frac{x - min(x)}{max(x) - min(x)} + a
        """

        norm_high_c = np.array([1] * self.num_constraints)
        norm_low_c = np.array([-1] * self.num_constraints)

        high_c = np.array([self.max_distance] * self.num_constraints)
        low_c = np.array([0] * self.num_constraints)

        norm_c = (norm_high_c - norm_low_c) * (
                (constraints - low_c) / (high_c - low_c)
        ) + norm_low_c

        return norm_c