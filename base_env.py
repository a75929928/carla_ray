"""
This is a sample carla environment. 
Only change env import
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gymnasium as gym
# import gym
# from gym.utils import seeding
# import sys
# sys.path.append("D:/Code/Ray_RLLib/env")
from core.CarlaCore_multi import CarlaCore
import time

# add to align with gymnasium API
from copy import copy
from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type

import random
import math
import carla
import numpy as np
from gymnasium import spaces
# from gym.spaces import Discrete
from helper.CarlaHelper import post_process_image
import time
from typing import Union, Dict, AnyStr, Optional, Tuple

# add route planning
import sys
sys.path.append('D:/Code/carla/CARLA_0.9.15/PythonAPI/carla') # tweak to where you put carla
from agents.navigation.global_route_planner import GlobalRoutePlanner

'''
    self.hero -> all heroes in {id: <carla.hero>} 
    _hero in function -> certain hero <carla.hero>
'''
from params import *

# from ray.rllib.env.multi_agent_env import MultiAgentEnv
# class BaseEnv(MultiAgentEnv):
class BaseEnv(gym.Env):

    def __init__(self, config):

        self.environment_config = config
        self.experiment_config = config["EXPERIMENT_CONFIG"]
        
        # ---- TODO need to be modified ---- 
        # self.observation[hero_id]["camera"] = [] # dict->dict how to initialize every dict inside?

        # Prameters
        self.max_idle = 40 # seconds
        self.max_ep_time = 120 # seconds
        self.speed_limit = 30 # kmh
        self.hero_model = ''.join(self.experiment_config["hero_vehicle_model"]) # TODO differ hero models

        self.allow_respawn = self.experiment_config["allow_respawn"]

        # ---- Unique variable ---- 
        self.spectator = None # spectator randomly choose hero

        # ---- Multi-agent part ----
        self.hero = {} # set default agent to avoid space error
        self.observation = {}
        self.route = {}
        self.action = {}
        
        self.spawn_point_list = []
        self.vehicle_list = []
        self.start_location = {}
        
        self.t_idle_start = {}
        self.t_ep_start = {}

        self.current_w = {}

        self.total_step = 0

        self.set_observation_space()
        self.set_action_space()
        # self.reset() # reset would be used when creating experiment

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        
        self.core = CarlaCore(self.environment_config, self.experiment_config)
        self.world = self.core.get_core_world()
        # CarlaCore.spawn_npcs(self.core, self.experiment_config["n_vehicles"],self.experiment_config["n_walkers"], hybrid = True)
        self.core.spawn_npcs(self.experiment_config["n_vehicles"],self.experiment_config["n_walkers"], hybrid=True)
        self.map = self.world.get_map()
        self.core.reset_sensors(self.experiment_config)

        # autopilot means to use Carla Expert but here is just decoration XD
        self.spawn_hero(self.world, autopilot=self.experiment_config["Autodrive_enabled"]) 
        self.core.setup_sensors(
            self.experiment_config,
            self.hero,
            self.world.get_settings().synchronous_mode,
        )

        self.initial_location = self.get_hero_location()

        self.initialize_reward(self.core)
        self.set_server_view(self.core)
        self.experiment_tick(self.core, self.world, action_dict=None)
        obs, infos = self.get_observation(self.core)
        obs = self.process_observation(self.core, obs)
        return obs, infos

    def step(self, action_dict):
        # assert action_dict in [0, 13], action_dict
        self.experiment_tick(self.core, self.world, action_dict)
        obs, infos = self.get_observation(self.core)
        # Observation is consisted of multiple sensor data including Collision/laneInvasion/Location
        processed_obs = self.process_observation(self.core, obs)
        rewards = self.compute_reward(self.core, obs, self.map)
        terminateds, need_respawn = self.get_done_status()
        
        if need_respawn and self.allow_respawn: 
            self.respawn_hero(self.core.world, False, terminateds)
        
        truncateds = copy(terminateds)
        for hero_id in truncateds:
            truncateds[hero_id] = False
        
        """
        # Just add truncateds to align with new gymnasium API
        self.total_step += 1
        truncateds = (self.total_step >= self.environment_config['horizon'])
        
        # Take single float and bollean as 'done' output
        reward = sum(reward_dict.values())
        terminateds = terminateds_dict["__all__"]
        """

        # return obs, rewards, terminateds, truncateds, infos
        return processed_obs, rewards, terminateds, truncateds, infos

#    def seed(self, seed=None):
#        self.np_random, seed = seeding.np_random(seed)
#        return [seed]
    
    def get_experiment_config(self):

        return self.experiment_config

    def set_observation_space(self):

        """
        observation_space_option: Camera Image
        :return: observation space:
        """
        raise NotImplementedError

    def get_observation_space(self):

        """
        :return: observation space
        """
        return self.observation_space

    def set_action_space(self):

        """
        :return: None. In this experiment, it is a discrete space
        """
        # self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))
        return NotImplementedError
    
    def get_action_space(self):

        """
        :return: action_space. In this experiment, it is a discrete space
        """
        return self.action_space
    
    def set_server_view(self, core):

        """
        Set server view to be behind the hero
        :param core:Carla Core
        :return:
        """
        # randomly select a hero
        hero_id = random.choice(list(self.hero.keys()))
        # spectator following the car
        hero = self.hero[hero_id]
        transforms = hero.get_transform()
        server_view_x = hero.get_location().x - 5 * transforms.get_forward_vector().x
        server_view_y = hero.get_location().y - 5 * transforms.get_forward_vector().y
        server_view_z = hero.get_location().z + 3
        server_view_pitch = transforms.rotation.pitch
        server_view_yaw = transforms.rotation.yaw
        server_view_roll = transforms.rotation.roll
        self.spectator = core.get_core_world().get_spectator()
        self.spectator.set_transform(
            carla.Transform(
                carla.Location(x=server_view_x, y=server_view_y, z=server_view_z),
                carla.Rotation(pitch=server_view_pitch,yaw=server_view_yaw,roll=server_view_roll),
            )
        )

    def get_speed(self, hero_id):
        """
        Compute speed of a vehicle in Km/h.

            :param vehicle: the vehicle for which speed is calculated
            :return: speed as a float in Km/h
        """
        hero = self.hero[hero_id]
        vel = hero.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    def get_done_status(self):
        done = {}
        all_done = True
        need_respawn = False
        for hero_id in self.hero:
            _done = self.get_done_status_single(hero_id)
            if _done == False: all_done = False
            else: need_respawn = True
            done.update({hero_id: _done}) # Don't stop env unless all agent is done
        done.update({"__all__": all_done})
        return done, need_respawn
    
    def get_done_status_single(self, hero_id):
        # ---- Done when vehicle crashes something ---- 
        done_collision = self.observation[hero_id]["collision"]
        # done = self.observation[hero_id]["collision"] or not self.check_lane_type(map)
        
        # ---- Done when hero keeps low speed for a long time ---- 
        # Every tick cost 0.1s, so t_idle_start could reflect the start time of hero being idle 
        if self.get_speed(hero_id) < 4.0:
            done_idle = (time.time() - self.t_idle_start[hero_id]) > self.max_idle
        else:
            done_idle = False
            self.t_idle_start[hero_id] = time.time()
        
        # ---- Done when vehicle lives too long ---- 
        done_max_time = (time.time() - self.t_ep_start[hero_id]) > self.max_ep_time

        # ---- Done when vehicle falls out the world ---- 
        hero = self.hero[hero_id]
        done_falling = hero.get_location().z < -0.5 
        
        return done_collision or done_idle or done_max_time or done_falling

    def process_observation(self, core, observation):

        """
        Main function to do all the post processing of observations. This is an example.
        :param core:
        :param observation:
        :return:
        """
        sensor_config = self.experiment_config["SENSOR_CONFIG"]
        observation['camera'] = post_process_image(
                                            observation['camera'],
                                            normalized = sensor_config["CAMERA_NORMALIZED"][0],
                                            grayscale = sensor_config["CAMERA_GRAYSCALE"][0]
            )

        return observation

    def get_observation(self, core):
        observation = {}
        infos = {}
        for hero_id in self.hero:
            _observation, _info = self.get_observation_single(core, hero_id)
            observation.update({hero_id: _observation})
            infos.update({hero_id: _info})
        return observation, infos
    
    def get_observation_single(self, core, hero_id):

        # Initialize
        if hero_id not in self.observation:
            self.observation.update({hero_id: {}})
        
        # Remind equation in Python means symbol rather than assignment 
        hero = self.hero[hero_id]
        sensor_config = self.experiment_config["SENSOR_CONFIG"]
        observation_config = self.experiment_config["OBSERVATION_CONFIG"]

        # ---- Observation ---- 
        obs = self.observation[hero_id]
        if len(sensor_config["SENSOR"]) != len(observation_config["CAMERA_OBSERVATION"]):
            raise Exception("You need to specify the CAMERA_OBSERVATION for each sensor.")
        
        for i in range(0,len(sensor_config["SENSOR"])):
            if observation_config["CAMERA_OBSERVATION"][i]:
                obs['camera'].append(core.get_camera_data(hero_id))
        if observation_config["COLLISION_OBSERVATION"]:
            obs["collision"] = core.get_collision_data(hero_id)
        if observation_config["LOCATION_OBSERVATION"]:
            loc = hero.get_transform().location
            obs["location"] = [loc.x, loc.y, loc.z]
        if observation_config["LANE_OBSERVATION"]:
            obs["lane_invasion"] = core.get_lane_data(hero_id)
        if observation_config["GNSS_OBSERVATION"]:
            obs["gnss"] = core.get_gnss_data(hero_id)
        if observation_config["IMU_OBSERVATION"]:
            obs["imu"] = core.get_imu_data(hero_id)
        if observation_config["RADAR_OBSERVATION"]:
            obs["radar"] = core.get_radar_data(hero_id)
        if observation_config["BIRDVIEW_OBSERVATION"]:
            obs["birdview"] = core.get_birdview_data(hero_id)
        if observation_config["COMMUNICATION_OBSERVATION"]:
            # Get action and observation of nearby
            # filter = ['autopilot', 'hero']
            filter = ['autopilot']
            obs["communication"] = core.get_nearby_vehicles(hero_id, hero, max_distance=50, filter=filter)
        
        # ---- Navigation waypoint ---- 
        # TODO overcome the drawback caused by ergodic route search
        '''
        angle, distance = None, None
        while angle is None:
            angle, distance = self.get_closest_wp_forward(hero_id)

        obs["waypoint_angle"] = angle
        obs["waypoint_distance"] = distance
        self.past_distance_to_route[hero_id] = distance
        '''

        # ---- Info ---- 
        info = {}
        # ---- Physics control ----
        # action_taken = hero.get_control()
        action_taken = self.action[hero_id]
        info["action_taken"] = {
            "steer": action_taken.steer,
            "throttle": action_taken.throttle,
            "brake": action_taken.brake,
            "reverse": action_taken.reverse,
            "hand_brake": action_taken.hand_brake,
        }

        # ---- Intension of Carla Expert ----
        
        '''
        tm_port = 8000
        # traffic_manager = core.traffic_manager
        traffic_manager = core.client.get_trafficmanager(tm_port)
            
        # TODO make it stable to use
        # Returns next [INTENSION, Waypoint] traffic manager would choose
        # NOT physics vehicle control, Represented as ['LaneFollow', <carla.Waypoint>]
        action_possible = traffic_manager.get_all_actions(hero)
        action_next = traffic_manager.get_next_action(hero)
        
        info["action_possible"] = {
            'intension': action_possible[0],
            'waypoint': action_possible[1]
        }
        info["action_next"] = {
            'intension': action_next[0],
            'waypoint': action_next[1]
        }
        '''

        return obs, info

    def update_measurements(self, core):
        
        sensor_config = self.experiment_config["SENSOR_CONFIG"]
        observation_config = self.experiment_config["OBSERVATION_CONFIG"]
        if len(sensor_config["SENSOR"]) != len(observation_config["CAMERA_OBSERVATION"]):
                raise Exception("You need to specify the CAMERA_OBSERVATION for each sensor.")
        
        for i in range(0,len(sensor_config["SENSOR"])):
            if observation_config["CAMERA_OBSERVATION"][i]:
                core.update_camera(self.hero)
        if observation_config["COLLISION_OBSERVATION"]:
            core.update_collision(self.hero)
        if observation_config["LANE_OBSERVATION"]:
            core.update_lane_invasion(self.hero)

    # action from step is int
    # self.action stores the past action taken
    def update_actions(self, action_dict):
        # add judgement for reset
        if action_dict is None:
            for hero_id in self.hero:
                self.action.update({hero_id: carla.VehicleControl()}) # default：do nothing 
        else:
            for hero_id in action_dict:
                self.update_action(action_dict[hero_id], hero_id)

    def update_action(self, action_single, hero_id):

        action_single = DISCRETE_ACTIONS[int(action_single)]

        _action = carla.VehicleControl()
        _action.throttle = float(np.clip(action_single[0], 0, 0.5))
        _action.steer = float(np.clip(action_single[1], -0.7, 0.7))
        _action.brake = float(np.clip(action_single[2], 0, 1))

        _action.reverse = action_single[3]
        _action.hand_brake = action_single[4]
        
        self.action.update({hero_id: _action})
        self.hero[hero_id].apply_control(_action)

    def compute_reward(self, core, observation, map):
        reward = {}
        for hero_id in self.hero:
            _reward = self.compute_reward_single(core, observation, map, hero_id)
            reward.update({hero_id: _reward})
        return reward
    
    def compute_reward_single(self, core, observation, map, hero_id):

        """
        :param core:
        :param observation:
        :return:
        """

        # ------------Reward------------
        reward = 0
        hero = self.hero[hero_id]
        obs = observation[hero_id]
        start_location = self.start_location[hero_id]

        speed_limit = self.speed_limit # default limit = 30kmh
        # ----Going farther-----
        # carla.Location.distance(): Returns Euclidean distance from this location to another one
        loc_hero = hero.get_location()
        vel_hero = self.get_speed(hero_id)
        distance_travelled = start_location.distance(loc_hero)
        reward += max(0, distance_travelled - self.previous_distance)
        self.previous_distance = max(self.previous_distance, distance_travelled)
        
        # ----Velocity is over speed limit by 10%----
        # Speed limit sign named as carla.traffic_sign.<speed_limit>
        sign_speed_limit = 120
        for actor in core.get_core_world().get_actors():
            loc_actor = actor.get_location()
            if loc_actor.distance(loc_hero) < 50 and 'speed_limit' in actor.type_id:
                sign_speed_limit = actor.type_id.split('.')[2] 
                break
        speed_limit = min(speed_limit, sign_speed_limit)
        
        if vel_hero <= 1.1 * speed_limit:
            reward += vel_hero
        else:
            reward -= vel_hero - 1.1 * speed_limit
        
        # TODO
        # ----Overtaking other vehicles----
        ###


        # ------------Penalty------------
        # ---- Collision & terminate immediately ----
        if obs["collision"]:
            reward -= 100
        
        # ---- Out of road & don't terminate ----
        # TODO add judgement of Solid/Broken Lane
        if obs["lane_invasion"]:
            reward -= 10
        
        # ---- Towards next waypoint ----
        # Penalty for deviating from the route 
        # TODO make it convincing
        # route_loss = obs["waypoint_distance"] - self.past_distance_to_route
        # reward -= max(0, route_loss/10)
        # self.past_distance_to_route = min(self.past_distance_to_route, obs["waypoint_distance"])
        
        # ---- TODO Traffic Light! ----
        ###

        return reward

    def initialize_reward(self, core):

        """
        Generic initialization of reward function
        :param core:
        :return:
        """
        print("This is a base experiment. Make sure you make you own reward initialization function")
        raise NotImplementedError


    # ==============================================================================
    # -------------------------- Hero -----------------------------------
    # ==============================================================================
    def spawn_hero(self, world, autopilot=False):

        """
        This function spawns the hero vehicle. It makes sure that if a hero exists, it destroys the hero and respawn
        :param core:
        :param transform: Hero start location
        :param autopilot: Autopilot Status
        :return:
        """
        num_hero = self.experiment_config['n_heroes']
        self.spawn_points = world.get_map().get_spawn_points()

        self.hero_blueprints = world.get_blueprint_library().find(self.hero_model)
        self.hero_blueprints.set_attribute("role_name", "hero")

        self.hero = {}

        i = 0
        random.shuffle(self.spawn_points, random.random)
        num_hero_exist = 0
        # server_port = core.tm_port
        # Keep finding spawn point until all self.hero are successfully spawned
        while num_hero_exist < num_hero:
            next_spawn_point = self.spawn_points[i % len(self.spawn_points)]
            hero = world.try_spawn_actor(self.hero_blueprints, next_spawn_point)
            if hero is not None:
                hero.set_autopilot(autopilot) # whether to use autopilot
                self.hero.update({hero.id: hero})
                self.start_location.update({hero.id: self.spawn_points[i].location})
                self.route.update({hero.id: self.select_random_route(world, hero.id)})
                self.t_idle_start.update({hero.id: time.time()})
                self.t_ep_start.update({hero.id: time.time()})
            else:
                # print("Could not spawn Hero, changing spawn point")
                i+=1
            num_hero_exist = len(self.hero)

        self.set_observation_space()
        self.set_action_space()

        world.tick()
        # print("Hero all spawned!")
    
    def update_hero(self, world):
        all_actors = world.get_actors()
        self.hero = {actor.id: actor for actor in all_actors if actor.attributes.get('role_name') == 'hero'}

    # TODO add respawn function to ensure there exist enough agents to communicate
    def respawn_hero(self, world, autopilot=False, terminateds={}):

        """
        This function respawns hero when some of them is terminated
        """
        num_hero = self.experiment_config['n_heroes']
        self.spawn_points = world.get_map().get_spawn_points()

        self.hero_blueprints = world.get_blueprint_library().find(self.hero_model)
        self.hero_blueprints.set_attribute("role_name", "hero")

        # Destroy done vehicles
        for hero_id in terminateds:
            if terminateds[hero_id]:
                self.core.destroy_sensors(self.experiment_config, hero_id)
                self.hero[hero_id].destroy()
                del self.hero[hero_id]

        i = 0
        random.shuffle(self.spawn_points, random.random)
        num_hero_exist = len(self.hero)
        # server_port = core.tm_port
        # Keep finding spawn point until all self.hero are successfully spawned
        while num_hero_exist < num_hero:
            next_spawn_point = self.spawn_points[i % len(self.spawn_points)]
            hero = world.try_spawn_actor(self.hero_blueprints, next_spawn_point)
            if hero is not None:
                hero.set_autopilot(autopilot) # whether to use autopilot
                self.hero.update({hero.id: hero})
                self.start_location.update({hero.id: self.spawn_points[i].location})
                self.route.update({hero.id: self.select_random_route(world, hero.id)})
                self.t_idle_start.update({hero.id: time.time()})
                self.t_ep_start.update({hero.id: time.time()})
            else:
                # print("Could not spawn Hero, changing spawn point")
                i+=1
            num_hero_exist = len(self.hero)

        # self.update_hero(world)
        # Sensors also should be updated!
        self.core.setup_sensors(
            self.experiment_config,
            self.hero,
            self.world.get_settings().synchronous_mode,
        )
        self.set_observation_space()
        self.set_action_space()

        world.tick()
        # print("Hero respawned!") # TODO realize respawn

    # ==============================================================================
    # -- Tick -----------------------------------------------------------
    # ==============================================================================

    def experiment_tick(self, core, world, action_dict):

        """
        This is the "tick" logic.
        :param core:
        :param action:
        :return:
        """

        world.tick()
        self.update_measurements(core)
        self.update_actions(action_dict)

    
    def select_random_route(self, world, hero_id):
        '''
        retruns a random route for the car/veh
        out of the list of possible locations locs
        where distance is longer than 100 waypoints
        '''  
        hero = self.hero[hero_id]
        point_a = hero.get_transform().location #we start at where the car is or last waypoint
        sampling_resolution = 1
        grp = GlobalRoutePlanner(world.get_map(), sampling_resolution)
        # now let' pick the longest possible route
        min_distance = 100
        result_route = None
        route_list = []
        for loc in world.get_map().get_spawn_points(): # we start trying all spawn points 
                                                            #but we just exclude first at zero index
            cur_route = grp.trace_route(point_a, loc.location)
            if len(cur_route) > min_distance:
                route_list.append(cur_route)
        result_route = random.choice(route_list)
        return result_route
    
    def get_closest_wp_forward(self, hero_id):
        '''
        this function is to find the closest point looking forward
        if there in no points behind, then we get first available
        '''

        # first we create a list of angles and distances to each waypoint
        # yeah - maybe a bit wastefull
        hero = self.hero[hero_id]
        vehicle_transform = hero.get_transform()
        points_ahead = []
        points_behind = []
        for i, wp in enumerate(self.route[hero_id]):
            #get angle  
            wp_transform = wp[0].transform
            distance = ((wp_transform.location.y - vehicle_transform.location.y)**2 + (wp_transform.location.x - vehicle_transform.location.x)**2)**0.5
            angle = math.degrees(math.atan2(wp_transform.location.y - vehicle_transform.location.y,
                                wp_transform.location.x - vehicle_transform.location.x)) -  vehicle_transform.rotation.yaw
            
            if angle > 360:
                angle = angle - 360
            elif angle < -360:
                angle = angle + 360

            if angle > 180:
                angle = -360 + angle
            elif angle < -180:
                angle = 360 - angle 

            if abs(angle) <= 90:
                points_ahead.append([i,distance,angle])
            else:
                points_behind.append([i,distance,angle])
        
        # now we pick a point we need to get angle to 
        if len(points_ahead) == 0:
            closest = min(points_behind, key=lambda x: x[1])
            if closest[2] > 0:
                closest = [closest[0], closest[1], 90]
            else:
                closest = [closest[0], closest[1], -90] 
        else:
            closest = min(points_ahead, key=lambda x: x[1])
            # move forward if too close
            for i, point in enumerate(points_ahead):
                if point[1] >= 10 and point[1] < 20:
                    closest = point
                    break
            return closest[2]/90.0, closest[1] # we convert angle to [-1 to +1] and also return distance
        
    def get_hero_location(self):
        loc = {}
        for hero_id in self.hero:
            _loc = self.hero[hero_id].get_location()
            loc.update({hero_id: _loc})
        return loc
    
    # add func to align with gymnasium API
    def render(self, mode='human', text: Optional[Union[dict, str]] = None, *args, **kwargs) -> Optional[np.ndarray]:
        pass

import random
if __name__ == '__main__':
    
    env_config = dict(
        RAY =  False,  # Are we running an experiment in Ray
        DEBUG_MODE = False,
        # Experiment = "experiment_birdview_multi",
        EXPERIMENT_CONFIG = BASE_EXPERIMENT_CONFIG,
        horizon = 300, # added for done judgement
    )
    
    env = BaseEnv(env_config)
    env.reset()
    while 1:
        heroes = env.get_hero()
        action_coast = {}
        for hero_id in heroes:
            action_coast.update({hero_id: 0})
        observation, reward, terminateds, truncateds, info = env.step(action_coast) # stay still to observe
        
        # random_hero_id = random.choice(list(heroes))
        # random_hero = heroes[random_hero_id]
        # surrounding_vehicles = env.core.get_nearby_vehicles(random_hero_id, random_hero, max_distance=200)