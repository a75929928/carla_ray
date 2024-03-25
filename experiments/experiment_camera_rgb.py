from experiments.base_experiment import *
from helper.CarlaHelper import spawn_vehicle_at, post_process_image, update_config
import random
import numpy as np
from gym import spaces
from itertools import cycle
import cv2
import time
import carla
import gc

SERVER_VIEW_CONFIG = {
}

SENSOR_CONFIG = {
    "SENSOR": [SensorsEnum.CAMERA_DISTORTED],
    "SENSOR_TRANSFORM": [SensorsTransformEnum.Transform_C],
    "CAMERA_X": 84,
    "CAMERA_Y": 84,
    "CAMERA_FOV": 60,
    "CAMERA_NORMALIZED": [True],# apparently doesnt work if set to false, its just for the image!
    "CAMERA_GRAYSCALE": [True],
    "FRAMESTACK": 1,
}

BIRDVIEW_CONFIG = {
    "SIZE": 190,
    "RADIUS": 15,
    "FRAMESTACK": 4,
    "RENDER": True
}

OBSERVATION_CONFIG = {
    "CAMERA_OBSERVATION": [True],
    # "RADAR_OBSERVATION": True, # TODO fix Radar
    # "BIRDVIEW_OBSERVATION": True,
}

EXPERIMENT_CONFIG = {
    "OBSERVATION_CONFIG": OBSERVATION_CONFIG,
    "Server_View": SERVER_VIEW_CONFIG,
    "SENSOR_CONFIG": SENSOR_CONFIG,
    # "server_map": "Town10HD",
    "server_map": "Town10HD_Opt", # layerd map - could remove buildings and trees to avoid disturbance
    "BIRDVIEW_CONFIG": BIRDVIEW_CONFIG,
    "n_vehicles": 40,
    "n_walkers": 15,
    # "hero_vehicle_model": "vehicle.tesla_m3",
    "hero_vehicle_model": "vehicle.lincoln.mkz_2017",

    "Disable_Rendering_Mode": False, # add to disable rendering
    # "Autodrive_enabled": True,
}

class Experiment(BaseExperiment):
    def __init__(self):
        config=update_config(BASE_EXPERIMENT_CONFIG, EXPERIMENT_CONFIG)
        super().__init__(config)

    def initialize_reward(self, core):
        """
        Generic initialization of reward function
        :param core:
        :return:
        """
        self.previous_distance = 0
        self.frame_stack = 4  # can be 1,2,3,4
        self.prev_image_0 = None
        self.prev_image_1 = None
        self.prev_image_2 = None
        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]

    def set_observation_space(self):
        birdview_size = self.experiment_config["BIRDVIEW_CONFIG"]["SIZE"]
        num_of_channels = 3
        framestack = self.experiment_config["BIRDVIEW_CONFIG"]["FRAMESTACK"]
        image_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(birdview_size, birdview_size, num_of_channels * framestack),
            dtype=np.float32,
        )
        angle_space = spaces.Box(low = -1.0,high = 1.0,shape = (1,),dtype=np.float32)
        # self.observation_space = spaces.Dict({"birdview": image_space})
        self.observation_space = spaces.Dict({"birdview": image_space, "waypoint": angle_space})

    def process_observation(self, core, observation):
        # Real observation sent to step
        # here concatenate birdview images in different seconds 
        """
        Process observations according to your experiment
        :param core:
        :param observation:
        :return:
        """
        self.set_server_view(core)
        image = post_process_image(observation['birdview'],
                                   normalized = False,
                                   grayscale = False
        )

        if self.prev_image_0 is None:
            self.prev_image_0 = image
            self.prev_image_1 = self.prev_image_0
            self.prev_image_2 = self.prev_image_1

        images = image

        if self.frame_stack >= 2:
            images = np.concatenate([self.prev_image_0, images], axis=2)
        if self.frame_stack >= 3 and images is not None:
            images = np.concatenate([self.prev_image_1, images], axis=2)
        if self.frame_stack >= 4 and images is not None:
            images = np.concatenate([self.prev_image_2, images], axis=2)

        self.prev_image_2 = self.prev_image_1
        self.prev_image_1 = self.prev_image_0
        self.prev_image_0 = image

        # return dict({'birdview': images/255.0})
        return dict({'birdview': images/255.0, 'waypoint': np.array([observation["waypoint_angle"]], dtype=np.float32)})

    def inside_lane(self, map):
        self.current_w = map.get_waypoint(self.hero.get_location(), lane_type=carla.LaneType.Any)
        return self.current_w.lane_type in self.allowed_types

    def compute_reward(self, core, observation, map):
        """
        Reward function
        :param observation:
        :param core:
        :return:
        """

        # ------------Reward------------
        reward = 0
        hero = self.hero
        speed_limit = self.speed_limit # default limit = 30kmh
        # ----Going farther-----
        # carla.Location.distance(): Returns Euclidean distance from this location to another one
        loc_hero = hero.get_location()
        vel_hero = self.get_speed()
        distance_travelled = self.start_location.distance(loc_hero)
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
        if self.observation["collision"]:
            reward -= 100
        # ---- Out of road & don't terminate ----
        # TODO add judgement of Solid/Broken Lane
        if self.observation["lane_invasion"]:
            reward -= 10
        # ---- Towards next waypoint ----
        # Penalty for deviating from the route
        route_loss = self.observation["waypoint_distance"] - self.past_distance_to_route
        reward -= max(0, route_loss/10)
        self.past_distance_to_route = min(self.past_distance_to_route, self.observation["waypoint_distance"])
        # TODO Traffic Light!
        ###
            
        return reward

