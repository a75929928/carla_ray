from experiments.base_experiment_multi import *
from helper.CarlaHelper import spawn_vehicle_at, post_process_image, update_config
import random
import numpy as np
from gymnasium import spaces
# from gym import spaces
from itertools import cycle
import cv2
import time
import carla
import gc

SERVER_VIEW_CONFIG = {
}

SENSOR_CONFIG = {
    "CAMERA_NORMALIZED": [True], # apparently doesnt work if set to false, its just for the image!
    "CAMERA_GRAYSCALE": [True],
    "FRAMESTACK": 4,
}

BIRDVIEW_CONFIG = {
    "SIZE": 190,
    "RADIUS": 15,
    "FRAMESTACK": 4,
    # "RENDER": True
}

OBSERVATION_CONFIG = {
    "CAMERA_OBSERVATION": [False],
    "COLLISION_OBSERVATION": True,
    "LOCATION_OBSERVATION": True,
    "RADAR_OBSERVATION": False,
    "IMU_OBSERVATION": False,
    "LANE_OBSERVATION": True,
    "GNSS_OBSERVATION": False,
    "BIRDVIEW_OBSERVATION": True,
    "COMMUNICATION_OBSERVATION": True
}

EXPERIMENT_CONFIG = {
    "OBSERVATION_CONFIG": OBSERVATION_CONFIG,
    "Server_View": SERVER_VIEW_CONFIG,
    "SENSOR_CONFIG": SENSOR_CONFIG,
    "server_map": "Town10HD_Opt", # layerd map - could remove buildings and trees to avoid disturbance
    "BIRDVIEW_CONFIG": BIRDVIEW_CONFIG,
    # "n_vehicles": 20,
    # "n_walkers": 15,
    # "hero_vehicle_model": "vehicle.tesla.model3", # catalogue find https://carla.readthedocs.io/en/0.9.15/catalogue_vehicles/
    "hero_vehicle_model": "vehicle.lincoln.mkz_2017",

    "Disable_Rendering_Mode": False, # add to disable rendering
    "n_heroes": 5,
    # "Autodrive_enabled": True,
}

class Experiment(BaseExperiment):
    def __init__(self):
        config=update_config(BASE_EXPERIMENT_CONFIG, EXPERIMENT_CONFIG)
        super().__init__(config)
        
        # initialize
        self.set_observation_spaces()
        self.set_action_spaces()

    def initialize_reward(self, core):
        """
        Generic initialization of reward function
        :param core:
        :return:
        """
        self.previous_distance = 0
        self.frame_stack = 4  # can be 1,2,3,4
        self.prev_image_0 = {}
        self.prev_image_1 = {}
        self.prev_image_2 = {}
        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]

    # default: every agent share same observation space and action space
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
        # self.observation_space = spaces.Dict({"birdview": image_space, "waypoint": angle_space})
        return spaces.Dict({"birdview": image_space})
        
    # add to align with multi-agent env
    def set_observation_spaces(self):
        self.observation_space = dict()
        if len(self.hero) == 0: pass
        else:
            for hero_id in self.hero:
                # default: every agent share the same obs & act space
                default_space = self.set_observation_space()
                self.observation_space.update({hero_id: default_space})
            # self.observation_space = spaces.Dict({"birdview": image_space, "waypoint": angle_space})
            
    def get_observation_spaces(self):

        """
        :return: observation space
        """
        return self.observation_space

    def process_observation(self, core, observation):
        processed_observation = {}
        for hero_id in self.hero:
            _processed_observation = self.process_observation_single(core, observation[hero_id], hero_id)
            processed_observation.update({hero_id: _processed_observation})
        return processed_observation
    
    def process_observation_single(self, core, observation, hero_id):
        # Real observation sent to step
        # here concatenate birdview images in different seconds 
        """
        Process observations according to your experiment
        :param core:
        :param observation:
        :return:
        """
        # self.set_server_view(core)
        image = post_process_image(observation['birdview'],
                                   normalized = False,
                                   grayscale = False
        )

        if hero_id not in self.prev_image_0:
            self.prev_image_0[hero_id] = image
            self.prev_image_1[hero_id] = self.prev_image_0[hero_id]
            self.prev_image_2[hero_id] = self.prev_image_1[hero_id]

        images = image

        if self.frame_stack >= 2:
            images = np.concatenate([self.prev_image_0[hero_id], images], axis=2)
        if self.frame_stack >= 3 and images is not None:
            images = np.concatenate([self.prev_image_1[hero_id], images], axis=2)
        if self.frame_stack >= 4 and images is not None:
            images = np.concatenate([self.prev_image_2[hero_id], images], axis=2)

        self.prev_image_2[hero_id] = self.prev_image_1[hero_id]
        self.prev_image_1[hero_id] = self.prev_image_0[hero_id]
        self.prev_image_0[hero_id] = image

        return dict({
                'birdview': images/255.0, 
                # 'waypoint': np.array([observation["waypoint_angle"]], dtype=np.float32)
                })
    
    def set_action_spaces(self):

        """
        :return: None. In this experiment, it is a spaces.Discrete space
        """
        self.action_spaces = dict()
        # self.action_spaces = spaces.Dict({})
        if len(self.hero) == 0: pass
        else:
            for hero_id in self.hero:
                action_space = self.set_action_space()
                self.action_spaces.update({hero_id: action_space})
            
    def get_action_spaces(self):

        """
        :return: action_space. In this experiment, it is a discrete space
        """
        return self.action_spaces
    
    def compute_reward_single(self, core, observation, map, hero_id):
        """
        Reward function
        :param observation:
        :param core:
        :return:
        """

        # Base reward concerning ground truth
        reward = super().compute_reward_single(core, observation, map, hero_id) 
        
        return reward

    def spawn_hero(self, world, transform, autopilot=False, core=None):
        
        super().spawn_hero(world, transform, autopilot, core)

        # Update dict of obs/act space when spawn hero
        self.set_action_spaces()
        self.set_observation_spaces()
