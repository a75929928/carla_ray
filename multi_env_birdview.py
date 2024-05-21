from base_env import *
from helper.CarlaHelper import spawn_vehicle_at, post_process_image, update_config
# import random
import numpy as np
from gymnasium import spaces
# from gym import spaces
# from itertools import cycle
# import cv2
# import time
import carla
# import gc

import os
CARLA_ROOT = os.getenv('1', 'D:\Code\carla\CARLA_0.9.15')
WORK_DIR = os.getenv('2', 'D:\Code\.SOTA\carla_garage')
CARLA_SERVER = f'{CARLA_ROOT}\\CarlaUE4.exe'

PYTHONPATH = os.environ.get('PYTHONPATH', '')

PYTHONPATH += f';{CARLA_ROOT}\\PythonAPI'
PYTHONPATH += f';{CARLA_ROOT}\\PythonAPI\\carla'
# PYTHONPATH += f';C:\\users\\forgiven\miniconda3\envs\garage\lib\site-packages\carla\libcarla.cp37-win_amd64.pyd'
PYTHONPATH += f';{CARLA_ROOT}\\PythonAPI\\carla\\dist\\carla-0.9.10-py3.7-win-amd64.egg'

# Use newest leaderboard and scenario_runner
PYTHONPATH += f';{WORK_DIR}\\scenario_runner'
PYTHONPATH += f';{WORK_DIR}\\leaderboard'
# PYTHONPATH += f';D:\Code\scenario_runner'
# PYTHONPATH += f';D:\Code\leaderboard' 

os.environ['PYTHONPATH'] = PYTHONPATH

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

SERVER_VIEW_CONFIG = {
    # "server_view_x_offset": 00,
    # "server_view_y_offset": 00,
    # "server_view_height": 200,
    # "server_view_pitch": -90,
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
    "n_vehicles": 20,
    "n_walkers": 15,
    # "hero_vehicle_model": "vehicle.tesla.model3", # catalogue find https://carla.readthedocs.io/en/0.9.15/catalogue_vehicles/
    "hero_vehicle_model": "vehicle.lincoln.mkz_2017",

    "Disable_Rendering_Mode": False, # add to disable rendering
    "n_heroes": 1,
    "allow_respawn": False,
    # "Autodrive_enabled": True,
}

class MultiEnvBirdview(BaseEnv):

    def __init__(self, config):
        super().__init__(config)

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

    def set_single_action_space(self):

        """
        :return: None. In this experiment, it is a discrete space
        """
        # self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))
        return spaces.Discrete(len(DISCRETE_ACTIONS))
    
    def set_action_space(self):

        """
        :return: None. In this experiment, it is a spaces.Discrete space
        """
        # self.action_spaces = dict()
        # if len(self.hero) == 0: pass
        # else:
        #     for hero_id in self.hero:
        #         action_space = self.set_action_space()
        #         self.action_spaces.update({hero_id: action_space})
        self.action_space = spaces.Dict({
            hero_id: self.set_single_action_space() for hero_id in self.hero
        })    

    # default: every agent share same observation space and action space
    def set_single_observation_space(self):
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
        # TODO add waypoint navigation
        # self.observation_space = spaces.Dict({"birdview": image_space, "waypoint": angle_space})
        return spaces.Dict({"birdview": image_space})
        
    # add to align with multi-agent env
    def set_observation_space(self):
        
        # self.observation_spaces = dict()
        # if len(self.hero) == 0: pass
        # else:
        #     for hero_id in self.hero:
        #         # default: every agent share the same obs & act space
        #         default_space = self.set_observation_space()
        #         self.observation_spaces.update({hero_id: default_space})
        
        self.observation_space = spaces.Dict({
            hero_id: self.set_single_observation_space() for hero_id in self.hero
        })

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
    
    def compute_reward_single(self, core, observation, map, hero_id):
        """
        Reward function
            Base reward concerning ground truth
            TODO add specific reward
        """

        reward = super().compute_reward_single(core, observation, map, hero_id) 
        
        return reward

from srunner.autoagents.autonomous_agent import AutonomousAgent
from examples.example_agents import AutopilotAgent
from srunner.scenariomanager.timer import GameTime
def get_policy_for_agent(agent: AutonomousAgent):
    def policy(obs):
        control = agent.run_step(input_data=obs, timestamp=GameTime.get_time())
        action = np.array([control.throttle, control.steer, control.brake])
        return action

    return policy

import random
if __name__ == '__main__':
    
    experiment_config = update_config(BASE_EXPERIMENT_CONFIG, EXPERIMENT_CONFIG)

    env_config = dict(
        RAY =  False,  # Are we running an experiment in Ray
        DEBUG_MODE = False,
        # Experiment = "experiment_birdview_multi",
        EXPERIMENT_CONFIG = experiment_config,
        horizon = 300, # added for done judgement
    )
    
    env = MultiEnvBirdview(env_config)
    
    for i in range(15):
        obs, info = env.reset()
        agent = AutopilotAgent(role_name="hero", carla_host="localhost", carla_port=env.core.tm_port)
        only_hero = list(obs.keys())[0]
        agent.setup(path_to_conf_file="", route=env.route[only_hero])
        policy = get_policy_for_agent(agent)
        done = False
        while not done:
            heroes = env.hero
            action_coast = {}
            action_rush = {}
            for hero_id in heroes:
                action_coast.update({hero_id: 0})
                action_rush.update({hero_id: 1})
            action_random = {agent_id: single_action_space.sample() for agent_id, single_action_space in env.action_space.items()}
            action_auto = {agent: policy(o) for agent, o in obs.items()}
            obs, reward, terminateds, truncateds, info = env.step(action_auto) 
            
            # random_hero_id = random.choice(list(heroes))
            # random_hero = heroes[random_hero_id]
            # surrounding_vehicles = env.core.get_nearby_vehicles(random_hero_id, random_hero, max_distance=200)