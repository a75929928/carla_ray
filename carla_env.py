"""
This is a sample carla environment. It does basic functionality.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
# from gym.utils import seeding

from core.CarlaCore import CarlaCore
import time


class CarlaEnv(gym.Env):

    def __init__(self, config):

        self.environment_config = config

        module = __import__("experiments.{}".format(self.environment_config["Experiment"] ))
        exec("self.experiment = module.{}.Experiment()".format(self.environment_config["Experiment"]))
        self.action_space = self.experiment.get_action_space()
        self.observation_space = self.experiment.get_observation_space()
        self.experiment_config = self.experiment.get_experiment_config()

        self.core = CarlaCore(self.environment_config, self.experiment_config)
        self.world = self.core.get_core_world()
        CarlaCore.spawn_npcs(self.core, self.experiment_config["n_vehicles"],self.experiment_config["n_walkers"], hybrid = True)
        self.map = self.world.get_map()
        self.reset()

    def reset(self):
        self.core.reset_sensors(self.experiment_config)

        # autopilot means to use Carla Expert but here is just decoration XD
        self.experiment.spawn_hero(self.world, self.experiment.start_location, autopilot=self.experiment_config["Autodrive_enabled"]) 
        self.core.setup_sensors(
            self.experiment.experiment_config,
            self.experiment.get_hero(),
            self.world.get_settings().synchronous_mode,
        )

        # add params to monitor training
        self.episode_start = time.time()
        self.steering_lock = False
        self.steering_lock_start = None # this is to count time in steering lock and start penalising for long time in steering lock
        self.step_counter = 0
        self.initial_location = self.experiment.hero.get_location()

        self.experiment.initialize_reward(self.core)
        self.experiment.set_server_view(self.core)
        self.experiment.experiment_tick(self.core, self.world, action=None)
        obs, info = self.experiment.get_observation(self.core)
        # obs = self.experiment.process_observation(self.core, obs)
        return obs

    def step(self, action):
        # assert action in [0, 13], action
        self.experiment.experiment_tick(self.core, self.world, action)
        observation, info = self.experiment.get_observation(self.core)
        # observation = self.experiment.process_observation(self.core, observation)
        reward = self.experiment.compute_reward(self.core,observation, self.map)
        done = self.experiment.get_done_status()
        return observation, reward, done, info

#    def seed(self, seed=None):
#        self.np_random, seed = seeding.np_random(seed)
#        return [seed]

if __name__ == '__main__':
    
    env_config = {
        "RAY": False,  # Are we running an experiment in Ray
        "DEBUG_MODE": False,
        "Experiment": "experiment_camera_rgb",
    }
    
    env = CarlaEnv(env_config)
    env.reset()
    while 1:
        # hero = env.experiment.hero.id
        observation, reward, done, info = env.step(0) # stay still to observe
        # surrounding_vehicles = env.core.get_nearby_vehicles(env.experiment.get_hero(), max_distance=200)