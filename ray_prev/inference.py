from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os

import ray
from ray.rllib.agents.dqn import DQNTrainer

from carla_env import CarlaEnv
from helper.CarlaHelper import kill_server


def main(args):
    # Load configuration from file params.json
    config_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(config_dir, "..", "params.json")
    with open(config_path) as f:
        config = json.load(f)
        del config["num_cpus_per_worker"]
        del config["num_gpus_per_worker"]
        config["num_workers"] = 0

    # Restore agent
    agent = DQNTrainer(env=CarlaEnv, config=config)
    agent.restore(args.checkpoint)

    env = agent.workers.local_worker().env

    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = agent.compute_action(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward

    print("Episode reward: {}".format(episode_reward))
    return episode_reward


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "checkpoint",
        default="/opt/ml/input/data/model/checkpoint",
        type=str,
        help="Checkpoint from which to roll out.")
    argparser.add_argument('--debug', action='store_true', help='enable debug messages')
    arguments = argparser.parse_args()

    if arguments.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    ray.init()
    main(arguments)
    ray.shutdown()
    kill_server()
