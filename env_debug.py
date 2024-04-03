# from env.multi_env_birdview import *
from multi_env_birdview import *

experiment_config = update_config(BASE_EXPERIMENT_CONFIG, EXPERIMENT_CONFIG)

env_config = dict(
    RAY =  False,  # Are we running an experiment in Ray
    DEBUG_MODE = False,
    # Experiment = "experiment_birdview_multi",
    EXPERIMENT_CONFIG = experiment_config,
    horizon = 300, # added for done judgement
)

env = MultiEnvBirdview(env_config)
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