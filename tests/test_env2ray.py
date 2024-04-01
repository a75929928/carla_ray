from ray.rllib.utils import check_env
from carla_env_multi import CarlaEnv

env_config = dict(
        RAY =  False,  # Are we running an experiment in Ray
        DEBUG_MODE = False,
        Experiment = "experiment_birdview_multi",

        horizon = 300, # added for done judgement
    )

check_env(CarlaEnv(env_config))