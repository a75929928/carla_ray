from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.sisl import waterworld_v4
# from carla_env_multi import CarlaEnv
from multi_env_birdview import *

if __name__ == "__main__":
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN

    # register_env("waterworld", lambda _: PettingZooEnv(waterworld_v4.env()))
    
    experiment_config = update_config(BASE_EXPERIMENT_CONFIG, EXPERIMENT_CONFIG)

    env_config = dict(
        RAY =  False,  # Are we running an experiment in Ray
        DEBUG_MODE = False,
        # Experiment = "experiment_birdview_multi",
        EXPERIMENT_CONFIG = experiment_config,
        horizon = 300, # added for done judgement
    )

    register_env("MyCarlaEnv", lambda _: MultiEnvBirdview(env_config))

    tune.Tuner(
        "PPO",
        # "APEX_DDPG",
        run_config=air.RunConfig(
            stop={"episodes_total": 60000},
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=10,
            ),
        ),
        param_space={
            # Enviroment specific.
            "env": "MyCarlaEnv",
            # General
            "num_gpus": 0,
            "num_workers": 1,
            "num_envs_per_worker": 2,
            # Specific model config
            "model": {
                "conv_filters": [
                        (32, 3, 2, 'relu'),
                        (64, 3, 2, 'relu'),
                        (128, 3, 2, 'relu'),
                        # 添加一个全连接层
                        (512,),
                    ],
                    "fcnet_activation": "relu",  # 指定全连接层的激活函数
                    "fcnet_hiddens": [256, 128]  # 指定全连接层的隐藏层大小
                },
            "num_steps_sampled_before_learning_starts": 1000,
            "compress_observations": True,
            "rollout_fragment_length": 20,
            "train_batch_size": 512,
            "gamma": 0.99,
            "n_step": 3,
            "lr": 0.0001,
            "target_network_update_freq": 50000,
            "min_sample_timesteps_per_iteration": 25000,
            # Method specific.
            # We only have one policy (calling it "shared").
            # Class, obs/act-spaces, and config will be derived
            # automatically.
            "policies": {"shared_policy": "CnnPolicy"},
            # Always use "shared" policy.
            "policy_mapping_fn": (
                # lambda agent_id, episode, worker, **kwargs: "shared_policy"
                lambda agent_id : "shared_policy"
            ),
        },
    ).fit()