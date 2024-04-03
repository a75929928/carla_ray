from pettingzoo.sisl.waterworld_v4 import raw_env
# env = waterworld_v4()
env = raw_env()
env.reset()
while 1:
    action_random = {agent_id: action_space.sample() for agent_id, action_space in env.action_spaces.items()}
    env.step(action_random)