from gymnasium import spaces

DISCRETE_ACTIONS_SMALL = {
    0: [0.0, 0.00, 0.0, False, False], # Coast
    1: [0.0, -0.1, 0.0, False, False], # Turn Left
    2: [0.0, 0.1, 0.0, False, False], # Turn Right
    3: [1.0, 0.00, 0.0, False, False], # Accelerate
    4: [0.0, 0.00, 1.0, False, False], # Brake
}

DISCRETE_ACTIONS = DISCRETE_ACTIONS_SMALL


action_space = spaces.Discrete(len(DISCRETE_ACTIONS))
# action_spaces = spaces.Dict({})
action_spaces = dict()
for i in range(10):
    action_spaces.update({i: action_space})
print(action_spaces)

# Conclusion: Just use Dict to organize action space!