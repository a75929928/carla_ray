
from gymnasium import spaces

def set_single_action_space(self):

        """
        :return: None. In this experiment, it is a discrete space
        """
        # self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))
        return spaces.Discrete(15)

hero = {}
action_space = spaces.Dict({
            hero_id: set_single_action_space() for hero_id in hero
    })    
print(action_space)