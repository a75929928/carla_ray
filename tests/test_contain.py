import numpy as np
from gym import spaces
a = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
b = np.array([0.5], dtype=np.float32)
if b in a:
    print("1")