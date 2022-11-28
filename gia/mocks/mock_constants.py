import numpy as np
from gym.spaces import Box, Dict, Discrete

MOCK_OBS_SPACE = Dict(
    image=Box(0, 1, shape=(3, 32, 40), dtype=np.float32), discrete=Discrete(4), vector=Box(0, 1, shape=(7,))
)
MOCK_ACTION_SPACE = Dict(cont=Box(0, 1, shape=(17,), dtype=np.float32), discrete=Discrete(3))
