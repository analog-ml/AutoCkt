import numpy as np
from gymnasium import spaces


class ActionNormalizer:
    """Rescale and relocate the actions."""

    def __init__(self, action_space_low, action_space_high):

        self.action_space_low = action_space_low
        self.action_space_high = action_space_high

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        low = self.action_space_low
        high = self.action_space_high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1)."""
        low = self.action_space_low
        high = self.action_space_high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action


action_space = spaces.Box(low=-1, high=1, shape=(24,), dtype=np.float64)
# print (action_space.sample())
# fmt: off
action_space_low = np.array(
    [
        0.13, 0.12, 1,
        0.13, 0.12, 1,    
        0.13, 0.12, 1,    
        0.13, 0.12, 1,    
        0.13, 0.12, 1,    
        0.13, 0.12, 1,    
        0.1, 
        0.1, 
        0.1, 
        0.1, 
        0.1,
        0.1
    ]
)

action_space_high = np.array(
    [
        50, 2, 100,
        50, 2, 100,    
        50, 2, 100,    
        50, 2, 100,    
        50, 2, 100,    
        50, 2, 100,    
        1.2, 
        1.2, 
        1.2, 
        1.2, 
        50,
        50
    ]
)
# fmt: on

action = ActionNormalizer(
    action_space_low=action_space_low, action_space_high=action_space_high
).action(action_space.sample())
print("action: ", action)

for idx in [2, 2 + 3, 5 + 3, 8 + 3, 11 + 3, 14 + 3, -1, -2]:
    action[idx] = int(action[idx])
print("action: ", action)
print("action: ", ",".join([str(x) for x in action]))
