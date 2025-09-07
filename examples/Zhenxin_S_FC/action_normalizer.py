import numpy as np
from gymnasium import spaces


class ActionNormalizer:
    """Rescale and relocate the actions."""

    def __init__(self, action_space_low, action_space_high):

        """
        Initialize the ActionNormalizer with per-dimension action bounds.
        
        Parameters:
            action_space_low (array-like): 1-D array of per-dimension minimum action values (lower bounds).
            action_space_high (array-like): 1-D array of per-dimension maximum action values (upper bounds).
        
        Both arrays must have the same shape and correspond elementwise; they are used to map actions between the canonical range (-1, 1) and the problem-specific [low, high] range.
        """
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Map an elementwise action from the canonical range (-1, 1) into the instance's per-dimension [low, high] bounds.
        
        The input `action` is expected to be an ndarray with the same shape as the normalizer's bounds. Each element x is transformed with a linear mapping:
            y = x * ((high - low) / 2) + (high - (high - low) / 2)
        and then clipped to the corresponding [low, high] interval.
        
        Parameters:
            action (np.ndarray): Elementwise action values in (−1, 1) to be scaled.
        
        Returns:
            np.ndarray: Action mapped and clipped to the per-dimension [low, high] range.
        """
        low = self.action_space_low
        high = self.action_space_high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """
        Map an action from the environment bounds [low, high] back into the canonical (-1, 1) range.
        
        Per-dimension inverse linear transform using this instance's action_space_low and action_space_high:
        scale = (high - low) / 2 and offset = high - scale, then result = (action - offset) / scale.
        The output is clipped elementwise to [-1.0, 1.0] and returned as an ndarray with the same shape as the input.
        """
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
