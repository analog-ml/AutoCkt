import numpy as np

cur_specs = np.array([7.20504133e03, 8.66115710e-04, 4.56972473e01, 2.69241239e07])
ideal_specs = np.array([8.02000000e02, 1.68455518e-03, 6.00000000e01, 1.90104525e06])

cur_specs = np.array([7.20504133e03, 8.66115710e-04, 6.56972473e01, 2.69241239e07])
ideal_specs = np.array([8.02000000e02, 1.68455518e-03, 6.00000000e01, 1.90104525e06])


def lookup(spec, goal_spec):
    goal_spec = [float(e) for e in goal_spec]
    norm_spec = (spec - goal_spec) / (goal_spec + spec)
    return norm_spec


specs_id = ["gain_min", "ibias_max", "phm_min", "ugbw_min"]


def reward(spec, goal_spec):
    """
    Reward: doesn't penalize for overshooting spec, is negative
    """
    # rel_specs = self.lookup(spec, goal_spec)
    # pos_val = []
    # reward = 0.0
    # for i, rel_spec in enumerate(rel_specs):
    #     if self.specs_id[i] == "ibias_max":
    #         rel_spec = rel_spec * -1.0  # /10.0
    #     if rel_spec < 0:
    #         reward += rel_spec
    #         pos_val.append(0)
    #     else:
    #         pos_val.append(1)

    # return reward if reward < -0.02 else 10

    norm_specs = lookup(spec, goal_spec)

    # pay attention to reward calculation, this is not quite the reward function in RL
    # but rather a penalty value for the optimization process
    reward = 0
    for i, rel_spec in enumerate(norm_specs):
        # For power,  smaller is better
        # For gain, larger (compared to the target/goal) is better
        # For other specs (pm, ugbw, etc.), smaller is better
        assert specs_id[i] in ["ibias_max", "gain_min", "ugbw_min", "phm_min"]
        if specs_id[i] == "ibias_max" and rel_spec > 0:
            reward += np.abs(rel_spec)  # /10
        elif specs_id[i] == "gain_min" and rel_spec < 0:
            reward += 3 * np.abs(rel_spec)  # /10
        elif specs_id[i] != "ibias_max" and rel_spec < 0:
            reward += np.abs(rel_spec)
    return -reward
    # return -reward if -reward < -1.0 else 10


print(reward(cur_specs, ideal_specs))
