"""
A new ckt environment based on a new structure of MDP
"""

from typing import Optional
import gymnasium as gym
import gymnasium.spaces as spaces

import numpy as np
import random
import psutil

from multiprocessing.dummy import Pool as ThreadPool
from collections import OrderedDict
import yaml
import yaml.constructor
import statistics
import os
import IPython
import itertools
from eval_engines.util.core import *
import pickle
import os

from eval_engines.ngspice.TwoStageClass import *

# ADD_CIRCUIT
# tip: comment un-used classes to quickly grasp errors
# from eval_engines.ngspice.LEDRO_D_FC45 import *
# from eval_engines.ngspice.LEDRO_D_FC import *
from eval_engines.ngspice.Zhenxin_S_FC import *
import datetime


from loguru import logger
import sys

from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Writer will output to ./runs/ directory by default

# get timestamp in form of string
date_time_obj = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# writer = SummaryWriter(date_time_obj)


# Custom format string
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# Clear default logger
logger.remove()

# Log to stdout
logger.add(sys.stdout, format=log_format, level="DEBUG")

# Log to file with rotation and retention
logger.add(
    "logs/ngspice_ledro_d_fc.log",
    format=log_format,
    level="DEBUG",
    rotation="1 day",
    retention="7 days",
)


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


# way of ordering the way a yaml file is read
class OrderedDictYAMLLoader(yaml.Loader):
    """
    A YAML loader that loads mappings into ordered dictionaries.
    """

    def __init__(self, *args, **kwargs):
        yaml.Loader.__init__(self, *args, **kwargs)

        self.add_constructor("tag:yaml.org,2002:map", type(self).construct_yaml_map)
        self.add_constructor("tag:yaml.org,2002:omap", type(self).construct_yaml_map)

    def construct_yaml_map(self, node):
        data = OrderedDict()
        yield data
        value = self.construct_mapping(node)
        data.update(value)

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
        else:
            raise yaml.constructor.ConstructorError(
                None,
                None,
                "expected a mapping node, but found %s" % node.id,
                node.start_mark,
            )

        mapping = OrderedDict()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping


# ADD_CIRCUIT
class Zhenxin_S_FC(gym.Env):
    metadata = {"render.modes": ["human"]}

    PERF_LOW = -1
    PERF_HIGH = 0

    # obtains yaml file
    path = os.getcwd()
    # ADD_CIRCUIT
    CIR_YAML = (
        path + "/eval_engines/ngspice/ngspice_inputs/yaml_files/zhenxin_s_fc.yaml"
    )

    def __init__(self, env_config):
        """
        Initialize the environment from a configuration dictionary, load circuit specs and parameter grids, set up the simulator, action/observation spaces, and initial state.

        env_config keys recognized:
        - "multi_goal" (bool): if True, allow multiple per-instance goal vectors; default False.
        - "generalize" (bool): if True, use precomputed generated specs instead of target_specs from YAML; default False.
        - "num_valid" (int): count used when optionally saving sampled specs; default 50.
        - "save_specs" (bool): if True, persist loaded specs to a pickle file; default False.
        - "run_valid" (bool): validation mode flag used when generalize is True; default False.

        Behavior and side effects:
        - Loads circuit/design YAML (CIR_YAML) using an ordered YAML loader and either reads target_specs or loads generated specs from disk (when generalize is True).
        - Constructs an ordered specs mapping, optional saving to a pickle file, and records spec identifiers and the fixed goal index.
        - Builds parameter value grids from YAML "params" and stores parameter identifiers.
        - Initializes the simulator interface (self.sim_env).
        - Defines an 11-dimensional continuous action space in [-1, 1] and an ActionNormalizer that maps actions to the configured physical ranges (action_space_low / action_space_high).
        - Defines the observation space combining normalized current specs, normalized ideal specs, and current parameter values.
        - Initializes runtime state containers: self.cur_specs, self.cur_params_idx, self.g_star (design goal values), self.global_g (normalization factors), and self.obj_idx (objective index for validation).

        No return value.
        """
        self.multi_goal = env_config.get("multi_goal", False)
        self.generalize = env_config.get("generalize", False)
        num_valid = env_config.get("num_valid", 50)
        self.specs_save = env_config.get("save_specs", False)
        self.valid = env_config.get("run_valid", False)

        self.env_steps = 0
        # ADD_CIRCUIT
        with open(Zhenxin_S_FC.CIR_YAML, "r") as f:
            yaml_data = yaml.load(f, OrderedDictYAMLLoader)

        # design specs
        if self.generalize == False:
            specs = yaml_data["target_specs"]
        else:
            load_specs_path = (
                Zhenxin_S_FC.path
                + "/autockt/gen_specs/ngspice_specs_gen_zhenxin_s_fc"  # ADD_CIRCUIT
            )
            with open(load_specs_path, "rb") as f:
                specs = pickle.load(f)

        self.specs = OrderedDict(sorted(specs.items(), key=lambda k: k[0]))
        if self.specs_save:
            with open(
                "specs_" + str(num_valid) + str(random.randint(1, 100000)), "wb"
            ) as f:
                pickle.dump(self.specs, f)

        self.specs_ideal = []
        self.specs_id = list(self.specs.keys())
        self.fixed_goal_idx = -1
        self.num_os = len(list(self.specs.values())[0])

        # param array
        params = yaml_data["params"]
        self.params = []
        self.params_id = list(params.keys())

        for value in params.values():
            param_vec = np.linspace(value[0], value[1], value[2])
            self.params.append(param_vec)

        # initialize sim environment
        # ADD CIRCUIT
        self.sim_env = Zhenxin_S_FC_Class(
            yaml_path=Zhenxin_S_FC.CIR_YAML, num_process=1, path=Zhenxin_S_FC.path
        )
        # self.action_meaning = [-1, 0, 2]
        # self.action_space = spaces.Tuple(
        #     [spaces.Discrete(len(self.action_meaning))] * len(self.params_id)
        # )

        # ADD_CIRCUIT
        self.action_space = spaces.Box(low=-1, high=1, shape=(11,), dtype=np.float64)

        action_space = spaces.Box(low=-1, high=1, shape=(11,), dtype=np.float64)
        # print (action_space.sample())

        # fmt: off
        action_space_low = np.array(
            [
                130,
                130, 
                130, 
                130,
                130,
                130, 
                0.0001, 
                0.0001, 
                0.0001, 
                0.0001, 
                0.01,
            ]
        )

        action_space_high = np.array(
            [
                100000,
                100000, 
                100000,
                100000, 
                100000,
                100000, 
                1.0, 
                1.0, 
                1.0, 
                1.0, 
                10,
            ]
        )

        # fmt: on

        self.action_normalizer = ActionNormalizer(
            action_space_low=action_space_low, action_space_high=action_space_high
        )

        # self.action_space = spaces.Discrete(len(self.action_meaning)**len(self.params_id))
        self.observation_space = spaces.Box(
            low=np.array(
                [Zhenxin_S_FC.PERF_LOW] * 2 * len(self.specs_id)
                + len(self.params_id) * [1]
            ),
            high=np.array(
                [Zhenxin_S_FC.PERF_HIGH] * 2 * len(self.specs_id)
                + len(self.params_id) * [1]
            ),
        )

        # initialize current param/spec observations
        self.cur_specs = np.zeros(len(self.specs_id), dtype=np.float32)
        self.cur_params_idx = np.zeros(len(self.params_id), dtype=np.int32)

        # Get the g* (overall design spec) you want to reach
        self.global_g = []
        for spec in list(self.specs.values()):
            self.global_g.append(float(spec[self.fixed_goal_idx]))
        self.g_star = np.array(self.global_g)
        self.global_g = np.array(yaml_data["normalize"])

        # objective number (used for validation)
        self.obj_idx = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # if multi-goal is selected, every time reset occurs, it will select a different design spec as objective
        """
        Reset the environment state and return the initial observation.

        Resets or (when generalization is enabled) selects a new target design specification, normalizes it, initializes the current parameter vector (hard-coded in this implementation), computes the initial simulated specs for those parameters, evaluates the initial reward, and constructs the initial observation.

        Behavior:
        - If generalize is True:
          - If valid is True, cycles through spec indices using self.obj_idx (wraps to 0 when exceeding available designs).
          - Otherwise selects a random spec index.
          - Sets self.specs_ideal to the selected column across all stored specs.
        - If generalize is False:
          - If multi_goal is False, sets self.specs_ideal to self.g_star (single global goal).
          - If multi_goal is True, selects a random spec index and sets self.specs_ideal accordingly.
        - Computes self.specs_ideal_norm via self.lookup against self.global_g.
        - Assigns a predefined initial parameter vector to self.cur_params_idx (overwrites multiple candidate vectors; final assignment used).
        - Calls self.update(self.cur_params_idx) to compute self.cur_specs and normalizes it.
        - Computes initial reward (via self.reward) and builds the initial observation self.ob by concatenating normalized current specs, normalized ideal specs, and current parameter values.

        Returns:
            numpy.ndarray: The initial observation vector (concatenation of current-specs-normalized, ideal-specs-normalized, and current parameter values).
        """
        if self.generalize == True:
            if self.valid == True:
                if self.obj_idx > self.num_os - 1:
                    self.obj_idx = 0
                idx = self.obj_idx
                self.obj_idx += 1
            else:
                idx = random.randint(0, self.num_os - 1)
            self.specs_ideal = []
            for spec in list(self.specs.values()):
                self.specs_ideal.append(spec[idx])
            self.specs_ideal = np.array(self.specs_ideal)
        else:
            if self.multi_goal == False:
                self.specs_ideal = self.g_star
            else:
                idx = random.randint(0, self.num_os - 1)
                self.specs_ideal = []
                for spec in list(self.specs.values()):
                    self.specs_ideal.append(spec[idx])
                self.specs_ideal = np.array(self.specs_ideal)
        # print("num total:"+str(self.num_os))

        # applicable only when you have multiple goals, normalizes everything to some global_g
        self.specs_ideal_norm = self.lookup(self.specs_ideal, self.global_g)

        # initialize current parameters
        # self.cur_params_idx = np.array([2] * 17)
        # self.cur_params_idx = np.array(
        #     # [2, 2, 2, 2, 2, 2] + [200, 200, 200, 200, 200, 200] + [10, 10, 10, 10, 10]
        #     [33, 33, 33, 33, 33, 33]
        #     + [10, 10, 10, 10, 10]
        # )

        # ADD_CIRCUIT
        # fmt: off
        self.cur_params_idx = np.array([3.74753369e+01 ,1.45339479e+02 ,8.10000000e+01 ,4.47246834e+01,
                5.42556293e+02 ,3.00000000e+01 ,7.92805812e+01, 6.73899490e+02,
                6.50000000e+01 ,5.03197719e+01 ,1.78390864e+02, 8.20000000e+01,
                7.67682715e+01 ,5.71772797e+02 ,7.40000000e+01, 7.27723837e+01,
                2.66384969e+02 ,7.10000000e+01 ,1.03974815e-01, 8.55390346e-01,
                2.17883575e-01 ,1.02317559e+00 ,1.00000000e+01, 1.00000000e+00])
        self.cur_params_idx = np.array([32.77338894433899,262.7480907119822,189.0,158.15349366404624,348.4724582030384,154.0,79.77405852298419,1111.4785699664237,86.0,99.43193620474902,1571.1634654290083,137.0,92.27007942763514,582.0961144404719,165.0,62.95195636704306,742.9661412949395,150.0,0.15006895370431445,0.707724000583208,0.34247109456861674,0.8081614628147477,24.0,18.0])
        self.cur_params_idx = np.array([1.306996077907531,1.3377967426358073,40.0,5.954494693818084,0.8992126892338544,37.0,25.488818790800927,0.4551653253998019,112.0,4.224049384633847,0.5619414658059543,140.0,25.698553139281227,1.0913253825818343,39.0,12.149631485977649,0.586970473291363,1.0,1.1773170908295258,0.2848438042937586,1.0786470679021434,0.8902246665998383,12.0,47.0])
        self.cur_params_idx = np.array([2.4284323390575366,1.338078576088182,78.0,29.264831036522917,1.9892456090951285,33.0,13.980927503463723,1.3113102449401783,17.0,32.12785582624383,1.8436928811518705,5.0,42.023886484328216,1.3431306578872302,46.0,35.47162817259426,1.3712540446476695,77.0,0.6568611016690267,0.17811360700059536,0.686108575948138,0.3053737857576733,1.0,1.0])
        self.cur_params_idx = np.array([242.84323390575366,
                                        133.8078576088182,
                                        292.64831036522917,
                                        198.92456090951285,
                                        139.80927503463723,
                                        131.13102449401783,
                                        0.6568611016690267,
                                        0.17811360700059536,
                                        0.686108575948138,
                                        0.3053737857576733,
                                        1.0,])

        # fmt: on

        self.cur_specs = self.update(self.cur_params_idx)
        cur_spec_norm = self.lookup(self.cur_specs, self.global_g)
        reward = self.reward(self.cur_specs, self.specs_ideal)

        # observation is a combination of current specs distance from ideal, ideal spec, and current param vals
        self.ob = np.concatenate(
            [cur_spec_norm, self.specs_ideal_norm, self.cur_params_idx]
        )
        return self.ob, {}

    def step(self, action):
        """
        Apply an action to update the environment's parameters, run the simulator, and return the next observation, reward, termination flag, and info.

        The provided `action` is expected in the agent's action space (typically values in [-1, 1]); it is first mapped to the environment's parameter value space using self.action_normalizer.action. The mapped values replace the current parameter vector, the simulator is invoked via self.update(...) to produce new specs, and a scalar reward is computed comparing the current specs to the environment goal. The environment's internal observation (self.ob) and step counter (self.env_steps) are updated.

        Parameters:
            action (array-like): Agent action vector (shape matches the environment action space, e.g., length 11). Values are in the agent's action range and will be converted to actual parameter values by the environment's ActionNormalizer.

        Returns:
            tuple:
                observation (np.ndarray): Concatenation of normalized current specs, normalized ideal specs, and the current parameter values.
                reward (float): Reward computed by self.reward(...) for the resulting specs.
                done (bool): True when a terminal condition is met (reward >= 10), otherwise False.
                info (dict): Empty dict (reserved for additional diagnostics).
        """

        # Take action that RL agent returns to change current params
        # action = list(np.reshape(np.array(action), (np.array(action).shape[0],)))
        # self.cur_params_idx = self.cur_params_idx + np.array(
        #     [self.action_meaning[a] for a in action]
        # )

        # #        self.cur_params_idx = self.cur_params_idx + np.array(self.action_arr[int(action)])
        # self.cur_params_idx = np.clip(
        #     self.cur_params_idx,
        #     [0] * len(self.params_id),
        #     [(len(param_vec) - 1) for param_vec in self.params],
        # )
        # # logger.debug(f"current param idx: {str(self.cur_params_idx)}")
        # # print(f"current param idx: {self.cur_params_idx=}")
        # logger.debug("current param idx simulation: " + str(self.cur_params_idx))
        action = self.action_normalizer.action(
            action
        )  # convert [-1.1] range back to normal range
        # action = action.astype(object)

        # ADD_CIRCUIT
        # for idx in [2, 2 + 3, 5 + 3, 8 + 3, 11 + 3, 14 + 3]:
        #     try:
        #         action[idx] = int(action[idx])
        #     except:
        #         logger.debug("error when rounding the M value")
        #         action[idx] = 1

        self.cur_params_idx = action

        # Get current specs and normalize
        self.cur_specs = self.update(self.cur_params_idx)
        # logger.info("current specs simulation: " + str(self.cur_specs))
        cur_spec_norm = self.lookup(self.cur_specs, self.global_g)
        reward = self.reward(self.cur_specs, self.specs_ideal)
        done = False

        # incentivize reaching goal state
        if reward >= 10:
            done = True
            print("-" * 10)
            print("params = ", self.cur_params_idx)
            print("specs:", self.cur_specs)
            print("ideal specs:", self.specs_ideal)
            print("re:", reward)
            print("-" * 10)

        self.ob = np.concatenate(
            [cur_spec_norm, self.specs_ideal_norm, self.cur_params_idx]
        )
        self.env_steps = self.env_steps + 1

        logger.info("current specs:" + str(self.cur_specs) + ", reward: " + str(reward))
        # writer.add_scalar('gain', self.cur_specs[0], self.env_steps)
        # writer.add_scalar('ugbw', self.cur_specs[1], self.env_steps)
        # writer.add_scalar('pm', self.cur_specs[2], self.env_steps)
        # writer.add_scalar('power', self.cur_specs[3], self.env_steps)
        # print('cur ob:' + str(self.cur_specs))
        # print('ideal spec:' + str(self.specs_ideal))
        # print(reward)
        truncated = False
        return self.ob, reward, done, truncated, {}

    def lookup(self, spec, goal_spec):
        goal_spec = [float(e) for e in goal_spec]
        norm_spec = (spec - goal_spec) / (goal_spec + spec)
        return norm_spec

    def reward(self, spec, goal_spec):
        """
        Compute a scalar objective for the current specs relative to a goal specification.

        This function:
        - Normalizes the difference between `spec` and `goal_spec` using self.lookup.
        - Accumulates a penalty according to per-spec rules:
          - "ibias_max": penalize only when the normalized value is positive (larger than goal).
          - "gain_min": penalize only when the normalized value is negative (smaller than goal).
          - All other tracked specs ("ugbw_min", "phm_min"): penalize when the normalized value is negative (smaller than goal).
        - Returns either the negated accumulated penalty (a negative value) or 10 when the negated penalty is above a small threshold, indicating a sufficiently good match.

        Parameters:
            spec (array-like): Current specification values (ordered to match self.specs_id).
            goal_spec (array-like): Target/ideal specification values.

        Returns:
            float: Either a negative penalty (-sum_of_violations) or 10 when the negated penalty is >= -0.02 (tolerance threshold).
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

        norm_specs = self.lookup(spec, goal_spec)

        # pay attention to reward calculation, this is not quite the reward function in RL
        # but rather a penalty value for the optimization process
        reward = 0
        for i, rel_spec in enumerate(norm_specs):
            # For power,  smaller is better
            # For gain, larger (compared to the target/goal) is better
            # For other specs (pm, ugbw, etc.), smaller is better
            assert self.specs_id[i] in ["ibias_max", "gain_min", "ugbw_min", "phm_min"]
            if self.specs_id[i] == "ibias_max" and rel_spec > 0:
                reward += np.abs(rel_spec)  # /10
            elif self.specs_id[i] == "gain_min" and rel_spec < 0:
                reward += 1 * np.abs(rel_spec)  # /10
            elif self.specs_id[i] != "ibias_max" and rel_spec < 0:
                reward += np.abs(rel_spec)
        # return -reward
        return -reward if -reward < -0.02 else 10

    def update(self, params_idx):
        """
        Update the circuit design using the provided parameter vector, run the simulator, and return the resulting specifications.

        Parameters:
            params_idx (Sequence[int|float]): Sequence of 11 parameter values (in the same order as the internal
                param_names: ["w_m12","w_m3","w_m45","w_m67","w_m89","w_m1011","vbp1","vbp2","vbn1","vbn2","cc"].
                These are treated as the parameter values passed to the simulator.

        Returns:
            numpy.ndarray: 1-D array of simulated specification values. The specs are taken from the simulator's
            output, sorted by specification name (ascending) before conversion to the array.
        """

        # params = [self.params[i][params_idx[i]] for i in range(len(self.params_id))]
        # param_val = [OrderedDict(list(zip(self.params_id, params)))]

        # ADD_CIRCUIT
        # fmt: off
        param_names = [
            "w_m12",
            "w_m3", 
            "w_m45",
            "w_m67", 
            "w_m89", 
            "w_m1011", 
            "vbp1",
            "vbp2",
            "vbn1",
            "vbn2",
            "cc"
            ]
        # fmt: on

        param_val = [OrderedDict(list(zip(param_names, params_idx)))]

        # run param vals and simulate
        cur_specs = OrderedDict(
            sorted(
                self.sim_env.create_design_and_simulate(param_val[0])[1].items(),
                key=lambda k: k[0],
            )
        )
        cur_specs = np.array(list(cur_specs.values()))

        return cur_specs


def main():
    env_config = {"generalize": True, "valid": True}
    env = Zhenxin_S_FC(env_config)
    env.reset()
    # env.step(
    #     [
    #         2,
    #         2,
    #         2,
    #         2,
    #         2,
    #         2,
    #         10 - 9,
    #         10 - 9,
    #         10 - 9,
    #         10 - 9,
    #         10 - 9,
    #         10 - 9,
    #         0.2,
    #         0.2,
    #         0.2,
    #         0.2,
    #         0.2,
    #     ]
    # )
    env.step([2] * 11)

    IPython.embed()


if __name__ == "__main__":
    main()
