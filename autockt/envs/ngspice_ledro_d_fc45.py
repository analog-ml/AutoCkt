"""
A new ckt environment based on a new structure of MDP
"""

import gym
from gym import spaces

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
from eval_engines.ngspice.LEDRO_D_FC import *
from eval_engines.ngspice.LEDRO_D_FC45 import *


from loguru import logger
import sys

from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()


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

class ActionNormalizer():
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


class LEDRO_D_FC45(gym.Env):
    metadata = {"render.modes": ["human"]}

    PERF_LOW = -1
    PERF_HIGH = 0

    # obtains yaml file
    path = os.getcwd()
    CIR_YAML = (
        path + "/eval_engines/ngspice/ngspice_inputs/yaml_files/ledro_d_fc45.yaml"
    )

    def __init__(self, env_config):
        self.multi_goal = env_config.get("multi_goal", False)
        self.generalize = env_config.get("generalize", False)
        num_valid = env_config.get("num_valid", 50)
        self.specs_save = env_config.get("save_specs", False)
        self.valid = env_config.get("run_valid", False)

        self.env_steps = 0
        with open(LEDRO_D_FC45.CIR_YAML, "r") as f:
            yaml_data = yaml.load(f, OrderedDictYAMLLoader)

        # design specs
        if self.generalize == False:
            specs = yaml_data["target_specs"]
        else:
            load_specs_path = (
                LEDRO_D_FC45.path + "/autockt/gen_specs/ngspice_specs_gen_ledro_d_fc45"
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
        self.sim_env = LEDRO_D_FC45_Class(
            yaml_path=LEDRO_D_FC45.CIR_YAML, num_process=1, path=LEDRO_D_FC45.path
        )
        # self.action_meaning = [-1, 0, 2]
        # self.action_space = spaces.Tuple(
        #     [spaces.Discrete(len(self.action_meaning))] * len(self.params_id)
        # )
        self.action_space = spaces.Box(low=-1, high=1, shape=(25, ), dtype=np.float64)
        # print (action_space.sample())

        # L: Rationale: start at ~2× technology minimum to reduce short-channel effects and improve matching.
        action_space_low = np.array(
            [
            0.12, 90, 1,
            0.12, 90, 1,    
            0.12, 90, 1,    
            0.12, 90, 1,    
            0.12, 90, 1,    
            0.12, 90, 1,    
            0.1, 
            0.1, 
            0.1, 
            0.1, 
            0.1, 

        1,
        1

            ]
        )

        action_space_high = np.array(
            [
            200, 2000, 100,
            200, 2000, 100,    
            200, 2000, 100,    
            200, 2000, 100,    
            200, 2000, 100,    
            200, 2000, 100,    
            1.2, 
            1.2, 
            1.2, 
            1.2, 
            1.2, 

        50,
        50

            ]
        )
        self.action_normalizer = ActionNormalizer(action_space_low=action_space_low, action_space_high =  action_space_high)


        # self.action_space = spaces.Discrete(len(self.action_meaning)**len(self.params_id))
        self.observation_space = spaces.Box(
            low=np.array(
                [LEDRO_D_FC45.PERF_LOW] * 2 * len(self.specs_id)
                + len(self.params_id) * [1]
            ),
            high=np.array(
                [LEDRO_D_FC45.PERF_HIGH] * 2 * len(self.specs_id)
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

    def reset(self):
        # if multi-goal is selected, every time reset occurs, it will select a different design spec as objective
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
        self.cur_params_idx = np.array([193.9020858253666 ,1318.8789684310007, 66, 106.3710338395701,
            517.342182701802, 69, 51.58973768557556 ,1148.0132888755084, 68,
            62.67064928753026, 1360.398953352503, 18, 73.15862009109797,
            1718.7658807907076, 61, 192.34328350363728, 1205.9109268448633, 27,
            0.7761091728857539, 1.1431755589519739, 0.6157432007527375,
            1.1177122194734337, 0.9243351863878987, 24, 9])

        self.cur_specs = self.update(self.cur_params_idx)
        cur_spec_norm = self.lookup(self.cur_specs, self.global_g)
        reward = self.reward(self.cur_specs, self.specs_ideal)

        # observation is a combination of current specs distance from ideal, ideal spec, and current param vals
        self.ob = np.concatenate(
            [cur_spec_norm, self.specs_ideal_norm, self.cur_params_idx]
        )
        return self.ob

    def step(self, action):
        """
        :param action: is vector with elements between 0 and 1 mapped to the index of the corresponding parameter
        :return:
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
        action = self.action_normalizer.action(action) # convert [-1.1] range back to normal range
        # action = action.astype(object)

        for idx in [2, 2+3, 5+3, 8+3, 11+3, 14+3, -1, -2]:
            action[idx] = int(action[idx])

        self.cur_params_idx = action

        # Get current specs and normalize
        self.cur_specs = self.update(self.cur_params_idx)
        #logger.info("current specs simulation: " + str(self.cur_specs))
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
        writer.add_scalar('gain', self.cur_specs[0], self.env_steps)
        writer.add_scalar('ugbw', self.cur_specs[1], self.env_steps)
        writer.add_scalar('pm', self.cur_specs[2], self.env_steps)
        writer.add_scalar('power', self.cur_specs[3], self.env_steps)
        # print('cur ob:' + str(self.cur_specs))
        # print('ideal spec:' + str(self.specs_ideal))
        # print(reward)
        return self.ob, reward, done, {}

    def lookup(self, spec, goal_spec):
        goal_spec = [float(e) for e in goal_spec]
        norm_spec = (spec - goal_spec) / (goal_spec + spec)
        return norm_spec

    def reward(self, spec, goal_spec):
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
                reward += 3 * np.abs(rel_spec)  # /10
            elif self.specs_id[i] != "ibias_max" and rel_spec < 0:
                reward += np.abs(rel_spec)
        return -reward

    def update(self, params_idx):
        """

        :param action: an int between 0 ... n-1
        :return:
        """

        # params = [self.params[i][params_idx[i]] for i in range(len(self.params_id))]
        # param_val = [OrderedDict(list(zip(self.params_id, params)))]


        param_names = [
            "wp1", "lp1", "mp1",
            "wp2", "lp2", "mp2",
            "wp3", "lp3", "mp3",
            "wp4", "lp4", "mp4",
            "wp5", "lp5", "mp5",
            "wp6", "lp6", "mp6",

            "vbiasp1",
            "vbiasp2",

            "vbiasn0",
            "vbiasn1",
            "vbiasn2",

            "cl",
            "cc"
            ]
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
    env = LEDRO_D_FC45(env_config)
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
