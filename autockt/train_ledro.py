import ray
import ray.tune as tune
from ray.rllib.agents import ppo
from autockt.envs.ngspice_vanilla_opamp import TwoStageAmp
from autockt.envs.ngspice_ledro_d_fc import LEDRO_D_FC

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", "-cpd", type=str)
args = parser.parse_args()
ray.init()

# configures training of the agent with associated hyperparameters
# See Ray documentation for details on each parameter

config_train = {
    "train_batch_size": 1200,
    "horizon": 200,  # 100 discrete values, starting from 33 (envs/ngspice_ledro_d_dc.py line 210), max step size = 2, -> 33 + 2*200 = 433.
    "num_gpus": 0,
    # "model": {"fcnet_hiddens": [64, 64]},
    "model": {"fcnet_hiddens": [128, 128, 128]},
    "num_workers": 6,
    "env_config": {"generalize": True, "run_valid": False},
}

# Runs training and saves the result in ~/ray_results/train_ngspice_45nm
# If checkpoint fails for any reason, training can be restored
if True:
    trials = tune.run_experiments(
        {
            "train_7nFinFET_LEDRO_D_FC": {
                "checkpoint_freq": 10,
                "run": "PPO",
                "env": LEDRO_D_FC,
                "stop": {"episode_reward_mean": -0.02},
                "config": config_train,
            },
        }
    )
else:
    print("RESTORING NOW!!!!!!")
    exit()  # do not restore for now
    tune.run_experiments(
        {
            "restore_ppo": {
                "run": "PPO",
                "config": config_train,
                "env": LEDRO_D_FC,
                # "restore": trials[0]._checkpoint.value},
                "restore": args.checkpoint_dir,
                "checkpoint_freq": 1,
            },
        }
    )
