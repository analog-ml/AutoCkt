import ray
import ray.tune as tune
from ray.rllib.agents import ppo
from autockt.envs.ngspice_vanilla_opamp import TwoStageAmp
from autockt.envs.ngspice_ledro_d_fc import LEDRO_D_FC

# ADD_CIRCUIT
from autockt.envs.ngspice_ledro_d_fc45 import LEDRO_D_FC45
from autockt.envs.ngspice_zhenxin_s_fc import Zhenxin_S_FC

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", "-cpd", type=str, default=None)
args = parser.parse_args()
ray.init()

# configures training of the agent with associated hyperparameters
# See Ray documentation for details on each parameter
config_train = {
    # "sample_batch_size": 200,
    "train_batch_size": 2400,
    # "sgd_minibatch_size": 1200,
    # "num_sgd_iter": 3,
    # "lr":1e-3,
    # "vf_loss_coeff": 0.5,
    "horizon": 400,
    # "rollout_fragment_length": 1200,
    # "rollout_fragment_length": 200,
    "num_gpus": 0,
    "model": {"fcnet_hiddens": [64, 64]},
    "num_workers": 6,
    "env_config": {"generalize": False, "run_valid": False},
}
config_train = {
    "train_batch_size": 1200,
    "horizon": 50,
    "num_gpus": 0,
    # "model": {"fcnet_hiddens": [64, 64]},
    "model": {"fcnet_hiddens": [128, 128, 128]},
    "num_workers": 6,
    "env_config": {"generalize": False, "run_valid": False},
}
# Runs training and saves the result in ~/ray_results/train_ngspice_45nm
# If checkpoint fails for any reason, training can be restored
if not args.checkpoint_dir:
    trials = tune.run_experiments(
        {
            "train_65nm_Zhenxin_S_FC": {
                "checkpoint_freq": 10,
                "run": "PPO",
                "env": Zhenxin_S_FC,  # ADD_CIRCUIT
                # "stop": {"episode_reward_mean": -0.02},
                # "stop": {"episode_reward_mean": -0.25},
                "config": config_train,
            },
        }
    )
    # trials = tune.run(
    #     "PPO",
    #     config=config_train,
    #     stop={"training_iteration": 1000},
    #     checkpoint_freq=10,
    #     name="train_7nFinFET_LEDRO_D_DC_1",
    # )

else:
    print("RESTORING NOW!!!!!!")
    tune.run_experiments(
        {
            "restore_ppo": {
                "run": "PPO",
                "config": config_train,
                "env": LEDRO_D_FC45,
                # "restore": trials[0]._checkpoint.value},
                "restore": args.checkpoint_dir,
                "checkpoint_freq": 1,
            },
        }
    )
