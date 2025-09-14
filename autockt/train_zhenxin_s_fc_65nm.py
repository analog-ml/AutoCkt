import ray
import ray.tune as tune
from ray.rllib.algorithms.ppo import PPO
from autockt.envs.ngspice_vanilla_opamp import TwoStageAmp
from autockt.envs.ngspice_ledro_d_fc import LEDRO_D_FC
import numpy as np
from ray.rllib.callbacks.callbacks import RLlibCallback


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


class RewardMonitor(RLlibCallback):
    def on_episode_created(self, *, episode, **kwargs):
        # Initialize an empty list in the `custom_data` property of `episode`.
        episode.custom_data["reward_0"] = []
        episode.custom_data["reward_1"] = []
        episode.custom_data["reward_2"] = []
        episode.custom_data["reward_3"] = []
        episode.custom_data["reward_4"] = []
        episode.custom_data["reward_5"] = []
        episode.custom_data["reward_idx"] = []

    def on_episode_step(self, *, episode, env, **kwargs):
        # Append the current reward to the list.
        episode.custom_data["reward_0"].append(env.envs[0].unwrapped.ret_reward_0)
        episode.custom_data["reward_1"].append(env.envs[0].unwrapped.ret_reward_1)
        episode.custom_data["reward_2"].append(env.envs[0].unwrapped.ret_reward_2)
        episode.custom_data["reward_3"].append(env.envs[0].unwrapped.ret_reward_3)
        episode.custom_data["reward_4"].append(env.envs[0].unwrapped.ret_reward_4)
        episode.custom_data["reward_5"].append(env.envs[0].unwrapped.ret_reward_5)
        episode.custom_data["reward_idx"].append(env.envs[0].unwrapped.reward_idx)

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        avg_reward_0 = np.mean(episode.custom_data["reward_0"])
        avg_reward_1 = np.mean(episode.custom_data["reward_1"])
        avg_reward_2 = np.mean(episode.custom_data["reward_2"])
        avg_reward_3 = np.mean(episode.custom_data["reward_3"])
        avg_reward_4 = np.mean(episode.custom_data["reward_4"])
        avg_reward_5 = np.mean(episode.custom_data["reward_5"])
        reward_idx = np.max(episode.custom_data["reward_idx"])

        metrics_logger.log_value("reward_org", avg_reward_0, reduce="mean", window=1)
        metrics_logger.log_value("reward_1_mean", avg_reward_1, reduce="mean", window=1)
        metrics_logger.log_value("reward_2_mean", avg_reward_2, reduce="mean", window=1)
        metrics_logger.log_value("reward_3_mean", avg_reward_3, reduce="mean", window=1)
        metrics_logger.log_value("reward_4_mean", avg_reward_4, reduce="mean", window=1)
        metrics_logger.log_value("reward_5_mean", avg_reward_5, reduce="mean", window=1)
        metrics_logger.log_value("reward_idx", reward_idx, reduce="max", window=1)


config_train = {
    "train_batch_size": 1200,
    "horizon": 50,
    "num_gpus": 0,
    "rollout_fragment_length": 50,
    # "model": {"fcnet_hiddens": [64, 64]},
    "model": {"fcnet_hiddens": [128, 128, 128]},
    "num_workers": 6,
    "env_config": {"generalize": False, "run_valid": False},
    "callbacks": RewardMonitor,
}
# Runs training and saves the result in ~/ray_results/train_ngspice_45nm
# If checkpoint fails for any reason, training can be restored
if not args.checkpoint_dir:
    trials = tune.run_experiments(
        {
            "train_65nm_Zhenxin_S_FC": {
                # "checkpoint_freq": 10,
                "run": "PPO",
                "env": Zhenxin_S_FC,  # ADD_CIRCUIT
                "stop": {"episode_reward_mean": -0.02},
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
