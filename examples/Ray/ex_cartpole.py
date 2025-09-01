import ray
from ray import tune
from ray.rllib.agents import ppo
import gym


# Minimal custom environment (wraps CartPole)
class TwoStageAmp(gym.Env):
    def __init__(self, config):
        self.env = gym.make("CartPole-v1")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


if __name__ == "__main__":
    ray.init()

    config_train = {
        "train_batch_size": 1200,
        "horizon": 30,
        "num_gpus": 0,
        "model": {"fcnet_hiddens": [64, 64]},
        "num_workers": 1,  # keep small for demo
        "env_config": {"generalize": True, "run_valid": False},
    }

    trials = tune.run_experiments(
        {
            "train_cartpole_demo": {
                "checkpoint_freq": 1,
                "run": "PPO",
                "env": TwoStageAmp,
                "stop": {"episode_reward_mean": 195},  # stop condition
                "config": config_train,
            },
        }
    )
