import os
import time
from robomaster_soccer_env import RobomasterSoccerEnv
import ray
from ray import tune
from ray.tune.registry import register_env
import gymnasium as gym
from pettingzoo.utils import ParallelEnv
from ray.rllib.algorithms.ppo import PPOConfig
import numpy as np


PPO_models_dir = f"models/PPO-{int(time.time())}"
#PPO_robot2_models_dir = f"models/robot2/PPO-{int(time.time())}"
#A2C_models_dir = f"models/A2C-{int(time.time())}"

PPO_logs_dir = f"logs/PPO-{int(time.time())}"
#PPO_robot2_logs_dir = f"logs/robot2/PPO-{int(time.time())}"
#A2C_logs_dir = f"logs/A2C-{int(time.time())}"

if not os.path.exists(PPO_models_dir):
    os.makedirs(PPO_models_dir)

#if not os.path.exists(PPO_robot2_models_dir):
#    os.makedirs(PPO_robot2_models_dir)

if not os.path.exists(PPO_logs_dir):
    os.makedirs(PPO_logs_dir)

#if not os.path.exists(PPO_robot2_logs_dir):
#    os.makedirs(PPO_robot2_logs_dir)

#if not os.path.exists(A2C_models_dir):
#    os.makedirs(A2C_models_dir)

#if not os.path.exists(A2C_logs_dir):
#    os.makedirs(A2C_logs_dir)
def create_env(env_config):
    env = RobomasterSoccerEnv()
    return env

register_env("RobomasterSoccer-v0", create_env)

config = (
    PPOConfig()
    .environment("RobomasterSoccer-v0")
    .multi_agent(
        policies={
            "robot1": (None,
                       gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
                       gym.spaces.Box(low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), dtype=np.float32),
                       {"learning_rate": tune.grid_search([0.0001, 0.0003, 0.001])}),
            "robot2": (None,
                       gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
                       gym.spaces.Box(low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), dtype=np.float32),
                       {"learning_rate": tune.grid_search([0.0001, 0.0003, 0.001])}),
        },
        policy_mapping_fn=lambda agent_id:agent_id,
        policies_to_train=["robot1", "robot2"]
    )
    .framework("torch")
    .evaluation(
        evaluation_interval=10,
        evaluation_config={
            "render_env": True,
            "env_config": {
                "render_mode": "human"
            }
        }
    )
)

try:
    ray.init()
    tune.run(
        "PPO",
        config=config,
        stop={"training_iteration": 1000},
        checkpoint_at_end=True,
        checkpoint_freq=500,
        verbose=1,
        storage_path=os.path.abspath(PPO_models_dir),
        metric="episode_reward_mean",
        mode="max"
    )
finally:
    ray.shutdown()