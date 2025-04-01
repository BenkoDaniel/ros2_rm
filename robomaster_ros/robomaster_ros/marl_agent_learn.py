import os
import time
import supersuit as ss
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
from pettingzoo.utils import parallel_to_aec
import gymnasium as gym
from gymnasium.wrappers import OrderEnforcing
from stable_baselines3 import PPO, A2C
from robomaster_soccer_env import RobomasterSoccerEnv


PPO_models_dir = f"models/PPO-{int(time.time())}"
#A2C_models_dir = f"models/A2C-{int(time.time())}"

PPO_logs_dir = f"logs/PPO-{int(time.time())}"
#A2C_logs_dir = f"logs/A2C-{int(time.time())}"

if not os.path.exists(PPO_models_dir):
    os.makedirs(PPO_models_dir)

if not os.path.exists(PPO_logs_dir):
    os.makedirs(PPO_logs_dir)

#if not os.path.exists(A2C_models_dir):
#    os.makedirs(A2C_models_dir)

#if not os.path.exists(A2C_logs_dir):
#    os.makedirs(A2C_logs_dir)


env = RobomasterSoccerEnv()
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

PPO_model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=PPO_logs_dir, ent_coef=0.01, learning_rate=0.0005, gamma=0.99, clip_range=0.2)
#A2C_model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=A2C_logs_dir)

timesteps = 10000
for i in range(1, 100):
    PPO_model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name="PPO")
    PPO_model.save(f"{PPO_models_dir}/{timesteps*i}")

env.close()
