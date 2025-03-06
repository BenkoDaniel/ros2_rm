from stable_baselines3 import PPO, A2C
import os
import time
from robomaster_soccer_env import RobomasterSoccerEnv


PPO_models_dir = f"models/PPO-{int(time.time())}"
A2C_models_dir = f"models/A2C-{int(time.time())}"

PPO_logs_dir = f"logs/PPO-{int(time.time())}"
A2C_logs_dir = f"logs/A2C-{int(time.time())}"

if not os.path.exists(PPO_models_dir):
    os.makedirs(PPO_models_dir)

if not os.path.exists(PPO_logs_dir):
    os.makedirs(PPO_logs_dir)

if not os.path.exists(A2C_models_dir):
    os.makedirs(A2C_models_dir)

if not os.path.exists(A2C_logs_dir):
    os.makedirs(A2C_logs_dir)

env = RobomasterSoccerEnv()
env.reset()

PPO_model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=PPO_logs_dir)
A2C_model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=A2C_logs_dir)

timesteps = 10000
for i in range(1, 100):
    PPO_model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name="PPO")
    PPO_model.save(f"{PPO_models_dir}/{timesteps*i}")

env.close()
