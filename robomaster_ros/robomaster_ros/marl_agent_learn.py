import os
import time
from robomaster_soccer_env import RobomasterSoccerEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

PPO_robot1_models_dir = f"models/robot1/PPO-{int(time.time())}"
PPO_robot2_models_dir = f"models/robot2/PPO-{int(time.time())}"
#A2C_models_dir = f"models/A2C-{int(time.time())}"

PPO_robot1_logs_dir = f"logs/robot1/PPO-{int(time.time())}"
PPO_robot2_logs_dir = f"logs/robot2/PPO-{int(time.time())}"
#A2C_logs_dir = f"logs/A2C-{int(time.time())}"

if not os.path.exists(PPO_robot1_models_dir):
    os.makedirs(PPO_robot1_models_dir)

if not os.path.exists(PPO_robot2_models_dir):
    os.makedirs(PPO_robot2_models_dir)

if not os.path.exists(PPO_robot1_logs_dir):
    os.makedirs(PPO_robot1_logs_dir)

if not os.path.exists(PPO_robot2_logs_dir):
    os.makedirs(PPO_robot2_logs_dir)

#if not os.path.exists(A2C_models_dir):
#    os.makedirs(A2C_models_dir)

#if not os.path.exists(A2C_logs_dir):
#    os.makedirs(A2C_logs_dir)


env = RobomasterSoccerEnv()
wrapped_env = make_vec_env(lambda: env, n_envs=1)

policies = {}
policies["robot1"] = PPO(
        "MlpPolicy",
        env=wrapped_env,
        verbose=1,
        tensorboard_log=PPO_robot1_logs_dir,
        policy_kwargs=dict(
            net_arch=[64, 64]
        )
    )
policies["robot2"] = PPO(
        "MlpPolicy",
        env=wrapped_env,
        verbose=1,
        tensorboard_log=PPO_robot2_logs_dir,
        policy_kwargs=dict(
            net_arch=[64, 64]
        )
    )

episode_rewards = {agent: [] for agent in env.possible_agents}

timesteps = 10000
for i in range(timesteps):
    obs = wrapped_env.reset()
    done = False
    episode_reward = {
        "robot1": 0,
        "robot2": 0
    }
    while not done:
        actions = {}
        for agent in env.possible_agents:
            action, _ = policies[agent].predict(obs[agent])
            actions[agent] = action
        observations, rewards, terminations, truncations, info = wrapped_env.step(actions)

        for agent in env.possible_agents:
            episode_reward[agent] += rewards[agent]

        for agent in env.possible_agents:
            policies[agent].learn(
                total_timesteps=100,
                reset_num_timesteps=False
            )
        done = terminations["robot1"] or terminations["robot2"] or truncations["robot1"] or truncations["robot2"]
        obs = observations
    
    for agent in env.possible_agents:
            episode_rewards[agent].append(episode_reward[agent])
            print(f"Episode {i+1}, Agent {agent} reward: {episode_reward[agent]:.2f}")
    policies["robot1"].save(f"{PPO_robot1_models_dir}/{timesteps*i}")
    policies["robot2"].save(f"{PPO_robot2_models_dir}/{timesteps*i}")


env.stop()
env.close()
