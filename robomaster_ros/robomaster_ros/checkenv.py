from stable_baselines3.common.env_checker import check_env
from robomaster_soccer_env import RobomasterSoccerEnv

env = RobomasterSoccerEnv()

check_env(env)
