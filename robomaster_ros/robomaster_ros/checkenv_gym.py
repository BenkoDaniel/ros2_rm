from stable_baselines3.common.env_checker import check_env
from robomaster_soccer_env_gym import RobomasterSoccerEnvGym

env = RobomasterSoccerEnvGym()

check_env(env)
