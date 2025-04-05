from pettingzoo.test import parallel_api_test
from robomaster_soccer_env import RobomasterSoccerEnv

env = RobomasterSoccerEnv()
parallel_api_test(env)