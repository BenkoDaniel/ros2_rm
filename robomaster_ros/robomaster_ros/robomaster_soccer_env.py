import gym
from gym import spaces
import numpy as np

class RobomasterSoccerEnv(gym.Env):
    '''Custom Environment that follows the gym interface'''
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(RobomasterSoccerEnv, self).__init__()
        self.action_space = spaces.Disrete(8)
        self.observation_space = spaces.Box(low=0, high=0, shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)

    def step(self, action):
        return self.observation, self.reward, self.done, self.info

    def reset(self):
        return self.observation   #reward, done, info can't be included

    def render(self, mode='human'):
        ...

    def close(self):
        ...

