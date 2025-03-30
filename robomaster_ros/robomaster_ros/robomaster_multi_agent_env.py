from pettingzoo.utils import ParallelEnv
from robomaster_soccer_env import RobomasterSoccerEnv

class MultiAgentEnv(ParallelEnv):
    def __init__(self):
        super().__init__()
        self.env = RobomasterSoccerEnv()
    
    def reset(self):
        obs_1, obs_2 = self.env.reset()
        return {"robot1": obs_1, "robot2": obs_2}
    
    def step(self, actions):
        (obs_1, obs_2), (rew_1, rew_2), done, _ = self.env.step([actions["robot1"], actions["robot2"]])
        return {"robot1": obs_1, "robot2": obs_2}, {"robot1": rew_1, "robot2": rew_2}, {"robot1": done, "robot2": done}, {}

    def render(self):
        self.env.render()