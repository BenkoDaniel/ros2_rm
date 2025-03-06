from robomaster_soccer_env import RobomasterSoccerEnv

env = RobomasterSoccerEnv()
episodes = 30

for episode in range(episodes):
    done = False
    obs = env.reset()
    while not done:
        random_action = env.action_space.sample()
        print("action", random_action)
        obs, reward, done, info = env.step(random_action)
        print("reward", reward)