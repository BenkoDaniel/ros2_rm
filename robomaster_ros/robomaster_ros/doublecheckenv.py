from robomaster_soccer_env import RobomasterSoccerEnv

env = RobomasterSoccerEnv()
episodes = 5

for episode in range(episodes):
    done = False
    obs = env.reset()
    while not done:
        random_action = env.action_space.sample()
        #print("action", random_action)
        obs, reward, terminated, truncated, info = env.step(random_action)
        done = terminated or truncated
        print("reward", reward)
        if done:
            print("done, reseting environment")

env.stop()