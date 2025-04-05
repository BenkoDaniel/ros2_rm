from robomaster_soccer_env_gym import RobomasterSoccerEnvGym

env = RobomasterSoccerEnvGym()
episodes = 3

for episode in range(episodes):
    done = False
    obs = env.reset()
    while not done:
        random_action = env.action_space.sample()
        print("action", random_action)
        obs, reward, terminated, truncated, info = env.step(random_action)
        done = terminated or truncated
        if done:
            print('episode ended')
        print("reward", reward)