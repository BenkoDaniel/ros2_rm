from robomaster_soccer_env import RobomasterSoccerEnv

env = RobomasterSoccerEnv()
episodes = 5

for episode in range(episodes):
    done = False
    observations = env.reset()
    while not done:
        space1 = env.action_space("robot1")
        space2 = env.action_space("robot2")

        random_action1 = space1.sample()
        random_action2 = space2.sample()
        #print("action", random_action)
        random_actions = {
            "robot1": random_action1,
            "robot2": random_action2
        }
        observations, rewards, terminations, truncations, info = env.step(random_actions)
        done = terminations["robot1"] or terminations["robot2"] or truncations["robot1"] or truncations["robot2"]
        print("reward", rewards)
        if done:
            print("done, reseting environment")

env.stop()