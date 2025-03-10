import gym
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
import numpy as np
from gym import spaces
from gazebo_msgs.msg import ModelState, EntityState, ModelStates, LinkStates
from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Empty

max_lifes = 5


class RobomasterSoccerEnv(gym.Env):
    '''Custom Environment that follows the gym interface'''

    def __init__(self):
        super(RobomasterSoccerEnv, self).__init__()
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
                                       high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                                       dtype=np.float32) #the whole twist msg
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14, 0), dtype=np.float32)
        #robot1_x, robot1_y,
        #robot2_x, robot2_y,
        #ball_x, ball_y, ball_z

        rclpy.init()
        self.node = Node("Robomaster_Soccer_env")

        self.robot1_vel = np.array([0, 0, 0, 0, 0, 0], float)
        self.robot2_vel = np.array([0, 0, 0, 0, 0, 0], float)

        self.set_state = self.node.create_client(SetEntityState, "/gazebo/set_entity_state")
        self.pause = self.node.create_client(Empty, "/pause_physics")
        self.unpause = self.node.create_client(Empty, "/unpause_physics")
        self.reset_world = self.node.create_client(Empty, "/reset_world")
        self.req = Empty.Request

        self.robot1_vel_publisher = self.node.create_publisher(Twist, '/robot1/cmd_vel_original', 10)
        self.robot2_vel_publisher = self.node.create_publisher(Twist, '/robot2/cmd_vel_original', 10)

        self.robot1_odom_sub = self.node.create_subscription(Odometry, "/robot1/odom", self.robot1_odom_callback, 10)
        self.robot2_odom_sub = self.node.create_subscription(Odometry, "/robot2/odom", self.robot2_odom_callback, 10)
        self.robot1_ball_sub = self.node.create_subscription(Point, "/robot1/detected_ball", self.robot1_ball_callback, 10)
        self.robot2_ball_sub = self.node.create_subscription(Point, "/robot2/detected_ball", self.robot2_ball_callback, 10)

        self.set_sphere_state = EntityState()
        self.set_sphere_state.name = "ball"
        self.set_sphere_state.pose.position.x = 0.0
        self.set_sphere_state.pose.position.y = 0.0
        self.set_sphere_state.pose.position.z = 0.2
        self.sphere_state = SetEntityState.Request()

        self.robot1_life = max_lifes
        self.robot2_life = max_lifes

        self.t = 0
        self.t_limit = 6000

        self.robot1_state = None
        self.robot2_state = None
        self.robot1_ball_state = None
        self.robot2_ball_state = None



        #(x_agent, y_agent, x_agent_next, y_agent_next,
        # x_ball, y_ball, x_ball_next, y_ball_next,
        # x_opponent, y_opponent, x_opponent_next, y_opponent_next)

        self.robot1_obs = np.array([])

    def robot1_odom_callback(self, msg):
        self.robot1_state = (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.twist.twist.linear.x, msg.twist.twist.linear.y)

    def robot2_odom_callback(self, msg):
        self.robot2_state = (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.twist.twist.linear.x, msg.twist.twist.linear.y)

    def robot1_ball_callback(self, msg):
        self.robot1_ball_state = (msg.x, msg.y)

    def robot2_ball_callback(self, msg):
        self.robot2_ball_state = (msg.x, msg.y)

    def step(self, action):
        return self.observation, self.reward, self.done, self.info

    def reset(self):
        return self.observation   #reward, done, info can't be included

    def render(self, mode='human'):
        ...

    def close(self):
        ...

