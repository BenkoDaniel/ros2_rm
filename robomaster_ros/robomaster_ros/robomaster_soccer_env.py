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

ball_position = np.array([0, 0], float)

class RobomasterSoccerEnv(gym.Env):
    '''Custom Environment that follows the gym interface'''

    def __init__(self):
        super(RobomasterSoccerEnv, self).__init__()
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
                                       high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                                       dtype=np.float32) #the whole twist msg
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        #robot1_x, robot1_y,
        #robot2_x, robot2_y,
        #ball_x, ball_y, ball_z

        rclpy.init()
        self.node = Node("Robomaster_Soccer_env")

        #to move a robot to initial position
        self.set_robot_1_state = EntityState()
        self.set_robot_1_state.pose.position.x = -0.5
        self.set_robot_1_state.pose.position.y = 0.0
        self.set_robot_1_state.pose.position.z = 0.0
        self.robot_1_state = SetEntityState.Request()

        self.set_robot_2_state = EntityState()
        self.set_robot_2_state.pose.position.x = 0.5
        self.set_robot_2_state.pose.position.y = 0.0
        self.set_robot_2_state.pose.position.z = 0.0
        self.robot_2_state = SetEntityState.Request()

        self.set_sphere_state = EntityState()
        self.set_sphere_state.name = "ball"
        self.set_sphere_state.pose.position.x = 0.0
        self.set_sphere_state.pose.position.y = 0.0
        self.set_sphere_state.pose.position.z = 0.175
        self.set_sphere_state.pose.orientation.x = 0.0
        self.set_sphere_state.pose.orientation.y = 0.0
        self.set_sphere_state.pose.orientation.z = 0.0
        self.set_sphere_state.pose.orientation.w = 1.0
        self.sphere_state = SetEntityState.Request()

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

        self.t = 0
        self.t_limit = 6000

        self.robot1_state = None
        self.robot2_state = None
        self.robot1_ball_relative = None
        self.robot2_ball_relative = None
        self.done = False
        self.reward = 0

        # robot1_odom (4),  robot2_odom (4), ??
        # robot1_ball_dx, robot1_ball_dy, robot2_ball_dx, robot2_ball_dy  - relative coordinates
        # ball_x, ball_y  - from gazebo

        self.observation = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


    def robot1_odom_callback(self, msg):
        self.robot1_state = [msg.pose.pose.position.x, msg.pose.pose.position.y,
                             msg.twist.twist.linear.x, msg.twist.twist.linear.y]

    def robot2_odom_callback(self, msg):
        self.robot2_state = [msg.pose.pose.position.x, msg.pose.pose.position.y,
                             msg.twist.twist.linear.x, msg.twist.twist.linear.y]

    def robot1_ball_callback(self, msg):
        self.robot1_ball_relative = [msg.x, msg.y]

    def robot2_ball_callback(self, msg):
        self.robot2_ball_relative = [msg.x, msg.y]

    def step(self, action):
        global ball_position
        self.t += 1

        robot1_command = Twist()
        robot2_command = Twist()

        robot1_command.linear.x = action[0]
        robot1_command.linear.y = action[1]
        robot1_command.angular.z = action[2]

        robot2_command.linear.x = action[3]
        robot2_command.linear.y = action[4]
        robot2_command.angular.z = action[5]

        self.robot1_vel_publisher.publish(robot1_command)
        self.robot2_vel_publisher.publish(robot2_command)

        self.node.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.1))

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        try:
            self.unpause.call_async(Empty.Request())
        except:
            self.get_logger().info("/unpause_physics service call failed")

        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        try:
            self.pause.call_async(Empty.Request())
        except rclpy.ServiceException as e:
            self.get_logger().info("/gazebo/pause_physics service call failed")

        if abs(ball_position[0]) > 1.55:
            if ball_position[0] < -1.55:
                self.robot1_life -= 1
                self.reward = -100
                self.get_logger().info('ROBOT1 LOST A POINT!')
            elif ball_position[0] > 1.55:
                self.robot2_life -= 1
                self.reward = 1000
                self.get_logger().info('ROBOT1 GET A POINT!')
            self.reset()
        else:
            self.reward = 0

        if self.t >= self.t_limit:
            self.done = True

        self.observation = np.array([
            self.robot_1_state[0],
            self.robot_1_state[1],
            self.robot_1_state[2],
            self.robot_1_state[3],
            self.robot_2_state[0],
            self.robot_2_state[1],
            self.robot_2_state[2],
            self.robot_2_state[3],
            self.robot1_ball_relative[0],
            self.robot1_ball_relative[1],
            self.robot2_ball_relative[0],
            self.robot2_ball_relative[1],
            ball_position[0],
            ball_position[1]
        ])

        return self.observation, self.reward, self.done, self.info

    def reset(self):
        return self.observation   #reward, done, info can't be included

    def render(self, mode='human'):
        ...

    def close(self):
        ...


class GetBallPosition(Node):

    def __init__(self):
        super().__init__('get_ball_position')
        self.subscription = self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self.ball_pos_callback,
            10)

    def ball_pos_callback(self, data):
        global ball_position

        unit_sphere_id = data.name.index('unit_sphere')

        ball_position[0] = data.pose[unit_sphere_id].position.x
        ball_position[1] = data.pose[unit_sphere_id].position.y
