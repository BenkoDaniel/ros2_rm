import random
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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        #robot1_x, robot1_y,
        #robot2_x, robot2_y,
        #ball_x, ball_y, ball_z

        rclpy.init()
        self.node = Node("Robomaster_Soccer_env")

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

        #to move a robot to initial position
        self.set_robot1_state = EntityState()
        self.set_robot1_state.name = "robomaster_1::chassis_base_link"
        self.set_robot1_state.pose.position.x = 0.0
        self.set_robot1_state.pose.position.y = -0.5
        self.set_robot1_state.pose.position.z = 0.0
        self.set_robot1_state.pose.orientation.z = 1.57
        self.robot1_state = SetEntityState.Request()

        self.set_robot2_state = EntityState()
        self.set_robot2_state.name = "robomaster_2::chassis_base_link"
        self.set_robot2_state.pose.position.x = 0.0
        self.set_robot2_state.pose.position.y = 0.5
        self.set_robot2_state.pose.position.z = 0.0
        self.set_robot2_state.pose.orientation.z = -1.57
        self.robot2_state = SetEntityState.Request()

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

        self.t = 0
        self.t_limit = 6000

        self.robot1_odom = None
        self.robot2_odom = None
        self.robot1_ball_relative = None
        self.done = False
        self.reward = None

        # robot1_odom (4),  robot2_odom (4)
        # robot1_ball_dx, robot1_ball_dy,
        # ball_x, ball_y  - from gazebo

        self.observation = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def robot1_odom_callback(self, msg):
        self.robot1_odom = [msg.pose.pose.position.x, msg.pose.pose.position.y,
                             msg.twist.twist.linear.x, msg.twist.twist.linear.y]

    def robot2_odom_callback(self, msg):
        self.robot2_odom = [msg.pose.pose.position.x, msg.pose.pose.position.y,
                             msg.twist.twist.linear.x, msg.twist.twist.linear.y]

    def robot1_ball_callback(self, msg):
        if msg.z == 1:
            self.robot1_ball_relative = [msg.x, msg.y]
        else:
            self.robot1_ball_relative = [0, 0]


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

        if abs(ball_position[0]) > 1.1:
            if ball_position[0] < -1.1:
                self.reward = -100
                self.get_logger().info('ROBOT1 LOST A POINT!')
            elif ball_position[0] > 1.1:
                self.reward = 10000
                self.get_logger().info('ROBOT1 GET A POINT!')
            self.reset()
        else:
            self.reward = 0

        if self.t >= self.t_limit:
            self.reward = self.count_reward_without_goal(self.robot1_odom, ball_position)
            self.done = True

        self.observation = np.array([
            self.robot1_odom[0],
            self.robot1_odom[1],
            self.robot1_odom[2],
            self.robot1_odom[3],
            self.robot2_odom[0],
            self.robot2_odom[1],
            self.robot2_odom[2],
            self.robot2_odom[3],
            self.robot1_ball_relative[0],
            self.robot1_ball_relative[1],
            ball_position[0],
            ball_position[1]
        ])

        return self.observation, self.reward, self.done, self.info

    def count_reward_without_goal(self, robot1_odom, ball):
        robot1_coord = np.array([robot1_odom[0], robot1_odom[1]])
        eucl_dist_robot1 = np.linalg.norm(robot1_coord, ball)
        reward = (0.5-eucl_dist_robot1)*10 + ball[0]*100   #my goal is to give more reward if it ends up closer to the ball, or give more reward, if it moves the ball closer to the goal
        return reward

    def reset(self):
        while not self.reset_world.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset : service not available, waiting again...')

        try:
            self.reset_world.call_async(Empty.Request())
        except:
            import traceback
            traceback.print_exc()

        if self.done:
            self.t = 0
            self.done = False

        self.robot1_state = SetEntityState.Request()
        self.robot1_state._state = self.set_robot_1_state
        while not self.set_state.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset : service not available, waiting again...')

        self.robot2_state = SetEntityState.Request()
        self.robot2_state._state = self.set_robot2_state
        while not self.set_state.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset : service not available, waiting again...')

        try:
            self.set_state.call_async(self.robot_1_state)
        except rclpy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        self.set_sphere_state.twist.linear.x = -1*(0.1 + 0.5*random.random())
        self.set_sphere_state.twist.linear.y = 0.6*(random.random()-0.5)
        self.set_sphere_state.twist.linear.z = 0.0

        self.sphere_state._state = self.set_sphere_state
        while not self.set_state.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset : service not available, waiting again...')
        try:
            self.set_state.call_async(self.sphere_state)
        except:
            import traceback
            traceback.print_exc()

        self.observation = np.array([
            self.robot1_odom[0],
            self.robot1_odom[1],
            self.robot1_odom[2],
            self.robot1_odom[3],
            self.robot2_odom[0],
            self.robot2_odom[1],
            self.robot2_odom[2],
            self.robot2_odom[3],
            self.robot1_ball_relative[0],
            self.robot1_ball_relative[1],
            ball_position[0],
            ball_position[1]
        ])

        return self.observation   #reward, done, info can't be included


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

        ball_id = data.name.index('ball')

        ball_position[0] = data.pose[ball_id].position.x
        ball_position[1] = data.pose[ball_id].position.y
