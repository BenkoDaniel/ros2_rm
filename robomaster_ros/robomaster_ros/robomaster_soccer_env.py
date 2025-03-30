import random
import gymnasium
import rclpy
import copy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
import numpy as np
from gymnasium import spaces
from gazebo_msgs.msg import ModelState, EntityState, ModelStates, LinkStates
from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Empty


class RobomasterSoccerEnv(gymnasium.Env):
    '''Custom Environment that follows the gym interface'''

    def __init__(self):
        super(RobomasterSoccerEnv, self).__init__()
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
                                       high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                                       dtype=np.float32) #the whole twist msg
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64)
        #robot1_x, robot1_y,
        #robot2_x, robot2_y,
        #ball_x, ball_y, ball_z

        rclpy.init()
        self.node = Node("Robomaster_Soccer_env")

        self.set_state = self.node.create_client(SetEntityState, "/gazebo/set_entity_state")
        self.pause = self.node.create_client(Empty, "/pause_physics")
        self.unpause = self.node.create_client(Empty, "/unpause_physics")
        self.req = Empty.Request

        self.robot1_vel_publisher = self.node.create_publisher(Twist, '/robot1/cmd_vel_original', 10)
        self.robot2_vel_publisher = self.node.create_publisher(Twist, '/robot2/cmd_vel_original', 10)

        self.robot1_odom_sub = self.node.create_subscription(Odometry, "/robot1/odom", self.robot1_odom_callback, 10)
        self.robot2_odom_sub = self.node.create_subscription(Odometry, "/robot2/odom", self.robot2_odom_callback, 10)
        self.robot1_ball_sub = self.node.create_subscription(Point, "/robot1/detected_ball", self.robot1_ball_callback, 10)
        self.ball_sub = self.node.create_subscription(ModelStates, "/gazebo/model_states", self.ball_pos_callback, 10)

        #to move a robot to initial position
        self.set_robot1_state = EntityState()
        self.set_robot1_state.name = "robomaster_1"
        self.set_robot1_state.pose.position.x = 0.0
        self.set_robot1_state.pose.position.y = -0.5
        self.set_robot1_state.pose.position.z = 0.0
        self.set_robot1_state.pose.orientation.z = 1.57
        self.robot1_state = SetEntityState.Request()

        self.set_robot2_state = EntityState()
        self.set_robot2_state.name = "robomaster_2"
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

        self.ball_position = np.array([0, 0], float)
        self.prev_ball_position = np.array([0, 0], float)
        self.t = 0
        self.t_limit = 6000

        # robot1_odom (4),  robot2_odom (4)
        # robot1_ball_dx, robot1_ball_dy,
        # ball_x, ball_y  - from gazebo

        self.robot1_odom = np.zeros(4)  # [x, y, vx, vy]
        self.robot2_odom = np.zeros(4)  # [x, y, vx, vy]
        self.robot1_ball_relative = np.zeros(2)  # [dx, dy]
        self.observation = np.zeros(12)
        self.terminated = False
        self.truncated = False
        self.reward = 0

        self.info = {}

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

    def ball_pos_callback(self, data):        
        if self.t%2 == 0:
            self.prev_ball_position = copy.copy(self.ball_position)

        ball_id = data.name.index('ball')

        self.ball_position[0] = data.pose[ball_id].position.x
        self.ball_position[1] = data.pose[ball_id].position.y
        #print(ball_position)
        #print(prev_ball_position)


    def step(self, action):
        self.t += 1

        robot1_command = Twist()
        robot2_command = Twist()

        robot1_command.linear.x = float(action[0])
        robot1_command.linear.y = float(action[1])
        robot1_command.angular.z = float(action[2])

        robot2_command.linear.x = float(action[3])
        robot2_command.linear.y = float(action[4])
        robot2_command.angular.z = float(action[5])

        self.robot1_vel_publisher.publish(robot1_command)
        self.robot2_vel_publisher.publish(robot2_command)

        rclpy.spin_once(self.node, timeout_sec=1)

        #while not self.unpause.wait_for_service(timeout_sec=1.0):
        #    self.node.get_logger().info('service not available, waiting again...')
        #try:
        #    self.unpause.call_async(Empty.Request())
        #except:
        #    self.node.get_logger().info("/unpause_physics service call failed")

        #while not self.pause.wait_for_service(timeout_sec=1.0):
        #    self.node.get_logger().info('service not available, waiting again...')
        #try:
        #    self.pause.call_async(Empty.Request())
        #except:
        #    self.node.get_logger().info("/gazebo/pause_physics service call failed")

            
        self.reward = self.count_reward(self.robot1_odom, self.ball_position, self.prev_ball_position)
            
        if self.t >= self.t_limit:
            self.terminated = True
            self.truncated = True


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
            self.ball_position[0],
            self.ball_position[1]
        ])

        return self.observation, self.reward, self.terminated, self.truncated, self.info

    def count_reward(self, robot1_odom, ball, prev_ball):
        reward = 0
        robot1_coord = np.array([robot1_odom[0], robot1_odom[1]])
        prev_distance = np.linalg.norm(prev_ball - robot1_coord)
        current_distance = np.linalg.norm(ball - robot1_coord)
        if current_distance < prev_distance:
            reward += 1 # to reward being closer to the ball
        else:
            reward += -1

        if abs(robot1_coord[0]) > 1:
            reward -= 10 #punish being outside the playground

        ball_direction = ball[1] - prev_ball[1]
        if ball_direction > 0:
            reward += 5  #to reward kicking the ball in the right direction
        else:
            reward += -5
        
        reward -=1  #for not scoring a goal

        if ball[1] < -1.0 and not self.terminated:
            reward = -100
            self.terminated = True
        elif ball[1] > 1.0 and not self.terminated:
            reward = 100
            self.terminated = True


        return reward
        """ robot1_coord = np.array([robot1_odom[0], robot1_odom[1]])
        eucl_dist_robot1 = np.linalg.norm(robot1_coord - ball)
        reward = (0.5-float(eucl_dist_robot1))*10 + ball[1]*100   #my goal is to give more reward if it ends up closer to the ball, or give more reward, if it moves the ball closer to the goal
        return reward """  
    
    def reset(self, *, seed=None, options = None):
        self.t = 0
        self.terminated = False
        self.truncated = False
        self.ball_position = np.array([0, 0], float)
        self.reward = 0
        self.prev_ball_position = np.array([0, 0], float)

        self.robot1_state = SetEntityState.Request()
        self.robot1_state._state = self.set_robot1_state
        while not self.set_state.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('reset : service not available, waiting again...')
        try:
            self.set_state.call_async(self.robot1_state)
        except rclpy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        self.robot2_state = SetEntityState.Request()
        self.robot2_state._state = self.set_robot2_state
        while not self.set_state.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('reset : service not available, waiting again...')
        try:
            self.set_state.call_async(self.robot2_state)
        except:
            print("/gazebo/reset_simulation service call failed")

        self.set_sphere_state.twist.linear.x = 0.6*(random.random()-0.5)
        self.set_sphere_state.twist.linear.y = -1*(0.1 + 0.5*random.random())
        self.set_sphere_state.twist.linear.z = 0.0

        self.sphere_state._state = self.set_sphere_state
        while not self.set_state.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('reset : service not available, waiting again...')
        try:
            self.set_state.call_async(self.sphere_state)
        except:
            self.node.get_logger().error(f"Reset world service call failed: {str(e)}")
            return None

        self.observation = np.zeros(12)
        
        rclpy.spin_once(self.node, timeout_sec=1)

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
            self.ball_position[0],
            self.ball_position[1]
        ])
        

        return self.observation, self.info  #reward, done, can't be included

    def render(self):
        return super().render()
    
    def close(self):
        return super().close()
    
    def stop(self):
        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting again...')
        try:
            self.pause.call_async(Empty.Request())
        except:
            self.node.get_logger().info("/gazebo/pause_physics service call failed")

