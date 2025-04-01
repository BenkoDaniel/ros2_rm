import random
import gymnasium
import rclpy
import copy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
import numpy as np
from gymnasium import spaces
from pettingzoo.utils import ParallelEnv
from gazebo_msgs.msg import ModelState, EntityState, ModelStates, LinkStates
from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Empty
from functools import lru_cache


class RobomasterSoccerEnv(ParallelEnv):
    '''Custom Environment that follows the pettingzoo interface'''

    def __init__(self):
        rclpy.init()
        super(RobomasterSoccerEnv, self).__init__()
        self.agents = ["robot1", "robot2"]
        self._observation_spaces = {
            "robot1": spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64),
            "robot2": spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64)
        }
        self._action_spaces = {
            "robot1": spaces.Box(low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), dtype=np.float32),
            "robot2": spaces.Box(low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), dtype=np.float32)
        }

        self.possible_agents = self.agents[:]
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
        self.robot2_ball_sub = self.node.create_subscription(Point, "/robot2/detected_ball", self.robot1_ball_callback, 10)
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
        self.robot2_ball_relative = np.zeros(2)  # [dx, dy]
        self.robot1_observation = np.zeros(12)
        self.robot2_observation = np.zeros(12)
        self.observations = None
        self.terminations = {"robot1": False, "robot2": False}
        self.truncations = {"robot1": False, "robot2": False}
        self.robot1_reward = 0
        self.robot2_reward = 0
        self.rewards = None

        self.infos = {
            "robot1": {
                "status": "active",
                "individual_reward": self.robot1_reward,
            },
            "robot2": {
                "status": "active",
                "individual_reward": self.robot2_reward,
            }
        }

    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]


    @lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]


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


    def robot2_ball_callback(self, msg):
        if msg.z == 1:
            self.robot2_ball_relative = [msg.x, msg.y]
        else:
            self.robot2_ball_relative = [0, 0]


    def ball_pos_callback(self, data):        
        if self.t%2 == 0:
            self.prev_ball_position = copy.copy(self.ball_position)
        ball_id = data.name.index('ball')
        self.ball_position[0] = data.pose[ball_id].position.x
        self.ball_position[1] = data.pose[ball_id].position.y


    def step(self, actions):
        self.t += 1

        robot1_command = Twist()
        robot2_command = Twist()

        action1 = actions["robot1"]
        action2 = actions["robot2"]

        robot1_command.linear.x = float(action1[0])
        robot1_command.linear.y = float(action1[1])
        robot1_command.angular.z = float(action1[2])

        robot2_command.linear.x = float(action2[0])
        robot2_command.linear.y = float(action2[1])
        robot2_command.angular.z = float(action2[2])

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

            
        self.robot1_reward = self.count_reward(self.robot1_odom, self.ball_position, self.prev_ball_position)
        self.robot2_reward = self.count_reward(self.robot2_odom, self.ball_position, self.prev_ball_position)

        self.rewards = {
            "robot1": self.robot1_reward,
            "robot2": self.robot2_reward
        }
            
        if self.t >= self.t_limit:
            self.truncations = {
                "robot1": True,
                "robot2": True
            }


        self.robot1_observation = np.array([
            self.robot1_odom[0], #own odom
            self.robot1_odom[1],
            self.robot1_odom[2],
            self.robot1_odom[3],
            self.robot2_odom[0], #enemy odom
            self.robot2_odom[1],
            self.robot2_odom[2],
            self.robot2_odom[3],
            self.robot1_ball_relative[0],
            self.robot1_ball_relative[1],
            self.ball_position[0],
            self.ball_position[1]
        ])

        self.robot2_observation = np.array([
            self.robot2_odom[0],
            self.robot2_odom[1],
            self.robot2_odom[2],
            self.robot2_odom[3],
            self.robot1_odom[0],
            self.robot1_odom[1],
            self.robot1_odom[2],
            self.robot1_odom[3],
            self.robot2_ball_relative[0],
            self.robot2_ball_relative[1],
            self.ball_position[0],
            self.ball_position[1]
        ])

        self.observations = {
            "robot1": self.robot1_observation,
            "robot2": self.robot2_observation,
        }

        return self.observations, self.rewards, self.terminations, self.truncations, self.infos



    def count_reward(self, robot_odom, ball, prev_ball):
        reward = 0
        robot_coord = np.array([robot_odom[0], robot_odom[1]])
        prev_distance = np.linalg.norm(prev_ball - robot_coord)
        current_distance = np.linalg.norm(ball - robot_coord)
        terminated = self.terminations["robot1"] or self.terminations["robot2"]
        if current_distance < prev_distance:
            reward += 1 # to reward being closer to the ball
        else:
            reward += -1

        if abs(robot_coord[0]) > 1:
            reward -= 10 #punish being outside the playground

        ball_direction = ball[1] - prev_ball[1]
        if ball_direction > 0:
            reward += 5  #to reward kicking the ball in the right direction
        else:
            reward += -5
        
        reward -=1  #for not scoring a goal

        if ball[1] < -1.0 and not terminated:
            reward = -100
            self.terminations = {"robot1": True, "robot2": True}
        elif ball[1] > 1.0 and not terminated:
            reward = 100
            self.terminations = {"robot1": True, "robot2": True}

        return reward
        """ robot1_coord = np.array([robot1_odom[0], robot1_odom[1]])
        eucl_dist_robot1 = np.linalg.norm(robot1_coord - ball)
        reward = (0.5-float(eucl_dist_robot1))*10 + ball[1]*100   #my goal is to give more reward if it ends up closer to the ball, or give more reward, if it moves the ball closer to the goal
        return reward """  
    
    def reset(self, *, seed=None, options = None):
        self.t = 0
        self.terminations = {"robot1": False, "robot2": False}
        self.truncations = {"robot1": False, "robot2": False}
        self.ball_position = np.array([0, 0], float)
        self.robot1_reward = 0
        self.robot1_reward = 0
        self.rewards = None
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

        self.robot1_observation = np.array([
            self.robot1_odom[0], #own odom
            self.robot1_odom[1],
            self.robot1_odom[2],
            self.robot1_odom[3],
            self.robot2_odom[0], #enemy odom
            self.robot2_odom[1],
            self.robot2_odom[2],
            self.robot2_odom[3],
            self.robot1_ball_relative[0],
            self.robot1_ball_relative[1],
            self.ball_position[0],
            self.ball_position[1]
        ])

        self.robot2_observation = np.array([
            self.robot2_odom[0],
            self.robot2_odom[1],
            self.robot2_odom[2],
            self.robot2_odom[3],
            self.robot1_odom[0],
            self.robot1_odom[1],
            self.robot1_odom[2],
            self.robot1_odom[3],
            self.robot2_ball_relative[0],
            self.robot2_ball_relative[1],
            self.ball_position[0],
            self.ball_position[1]
        ])

        self.observations = {
            "robot1": self.robot1_observation,
            "robot2": self.robot2_observation,
        }

        return self.observations, self.infos  #reward, done, can't be included

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

