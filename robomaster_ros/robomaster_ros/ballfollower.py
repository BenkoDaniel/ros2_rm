import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from robomaster_msgs.msg import GimbalCommand
import time


class BallFollower(Node):
    def __init__(self):
        super().__init__('ball_follower')
        self.subscription = self.create_subscription(
            Point,
            "detected_ball",
            self.cb_detectedball,
            10)
        self.publisher_ = self.create_publisher(GimbalCommand, "cmd_gimbal", 10)
        self.rcv_timeout_secs = 10
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.target_x = 0.0
        self.target_y = 0.0
        self.datachecker = 1
        self.lastrcvtime = time.time() - 10000

    def cb_detectedball(self, msg):
        self.target_x = msg.x
        self.target_y = msg.y
        self.datachecker = msg.z
        self.lastrcvtime = time.time()

    def timer_callback(self):
        gimbalmsg = GimbalCommand()
        if self.datachecker == 1:
            gimbalmsg.yaw_speed = self.target_x
            gimbalmsg.pitch_speed = self.target_y
        self.publisher_.publish(gimbalmsg)


def main(args=None):
    rclpy.init(args=args)
    follow_ball = BallFollower()
    rclpy.spin(follow_ball)
    follow_ball.destroy_node()
    rclpy.shutdown()