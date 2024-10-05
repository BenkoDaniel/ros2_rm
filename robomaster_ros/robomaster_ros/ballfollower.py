import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from robomaster_msgs.msg import GimbalCommand
import time


class BallFollower(Node):
    def __init__(self):
        super().__init__('balltracker')
        self.subscription = self.create_subscription(
            Point,
            '/detected_ball',
            self.cb_detectedball,
            10)
        self.publisher_ = self.create_publisher(GimbalCommand, '/cmd_gimbal', 10)

        self.rcv_timeout_secs = 10
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.target_x = 0.0
        self.target_y = 0.0
        self.errorchecker = 1
        self.lastrcvtime = time.time() - 10000
        self.lookaround_speed = 0.2


    def cb_detectedball(self, msg):
        self.target_x = msg.x
        self.target_y = msg.y
        self.errorchecker = msg.z
        self.lastrcvtime = time.time()

    def timer_callback(self):
        gimbalmsg = GimbalCommand()
        if (self.errorchecker == 1):
            gimbalmsg.yaw_speed = self.target_x
            gimbalmsg.pitch_speed = self.target_y
        else:
            self.get_logger().info('Target lost')
            #gimbalmsg.yaw_speed = self.lookaround_speed
        self.publisher_.publish(gimbalmsg)


def main(args=None):
    rclpy.init(args=args)
    follow_ball = BallFollower()
    rclpy.spin(follow_ball)
    follow_ball.destroy_node()
    rclpy.shutdown()