import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

class BallTracker(Node):

    def __init__(self):
        super().__init__('Ball_tracker')
        self.bridge = CvBridge()
        self.camera = None
        self.subscripiton = self.create_subscription(
            Image,
            '/camera/image_color',
            self.cb_camera,
            10)
    def cb_camera(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            cv2.imshow("Camera", cv_image)
            cv2.waitKey(1)

        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge error: {e}")

def main(args=None):
    rclpy.init(args=args)
    bt = BallTracker()
    try:
        rclpy.spin(bt)
    finally:
        bt.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
