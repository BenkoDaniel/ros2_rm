import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image


class BallTrackerSim(Node):

    def __init__(self):
        super().__init__('Ball_tracker_sim')

        self.declare_parameter('camera_number', 'camera1')

        cam_topic_name = self.get_parameter('camera_number').get_parameter_value().string_value + "/image_raw"

        self.bridge = CvBridge()
        self.camera = None
        self.subscripiton = self.create_subscription(
            Image,
            cam_topic_name,
            self.cb_camera,
            10)

        self.ballpub = self.create_publisher(Point, "detected_ball", 1)

    def cb_camera(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            im_with_ballmarker, ball_x, ball_y = self.find_ball(cv_image)

            cv2.imshow("Camera", im_with_ballmarker)
            cv2.waitKey(1)
            point = Point()
            rows = float(cv_image.shape[0])
            cols = float(cv_image.shape[1])
            center_x = 0.5*cols
            center_y = 0.5*rows
            point.x = (ball_x - center_x)/center_x
            point.y = (ball_y - center_y)/center_y
            if point.x == -1.0 and point.y == -1.0:
                point.z = 0.0
                self.ballpub.publish(point)
            else:
                point.z = 1.0
                self.ballpub.publish(point)



        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge error: {e}")


    def find_ball(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_green = np.array([45, 150, 100])
        upper_green = np.array([75, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        x_coord, y_coord, w, h = 0, 0, 0, 0

        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x_coord, y_coord, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image, (x_coord, y_coord), (x_coord + w, y_coord + h), (255, 0, 0), 2)
                break

        b_center_x = x_coord + (w//2)
        b_center_y = y_coord + (h//2)

        return image, b_center_x, b_center_y


def main(args=None):
    rclpy.init(args=args)
    bt = BallTrackerSim()
    try:
        rclpy.spin(bt)
    finally:
        bt.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
