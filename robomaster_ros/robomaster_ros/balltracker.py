import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import GimbalCommand


class BallTracker(Node):

    def __init__(self):
        super().__init__('Ball_tracker')
        self.bridge = CvBridge()
        self.yolomodel = torch.hub.load("ultralytics/yolov5", "yolov5s")
        self.camera = None
        self.subscripiton = self.create_subscription(
            Image,
            '/camera/image_color',
            self.cb_camera,
            10)
        self.gimbal_pub = self.create_publisher(GimbalCommand, '/cmd_gimbal', 10)

    def cb_camera(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            im_with_ballmarker = self.find_ball(cv_image)
            cv2.imshow("Camera", im_with_ballmarker)
            cv2.waitKey(1)

        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge error: {e}")

    def find_ball(self, image):
        results = self.yolomodel(image)
        df = results.pandas().xyxy[0]
        for _, row in df.iterrows():

            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

            if row['name'] == "sports ball":
                if row['confidence'] > 0.50:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f"{row['name']} {row['confidence']:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


        loop_rate = self.create_rate(100, self.get_clock())
        speed = self.get_parameter('speed').get_parameter_value().double_value
        gimbal_msg = GimbalCommand()
        gimbal_msg.pitch_speed = 0.0
        gimbal_msg.jaw_speed = 0.0

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        dx =


        return image


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
