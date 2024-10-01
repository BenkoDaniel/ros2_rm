import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
from robomaster_msgs.msg import GimbalCommand
from std_msgs.msg import Bool
import time

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
        self.gimbal_statepub = self.create_publisher(Bool, '/gimbal/engage', 10)
        self.pic_center_x = 320
        self.pic_center_y = 130
        self.timer_period = 0.1
        #self.timer = self.create_timer(self.timer_period, self.timer_callback)
        #self.lastrcvtime = time.time() - 10000
        self.rightspeed = 0.3
        self.leftspeed = -0.3
        self.upspeed = 0.3
        self.downspeed = -0.3


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
        x1, x2, = 320, 320
        y1, y2 = 130, 130

        for _, row in df.iterrows():

            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

            if row['name'] == "sports ball":
                if row['confidence'] > 0.50:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    #center_x = int((x1 + x2) / 2)
                    #center_y = int((y1 + y2) / 2)
                    #print(center_x, center_y)
                    cv2.putText(image, f"{row['name']} {row['confidence']:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        #loop_rate = self.create_rate(100, self.get_clock())
        gimbal_msg = GimbalCommand()
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        dx = center_x - self.pic_center_x
        dy = center_y - self.pic_center_y

        if dx < 0:
            gimbal_msg.yaw_speed = self.leftspeed
        elif dx > 0:
            gimbal_msg.yaw_speed = self.rightspeed
        else:
            gimbal_msg.yaw_speed = 0.0

        if dy < 0:
            gimbal_msg.pitch_speed = self.upspeed
        elif dy > 0:
            gimbal_msg.pitch_speed = self.downspeed
        else:
            gimbal_msg.pitch_speed = 0.0

        self.gimbal_pub.publish(gimbal_msg)


        return image
    def shutdown(self):
        gimbal_msg = GimbalCommand()
        gimbal_msg.pitch_speed = 0.0
        gimbal_msg.yaw_speed = 0.0
        self.gimbal_pub.publish(gimbal_msg)
        gimbal_false = Bool()
        gimbal_false.data = False
        self.gimbal_statepub.publish(gimbal_false)


def main(args=None):
    rclpy.init(args=args)
    bt = BallTracker()
    try:
        rclpy.spin(bt)
    finally:
        bt.shutdown()
        bt.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
