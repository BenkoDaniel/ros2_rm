import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point
from ultralytics import YOLO

class BallTracker(Node):

    def __init__(self):
        super().__init__('Ball_tracker')
        self.bridge = CvBridge()
        self.yolomodel = YOLO("yolov5s.pt") #YOLOv5 Nano model: yolov5n
        self.subscripiton = self.create_subscription(
            Image,
            '/camera/image_color',
            self.cb_camera,
            10)

        self.ballpub = self.create_publisher(Point, "/detected_ball", 1)

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
        results = self.yolomodel(image)
        center_x, center_y = -1.0, -1.0

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            class_names = result.names 

            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                if class_names[cls_id] == "sports ball" and conf > 0.50:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    cv2.putText(image, f"{class_names[cls_id]} {conf:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),2)

        return image, center_x, center_y


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
