import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class VelCommandConverter(Node):
    def __init__(self):
        super().__init__('vel_command_converter')
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.command_callback,
            10
        )
        self.publisher = self.create_publisher(Twist, 'converted_cmd_vel', 10)

    def command_callback(self, msg):
        conv_command = Twist()
        conv_command.linear.x = msg.linear.x
        if conv_command.linear.x == 0:
            conv_command.linear.y = msg.linear.y * 2.0
        else:
            conv_command.linear.y = msg.linear.y
        conv_command.linear.z = msg.linear.z

        conv_command.angular.x = msg.angular.x
        conv_command.angular.y = msg.angular.y
        conv_command.angular.z = msg.angular.z * 5.5

        self.publisher.publish(conv_command)

def main(args=None):
    rclpy.init(args=args)
    node = VelCommandConverter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()