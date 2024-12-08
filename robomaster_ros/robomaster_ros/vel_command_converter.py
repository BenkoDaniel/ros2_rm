import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class VelCommandConverter(Node):
    def __init__(self):
        super().__init__('vel_command_converter')
        self.declare_parameter('use_sim_conversion', 'false')
        self.use_sim_conv = self.get_parameter('use_sim_conversion').get_parameter_value().string_value

        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel_original',
            self.command_callback,
            10
        )
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)

    def command_callback(self, msg):
        if self.use_sim_conv == 'true':
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

        if self.use_sim_conv == 'false':
            self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = VelCommandConverter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()