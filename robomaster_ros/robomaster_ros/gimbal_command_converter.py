import rclpy
from rclpy.node import Node
from robomaster_msgs.msg import GimbalCommand
from std_msgs.msg import Float64MultiArray

class GimbalCommandToStd(Node):
    def __init__(self):
        super().__init__('gimbal_command_converter')
        self.subscription = self.create_subscription(
            GimbalCommand,
            '/cmd_gimbal',
            self.command_callback,
            10
        )
        self.publisher = self.create_publisher(Float64MultiArray, '/gim_controller/commands', 10)

    def command_callback(self, msg):
        std_command = Float64MultiArray()
        std_command.data = [-msg.yaw_speed, msg.pitch_speed]
        self.publisher.publish(std_command)

def main(args=None):
    rclpy.init(args=args)
    node = GimbalCommandToStd()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()