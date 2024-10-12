from launch import LaunchDescription
from launch_ros.atctions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robomaster_ros',
            executable='balltracker.py',
            name='balltracker'
        ),
        Node(
            package='robomaster_ros',
            executable='ballfollower.py',
            name='ballfollower'
        )
    ])