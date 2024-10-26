import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
import launch_ros

def generate_launch_description():
    pkg_src = launch_ros.substitutions.FindPackageShare(package='robomaster_ros').find('robomaster_ros')
    sim = os.path.join(pkg_src, 'gazebo/simulation.sdf')

    return LaunchDescription([
        ExecuteProcess(
            cmd=['gzserver', '-s', 'libgazebo_ros_factory.so', sim, '-u'],
            output='screen',
            shell=True
        ),
        launch_ros.actions.Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
        ),

        launch_ros.actions.Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher'
        ),
    ])
