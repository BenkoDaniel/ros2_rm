import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(get_package_share_directory('gazebo_ros'), 'launch'), '/gazebo.launch.py']),
            launch_arguments={
                'world': '/home/benko/ros2_rm/src/robomaster_ros/robomaster_description/Gazebo/robomaster.world'
            }.items()
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                #'robot_description': open('/home/benko/.gazebo/models/robomaster_s1/robot.sdf').read()
                'robot_description': open('/home/benko/ros2_rm/src/robomaster_ros/robomaster_description/urdf/robomaster.urdf').read(),
                'use_sim_time': True
            }]
        ),
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            output='screen',
        ),
    ])

