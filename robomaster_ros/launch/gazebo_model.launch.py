import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.substitutions import Command, LaunchConfiguration
import launch_ros

def generate_launch_description():
    pkg_src = launch_ros.substitutions.FindPackageShare(package='robomaster_description').find('robomaster_description')
    world_file = os.path.join(pkg_src, 'Gazebo/simulation.sdf')

    return LaunchDescription([
        ExecuteProcess(
            cmd=['gzserver', '-s', 'libgazebo_ros_factory.so', world_file, '-u'],
            output='screen',
            shell=True
        ),


    ])