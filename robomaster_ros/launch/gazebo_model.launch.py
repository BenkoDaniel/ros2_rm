import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
import launch_ros

def generate_launch_description():
    pkg_share = launch_ros.substitutions.FindPackageShare(package='robomaster_ros').find('robomaster_ros')
    world_file = os.path.join(pkg_share, 'gazebo/simulation.sdf')

    return LaunchDescription([
        ExecuteProcess(
            cmd=['gazebo', '--verbose', world_file, '-u'],
            shell=True
        ),
        launch_ros.actions.Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-entity', 'robomaster_ros', '-topic', 'robot_description'],
            output='screen')
    ])