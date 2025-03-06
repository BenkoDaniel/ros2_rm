from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from ament_index_python import get_package_share_directory
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
import os


def generate_launch_description():

    full_simulation_launcher = IncludeLaunchDescription(
        XMLLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('robomaster_ros'),
                'launch/simulation.launch'
            )
        )
    )



    return LaunchDescription([
        full_simulation_launcher
    ])
