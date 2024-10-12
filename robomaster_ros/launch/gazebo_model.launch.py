import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    robotXacroname='robomaster_s1'
    namePackage = 'robomaster_ros'
    modelFileRelativePath = 'urdf/urdf/robomaster_s1.urdf.xacro'
    worldFileRealtivePath = 'Gazebo/emptyworld.world'
    pathModelFile = os.path.join(get_package_share_directory(namePackage), modelFileRelativePath)
    pathWorldFile = os.path.join(get_package_share_directory(namePackage), worldFileRealtivePath)
    robotDescription = xacro.process_file(pathModelFile).toxml()

    gazebo_rosPackageLaunch=PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('gazebo_ros'),'launch','model.launch.py'))
    gazeboLaunch = IncludeLaunchDescription(gazebo_rosPackageLaunch, launch_arguments={'world':pathWorldFile}.items())

    spawnModelNode = Node(package='robomaster_ros', executable='spawn_entity.py', arguments=['-topic','robot_description','-entity',robotXacroname], output='screen')

    nodeRpbotStatePublisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robotDescription,
                     'use_sim_time': True}]
    )

    launchDescriptionObject = LaunchDescription()

    launchDescriptionObject.add_action(gazeboLaunch)
    launchDescriptionObject.add_action(spawnModelNode)
    launchDescriptionObject.add_action(nodeRpbotStatePublisher)

    return launchDescriptionObject