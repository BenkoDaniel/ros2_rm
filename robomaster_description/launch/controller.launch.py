from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue, Parameter
import os


def generate_launch_description():
    robot_controllers = PathJoinSubstitution(
        [
            FindPackageShare("robomaster_description"),
            "config",
            "simcontroller.yaml",
        ]
    )

    pkg_src = FindPackageShare(package='robomaster_description').find('robomaster_description')
    urdf_path = os.path.join(pkg_src, 'urdf/robomastersim.urdf')

    with open(urdf_path, 'r') as urdf_file:
        urdf_content = urdf_file.read()

    # Create a Parameter to pass the URDF content as a string
    robot_description = Parameter("robot_description", value=urdf_content)

    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, robot_controllers],
        output="both",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"]
    )
    gimbal_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gimbal_controller", "--controller-manager", "/controller_manager"]
    )

    delay_robot_controller_spawner_after_joint_state_broadcaster_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[gimbal_controller_spawner]
        )
    )

    return LaunchDescription([
        controller_manager,
        joint_state_broadcaster_spawner,
        delay_robot_controller_spawner_after_joint_state_broadcaster_spawner,
    ])
