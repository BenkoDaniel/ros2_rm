from launch import LaunchDescription
from launch.actions import GroupAction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue, Parameter
import os


def generate_launch_description():

    pkg_src = FindPackageShare(package='robomaster_description').find('robomaster_description')
    robot2_urdf_path = os.path.join(pkg_src, 'urdf/robomastersim_robot2.urdf')

    with open(robot2_urdf_path, 'r') as urdf_file2:
        robot2_urdf_content = urdf_file2.read()

    robot2_robot_description = Parameter("robot_description", value=robot2_urdf_content)

    robot2_namespace = "robot2"

#region robot2
    robot2_spawn_entity_node = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            '-entity', 'robomaster_2',
            '-file', robot2_urdf_path,
            '-x', '0',
            '-y', '0.5',
            '-z', '0',
            '-R', '0',
            '-P', '0',
            '-Y', '-1.57',
        ],
        output="screen"
    )

    robot2_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot2_robot_description],
        remappings=[("/robot_description", '/robot2/robot_description'),
                    ("/joint_states", '/robot2/joint_states')],
    )

    robot2_joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "controller_manager"]
    )
    robot2_gimbal_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        namespace=robot2_namespace,
        arguments=["gim_controller", "--controller-manager", "controller_manager"],
    )
    robot2_command_converter = Node(
        package="robomaster_ros",
        executable="gimbal_command_converter",
    )
    robot2_simballtracker = Node(
        package="robomaster_ros",
        executable="simballtracker",
        parameters=[{'camera_number': 'camera2'}]
    )
    robot2_simballfollower = Node(
        package="robomaster_ros",
        executable="ballfollower",
    )
#endregion


    robot2_delay_gimbal_controller_spawner_after_joint_state_broadcaster_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=robot2_joint_state_broadcaster_spawner,
            on_exit=[robot2_gimbal_controller_spawner]
        )
    )

    return LaunchDescription([
        GroupAction([
            PushRosNamespace(robot2_namespace),
            robot2_spawn_entity_node,
            robot2_robot_state_publisher,
            robot2_joint_state_broadcaster_spawner,
            robot2_delay_gimbal_controller_spawner_after_joint_state_broadcaster_spawner,
            robot2_command_converter,
            #robot2_simballtracker,
            #robot2_simballfollower
        ])
    ])
