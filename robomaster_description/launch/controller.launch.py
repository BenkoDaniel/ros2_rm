from launch import LaunchDescription
from launch.actions import GroupAction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue, Parameter
import os


def generate_launch_description():

    pkg_src = FindPackageShare(package='robomaster_description').find('robomaster_description')
    urdf_path = os.path.join(pkg_src, 'urdf/robomastersim.urdf')

    with open(urdf_path, 'r') as urdf_file:
        urdf_content = urdf_file.read()

    robot_description = Parameter("robot_description", value=urdf_content)
    robot1_namespace = "robot1"
    robot2_namespace = "robot2"

#region robot1
    spawn_entity_node = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        namespace=robot1_namespace,
        arguments=[
            '-entity', 'robomaster_1',
            '-file', urdf_path,
            '-x', '0',
            '-y', '-0.5',
            '-z', '0',
            '-R', '0',
            '-P', '0',
            '-Y', '1.57',
        ],
        output="screen"
    )


    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
        remappings=[("/robot_description", '/robot1/robot_description'),
                    ("/joint_states", '/robot1/joint_states')],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "controller_manager"]
    )
    gimbal_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        namespace="robot1",
        arguments=["gim_controller", "--controller-manager", "controller_manager"],
    )
    command_converter = Node(
        package="robomaster_ros",
        executable="gimbal_command_converter",
    )
#endregion
#region robot2
    spawn_entity_node_2 = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        namespace=robot2_namespace,
        arguments=[
            '-entity', 'robomaster_2',
            '-file', urdf_path,
            '-x', '0',
            '-y', '-0.5',
            '-z', '0',
            '-R', '0',
            '-P', '0',
            '-Y', '1.57',
        ],
        output="screen"
    )

    robot_state_publisher_2 = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace=robot2_namespace,
        output="screen",
        parameters=[robot_description]
    )

    joint_state_broadcaster_spawner_2 = Node(
        package="controller_manager",
        executable="spawner",
        namespace=robot2_namespace,
        arguments=["joint_state_broadcaster"]
    )
    gimbal_controller_spawner_2 = Node(
        package="controller_manager",
        executable="spawner",
        namespace=robot2_namespace,
        arguments=["gim_controller"],
    )
    command_converter_2 = Node(
        package="robomaster_ros",
        executable="gimbal_command_converter",
        namespace=robot2_namespace,
    )
#endregion


    delay_gimbal_controller_spawner_after_joint_state_broadcaster_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[gimbal_controller_spawner]
        )
    )
    delay_gimbal_controller_spawner_after_joint_state_broadcaster_spawner_2 = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner_2,
            on_exit=[gimbal_controller_spawner_2]
        )
    )


    return LaunchDescription([
        #spawn_entity_node,
        #robot_state_publisher,
        #joint_state_broadcaster_spawner,
        #delay_gimbal_controller_spawner_after_joint_state_broadcaster_spawner,
        #command_converter,
        GroupAction([
            PushRosNamespace(robot1_namespace),
            spawn_entity_node,
            robot_state_publisher,
            joint_state_broadcaster_spawner,
            delay_gimbal_controller_spawner_after_joint_state_broadcaster_spawner,
            command_converter
        ]),
    ])
