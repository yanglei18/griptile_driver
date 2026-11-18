"""Rebuild state in a separate namespace during ros2 bag play --clock."""

from griptile_driver import contact_launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    sensor_id = LaunchConfiguration("sensor_id")
    source = LaunchConfiguration("source_namespace")
    output = LaunchConfiguration("namespace")
    identity = {"sensor_id": ParameterValue(sensor_id, value_type=str), "use_sim_time": True}
    return LaunchDescription([
        *contact_launch.arguments(),
        DeclareLaunchArgument("sensor_id", default_value="sensor_0"),
        DeclareLaunchArgument("source_namespace", default_value=["/tactile/", sensor_id]),
        DeclareLaunchArgument("namespace", default_value="/tactile/replay"),
        DeclareLaunchArgument("gui", default_value="false"),
        DeclareLaunchArgument("max_cell_age_ms", default_value="100.0"),
        DeclareLaunchArgument("state_rate_hz", default_value="30.0"),
        Node(package="griptile_driver", executable="griptile_replay", namespace=output,
             output="screen", remappings=[("samples", [source, "/samples"]),
                                          ("source_info", [source, "/info"]),
                                          ("recorded_contact_reference", [source, "/contact_reference"])],
             parameters=[{
                 **identity,
                 **contact_launch.parameters(),
                 "max_cell_age_ms": ParameterValue(LaunchConfiguration("max_cell_age_ms"), value_type=float),
                 "state_rate_hz": ParameterValue(LaunchConfiguration("state_rate_hz"), value_type=float),
             }]),
        Node(package="griptile_viewer", executable="viewer", namespace=output, output="screen",
             condition=IfCondition(LaunchConfiguration("gui")),
             remappings=[("diagnostics", [source, "/diagnostics"])],
             parameters=[identity]),
    ])
