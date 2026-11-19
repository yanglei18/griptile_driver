"""Launch the serial driver and optional control and visualization window."""

from griptile_driver import contact_launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    sensor_id = LaunchConfiguration("sensor_id")
    namespace = LaunchConfiguration("namespace")
    identity = {"sensor_id": ParameterValue(sensor_id, value_type=str)}
    return LaunchDescription([
        *contact_launch.arguments(),
        DeclareLaunchArgument("sensor_id", default_value="sensor_0"),
        DeclareLaunchArgument("namespace", default_value=["/tactile/", sensor_id]),
        DeclareLaunchArgument("port", default_value="", description="Explicit serial device path"),
        DeclareLaunchArgument("baudrate", default_value="115200"),
        DeclareLaunchArgument("publish_heatmap", default_value="false"),
        DeclareLaunchArgument("state_rate_hz", default_value="30.0"),
        DeclareLaunchArgument("max_cell_age_ms", default_value="100.0"),
        Node(package="griptile_driver", executable="griptile_node", namespace=namespace,
             output="screen", parameters=[{
                 **identity,
                 **contact_launch.parameters(),
                 "port": ParameterValue(LaunchConfiguration("port"), value_type=str),
                 "baudrate": ParameterValue(LaunchConfiguration("baudrate"), value_type=int),
                 "publish_heatmap": ParameterValue(LaunchConfiguration("publish_heatmap"), value_type=bool),
                 "state_rate_hz": ParameterValue(LaunchConfiguration("state_rate_hz"), value_type=float),
                 "max_cell_age_ms": ParameterValue(LaunchConfiguration("max_cell_age_ms"), value_type=float),
             }]),
        Node(package="griptile_viewer", executable="viewer", namespace=namespace, output="screen",
             condition=IfCondition(LaunchConfiguration("gui")),
             parameters=[identity, {"imprint": ParameterValue(LaunchConfiguration("imprint"), value_type=bool)}]),
    ])
