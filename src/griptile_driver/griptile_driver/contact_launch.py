"""Contact image and viewer launch settings shared by acquisition and replay."""

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue

from griptile_core.contact_image import DEFAULT_CONTACT_SPAN


def arguments():
    return [
        DeclareLaunchArgument("imprint", default_value="false",
                              description="Show the contact print page in the main viewer at startup"),
        DeclareLaunchArgument("gui", default_value=LaunchConfiguration("imprint"),
                              description="Open the viewer (defaults to the imprint setting)"),
        DeclareLaunchArgument("contact_span_adc", default_value=str(DEFAULT_CONTACT_SPAN),
                              description="Noise-gated ADC response at full red"),
        DeclareLaunchArgument("contact_direction", default_value="both",
                              choices=["both", "increase", "decrease"]),
        DeclareLaunchArgument("contact_auto_reference", default_value="false",
                              description="Capture the first three seconds as unloaded on each connection"),
    ]


def parameters():
    return {name: ParameterValue(LaunchConfiguration(name), value_type=kind)
            for name, kind in (("contact_span_adc", float), ("contact_direction", str),
                               ("contact_auto_reference", bool))}
