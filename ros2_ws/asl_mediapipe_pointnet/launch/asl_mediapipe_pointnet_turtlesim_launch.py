from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='v4l2_camera',
            executable='v4l2_camera_node',
        ),
        Node(
            package='asl_mediapipe_pointnet',
            executable='asl_mediapipe_pointnet_demo',
        ),
        Node(
            package='turtlesim',
            executable='turtlesim_node',
        )
    ])
