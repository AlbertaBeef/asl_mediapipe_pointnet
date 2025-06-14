from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='v4l2_camera',
            executable='v4l2_camera_node',
            name="my_camera",
            remappings=[("image_raw", "my_input_image")]                        
        ),
        Node(
            package='asl_mediapipe_pointnet',
            executable='asl_mediapipe_pointnet_demo',
            name="my_asl_controller",
            parameters=[{"model_path","/media/albertabeef/Tycho/asl_mediapipe_pointnet/model"},{"model_name","point_net_1.pth"}],
            remappings=[("image_raw", "my_input_image"),("cmd_vel", "turtle1/cmd_vel")]            
        ),
        Node(
            package='turtlesim',
            executable='turtlesim_node',
            name="my_turtlesim",
        )
    ])
