from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "model_path",
            default_value="/root/asl_mediapipe_pointnet/model",
            description="Path (absolute) to PointNet model."
        ),
        DeclareLaunchArgument(
            "model_name",
            default_value="point_net_1.pth",
            description="Name of PointNet model."
        ),
        Node(
            package='v4l2_camera',
            executable='v4l2_camera_node',
            name="my_camera",
            remappings=[("image_raw", "my_input_image")]            
        ),
        Node(
            package='asl_mediapipe_pointnet',
            executable='asl_controller_pose_node',
            name="my_asl_controller",
            parameters=[
               {"model_path":LaunchConfiguration("model_path")},
               {"model_name":LaunchConfiguration("model_name")}
            ],
            remappings=[
               ("image_raw", "my_input_image"),
               ("asl_controller/target_pose", "target_pose")
            ]
        ),
        Node(
            package='asl_moveit_demos',
            executable='pose_controlled_moveit',
            name="my_pose_controller",
            remappings=[("target_pose", "target_pose")]
        )
    ])
