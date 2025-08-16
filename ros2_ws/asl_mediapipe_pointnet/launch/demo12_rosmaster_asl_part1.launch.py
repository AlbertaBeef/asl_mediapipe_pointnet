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
            package='asl_mediapipe_pointnet',
            executable='usbcam_publisher_node',
            name="usbcam_publisher",
            remappings=[("image_raw", "usbcam_image")]            
        ),
        Node(
            package='asl_mediapipe_pointnet',
            executable='asl_controller_twist_node',
            name="asl_controller",
            parameters=[
               {"model_path":LaunchConfiguration("model_path")},
               {"model_name":LaunchConfiguration("model_name")},
               {"use_imshow":False}
            ],
            remappings=[
               ("image_raw", "usbcam_image"),
               ("asl_controller/cmd_vel", "cmd_vel")
            ]            
        )
        # not available on QIRP v1.4
        #Node(
        #    package='twist_stamper',
        #    executable='twist_stamper',
        #    remappings=[("cmd_vel_in", "cmd_vel"),("cmd_vel_out", "mecanum_drive_controller/cmd_vel")]            
        #)        
    ])
