  
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    pkg_bme_gazebo_basics = get_package_share_directory('bme_gazebo_basics')

    gazebo_models_path, ignore_last_dir = os.path.split(pkg_bme_gazebo_basics)
    os.environ["GZ_SIM_RESOURCE_PATH"] += os.pathsep + gazebo_models_path

    asl_model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value="/root/asl_mediapipe_pointnet/model",
        description="Path (absolute) to PointNet model."
    )
    asl_model_name_arg = DeclareLaunchArgument(
        "model_name",
        default_value="point_net_1.pth",
        description="Name of PointNet model."
    )
        
    rviz_launch_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Open RViz.'
    )

    world_arg = DeclareLaunchArgument(
        'world', default_value='world.sdf',
        description='Name of the Gazebo world file to load'
    )

    model_arg = DeclareLaunchArgument(
        'model', default_value='mogi_bot.urdf',
        description='Name of the URDF description to load'
    )

    # Define the path to your URDF or Xacro file
    urdf_file_path = PathJoinSubstitution([
        pkg_bme_gazebo_basics,  # Replace with your package name
        "urdf",
        LaunchConfiguration('model')  # Replace with your URDF or Xacro file
    ])

    world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bme_gazebo_basics, 'launch', 'world.launch.py'),
        ),
        launch_arguments={
        'world': LaunchConfiguration('world'),
        }.items()
    )
    
    # Launch ASL controller
    v4l2_camera_node = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        name="my_camera",
        remappings=[("image_raw", "my_input_image")]            
    )
    asl_controller_node = Node(
        package='asl_mediapipe_pointnet',
        executable='asl_mediapipe_pointnet_demo',
        name="my_asl_controller",
        parameters=[{"model_path":LaunchConfiguration("model_path")},{"model_name":LaunchConfiguration("model_name")}],
        remappings=[("image_raw", "my_input_image"),("cmd_vel", "cmd_vel")]            
    )    

    # Launch rviz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(pkg_bme_gazebo_basics, 'rviz', 'rviz.rviz')],
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[
            {'use_sim_time': True},
        ]
    )

    # Spawn the URDF model using the `/world/<world_name>/create` service
    spawn_urdf_node = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "my_robot",
            "-topic", "robot_description",
            "-x", "0.0", "-y", "0.0", "-z", "0.5", "-Y", "0.0"  # Initial spawn position
        ],
        output="screen",
        parameters=[
            {'use_sim_time': True},
        ]
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'robot_description': Command(['xacro', ' ', urdf_file_path]),
             'use_sim_time': True},
        ],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static')
        ]
    )

    # Node to bridge messages like /cmd_vel and /odom
    gz_bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            "/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model",
            "/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V"
        ],
        output="screen",
        parameters=[
            {'use_sim_time': True},
        ]
    )

    trajectory_node = Node(
        package='mogi_trajectory_server',
        executable='mogi_trajectory_server',
        name='mogi_trajectory_server',
    )

    launchDescriptionObject = LaunchDescription()

    launchDescriptionObject.add_action(asl_model_path_arg)
    launchDescriptionObject.add_action(asl_model_name_arg)
    launchDescriptionObject.add_action(rviz_launch_arg)
    launchDescriptionObject.add_action(world_arg)
    launchDescriptionObject.add_action(model_arg)
    launchDescriptionObject.add_action(world_launch)
    launchDescriptionObject.add_action(v4l2_camera_node)
    launchDescriptionObject.add_action(asl_controller_node)
    launchDescriptionObject.add_action(rviz_node)
    launchDescriptionObject.add_action(spawn_urdf_node)
    launchDescriptionObject.add_action(robot_state_publisher_node)
    launchDescriptionObject.add_action(gz_bridge_node)
    launchDescriptionObject.add_action(trajectory_node)
    
    return launchDescriptionObject    
