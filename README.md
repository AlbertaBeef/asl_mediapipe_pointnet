# Overview

ASL Recognition using MediaPipe and Pointnet.

## MediaPipe

References:

- Documentation [google](https://google.github.io/mediapipe/solutions/hands.html)


## PointNet (for Hands)

References:

- Article [medium](https://medium.com/@er_95882/asl-recognition-using-pointnet-and-mediapipe-f2efda78d089)
- Dataset [kaggle](https://www.kaggle.com/datasets/ayuraj/asl-dataset)
- Source [github](https://github.com/e-roe/pointnet_hands/tree/main)

The PointNet model can be re-trained based on the previous references.

The pre-trained model can be downloaded as follows:

   - cd model
   - source ./get_model.sh
   - ..

## Stand-Alone Operation

The stand-alone MediaPipe + PointNet pipeline can be launched as follows:

   - python3 asl_mediapipe_pointnet_live.py

![](images/asl_mediapipe_pointnet_demo01_standalone.gif)

## Use as ROS2 Node

Itialize the ROS2 environment:

   - /opt/ros2/humble/setup.bash

Build and Install the ROS2 packages:

   - cd ros2_ws/asl_mediapipe_pointnet
   - colcon build
   - source install/setup.bash

The packages provides two controllers:

   - asl_controller_twist_node : used to control a vehicle
      - recognized hand signs (A:advance, B:back-up, L:turn-left, R:turn-right)
      - generates Twist messages
   - asl_controller_pose_node : used to control a robotic arm
      - recognized hand signs (A:advance, B:back-up, L:left, R:right, U:up, D:down)
      - reads current pose of robotic arm (position of gripper wrt base)
      - generates Pose messages for current position and target position
      - communicates with MoveIt2 to plan/execute robotic arm movement
      
Launch the asl_mediapipe_pointnet_twist node with v4l2_camera only: (requires an additional camera source)

   - ros2 run asl_mediapipe_pointnet asl_mediapipe_pointnet_twist_node


Launch the asl_controller_twist node with v4l2_camera and turtlesim nodes:

   - ros2 launch asl_mediapipe_pointnet asl_mediapipe_pointnet_turtlesim_launch.py

![](images/asl_mediapipe_pointnet_demo01_ros2_turtlesim.gif)



## Use as ROS2 Node with Gazebo simulator

Launch the asl_controller_twist node with MOGI-ROS vehicle:

   - ros2 launch asl_mediapipe_pointnet asl_mediapipe_pointnet_mogiros.launch.py

![](images/asl_mediapipe_pointnet_demo01_ros2_gazebo.gif)

Launch the asl_controller_twist node with ROSMASTER-X3 vehicle:

   - ros2 launch asl_mediapipe_pointnet asl_mediapipe_pointnet_rosmaster.launch.py

![](images/asl_mediapipe_pointnet_demo02_ros2_gazebo_rosmaster.gif)

Launch the asl_controller_pose node with MYCOBOT-280 robotic arm:

   - ros2 launch asl_mediapipe_pointnet asl_mediapipe_pointnet_mycobot.launch.py

![](images/asl_mediapipe_pointnet_demo03_ros2_gazebo_mycobot.gif)

