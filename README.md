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

Build and Install the ROS2 package:

   - cd ros2_ws/asl_mediapipe_pointnet
   - colcon build
   - source install/setup.bash

Run the asl_mediapipe_pointnet_demo node (requires an additional camera source)

   - ros2 run asl_mediapipe_pointnet asl_mediapipe_pointnet_demo

Launch the asl_mediapipe_pointnet node with v4l2_camera and turtlesim nodes:

   - ros2 launch asl_mediapipe_pointnet asl_mediapipe_pointnet_turtlesim_launch.py

![](images/asl_mediapipe_pointnet_demo01_ros2_turtlesim.gif)

Launch the asl_mediapipe_pointnet node with v4l2_camera only:

   - ros2 launch asl_mediapipe_pointnet asl_mediapipe_pointnet_launch.py

![](images/asl_mediapipe_pointnet_demo01_ros2_gazebo.gif)

