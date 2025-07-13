# Copyright 2025 Tria Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import sys
import os

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
# import CV BRIDGE
from cv_bridge import CvBridge, CvBridgeError
import cv2

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# MediaPipe references : 
#   https://google.github.io/mediapipe/solutions/hands.html
import mediapipe as mp


# PointNet (for Hands) references : 
#    https://medium.com/@er_95882/asl-recognition-using-pointnet-and-mediapipe-f2efda78d089
#    https://www.kaggle.com/datasets/ayuraj/asl-dataset
#    https://github.com/e-roe/pointnet_hands/tree/main
import torch

#sys.path.append('/media/albertabeef/Tycho/asl_mediapipe_pointnet/model')
#from point_net import PointNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#char2int = {
#            "a":0, "b":1, "c":2, "d":3, "e":4, "f":5, "g":6, "h":7, "i":8, "k":9, "l":10, "m":11,
#            "n":12, "o":13, "p":14, "q":15, "r":16, "s":17, "t":18, "u":19, "v":20, "w":21, "x":22, "y":23
#            }
char2int = {
            "A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6, "H":7, "I":8, "K":9, "L":10, "M":11,
            "N":12, "O":13, "P":14, "Q":15, "R":16, "S":17, "T":18, "U":19, "V":20, "W":21, "X":22, "Y":23
            }


@torch.no_grad()        
class AslControllerJointsNode(Node):

    def __init__(self):
        super().__init__('asl_controller_joints_node')
        self.subscriber1_ = self.create_subscription(Image,'image_raw',self.listener_callback,10)
        self.subscriber1_  # prevent unused variable warning
        self.publisher1_ = self.create_publisher(Image, 'asl_controller/image_annotated', 10)
        
        # Open MediaPipe Hands model
        self.mp_hands = mp.solutions.hands
        #self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5,static_image_mode=True,max_num_hands=2)
        #
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        #
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # PointNet model
        self.declare_parameter("model_path", "/root/asl_mediapipe_pointnet/model")
        self.declare_parameter("model_name", "point_net_1.pth")
        self.model_path = self.get_parameter('model_path').value        
        self.model_name = self.get_parameter('model_name').value  
        self.get_logger().info('Model path/name : "%s"' % os.path.join(self.model_path, self.model_name))      
        sys.path.append(self.model_path)
        self.model = torch.load(os.path.join(self.model_path, self.model_name),weights_only=False,map_location=device)
        #self.model.eval() # set dropout and batch normalization layers to evaluation mode before running inference
        
        # Sign Detection status
        self.asl_sign = ""
        self.actionDetected = ""        

        # Parameters (for text overlay)
        self.scale = 1.0
        self.text_fontType = cv2.FONT_HERSHEY_SIMPLEX
        self.text_fontSize = 0.75*self.scale
        self.text_color    = (255,0,0)
        self.text_lineSize = max( 1, int(2*self.scale) )
        self.text_lineType = cv2.LINE_AA
        self.text_x = int(10*self.scale)
        self.text_y = int(30*self.scale)        

        # Create a publisher for the '/arm_controller/joint_trajectory' topic
        self.publisher2_ = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.publisher3_ = self.create_publisher(JointTrajectory, '/gripper_controller/joint_trajectory', 10)

        # Create the JointTrajectory messages
        self.arm_trajectory_command = JointTrajectory()
        arm_joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_joint', 'virtual_roll_joint', 'virtual_yaw_joint']
        self.arm_trajectory_command.joint_names = arm_joint_names
        #
        self.gripper_trajectory_command = JointTrajectory()
        gripper_joint_names = ['left_finger_joint', 'right_finger_joint']
        self.gripper_trajectory_command.joint_names = gripper_joint_names
        

        arm_point= JointTrajectoryPoint()
        #['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_joint', 'virtual_roll_joint', 'virtual_yaw_joint']
        arm_point.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        arm_point.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        arm_point.time_from_start.sec = 1 #2
        
        self.arm_point = arm_point
        self.arm_trajectory_command.points = [arm_point]
        
        # Publish the message
        self.get_logger().info(f"Publishing arm joint angles : {self.arm_trajectory_command.points}")
        self.publisher2_.publish(self.arm_trajectory_command)

        gripper_point = JointTrajectoryPoint()
        #['left_finger_joint', 'right_finger_joint']
        gripper_point.positions = [0.04, 0.04]
        gripper_point.velocities = [0.0, 0.0]
        gripper_point.time_from_start.sec = 1 #2
        
        self.gripper_point = gripper_point
        self.gripper_trajectory_command.points = [gripper_point]
        
        # Publish the message
        self.get_logger().info(f"Publishing gripper joint angles : {self.gripper_trajectory_command.points}")
        self.publisher3_.publish(self.gripper_trajectory_command)
        

    def listener_callback(self, msg):
        bridge = CvBridge()
        cv2_image = bridge.imgmsg_to_cv2(msg,desired_encoding = "rgb8")
        
        # Mirror horizontally for selfie-mode
        cv2_image = cv2.flip(cv2_image, 1)

        # Process it with MediaPipe Hands.
        image = cv2_image
        results = self.hands.process(image)

        image_height, image_width, _ = image.shape
        annotated_image = image.copy()

        if results.multi_hand_landmarks:
        
            #self.get_logger().info('Detected Handedness: "%s"' % results.multi_handedness)
            #self.get_logger().info('Detected Landmarks : "%s"' % results.multi_hand_landmarks)
                
            #for hand_landmarks in results.multi_hand_landmarks:
            for hand_id in range(len(results.multi_hand_landmarks)):
                hand_handedness = results.multi_handedness[hand_id]
                hand_landmarks = results.multi_hand_landmarks[hand_id]
                
                handedness = hand_handedness.classification[0].label
                #self.get_logger().info('Detected Hand: "%s"' % handedness)
                
                hand_x = self.text_x
                hand_y = self.text_y
                hand_color = self.text_color
                if handedness == "Left":
                    hand_x = 10
                    hand_y = 30
                    #hand_color = (0, 0, 255) # RGB : Blue
                    hand_color = (0, 255, 0) # RBG : Green
                    hand_msg = 'LEFT='
                if handedness == "Right":
                    hand_x = image_width-128
                    hand_y = 30
                    #hand_color = (0, 255, 0) # RBG : Green
                    #hand_color = (0, 0, 255) # RGB : Blue
                    hand_color = (255, 0, 0) # RGB : Red
                    hand_msg = 'RIGHT='
                          
                # Determine bounding box of hand
                points_raw=[]
                for lm in hand_landmarks.landmark:
                    points_raw.append([lm.x, lm.y, lm.z])
                points_raw = np.array(points_raw)
                #self.get_logger().info('Points: "%s"' % points_raw)     

                # Normalize point cloud of hand
                points_norm = points_raw.copy()
                min_x = np.min(points_raw[:, 0])
                max_x = np.max(points_raw[:, 0])
                min_y = np.min(points_raw[:, 1])
                max_y = np.max(points_raw[:, 1])
                for i in range(len(points_raw)):
                    points_norm[i][0] = (points_norm[i][0] - min_x) / (max_x - min_x)
                    points_norm[i][1] = (points_norm[i][1] - min_y) / (max_y - min_y)
                    # PointNet model was trained on left hands, so need to mirror right hand landmarks
                    if handedness == "Right":
                        points_norm[i][0] = 1.0 - points_norm[i][0]
                    
                                        
                # Draw hand landmarks of each hand.
                for hc in self.mp_hands.HAND_CONNECTIONS:
                    cv2.line(annotated_image,
                            (int((points_raw[hc[0]][0]) * image_width), 
                             int((points_raw[hc[0]][1]) * image_height)),
                            (int((points_raw[hc[1]][0]) * image_width), 
                             int((points_raw[hc[1]][1]) * image_height)), 
                             hand_color, 4)
	
                asl_sign = ""
                self.actionDetected = ""
                try:
                    pointst = torch.tensor([points_norm]).float().to(device)
                    label = self.model(pointst)
                    label = label.detach().cpu().numpy()
                    #self.get_logger().info('Detected Labels: "%s"' % label)                    
                    asl_id = np.argmax(label)
                    asl_sign = list(char2int.keys())[list(char2int.values()).index(asl_id)]                
                    #self.get_logger().info('Detected Sign: "%s"' % asl_sign)

                    #asl_text = '['+str(asl_id)+']='+asl_sign
                    asl_text = hand_msg+asl_sign
                    cv2.putText(annotated_image,asl_text,
                    	(hand_x,hand_y),
                    	self.text_fontType,self.text_fontSize,
                    	hand_color,self.text_lineSize,self.text_lineType)
        
                    if handedness == "Left":
                        self.get_logger().info('Left Hand Sign: "%s"' % asl_sign)
                        
                        # Define action
                        if asl_sign == 'A':
                          self.actionDetected = "A : Advance"
                        if asl_sign == 'B':
                          self.actionDetected = "B : Back-Up"
                        if asl_sign == 'L':
                          self.actionDetected = "L : Left"
                        if asl_sign == 'R':
                          self.actionDetected = "R : Right"
                        if asl_sign == 'U':
                          self.actionDetected = "U : Up"
                        if asl_sign == 'D':
                          self.actionDetected = "D : Down"

                    if handedness == "Right":
                        self.get_logger().info('Right Hand Sign: "%s"' % asl_sign)

                        # Define action
                        if asl_sign == 'A':
                          self.actionDetected = "A : Close Gripper"
                        if asl_sign == 'B':
                          self.actionDetected = "B : Open Gripper"
                                

                except:
                    #print("[ERROR] Exception occured during ASL classification ...")
                    self.get_logger().warning('Exception occured during ASL Classification ...') 

                if handedness == "Left" and self.actionDetected != "":
                    try:
                        arm_point = self.arm_point

                        # shoulder pan joint : index 0, range +3.14(L) to -3.14(R)
                        shoulder_pan_joint = arm_point.positions[0]
                        if self.actionDetected == "L : Left":
                            shoulder_pan_joint += 0.01
                            if shoulder_pan_joint > +3.14:
                                shoulder_pan_joint = +3.14
                        if self.actionDetected == "R : Right":
                            shoulder_pan_joint -= 0.01
                            if shoulder_pan_joint < -3.14:
                                shoulder_pan_joint = -3.14
                        arm_point.positions[0] = shoulder_pan_joint
                                                
                        # shoulder lift joint : index 1, range +1.57(A) to -1.57(B)
                        shoulder_lift_joint = arm_point.positions[1]
                        if self.actionDetected == "A : Advance":
                            shoulder_lift_joint += 0.01
                            if shoulder_lift_joint > +1.57:
                                shoulder_lift_joint = +1.57
                        if self.actionDetected == "B : Back-Up":
                            shoulder_lift_joint -= 0.01
                            if shoulder_lift_joint < -1.57:
                                shoulder_lift_joint = -1.57
                        arm_point.positions[1] = shoulder_lift_joint

                        # elbow joint : index 2, range -2.35(U) to +2.34(D)
                        elbow_joint = arm_point.positions[2]
                        if self.actionDetected == "U : Up":
                            elbow_joint -= 0.01
                            if elbow_joint < -2.35:
                                elbow_joint = -2.35
                        if self.actionDetected == "D : Down":
                            elbow_joint += 0.01
                            if elbow_joint > +2.35:
                                elbow_joint = +2.35
                        arm_point.positions[2] = elbow_joint
                        
                        self.arm_point = arm_point

                        self.arm_trajectory_command.points = [arm_point]
        
                        # Publish the message
                        self.get_logger().info(f"Publishing arm joint angles : {self.arm_trajectory_command.points}")        
                        self.publisher2_.publish(self.arm_trajectory_command)
                        

                    except Exception as e:
                        self.get_logger().warn(f"Error publishing arm joint angles: {e}")

                if handedness == "Right" and self.actionDetected != "":
                    try:
                        gripper_point = self.gripper_point

                        # left/right finger : index 0/1, range +1.57(A) to -1.57(B)
                        if self.actionDetected == "A : Close Gripper":
                            finger_joint = 0.00
                        if self.actionDetected == "B : Open Gripper":
                            finger_joint = 0.04
                            
                        gripper_point.positions[0] = finger_joint
                        gripper_point.positions[1] = finger_joint
                        
                        self.gripper_point = gripper_point

                        self.gripper_trajectory_command.points = [gripper_point]
        
                        # Publish the message
                        self.get_logger().info(f"Publishing gripper joint angles : {self.gripper_trajectory_command.points}")        
                        self.publisher3_.publish(self.gripper_trajectory_command)

                    except Exception as e:
                        self.get_logger().warn(f"Error publishing gripper joint angles: {e}")
                
        # DISPLAY
        cv2_bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('asl_controller_joints_node',cv2_bgr_image)
        cv2.waitKey(1)                    
        
        # CONVERT BACK TO ROS & PUBLISH
        image_ros = bridge.cv2_to_imgmsg(cv2_image, encoding="rgb8")        
        self.publisher1_.publish(image_ros)


def main(args=None):
    rclpy.init(args=args)

    asl_controller_joints_node = AslControllerJointsNode()

    rclpy.spin(asl_controller_joints_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    asl_controller_joints_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
