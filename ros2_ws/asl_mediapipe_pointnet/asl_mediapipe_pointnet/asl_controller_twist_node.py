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

#from std_msgs.msg import String
from geometry_msgs.msg import Twist

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
class AslControllerTwistNode(Node):

    def __init__(self):
        super().__init__('asl_controller_twist_node')
        self.subscriber1_ = self.create_subscription(Image,'image_raw',self.listener_callback,10)
        self.subscriber1_  # prevent unused variable warning
        self.publisher1_ = self.create_publisher(Image, 'asl_controller/image_annotated', 10)
        # Create publisher for velocity command (twist)
        self.publisher2_ = self.create_publisher(Twist, 'asl_controller/cmd_vel', 10)        

        # verbose
        self.declare_parameter("verbose", True)
        self.verbose = self.get_parameter('verbose').value          
        self.get_logger().info('Verbose : "%s"' % self.verbose)

        # use_imshow
        self.declare_parameter("use_imshow", True)
        self.use_imshow = self.get_parameter('use_imshow').value          
        self.get_logger().info('Use imshow : "%s"' % self.use_imshow)

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
        self.asl_model = torch.load(os.path.join(self.model_path, self.model_name),weights_only=False,map_location=device)
        #self.asl_model.eval() # set dropout and batch normalization layers to evaluation mode before running inference

        # Sign Detection status
        self.asl_sign = ""
        self.actionDetected = ""        

        # Additional Settings (for text overlay)
        self.scale = 1.0
        self.text_fontType = cv2.FONT_HERSHEY_SIMPLEX
        self.text_fontSize = 0.75*self.scale
        self.text_color    = (255,0,0)
        self.text_lineSize = max( 1, int(2*self.scale) )
        self.text_lineType = cv2.LINE_AA
        self.text_x = int(10*self.scale)
        self.text_y = int(30*self.scale)        

        self.get_logger().info("Initialization Successful")


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
                    hand_color = (0, 255, 0) # RGB : Green
                    hand_msg = 'LEFT='
                if handedness == "Right":
                    hand_x = image_width-256
                    hand_y = 30
                    hand_color = (255, 0, 0) # RGB : Red
                    hand_msg = 'RIGHT='
                          
                # Determine point cloud of hand
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
                    label = self.asl_model(pointst)
                    label = label.detach().cpu().numpy()
                    #self.get_logger().info('Detected Labels: "%s"' % label)                    
                    asl_id = np.argmax(label)
                    asl_sign = list(char2int.keys())[list(char2int.values()).index(asl_id)]                
                    #self.get_logger().info('Detected Sign: "%s"' % asl_sign)

                    asl_text = hand_msg+asl_sign
                    cv2.putText(annotated_image,asl_text,
                        (hand_x,hand_y),
                        self.text_fontType,self.text_fontSize,
                        hand_color,self.text_lineSize,self.text_lineType)

                    if handedness == "Left":
                        # Define action
                        if asl_sign == 'A':
                          self.actionDetected = "A : Advance"
                        if asl_sign == 'B':
                          self.actionDetected = "B : Back-Up"
                        if asl_sign == 'L':
                          self.actionDetected = "L : Turn Left"
                        if asl_sign == 'R':
                          self.actionDetected = "R : Turn Right"

                        action_text = '['+self.actionDetected+']'
                        cv2.putText(annotated_image,action_text,
                            (hand_x,hand_y*2),
                            self.text_fontType,self.text_fontSize,
                            hand_color,self.text_lineSize,self.text_lineType)

                        if self.verbose:
                            self.get_logger().info(f"{asl_text} => {action_text}")

 
                    if handedness == "Right":
                        # Define action
                        # ... TBD ...

                        action_text = '['+self.actionDetected+']'
                        cv2.putText(annotated_image,action_text,
                            (hand_x,hand_y*2),
                            self.text_fontType,self.text_fontSize,
                            hand_color,self.text_lineSize,self.text_lineType)

                        if self.verbose:
                            self.get_logger().info(f"{asl_text} => {action_text}")


                except:
                    #print("[ERROR] Exception occured during ASL classification ...")
                    self.get_logger().warning('Exception occured during ASL Classification ...') 

                if handedness == "Left" and self.actionDetected != "":
                    try:
                        # Create message
                        msg = Twist()
                        msg.linear.x = 0.0
                        msg.linear.y = 0.0
                        msg.linear.z = 0.0
                        msg.angular.x = 0.0
                        msg.angular.y = 0.0
                        msg.angular.z = 0.0

                        if self.actionDetected == "A : Advance":
                          # Modify message to advance (+ve value on x axis)
                          msg.linear.x = 2.0
                        if self.actionDetected == "B : Back-Up":
                          # Modify message to backup (-ve value on x axis)
                          msg.linear.x = -2.0

                        if self.actionDetected == "L : Turn Left":
                          # Modify message to advance (+ve value on x axis)
                          msg.linear.x = 2.0
                          # Modify message to turn left (+ve value on z axis)
                          msg.angular.z = 2.0
                        if self.actionDetected == "R : Turn Right":
                          # Modify message to advance (+ve value on x axis)
                          msg.linear.x = 2.0
                          # Modify message to turn right (-ve value on z axis)
                          msg.angular.z = -2.0

                        self.publisher2_.publish(msg)

                    except Exception as e:
                        self.get_logger().warn(f"Error publishing twist message: {e}")

        if self.use_imshow == True:
            # DISPLAY
            cv2_bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('asl_controller_twist_node',cv2_bgr_image)
            cv2.waitKey(1)                    
        
        # CONVERT BACK TO ROS & PUBLISH
        image_ros = bridge.cv2_to_imgmsg(annotated_image, encoding="rgb8")        
        self.publisher1_.publish(image_ros)


def main(args=None):
    rclpy.init(args=args)

    asl_controller_twist_node = AslControllerTwistNode()

    rclpy.spin(asl_controller_twist_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    asl_controller_twist_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
