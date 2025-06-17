
import os
import sys
import numpy as np
import torch
import cv2
import mediapipe as mp
import itertools
import matplotlib.pyplot as plt

from tqdm import tqdm

sys.path.append('./model')
from point_net import PointNet

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#mp_holistic = mp.solutions.holistic
#holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#char2int = {
#            "a":0, "b":1, "c":2, "d":3, "e":4, "f":5, "g":6, "h":7, "i":8, "k":9, "l":10, "m":11,
#            "n":12, "o":13, "p":14, "q":15, "r":16, "s":17, "t":18, "u":19, "v":20, "w":21, "x":22, "y":23
#            }
char2int = {
            "A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6, "H":7, "I":8, "K":9, "L":10, "M":11,
            "N":12, "O":13, "P":14, "Q":15, "R":16, "S":17, "T":18, "U":19, "V":20, "W":21, "X":22, "Y":23
            }

# Parameters (for text overlay)
scale = 1.0
text_fontType = cv2.FONT_HERSHEY_SIMPLEX
text_fontSize = 0.75*scale
text_color    = (255,0,0)
text_lineSize = max( 1, int(2*scale) )
text_lineType = cv2.LINE_AA
text_x = int(10*scale)
text_y = int(30*scale)  

def predict_live():
    model_name = 'point_net_1.pth'
    model_path = './model'

    model = torch.load(os.path.join(model_path, model_name),weights_only=False,map_location=device)
    
    # Open video
    input_video = 0
    cap = cv2.VideoCapture(input_video)
    frame_width = 640
    frame_height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)
    #frame_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    #frame_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("[INFO] input : camera",input_video," (",frame_width,",",frame_height,")")

    while True:    
        flag, frame = cap.read()
        if not flag:
            print("[ERROR] cap.read() FAILEd !")
            break    

        #cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2_image = frame.copy()
        
        # Mirror horizontally for selfie-mode
        cv2_image = cv2.flip(cv2_image, 1)

        # Process it with MediaPipe Hands.
        image = cv2_image
        results = hands.process(image)

        image_height, image_width, _ = image.shape
        annotated_image = image.copy()

        if results.multi_hand_landmarks:
                
            #for hand_landmarks in results.multi_hand_landmarks:
            print("Detected Hands : ",len(results.multi_hand_landmarks))
            for hand_id in range(len(results.multi_hand_landmarks)):
                hand_handedness = results.multi_handedness[hand_id]
                hand_landmarks = results.multi_hand_landmarks[hand_id]
                
                handedness = hand_handedness.classification[0].label
                print('[INFO] Detected Hand: "%s"' % handedness)
                
                hand_x = text_x
                hand_y = text_y
                hand_color = text_color
                if handedness == "Left":
                    hand_x = 10
                    hand_y = 30
                    #hand_color = (0, 0, 255) # BGR : Red
                    hand_color = (0, 255, 0) # BGR : Green
                    #hand_color = (0, 0, 255) # BGR : Blue
                    hand_msg = 'LEFT='
                if handedness == "Right":
                    hand_x = image_width-128
                    hand_y = 30
                    hand_color = (0, 0, 255) # BGR : Red
                    #hand_color = (0, 255, 0) # BGR : Green
                    #hand_color = (255, 0, 0) # BGR : Blue
                    hand_msg = 'RIGHT='
                          
                # Determine point cloud of hand
                points_raw=[]
                for lm in hand_landmarks.landmark:
                    points_raw.append([lm.x, lm.y, lm.z])
                points_raw = np.array(points_raw)

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
                for hc in mp_hands.HAND_CONNECTIONS:
                    cv2.line(annotated_image,
                            (int((points_raw[hc[0]][0]) * image_width), 
                             int((points_raw[hc[0]][1]) * image_height)),
                            (int((points_raw[hc[1]][0]) * image_width), 
                             int((points_raw[hc[1]][1]) * image_height)), 
                             hand_color, 4)
                #try:
                if True:
                    pointst = torch.tensor([points_norm]).float().to(device)
                    label = model(pointst)
                    label = label.detach().cpu().numpy()
                    asl_id = np.argmax(label)
                    asl_sign = list(char2int.keys())[list(char2int.values()).index(asl_id)]                
                                    
                    #asl_text = '['+str(asl_id)+']='+asl_sign
                    asl_text = hand_msg+asl_sign
                    cv2.putText(annotated_image,asl_text,
                    	(hand_x,hand_y),
                    	text_fontType,text_fontSize,
                    	hand_color,text_lineSize,text_lineType)
        
                    actionDetected = ""
                    if asl_sign == 'A':
                      actionDetected = "A : Advance"
                    if asl_sign == 'B':
                      actionDetected = "B : Back-Up"
                    if asl_sign == 'L':
                      actionDetected = "L : Turn Left"
                    if asl_sign == 'R':
                      actionDetected = "R : Turn Right"
                      
                    print("Action = ",actionDetected)
                            
                #except:
                else:
                    print("[ERROR] Exception occured during ASL classification ...")
                           
        cv2.imshow("pointnet_hands",annotated_image)
        key = cv2.waitKey(10)
        if key == 27 or key == 113: # ESC or 'q':
            break    

if __name__ == '__main__':
    predict_live()
