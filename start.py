import numpy as np
import cv2
import mediapipe as mp
import pyautogui
import math
from enum import IntEnum
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import subprocess

mp_drawing= mp.solutions.drawing_utils #
mp_hands = mp.solutions.hands #


class Gest(IntEnum):
    FIST = 0
    PINKY = 1
    RING = 2
    MID = 4
    LAST3 = 7
    INDEX = 8
    FIRST2 = 12
    LAST4 = 15
    THUMB = 16    
    PALM = 31
    V_GEST = 33
    TWO_FINGER_CLOSED = 34
    PINCH_MAJOR = 35
    PINCH_MINOR = 36



class HandRecog:

    def __init__(self):
        self.finger = 0
        self.ori_gesture = Gest.PALM
        self.prev_gesture = Gest.PALM
        self.frame_count = 0
        self.hand_result = None

    def update_hand_result(self, hand_result):
        self.hand_result = hand_result
       


    def get_signed_dist(self, point):
        sign = -1
        if self.hand_result.landmark[point[0]].y < self.hand_result.landmark[point[1]].y:
            sign = 1
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        dist = math.sqrt(dist)
        return dist*sign


    def get_dist(self,point):
      
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        dist = math.sqrt(dist)
        return dist
        
    def get_dz(self,point):
        

        return abs(self.hand_result.landmark[point[0]].z - self.hand_result.landmark[point[1]].z)


    def set_finger_state(self):
        if self.hand_result == None:
            return

        points = [[8,5,0],[12,9,0],[16,13,0],[20,17,0]]
        self.finger = 0
        self.finger = self.finger | 0 #thumb
        for idx,point in enumerate(points):
            
            dist = self.get_signed_dist(point[:2])
            dist2 = self.get_signed_dist(point[1:])
            
            try:
                ratio = round(dist/dist2,1)
            except:
                ratio = round(dist/0.01,1)

            self.finger = self.finger << 1
            if ratio > 0.5 :
                self.finger = self.finger | 1



    def get_gesture(self):

        
       

        if Gest.FIRST2 == self.finger :
            point = [[8,12],[5,9]]
            dist1 = self.get_dist(point[0])
            dist2 = self.get_dist(point[1])
            ratio = dist1/dist2
            if ratio > 1.7:
                current_gesture = 0
            else:
                if self.get_dz([8,12]) < 0.1:
                    current_gesture =  1
                
            
            
        else:
            current_gesture =  self.finger

        
        #print(bin(self.finger))

        
        
        
        if current_gesture == self.prev_gesture:
            self.frame_count += 1
        else:
            self.frame_count = 0

        self.prev_gesture = current_gesture
       

        if self.frame_count > 4 :
            self.ori_gesture = current_gesture
        return self.ori_gesture



class Controller:
    tx_old = 0
    ty_old = 0
    trial = True
    flag = False
    grabflag = False
    framecount = 0
    prev_hand = None

    def get_position(hand_result):
        point = 9
        position = [hand_result.landmark[point].x ,hand_result.landmark[point].y]
        sx,sy = pyautogui.size()
        x_old,y_old = pyautogui.position()
        x = int(position[0]*sx)
        y = int(position[1]*sy)
        if Controller.prev_hand is None:
            Controller.prev_hand = x,y
        delta_x = x - Controller.prev_hand[0]
        delta_y = y - Controller.prev_hand[1]

        distsq = delta_x**2 + delta_y**2
        ratio = 1
        Controller.prev_hand = [x,y]

        if distsq <= 25:
            ratio = 0
        elif distsq <= 900:
            ratio = 0.07 * (distsq ** (1/2))
        else:
            ratio = 2.1
        x , y = x_old + delta_x*ratio , y_old + delta_y*ratio
        return (x,y)



    def handle_controls(gesture, hand_result):  
           
        x,y = None,None
        if gesture != Gest.PALM :
            x,y = Controller.get_position(hand_result)
        
        # flag reset
        if gesture != Gest.FIST and Controller.grabflag:
            Controller.grabflag = False
            pyautogui.mouseUp(button = "left")

        
        # implementation
        if gesture == Gest.V_GEST:
            Controller.flag = True
            pyautogui.moveTo(x, y, duration = 0.1)

        elif gesture == Gest.FIST:
            if not Controller.grabflag : 
                Controller.grabflag = True
                pyautogui.mouseDown(button = "left")
            pyautogui.moveTo(x, y, duration = 0.1)

        elif gesture == Gest.MID and Controller.flag:
            pyautogui.click()
            Controller.flag = False

        elif gesture == Gest.INDEX and Controller.flag:
            pyautogui.click(button='right')
            Controller.flag = False

        elif gesture == Gest.TWO_FINGER_CLOSED and Controller.flag:
            pyautogui.doubleClick()
            Controller.flag = False

        


def main():

    capture = cv2.VideoCapture(0)
    goMouse=0
    goKeyboard=0
    goUtil=0

    #the detection confidence initially detects the hands and the tracking confidence tracks the hands
    with mp_hands.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.5) as hands:

        handsRecog=HandRecog()  

        while capture.isOpened():
            ret,frame = capture.read()

            #detections
            image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #in order to work with mediapipe we need rgb

            image = cv2.flip(image,1)

            #set flags
            image.flags.writeable=False


           

            #detections
            results = hands.process(image)


            x=130
            my=50
            mk=130
            mu=210


         
        

            cv2.rectangle(image,(x,my),(x+400,my+50),(0,0,128),-1)
            cv2.putText(image,"Mouse",(245,80),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            cv2.rectangle(image,(x,mk),(x+400,mk+50),((0,0,128)),-1)
            cv2.putText(image,"keyboard",(245,160),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            cv2.rectangle(image,(x,mu),(x+400,mu+50),((0,0,128)),-1)
            cv2.putText(image,"Utilities",(245,240),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)


            cv2.circle(image, (530,100), 2,(255,0,255), 1)


            
            if results.multi_hand_landmarks:
                handsRecog.update_hand_result(results.multi_hand_landmarks[0])           
                handsRecog.set_finger_state()
                

                hand_landmarks = results.multi_hand_landmarks[0]
                hx = int(hand_landmarks.landmark[8].x * image.shape[1])
                hy = int(hand_landmarks.landmark[8].y * image.shape[0])

                print(hx)
                print(hy)

                if((x<hx<x+400) and (my<hy<my+50) ):
                        cv2.rectangle(image,(130,50),(530,100),(235,12,145),-1)
                        cv2.putText(image,"Mouse",(245,80),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                        gesture = handsRecog.get_gesture()
                        if(gesture==1):
                            goMouse=1
                            
                            

                elif( (x<hx<x+400) and (mk<hy<mk+50) ):
                        cv2.rectangle(image,(130,130),(530,180),(235,12,145),-1)
                        cv2.putText(image,"keyboard",(245,160),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                        gesture = handsRecog.get_gesture()
                        if(gesture==1):
                            goKeyboard=1
                
                elif( (x<hx<x+400) and (mu<hy<mu+50) ):
                        cv2.rectangle(image,(130,mu),(530,mu+50),(235,12,145),-1)
                        cv2.putText(image,"Utilities",(245,240),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                        gesture = handsRecog.get_gesture()
                        if(gesture==1):
                            goUtil=1
                        







                
                             

            #set flags
            image.flags.writeable = True

            #rgb2bgr
            image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

            

            if results.multi_hand_landmarks:
                for num,hand in enumerate(results.multi_hand_landmarks):                
                    mp_drawing.draw_landmarks(image,hand,mp_hands.HAND_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(121,22,76),thickness=2,circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(121,44,250),thickness=2,circle_radius=2))
                    
            

            cv2.imshow("Hand Tracking",image)

            if(goUtil or goKeyboard or goMouse ):
                break

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break


    capture.release()
    cv2.destroyAllWindows()
    if goMouse:
        subprocess.Popen(['python', 'input_window.py'])
    elif goKeyboard:
        subprocess.Popen(['python', 'keyboard.py'])
    elif goUtil:
        subprocess.Popen(['python', 'utils.py'])

main()  




