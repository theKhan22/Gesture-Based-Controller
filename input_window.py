import cv2
import mediapipe as mp
from time import sleep
import pyautogui
import math
from enum import IntEnum
from google.protobuf.json_format import MessageToDict
from pynput.keyboard import Controller,Key
import subprocess

pyautogui.FAILSAFE = False
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


#keyboard part:
def drawAll(img,buttonList):
        
        for button in buttonList:
            x,y=button.pos
            w,h=button.size
            cv2.rectangle(img,button.pos,(x+w,y+h),(8,0,0),cv2.FILLED)
            cv2.putText(img,button.text,(button.pos[0]+12,button.pos[1]+30),cv2.FONT_HERSHEY_COMPLEX,.9,(255,255,255),1)

        return img


class button():
    def __init__(self,pos,text,size=[60,60]):
        self.pos=pos
        self.text=text
        self.size=size
        



keys=[["Q","W","E","R","T","Y","U","I","O","P"],
      ["A","S","D","F","G","H","J","K","L",";"],
      ["Z","X","C","V","B","N","M","<-","?","."]]
      

keyBoard=Controller()

buttonList=[]

for i in range(len(keys)):         
    for j,key in enumerate(keys[i]):            
        buttonList.append(button([(j*65),(75*i)],key))



def get_signed_dist(point,hand_result):
        sign = -1
        if hand_result.landmark[point[0]].y < hand_result.landmark[point[1]].y:
            sign = 1
        dist = (hand_result.landmark[point[0]].x - hand_result.landmark[point[1]].x)**2
        dist += (hand_result.landmark[point[0]].y - hand_result.landmark[point[1]].y)**2
        dist = math.sqrt(dist)
        return dist*sign

def get_dist(point,hand_result):
    
    dist = (hand_result.landmark[point[0]].x - hand_result.landmark[point[1]].x)**2
    dist += (hand_result.landmark[point[0]].y - hand_result.landmark[point[1]].y)**2
    dist = math.sqrt(dist)
    return dist
        
def get_dz(point,hand_result):
    

    return abs(hand_result.landmark[point[0]].z - hand_result.landmark[point[1]].z)



def getGesture(results):
     

    if results==None:
        return None
     

    points = [[8,5,0],[12,9,0],[16,13,0],[20,17,0]]
    finger = 0
    finger = finger | 0 #thumb
    for idx,point in enumerate(points):
        
        dist1 = get_signed_dist(point[:2],results)
        dist2 = get_signed_dist(point[1:],results)
        
        try:
            ratio = round(dist1/dist2,1)
        except:
            ratio = round(dist1/0.01,1)

        finger = finger << 1
        if ratio > 0.5 :
            finger = finger | 1

    current_gesture ="None"
    
    if finger==12:
        point = [[8,12],[5,9]]
        dist1 = get_dist(point[0],results)
        dist2 = get_dist(point[1],results)
        ratio = dist1/dist2        
        if ratio < 1.7:
            if get_dz([8,12],results) < 0.1:
                current_gesture="V"
    
    elif finger==14:
        current_gesture=14
    
    elif finger==7:
        current_gesture=7
    
    return current_gesture



# Gesture Encodings 
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
   

class HLabel(IntEnum):
    MINOR = 0
    MAJOR = 1

class HandRecog:
    def __init__(self, hand_label):
        self.finger = 0
        self.ori_gesture = Gest.PALM
        self.prev_gesture = Gest.PALM
        self.frame_count = 0
        self.hand_result = None
        self.hand_label = hand_label
    
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
    
    def get_dist(self, point):
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
            
            dist1 = self.get_signed_dist(point[:2])
            dist2 = self.get_signed_dist(point[1:])
            
            try:
                ratio = round(dist1/dist2,1)
            except:
                ratio = round(dist1/0.01,1)

            self.finger = self.finger << 1
            if ratio > 0.5 :
                self.finger = self.finger | 1
    

    # Handling Fluctations due to noise
    def get_gesture(self):
        if self.hand_result == None:
            return Gest.PALM

        current_gesture = Gest.PALM
       

        if Gest.FIRST2 == self.finger :
            point = [[8,12],[5,9]]
            dist1 = self.get_dist(point[0])
            dist2 = self.get_dist(point[1])
            ratio = dist1/dist2
            if ratio > 1.7:
                current_gesture = Gest.V_GEST
            else:
                if self.get_dz([8,12]) < 0.1:
                    current_gesture =  Gest.TWO_FINGER_CLOSED
                
            
        else:
            current_gesture =  self.finger
        
        if current_gesture == self.prev_gesture:
            self.frame_count += 1
        else:
            self.frame_count = 0

        self.prev_gesture = current_gesture

        if self.frame_count > 1 :
            self.ori_gesture = current_gesture
        return self.ori_gesture



# Executes commands according to detected gestures
class Controller:
    tx_old = 0
    ty_old = 0
    trial = True
    flag = False
    grabflag = False
    pinchmajorflag = False
    pinchminorflag = False
    pinchstartxcoord = None
    pinchstartycoord = None
    pinchdirectionflag = None
    prevpinchlv = 0
    pinchlv = 0
    framecount = 0
    prev_hand = None
    pinch_threshold = 0.3
     

    # Locate Hand to get Cursor Position
    # Stabilize cursor by Dampening
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
        """Impliments all gesture functionality."""      
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


class GestureController:
    gc_mode = 0
    cap = None
    CAM_HEIGHT = None
    CAM_WIDTH = None
    hr_major = None # Right Hand by default
    hr_minor = None # Left hand by default
    dom_hand = True

    def __init__(self):
        """Initilaizes attributes."""
        GestureController.gc_mode = 1
        GestureController.cap = cv2.VideoCapture(0)
        GestureController.CAM_HEIGHT = GestureController.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        GestureController.CAM_WIDTH = GestureController.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    
    def classify_hands(results):
        left , right = None,None
        try:
            handedness_dict = MessageToDict(results.multi_handedness[0])
            if handedness_dict['classification'][0]['label'] == 'Right':
                right = results.multi_hand_landmarks[0]
            else :
                left = results.multi_hand_landmarks[0]
        except:
            pass

        try:
            handedness_dict = MessageToDict(results.multi_handedness[1])
            if handedness_dict['classification'][0]['label'] == 'Right':
                right = results.multi_hand_landmarks[1]
            else :
                left = results.multi_hand_landmarks[1]
        except:
            pass
        
        if GestureController.dom_hand == True:
            GestureController.hr_major = right
            GestureController.hr_minor = left
        else :
            GestureController.hr_major = left
            GestureController.hr_minor = right

        

    def start(self):

        prevClick=None
        prevClick1=None
        prevClick2=None
        goBack=0

       

        
        handmajor = HandRecog(HLabel.MAJOR)
        
        keyOn=0
        with mp_hands.Hands(max_num_hands = 2,min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while GestureController.cap.isOpened() and GestureController.gc_mode:
                
                success, image = GestureController.cap.read()

                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if keyOn==0:
                    if results.multi_hand_landmarks:                   
                        GestureController.classify_hands(results)
                        handmajor.update_hand_result(GestureController.hr_major)
                        

                        handmajor.set_finger_state()
                        
                                
                        gest_name = handmajor.get_gesture()
                        Controller.handle_controls(gest_name, handmajor.hand_result)

                        print(gest_name)
                        
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    else:
                        Controller.prev_hand = None
                else:
                    if results.multi_hand_landmarks:                                        
                                
                        gest_name = getGesture(results.multi_hand_landmarks[0])
                   

                        print(gest_name)
                        print("hi")
                        
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    


                
                #the keyboard button
                if keyOn:
                    x=40
                    text_kbd="MSE"
                else:
                    text_kbd="KBD"
                    x=550
                
                y=230
                w=65
                h=50
                cv2.rectangle(image,(x,y),(x+w,y+h),(220,220,227),cv2.FILLED)
                cv2.putText(image,text_kbd,(x,y+40),cv2.FONT_HERSHEY_DUPLEX,1,(10,10,10))

                res =results.multi_hand_landmarks



                if res:     
                    hand_landmarks = results.multi_hand_landmarks[0]
                    hx = int(hand_landmarks.landmark[12].x * image.shape[1])
                    hy = int(hand_landmarks.landmark[12].y * image.shape[0])

                    
                    if x<hx<x+w and y<hy<y+h and gest_name==7:
                        if(keyOn==0):
                            keyOn=1
                        else:
                            keyOn=0
                        sleep(1)

                
                #the scroll up and down button and back button

                if not keyOn:

                    x=550
                    y=120
                    w=60
                    h=30
                    cv2.rectangle(image,(x,y),(x+w,y+h),(220,220,227),cv2.FILLED)
                    cv2.putText(image,"U",(x+10,y+25),cv2.FONT_HERSHEY_DUPLEX,1,(10,10,10))


                    x2=550
                    y2=180
                    w2=60
                    h2=30
                    
                    cv2.rectangle(image,(x2,y2),(x2+w2,y2+h2),(220,220,227),cv2.FILLED)
                    cv2.putText(image,"D",(x2+10,y2+25),cv2.FONT_HERSHEY_DUPLEX,1,(10,10,10))


                    x3=10
                    y3=180
                    w3=60
                    h3=30
                    
                    cv2.rectangle(image,(x3,y3),(x3+w3,y3+h3),(220,220,227),cv2.FILLED)
                    cv2.putText(image,"B",(x3+10,y3+25),cv2.FONT_HERSHEY_DUPLEX,1,(10,10,10))

                    if res:
                        
                        if x<hx<x+w and y<hy<y+h and gest_name==7:
                            cv2.rectangle(image,(x,y),(x+w,y+h),(224,11,139),cv2.FILLED)
                            cv2.putText(image,"U",(x+10,y+25),cv2.FONT_HERSHEY_DUPLEX,1,(247,247,250))

                            pyautogui.scroll(100)  # Positive value indicates scrolling up
                        elif x2<hx<x2+w2 and y2<hy<y2+h2 and gest_name==7:
                            cv2.rectangle(image,(x2,y2),(x2+w2,y2+h2),(224,11,139),cv2.FILLED)
                            cv2.putText(image,"D",(x2+10,y2+25),cv2.FONT_HERSHEY_DUPLEX,1,(247,247,250))
                            pyautogui.scroll(-100)  # Positive value indicates scrolling up
                        elif x3<hx<x3+w3 and y3<hy<y3+h3 and gest_name==7:
                            cv2.rectangle(image,(x2,y2),(x2+w2,y2+h2),(224,11,139),cv2.FILLED)
                            cv2.putText(image,"B",(x2+10,y2+25),cv2.FONT_HERSHEY_DUPLEX,1,(247,247,250))
                            goBack=1
                            



                

                        
                        

                       

                #keyboard code start
                if keyOn==1:
                    image=drawAll(image,buttonList)
                    x1=200
                    y1=220
                    cv2.rectangle(image,(x1,y1),(x1+180,y1+40),((8,0,0)),cv2.FILLED)
                    cv2.putText(image,"SPACE",(x1+25,y1+30),cv2.FONT_HERSHEY_COMPLEX,.9,(255,255,255),1)

                    cv2.rectangle(image,(x1+250,y1),(x1+180+195,y1+40),((8,0,0)),cv2.FILLED)
                    cv2.putText(image,"Delete",(x1+250,y1+30),cv2.FONT_HERSHEY_COMPLEX,.9,(255,255,255),1)
                    if results.multi_hand_landmarks:

                        numberOfHands=len(results.multi_hand_landmarks)


                        if numberOfHands==2:
                            hand_landmarks1 = results.multi_hand_landmarks[0]
                            hand_landmarks2 = results.multi_hand_landmarks[1]

                            hx1 = int(hand_landmarks1.landmark[8].x * image.shape[1])
                            hy1 = int(hand_landmarks1.landmark[8].y * image.shape[0])
                            hx2 = int(hand_landmarks2.landmark[8].x * image.shape[1])
                            hy2 = int(hand_landmarks2.landmark[8].y * image.shape[0])

                            gesture1 = getGesture(hand_landmarks1)
                            gesture2 = getGesture(hand_landmarks2)




                            for button in buttonList:
                                x,y=button.pos
                                w,h=button.size
                                if x<hx1<x+w and y<hy1<y+h:
                                    cv2.rectangle(image,button.pos,(x+w,y+h),(46, 44, 44),cv2.FILLED)
                                    cv2.putText(image,button.text,(button.pos[0]+12,button.pos[1]+30),cv2.FONT_HERSHEY_COMPLEX,.9,(255,255,255),1)
                                    if prevClick1!=button.text:
                                        prevClick1= None
                                    
                                    if(gesture1=="V" and prevClick1!=button.text):
                                        prevClick1= button.text
                                        
                                        
                                        cv2.rectangle(image,button.pos,(x+w,y+h),(135, 130, 130),cv2.FILLED)
                                        cv2.putText(image,button.text,(button.pos[0]+12,button.pos[1]+30),cv2.FONT_HERSHEY_COMPLEX,.9,(255,255,255),1)
                                        if button.text=="<-":
                                            keyBoard.press(Key.enter)
                                            #sleep(0.15)
                                        else:
                                            keyBoard.press(button.text)                        
                                            #sleep(0.15)

                                if x<hx2<x+w and y<hy2<y+h:
                                    cv2.rectangle(image,button.pos,(x+w,y+h),(46, 44, 44),cv2.FILLED)
                                    cv2.putText(image,button.text,(button.pos[0]+12,button.pos[1]+30),cv2.FONT_HERSHEY_COMPLEX,.9,(255,255,255),1)
                                    if prevClick2!=button.text:
                                        prevClick2= None
                                
                                    
                                    if(gesture2=="V" and prevClick2!=button.text):
                                        prevClick2= button.text
                                        
                                        cv2.rectangle(image,button.pos,(x+w,y+h),(135, 130, 130),cv2.FILLED)
                                        cv2.putText(image,button.text,(button.pos[0]+12,button.pos[1]+30),cv2.FONT_HERSHEY_COMPLEX,.9,(255,255,255),1)
                                        if button.text=="<-":
                                            keyBoard.press(Key.enter)
                                            #sleep(0.15)
                                        else:
                                            keyBoard.press(button.text)                        
                                            #sleep(0.15)
                                

                            
                            
                                
                            if x1<hx1<x1+180 and y1<hy1<y1+40:
                                cv2.rectangle(image,(x1,y1),(x1+180,y1+40),(245,183,15),cv2.FILLED)
                                cv2.putText(image,"SPACE",(x1+25,y1+30),cv2.FONT_HERSHEY_DUPLEX,.9,(255,255,255),2)
                            
                                if(gesture1=="V"):
                                    keyBoard.press(Key.space)
                                    sleep(0.25)
                            
                            if x1<hx2<x1+180 and y1<hy2<y1+40:
                                cv2.rectangle(image,(x1,y1),(x1+180,y1+40),(245,183,15),cv2.FILLED)
                                cv2.putText(image,"SPACE",(x1+25,y1+30),cv2.FONT_HERSHEY_DUPLEX,.9,(255,255,255),2)
                            
                                if(gesture2=="V"):
                                    keyBoard.press(Key.space)
                                    sleep(0.25)
                            
                            

                    
                            if x1+250<hx1<x1+180+195 and y1<hy1<y1+40:

                                cv2.rectangle(image,(x1+250,y1),(x1+180+195,y1+40),(245,183,15),cv2.FILLED)
                                cv2.putText(image,"Delete",(x1+250,y1+30),cv2.FONT_HERSHEY_COMPLEX,.9,(255,255,255),1)                
                                if(gesture1=="V"):
                                    keyBoard.press(Key.backspace)
                                    sleep(0.25)
                            
                            if x1+250<hx2<x1+180+195 and y1<hy2<y1+40:

                                cv2.rectangle(image,(x1+250,y1),(x1+180+195,y1+40),(245,183,15),cv2.FILLED)
                                cv2.putText(image,"Delete",(x1+250,y1+30),cv2.FONT_HERSHEY_COMPLEX,.9,(255,255,255),1)                
                                if(gesture2=="V"):
                                    keyBoard.press(Key.backspace)
                                    sleep(0.25)






















                        else:

                            hand_landmarks1 = results.multi_hand_landmarks[0]
                            hx1 = int(hand_landmarks1.landmark[8].x * image.shape[1])
                            hy1 = int(hand_landmarks1.landmark[8].y * image.shape[0])
                            gesture = getGesture(hand_landmarks1)
                            for button in buttonList:
                                x,y=button.pos
                                w,h=button.size
                                if x<hx1<x+w and y<hy1<y+h:
                                    cv2.rectangle(image,button.pos,(x+w,y+h),(46, 44, 44),cv2.FILLED)
                                    cv2.putText(image,button.text,(button.pos[0]+12,button.pos[1]+30),cv2.FONT_HERSHEY_COMPLEX,.9,(255,255,255),1)

                                    if prevClick!=button.text:
                                        prevClick= None                     
                                    
                                    if(gesture=="V" and prevClick!=button.text):
                                        prevClick= button.text
                                        cv2.rectangle(image,button.pos,(x+w,y+h),(135, 130, 130),cv2.FILLED)
                                        cv2.putText(image,button.text,(button.pos[0]+12,button.pos[1]+30),cv2.FONT_HERSHEY_COMPLEX,.9,(255,255,255),1)
                                        if button.text=="<-":
                                            keyBoard.press(Key.enter)
                                            #sleep(0.15)
                                        else:
                                            keyBoard.press(button.text)                        
                                            #sleep(0.15)
                            
                            
                                
                            if x1<hx1<x1+180 and y1<hy1<y1+40:
                                cv2.rectangle(image,(x1,y1),(x1+180,y1+40),(245,183,15),cv2.FILLED)
                                cv2.putText(image,"SPACE",(x1+25,y1+30),cv2.FONT_HERSHEY_DUPLEX,.9,(255,255,255),2)
                                                
                                if(gesture=="V"):
                                    
                                    keyBoard.press(Key.space)
                                    sleep(0.25)
                            
                            

                    
                            if x1+250<hx1<x1+180+195 and y1<hy1<y1+40:

                                cv2.rectangle(image,(x1+250,y1),(x1+180+195,y1+40),(245,183,15),cv2.FILLED)
                                cv2.putText(image,"Delete",(x1+250,y1+30),cv2.FONT_HERSHEY_COMPLEX,.9,(255,255,255),1)

                                
                                if(gesture=="V"):
                                    keyBoard.press(Key.backspace)
                                    sleep(0.25)

            
                            
                #keyboard code end

               
                        
                
                
                if(goBack):
                    break
                cv2.imshow('Input Window', image)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        

        
        GestureController.cap.release()
        cv2.destroyAllWindows()
        if(goBack):
            subprocess.Popen(['python', 'start.py'])









gc1 = GestureController()
gc1.start()


