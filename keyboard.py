import cv2
from time import sleep
from pynput.keyboard import Controller,Key
import mediapipe as mp
import math

mp_drawing= mp.solutions.drawing_utils #
mp_hands = mp.solutions.hands #

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
    
    return current_gesture
        
    

     


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


cap =cv2.VideoCapture(0)
prevClick=None
prevClick1=None
prevClick2=None


with mp_hands.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.5) as hands:
    cap.set(3,1280)
    cap.set(4,720)

    while True:
       
        success,image= cap.read()
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #in order to work with mediapipe we need rgb
        image = cv2.flip(image,1)
        image.flags.writeable=False
        results = hands.process(image)


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

                   
                

        image.flags.writeable = True
        #rgb2bgr
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for num,hand in enumerate(results.multi_hand_landmarks):                
                    mp_drawing.draw_landmarks(image,hand,mp_hands.HAND_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(121,22,76),thickness=2,circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(121,44,250),thickness=2,circle_radius=2))
                    
        
        cv2.imshow('Image', image)

        

        if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
