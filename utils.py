import mediapipe as mp
from pynput.keyboard import Controller,Key
import cv2
import math
import subprocess



keyboard=Controller()

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

mp_drawing= mp.solutions.drawing_utils #
mp_hands = mp.solutions.hands #


def main():
    goCalc=0
    goShort=0
    goPres=0
    goStart=0
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            success,image=cap.read()
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image=cv2.flip(image,1)
            image.flags.writeable=False

            results = hands.process(image)

            x=130
            my=50
            mk=130
            mu=210


         
        

            cv2.rectangle(image,(x,my),(x+400,my+50),(0,0,128),-1)
            cv2.putText(image,"Calculator",(245,80),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            cv2.rectangle(image,(x,mk),(x+400,mk+50),((0,0,128)),-1)
            cv2.putText(image,"Shortcuts",(245,160),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            cv2.rectangle(image,(x,mu),(x+400,mu+50),((0,0,128)),-1)
            cv2.putText(image,"Presentation",(245,240),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)


            cv2.circle(image, (530,100), 2,(255,0,255), 1)

            x3=10
            y3=180
            w3=60
            h3=30
            
            cv2.rectangle(image,(x3,y3),(x3+w3,y3+h3),(220,220,227),cv2.FILLED)
            cv2.putText(image,"H",(x3+10,y3+25),cv2.FONT_HERSHEY_DUPLEX,1,(10,10,10))

            if results.multi_hand_landmarks:
                hand_landmarks=results.multi_hand_landmarks[0]
                gesture = getGesture(hand_landmarks)
                print(gesture)               
                hx = int(hand_landmarks.landmark[8].x * image.shape[1])
                hy = int(hand_landmarks.landmark[8].y * image.shape[0])



                if((x<hx<x+400) and (my<hy<my+50) ):
                        cv2.rectangle(image,(130,50),(530,100),(235,12,145),-1)
                        cv2.putText(image,"Calculator",(245,80),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                        
                        if(gesture=='V'):
                            goCalc=1
                            
                            

                elif( (x<hx<x+400) and (mk<hy<mk+50) ):
                        cv2.rectangle(image,(130,130),(530,180),(235,12,145),-1)
                        cv2.putText(image,"Shortcuts",(245,160),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                        
                        if(gesture=='V'):
                            goShort=1
                
                elif( (x<hx<x+400) and (mu<hy<mu+50) ):
                        cv2.rectangle(image,(130,mu),(530,mu+50),(235,12,145),-1)
                        cv2.putText(image,"Presentation",(245,240),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                        
                        if(gesture=='V'):
                            goPres=1


                elif( (x3<hx<x3+w3) and (y3<hy<y3+h3) ):
                        cv2.rectangle(image,(x3,y3),(x3+w3,y3+h3),(235,12,145),cv2.FILLED)
                        cv2.putText(image,"H",(x3+10,y3+25),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)
                        
                        
                        if(gesture=='V'):
                            goStart=1                
                




            image.flags.writeable = True
            image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for num,hand in enumerate(results.multi_hand_landmarks):                
                    mp_drawing.draw_landmarks(image,hand,mp_hands.HAND_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(121,22,76),thickness=2,circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(121,44,250),thickness=2,circle_radius=2))
                    
            
            

            cv2.imshow("Image",image)
            if(goCalc or goPres or goShort or goStart ):
                break
            
            if cv2.waitKey(10) & 0xFF==ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    if goCalc:
        subprocess.Popen(['python', 'calc.py'])
    elif goPres:
        subprocess.Popen(['python', 'presentation.py'])
    elif goShort:
        subprocess.Popen(['python', 'shortcuts.py'])
    elif goStart:
        subprocess.Popen(['python', 'start.py'])


   

main()