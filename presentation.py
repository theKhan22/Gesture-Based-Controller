import mediapipe as mp
from pynput.keyboard import Controller,Key
import cv2
import math
from time import sleep


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

    current_gesture =finger
    
   
    
    return current_gesture

mp_drawing= mp.solutions.drawing_utils #
mp_hands = mp.solutions.hands #


def main():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            success,image=cap.read()
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image=cv2.flip(image,1)
            image.flags.writeable=False

            results = hands.process(image)

            if results.multi_hand_landmarks:
                hand_land_marks=results.multi_hand_landmarks[0]
                gesture = getGesture(hand_land_marks)
                print(gesture)

                

                if gesture==8:
                    keyboard.press(Key.right)
                    sleep(1)
                elif gesture==4:
                    keyboard.press(Key.left)
                    
                    sleep(1)




            image.flags.writeable = True
            image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for num,hand in enumerate(results.multi_hand_landmarks):                
                    mp_drawing.draw_landmarks(image,hand,mp_hands.HAND_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(121,22,76),thickness=2,circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(121,44,250),thickness=2,circle_radius=2))
                    
            
            

            cv2.imshow("Presentation Mode",image)
            
            if cv2.waitKey(10) & 0xFF==ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()


   

main()