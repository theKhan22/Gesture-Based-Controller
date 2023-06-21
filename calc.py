import cv2
import math
import mediapipe as mp
import subprocess


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


class Button:
    def __init__(self, pos, width, height, value):
        self.pos = pos
        self.width = width
        self.height = height
        self.value = value

    def draw(self, img):
        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),
                      (255, 255, 153), cv2.FILLED)
        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),
                      (50, 50, 50), 3)
        cv2.putText(img, self.value, (self.pos[0] + 30, self.pos[1] + 20), cv2.FONT_HERSHEY_PLAIN,
                    1, (50, 50, 50), 2)

    def checkClick(self, x, y):
        if self.pos[0] < x < self.pos[0] + self.width and \
                self.pos[1] < y < self.pos[1] + self.height:
            cv2.rectangle(img, (self.pos[0] + 3, self.pos[1] + 3),
                          (self.pos[0] + self.width - 3, self.pos[1] + self.height - 3),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(img, self.value, (self.pos[0] + 25, self.pos[1] + 80), cv2.FONT_HERSHEY_PLAIN,
                        5, (0, 0, 0), 5)
            return True
        else:
            return False


# Buttons
buttonListValues = [['7', '8', '9', '*'],
                    ['4', '5', '6', '-'],
                    ['1', '2', '3', '+'],
                    ['0', '/', '.', '='],
                    ['00', '(',')','%']]


buttonList = []
for x in range(4):
    for y in range(5):
        xpos = x * 70 + 100
        ypos = y * 65 + 50

        buttonList.append(Button((xpos, ypos), 60, 60, buttonListValues[y][x]))


myEquation = ''
delayCounter = 0
goStart=0

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.5) as hands:

    while True:
        # Get image frame
        success, img = cap.read()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
        img = cv2.flip(img, 1)
        img.flags.writeable=False
        results = hands.process(img)
        

        # Draw All

        x3=10
        y3=180
        w3=60
        h3=30
        
        cv2.rectangle(img,(x3,y3),(x3+w3,y3+h3),(220,220,227),cv2.FILLED)
        cv2.putText(img,"H",(x3+10,y3+25),cv2.FONT_HERSHEY_DUPLEX,1,(10,10,10))



        x=100
        cv2.rectangle(img, (x, 15), (x + 270,50),
                    (204, 255, 255), cv2.FILLED)

        cv2.rectangle(img, (x, 15), (x + 270, 50),
                    (50, 50, 50), 3)
        for button in buttonList:
            button.draw(img)
        
        x1=400
        y1=200
        cv2.rectangle(img, (x1, y1), (x1+60,y1+60),(204, 255, 255), cv2.FILLED)
        cv2.putText(img, '<-', (x1,y1+30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)

        # Check for Hand
        if results.multi_hand_landmarks:
            # Find distance between fingers
            hand_landmarks=results.multi_hand_landmarks[0]
            gesture = getGesture(hand_landmarks)


            hx1 = int(hand_landmarks.landmark[8].x * img.shape[1])
            hy1 = int(hand_landmarks.landmark[8].y * img.shape[0])
            print(gesture)

            

            # If clicked check which button and perform action
            if gesture=='V' and delayCounter == 0:
                for i, button in enumerate(buttonList):
                    if button.checkClick(hx1, hy1):
                        myValue = buttonListValues[int(i % 5)][int(i / 5)]  # get correct value
                        if isinstance(myValue, int):
                            myValue = str(myValue)

                        if myValue == '=':
                            # replace % with /100 and evaluate the expression
                            myEquation = myEquation.replace('%', '/100')
                            myEquation = str(eval(myEquation))
                            
                            

                        else:
                            myEquation += myValue
                        delayCounter = 1

            if gesture=='V' and x1<hx1<x1+60 and y1<hy1<y1+60:
                myEquation = ''
            
            if( (x3<hx1<x3+w3) and (y3<hy1<y3+h3) ):

                cv2.rectangle(img,(x3,y3),(x3+w3,y3+h3),(235,12,145),cv2.FILLED)
                cv2.putText(img,"H",(x3+10,y3+25),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)
                
                
                if(gesture=='V'):
                    goStart=1   

        # to avoid multiple clicks
        if delayCounter != 0:
            delayCounter += 1
            if delayCounter > 10:
                delayCounter = 0

        # Write the Final answer

        text_x=100
        cv2.putText(img, myEquation, (text_x, 40), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 0), 3)

        # Display
        key = cv2.waitKey(1)

        if results.multi_hand_landmarks:
            for num,hand in enumerate(results.multi_hand_landmarks):                
                    mp_drawing.draw_landmarks(img,hand,mp_hands.HAND_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(121,22,76),thickness=2,circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(121,44,250),thickness=2,circle_radius=2))
                    
        
        #draw the delete button 
        

        img.flags.writeable = True
        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        cv2.imshow("Calculator", img)

        if goStart:
            break
        if cv2.waitKey(10) & 0xFF == ord('q'):        
            break

    
        
cap.release()
cv2.destroyAllWindows()
if goStart:
    subprocess.Popen(['python', 'start.py'])

   