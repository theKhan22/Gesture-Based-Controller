import cv2
import mediapipe as mp
import subprocess



with open("c:/Users/User/Desktop/paths.txt", "r") as file:
    # Read the contents of the file
    lines = file.readlines()


path1 = ""
path2 = ""
path3 = ""


for line in lines:
   
    if "Path 1:" in line:
        path1 = line.split("Path 1: ")[1].strip()
    elif "Path 2:" in line:
        path2 = line.split("Path 2: ")[1].strip()
    elif "Path 3:" in line:
        path3 = line.split("Path 3: ")[1].strip()


print("Path 1:", path1)
print("Path 2:", path2)
print("Path 3:", path3)





mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)



# Gesture detection variables

gesture = None
check = False

# Main loop
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    
    results = hands.process(image)

    
    cv2.imshow('Shortcuts', frame)

   
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            
            
            if (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y):
                    gesture_detected = True
                    gesture = 'X'
                
            
            
                 
            
            if (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
                                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
                                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
                                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y) and gesture_detected:
                            check = True
                        
            if (check):
               
                            if (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
                                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
                                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
                                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y):            
                                
                                    gesture = 'A'
                            elif (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
                                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
                                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
                                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y):
                               
                                    gesture = 'B'
                            elif (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
                                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
                                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
                                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y):
                                
                                    gesture = 'C'
                                    
                            print(gesture)
                       

                           
                            if (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
                                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
                                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
                                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y) & (gesture == 'A'):
                                 subprocess.Popen(path1)
                                 check = False
                                 
                            elif (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
                                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
                                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
                                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y) & (gesture == 'B'):
                                 subprocess.Popen(path2)
                                 check = False
                                 
                            elif (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
                                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
                                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
                                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y) & (gesture == 'C'):
                                 subprocess.Popen(path3)
                                 check = False
                                 

            
            gesture = None
            



    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
