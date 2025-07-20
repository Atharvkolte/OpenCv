import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands() 
mpDraw=mp.solutions.drawing_utils


pTime=0
cTime=0

while True:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)

    if not ret:
        print("failed to grab frame")
        break
        
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h,w,c=frame.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
                if id==0:
                    cv2.circle(frame,(cx,cy),10,(255,0,0),cv2.FILLED)
            mpDraw.draw_landmarks(frame,handLms,mpHands.HAND_CONNECTIONS)
    
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    
    cv2.putText(frame,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    
    
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
    
    
    
cap.release()
cv2.destroyAllWindows()