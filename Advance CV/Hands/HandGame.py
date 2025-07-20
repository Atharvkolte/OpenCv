import cv2
import mediapipe as mp
import time
import handDetectorMod as hm


pTime=0
cTime=0
cap=cv2.VideoCapture(0)
detector=hm.HandDetector()
while True:
    ret,frame=cap.read()
    if not ret:
        print("failed to grab frame")
        break
    frame = cv2.flip(frame, 1)
    frame=detector.findHands(frame)
    lmlist=detector.findPosition(frame)
    if len(lmlist) >4:
        print(lmlist[4])
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    
    cv2.putText(frame,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    
    
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()