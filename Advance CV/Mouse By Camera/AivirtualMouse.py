import cv2
import numpy as np
import handDetectorMod as htm 
import time
import autopy

########################
wCam,hCam = 640, 480
frameR = 100  # Frame Reduction
smoothing = 12  # For mouse movement smoothing
########################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)  # Set width
cap.set(4, hCam)  # Set height
pTime = 0
ploxx, ploy = 0, 0  # Previous mouse position
clocx,clocy = 0, 0  # Current mouse position
Detector=htm.HandDetector(maxHands=1)
wScr, hScr = autopy.screen.size()  # Get screen size

while True:
    ret,frame=cap.read()
    if not ret:
        print("failed to grab frame")
        break
    #frame = cv2.flip(frame, 1)
    
    frame=Detector.findHands(frame)
    lmlist,bbox=Detector.findPosition(frame)
    
    if lmlist and len(lmlist) > 12:
        x1, y1 = lmlist[8][1], lmlist[8][2]
        x2, y2 = lmlist[12][1], lmlist[12][2]
        #print(x1, y1, x2, y2)
        # Uncomment and implement Detector.fingersUp() if needed
        fingers = Detector.fingersUp()
        #print(fingers)
    
        cv2.rectangle(frame,(frameR,frameR),(wCam-frameR, hCam-frameR), (255, 0, 255), 2)

        if fingers[1] == 1 and fingers[2] == 0:  
            x3=np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3=np.interp(y1, (frameR, hCam-frameR), (0, hScr))
            autopy.mouse.move(wScr-x3, y3)

            clocx = ploxx + (x3 - ploxx) / smoothing
            clocy = ploy + (y3 - ploy) / smoothing
            
        if fingers[1] == 1 and fingers[2] == 1: 
            length, frame, lineInfo = Detector.findDistance(8, 12, frame)
            #print(length)
            if length < 40:
                cv2.circle(frame, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()
                

    cTime = time.time()
    fps= 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()