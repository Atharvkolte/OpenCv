import cv2
import mediapipe as mp
import time

mpDraw=mp.solutions.drawing_utils
mpPose=mp.solutions.pose
pose=mpPose.Pose()

cap=cv2.VideoCapture(1)
pTime=0

while True:
    ret,frame=cap.read()
    
    if not ret:
        print("Cannot caputre")
        break
    
    imgRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=pose.process(imgRGB)
    print(result.pose_landmarks)
    if result.pose_landmarks:
        mpDraw.draw_landmarks(frame,result.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id,lm in enumerate(result.pose_landmarks.landmark):
            h,w,c=frame.shape
            print(id,lm)
            cx,cy=int(lm.x*w),int(lm.y*h)
            cv2.circle(frame,(cx,cy),5,(225,0,0),cv2.FILLED)
    
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    
    cv2.putText(frame,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    
    cv2.imshow("Atharva",frame)
    
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()