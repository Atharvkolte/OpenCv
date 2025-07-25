import cv2
import time
import mediapipe as mp

cap=cv2.VideoCapture(0)
pTime=0

mpDraw=mp.solutions.drawing_utils
mpFaceMesh=mp.solutions.face_mesh
faceMesh=mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec=mpDraw.DrawingSpec(thickness=1,circle_radius=2)

while True:
    ret,frame=cap.read()
    if not ret:
        print("somethings Wrong")
        break
    
    imgRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=faceMesh.process(imgRGB)  
    if result.multi_face_landmarks:
        for facelms in result.multi_face_landmarks:
            #mpDraw.draw_landmarks(frame, facelms, mpFaceMesh.FACEMESH_TESSELATION)
            mpDraw.draw_landmarks(frame, facelms, mpFaceMesh.FACEMESH_CONTOURS,drawSpec,drawSpec)

    
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    
    cv2.putText(frame,f"Fps:{int(fps)}",(20,70),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
    cv2.imshow("My Webcam",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()