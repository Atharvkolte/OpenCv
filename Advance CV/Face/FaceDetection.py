import cv2
import time
import mediapipe as mp


cap=cv2.VideoCapture(0)
pTime=0
mpFaceDetection=mp.solutions.face_detection
mpDraw=mp.solutions.drawing_utils
faceDetection=mpFaceDetection.FaceDetection()

while True:
    ret,frame=cap.read()
    if not ret:
        print("Doesn't working")
        break
    
    
    imgRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    
    result=faceDetection.process(imgRGB)
    #print(result)
    
    if result.detections:
        for id,detection in enumerate(result.detections):
            #mpDraw.draw_detection(frame,detection)
            # print(id,detection)
            # print(detection.score)
            #print(detection.location_data.relative_bounding_box)
            bboxc=detection.location_data.relative_bounding_box
            ih,iw,ic=frame.shape
            bbox=int(bboxc.xmin*iw),int(bboxc.ymin*ih),\
                 int(bboxc.width*iw),int(bboxc.height*ih)
                 
            cv2.rectangle(frame,bbox,(255,0,0),2)
            
    
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    
    #frame=cv2.flip(frame,1)
    cv2.putText(frame,f"Fps:{int(fps)}",(20,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
    cv2.putText(frame,f"Score:{int(detection.score[0]*100)}%",(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,1,(255,255,0),2)
    cv2.imshow("Webcam",frame)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
    
cap.release
cv2.destroyAllWindows()