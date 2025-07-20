import cv2
import mediapipe as mp
import time

# cap=cv2.VideoCapture(0)
# mpHands = mp.solutions.hands
# hands = mpHands.Hands() 
# mpDraw=mp.solutions.drawing_utils


# pTime=0
# cTime=0

# while True:
#     ret,frame=cap.read()
#     frame=cv2.flip(frame,1)
#     imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results=hands.process(imgRGB)

#     if not ret:
#         print("failed to grab frame")
#         break
        
#     #print(results.multi_hand_landmarks)
#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#             for id, lm in enumerate(handLms.landmark):
#                 #print(id,lm)
#                 h,w,c=frame.shape
#                 cx,cy=int(lm.x*w),int(lm.y*h)
#                 print(id,cx,cy)
#                 if id==0:
#                     cv2.circle(frame,(cx,cy),10,(255,0,0),cv2.FILLED)
#             mpDraw.draw_landmarks(frame,handLms,mpHands.HAND_CONNECTIONS)
    
#     cTime=time.time()
#     fps=1/(cTime-pTime)
#     pTime=cTime
    
#     cv2.putText(frame,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    
    
#     cv2.imshow("Webcam", frame)
#     if cv2.waitKey(1) & 0xff==ord('q'):
#         break
    
    
    
# cap.release()
# cv2.destroyAllWindows()


class HandDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = min_detection_confidence
        self.trackCon = min_tracking_confidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=self.modelComplexity,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    
    def findHands(self,frame,draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)
            
        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                   if draw: 
                        self.mpDraw.draw_landmarks(frame,handLms,self.mpHands.HAND_CONNECTIONS)
        return frame
    
    def findPosition(self,frame,handNo=0,draw=True):
        lmlist=[]
        
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
            #print(id,lm)
                h,w,c=frame.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                #print(id,cx,cy)
                lmlist.append([id,cx,cy])
                if id==0:
                    cv2.circle(frame,(cx,cy),10,(255,0,0),cv2.FILLED)
        return lmlist
            

def main():
    pTime=0
    cTime=0
    cap=cv2.VideoCapture(0)
    detector=HandDetector()
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
    
    
if __name__=="__main__":
    main()