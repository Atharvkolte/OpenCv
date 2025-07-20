import cv2
import time
import numpy as np
import handDetectorMod as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
#####################
wCam,hCam=1280,720
#####################


cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime=0

detector=htm.HandDetector()

"""device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
#print(f"- Muted: {bool(volume.GetMute())}")
#print(f"Audio output: {device.FriendlyName}")
#print(f"- Volume level: {volume.GetMasterVolumeLevel()} dB")
#print(f"- Volume range: {volume.GetVolumeRange()[0]} dB - {volume.GetVolumeRange()[1]} dB")
volRange=volume.GetVolumeRange()
volume.SetMasterVolumeLevel(0, None)
minvol=volRange[0]
maxvol=volRange[1]
"""

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

while True:
    ret,frame=cap.read()
    if not ret:
        print("Sonething is wrong")
    
    frame=detector.findHands(frame)
    lmlist=detector.findPosition(frame,draw=False)
    
    if(len(lmlist)!=0):
        # print(lmlist[4],lmlist[8])
        x1,y1=lmlist[4][1],lmlist[4][2]
        x2,y2=lmlist[8][1],lmlist[8][2]
        
        cx,cy=(x1+x2)//2,(y1+y2)//2
        
        cv2.circle(frame,(x1,y1),15,(255,0,255),cv2.FILLED)
        cv2.circle(frame,(x2,y2),15,(255,0,255),cv2.FILLED)
        cv2.line(frame,(x1,y1),(x2,y2),(255,0,255),3)
        cv2.circle(frame,(cx,cy),15,(255,0,255),cv2.FILLED)
        
        length=math.hypot(x2-x1,y2-y1)
        #print(length)

        # hand range=50,300
        #volume Range -65,0
        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])
        print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)
        
        if(length<=50):
            cv2.circle(frame,(cx,cy),15,(0,255,0),cv2.FILLED)
        cv2.rectangle(frame, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(frame, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(frame, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,1, (255, 0, 0), 3)    
        
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    
    cv2.putText(frame,f"Fps:{int(fps)}",(20,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
    cv2.imshow("WebCam",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()