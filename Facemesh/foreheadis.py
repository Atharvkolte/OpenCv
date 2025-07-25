import cv2
import time
import mediapipe as mp
import math

def lm_to_pixel(lm, w, h):
    return int(lm.x * w), int(lm.y * h), lm.z  # z stays normalized

def pixel_distance(p1, p2):
    (x1, y1), (x2, y2) = p1[:2], p2[:2]
    return math.hypot(x2 - x1, y2 - y1)

def distance_3d(p1, p2, w, h):
    # Scale x/y to pixels, keep z in the same normalized space as x/y (MediaPipe docs say
    # z is roughly in the same scale as x)
    x1, y1, z1 = p1[0], p1[1], p1[2]
    x2, y2, z2 = p2[0], p2[1], p2[2]
    dx = (x2 - x1)
    dy = (y2 - y1)
    dz = (z2 - z1)
    # If you want everything in "pixel units", multiply dx,dy by 1 (already pixels)
    # and dz by image width (common hack). Otherwise just return the normalized 3D distance.
    return math.sqrt(dx*dx + dy*dy + (dz * w)*(dz * w))


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

    if result.multi_face_landmarks:
        h, w, _ = frame.shape
        pts = result.multi_face_landmarks[0].landmark

        i1, i2 = 10, 338  # choose the two landmarks you want
        p1 = lm_to_pixel(pts[i1], w, h)
        p2 = lm_to_pixel(pts[i2], w, h)

        dist_px = pixel_distance(p1, p2)
        dist_3d_norm = distance_3d(p1, p2, w, h)

        cv2.putText(frame, f"Forehead dist: {dist_px:.1f}px",(20, 110), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    
    cv2.putText(frame,f"Fps:{int(fps)}",(20,70),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
    cv2.imshow("My Webcam",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()