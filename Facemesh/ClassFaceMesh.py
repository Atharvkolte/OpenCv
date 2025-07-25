import cv2
import mediapipe as mp
import time
import math

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, refine_landmarks=False):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refine_landmarks = refine_landmarks

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                        self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y, lm.z])  # Now including z coordinate
                faces.append(face)
        return img, faces, self.results.multi_face_landmarks

    @staticmethod
    def lm_to_pixel(lm, w, h):
        return int(lm.x * w), int(lm.y * h), lm.z

    @staticmethod
    def pixel_distance(p1, p2):
        (x1, y1), (x2, y2) = p1[:2], p2[:2]
        return math.hypot(x2 - x1, y2 - y1)

    @staticmethod
    def distance_3d(p1, p2, w, h):
        x1, y1, z1 = p1[0], p1[1], p1[2]
        x2, y2, z2 = p2[0], p2[1], p2[2]
        dx = (x2 - x1)
        dy = (y2 - y1)
        dz = (z2 - z1)
        return math.sqrt(dx*dx + dy*dy + (dz * w)*(dz * w))

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector(maxFaces=2)
    
    # Landmark indices for measurement (example: forehead width)
    landmark_indices = [10, 338]  # Left and right forehead points
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        img, faces, multi_face_landmarks = detector.findFaceMesh(img)

        if multi_face_landmarks:
            h, w, _ = img.shape
            pts = multi_face_landmarks[0].landmark

            # Get the specified landmarks
            p1 = detector.lm_to_pixel(pts[landmark_indices[0]], w, h)
            p2 = detector.lm_to_pixel(pts[landmark_indices[1]], w, h)

            # Calculate distances
            dist_px = detector.pixel_distance(p1, p2)
            dist_3d_norm = detector.distance_3d(p1, p2, w, h)

            # Display measurements
            cv2.putText(img, f"Forehead dist: {dist_px:.1f}px", (20, 110), 
                       cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.putText(img, f"3D dist: {dist_3d_norm:.1f}", (20, 140), 
                       cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (20, 70), 
                   cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.imshow("Face Measurement", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()