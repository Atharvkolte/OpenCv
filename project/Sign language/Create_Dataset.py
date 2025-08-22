import os
#import mediapipe as mp
#import cv2
import matplotlib.pyplot as plt

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

class HandSign:
    def __init__(self, data_dir="./dataset"):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.data_dir = data_dir

    def getInfo(self):
        self.hand_signs = input("Enter the hand signs you want to capture: ")
        self.hand_signs = self.hand_signs.upper()
        if os.path.exists(os.path.join(self.data_dir, self.hand_signs)):
            print(f"Dataset for {self.hand_signs} already exists.")
        else:
            os.makedirs(os.path.join(self.data_dir, self.hand_signs))
    
    def captureHandSign(self):
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        while True:
            ret,frame=cap.read()
            if not ret:
                break
            
            frame=cv2.flip(frame,1)
            self.detectHandLandmarks(frame)
            cv2.imshow("Hand Sign Capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        
    def detectHandLandmarks(self,frame):
        # Initialize mediapipe hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        mp_drawing = mp.solutions.drawing_utils

        # Finger landmark mapping
        finger_mapping = {
            "Wrist": [0],
            "Thumb": [1, 2, 3, 4],
            "Index": [5, 6, 7, 8],
            "Middle": [9, 10, 11, 12],
            "Ring": [13, 14, 15, 16],
            "Pinky": [17, 18, 19, 20]
        }
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, c = frame.shape

                # Grouped landmark coordinates
                grouped_landmarks = {}

                for finger, indices in finger_mapping.items():
                    grouped_landmarks[finger] = []
                    for idx in indices:
                        lm = hand_landmarks.landmark[idx]
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        grouped_landmarks[finger].append((idx, cx, cy))

                # Print nicely
                print("\n=== Hand Landmarks ===")
                for finger, coords in grouped_landmarks.items():
                    print(f"{finger}: {coords}")

                # Optional: draw on frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
def main():
    HS = HandSign()
    HS.getInfo()
    #HS.captureHandSign()

if __name__ == "__main__":
    main()