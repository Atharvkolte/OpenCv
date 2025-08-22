import cv2
import mediapipe as mp
import polars as pl
import pandas as pd # Import pandas for conversion
import joblib
import time

# Initialize mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Load the trained Decision Tree model
try:
    dt_classifier = joblib.load("hand_finger_dt_model.pkl")
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: The model file 'hand_finger_dt_model.pkl' was not found. Please train the model first.")
    exit()

# Finger landmark mapping
finger_mapping = {
    "Wrist": [0],
    "Thumb": [1, 2, 3, 4],
    "Index": [5, 6, 7, 8],
    "Middle": [9, 10, 11, 12],
    "Ring": [13, 14, 15, 16],
    "Pinky": [17, 18, 19, 20]
}

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

# ... (rest of the code remains the same)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, c = frame.shape
            
            input_data = {}
            for finger, indices in finger_mapping.items():
                for i, idx in enumerate(indices):
                    lm = hand_landmarks.landmark[idx]
                    
                    # 🔹 Corrected: Use raw pixel coordinates (like the training data)
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    input_data[f"{finger.lower()}{i+1}_x"] = cx
                    input_data[f"{finger.lower()}{i+1}_y"] = cy

            pl_df = pl.DataFrame([input_data])
            pd_df = pl_df.to_pandas()
            
            # Make the prediction
            predicted_label = dt_classifier.predict(pd_df)[0]

            # Draw landmarks and the predicted label on the frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            cv2.putText(frame, 
                        f"Prediction: {predicted_label}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA)
            
    cv2.imshow("Hand Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()