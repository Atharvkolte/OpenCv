import cv2
import mediapipe as mp
import polars as pl
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

# Finger landmark mapping
finger_mapping = {
    "Wrist": [0],
    "Thumb": [1, 2, 3, 4],
    "Index": [5, 6, 7, 8],
    "Middle": [9, 10, 11, 12],
    "Ring": [13, 14, 15, 16],
    "Pinky": [17, 18, 19, 20]
}

# Define dataset columns
columns = ["timestamp", "hand_id", "frame_no"]
for finger, indices in finger_mapping.items():
    for i, idx in enumerate(indices):
        columns.extend([f"{finger.lower()}{i+1}_x", f"{finger.lower()}{i+1}_y"])

# Store rows here
data_rows = []

cap = cv2.VideoCapture(0)
frame_no = 0
start_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame_no += 1
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
            h, w, c = frame.shape
            row = {
                "timestamp": round(time.time() - start_time, 2),
                "hand_id": hand_id,
                "frame_no": frame_no
            }

            # Collect landmark points
            for finger, indices in finger_mapping.items():
                for i, idx in enumerate(indices):
                    lm = hand_landmarks.landmark[idx]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    row[f"{finger.lower()}{i+1}_x"] = cx
                    row[f"{finger.lower()}{i+1}_y"] = cy

            data_rows.append(row)

            # 🔹 Draw landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

    # Show video with landmarks
    cv2.imshow("Hand Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

# Convert collected rows into Polars DataFrame
df = pl.DataFrame(data_rows)

# Save dataset
df.write_csv("hand_landmarks.csv")
print("Dataset saved as hand_landmarks.csv")
print(df.head())
