import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_w, screen_h = pyautogui.size()

# Start webcam
cap = cv2.VideoCapture(0)

# Helper function: get hand positions
def get_position(landmark, shape):
    h, w = shape
    return int(landmark.x * w), int(landmark.y * h)

# Finger tip landmark indices
finger_tips = [8, 12, 16, 20]  # index, middle, ring, pinky
finger_dips = [6, 10, 14, 18]  # their DIP joints

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        # Draw hand
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get landmarks
        lm = hand_landmarks.landmark
        index_tip = lm[8]
        thumb_tip = lm[4]
        middle_tip = lm[12]

        # Get mouse position
        index_x, index_y = get_position(index_tip, (h, w))
        screen_x = np.interp(index_x, [0, w], [0, screen_w])
        screen_y = np.interp(index_y, [0, h], [0, screen_h])
        pyautogui.moveTo(screen_x, screen_y)

        # --- Gesture Detection ---
        def distance(a, b):
            return np.hypot(a.x - b.x, a.y - b.y)

        # 1. Left Click (index + thumb touch)
        if distance(index_tip, thumb_tip) < 0.03:
            pyautogui.click()
            time.sleep(0.3)

        # 2. Right Click (index + middle touch)
        elif distance(index_tip, middle_tip) < 0.03:
            pyautogui.rightClick()
            time.sleep(0.3)

        # 3. Finger state detection (check if finger is up)
        finger_up = []
        for tip, dip in zip(finger_tips, finger_dips):
            finger_up.append(lm[tip].y < lm[dip].y)

        # finger_up = [Index, Middle, Ring, Pinky]
        if finger_up[0] and not any(finger_up[1:]):  # Only index up
            pass  # Just move mouse
        elif finger_up[0] and finger_up[3] and not finger_up[1] and not finger_up[2]:
            pyautogui.scroll(100)  # Scroll up
            time.sleep(0.3)
        elif finger_up[0] and finger_up[2] and not finger_up[1] and not finger_up[3]:
            pyautogui.scroll(-100)  # Scroll down
            time.sleep(0.3)

    cv2.imshow("Hand Mouse Control", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
