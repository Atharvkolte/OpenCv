import cv2
import numpy as np

path="C:\\DS\\Atharva\\Face senti analysis\\learning process\\OpenCV\\Images\\AK.jpg"

image = cv2.imread(path)
if image is None:
    print("Error: Image not found.")
image=cv2.medianBlur(image, 3)
image=cv2.GaussianBlur(image, (5,5), 0)
kernel = np.array([
            [0, -1, 0], 
            [-1, 5, -1], 
            [0, -1, 0]
            ])
        
image = cv2.filter2D(image, -1, kernel)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh=cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
cv2.imshow("Contours", image)
cv2.waitKey(0)
