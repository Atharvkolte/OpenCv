import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0,255,0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        #gray[150:150+80, 100:100+80]
        roi_color = frame[y:y+h, x:x+w]
    """
    x = 100
    y = 150
    w = 80
    h = 80

    (100,150)
    w = 80 > 180
    h = 80 > 230
    """

    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
    if len(eyes) > 0:
        cv2.putText(frame, "Eyes Detected", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 20)
    if len(smiles) > 0:
        cv2.putText(frame, "Smiling", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    
    cv2.imshow("Smart Face Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    