from email.mime import image
import numpy as np
import cv2

### Yaha height and width nahi hai yaha width and height hai
### Image ka shape height, width and channels me hota hai

class ClassOpencv:
    #Photo operations
    def showimg(self,path):
        image=cv2.imread(path)

        if image is None:
            print("Error: Image not found.")
        else:
            print(image.shape)
            cv2.imshow("AK", image) # here "AK" is the window name and image is the image to be displayed
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    def saveimg(self,path):
        image=cv2.imread(path)
        if image is not None:
            success = cv2.imwrite("output_python.jpg", image) # Save the image
            if success:
                print("Image saved successfully as 'output_python.jpg'")
            else:
                print("Failed to save an image")
        else:
            print("Error: Could not load image")
    def shape(self,path):
        image=cv2.imread(path)
        if image is not None:
            h, w, c = image.shape
            print(f"Image shape:\nHeight: {h}\nWidth: {w}\nChannels: {c}")
        else:
            print("Error: Could not load image")
    def toGray(self,path):
        image=cv2.imread(path)
        if image is not None:
            grayImg=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            grayImg = grayImg.astype('float64') / 255 # Normalize the image to [0, 1] range
            cv2.imshow("Gray Image", grayImg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: Could not load image")
    def rezied(self,path):
        image=cv2.imread(path)
        if image is not None:
            resized_image = cv2.resize(image, (500, 500))#(500, 500) widht and height
            cv2.imshow("Resized Image", resized_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: Could not load image")
    def crop(self,path):
        image=cv2.imread(path)
        if image is not None:
            cropped_image = image[50:200, 50:200]
            cv2.imshow("Cropped Image", cropped_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: Could not load image")
    def rotate(self, path):
        image = cv2.imread(path)
        if image is not None:
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            angle = 45  # Rotate by 45 degrees
            scale = 1.0  # No scaling
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
            rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
            cv2.imshow("Rotated Image", rotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: Could not load image")
    def flip(self, path):
        image = cv2.imread(path)
        if image is not None:
            flipped_image = cv2.flip(image, 1)# 1 for horizontal flip, 0 for vertical flip, -1 for both
            cv2.imshow("Flipped Image", flipped_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: Could not load image")
    # DRAWING SHAPES
    def drawLine(self, path):
        image = cv2.imread(path)
        if image is not None:
            pt1= (50, 100)
            pt2= (500, 600)
            # Draw a line from pt1 to pt2 with color (255, 0, 0) (blue in BGR format) and thickness 5
            cv2.line(image, pt1, pt2, (255, 0, 0), 5)
            cv2.imshow("Line", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: Could not load image")
    def drawRectangle(self, path):
        image = cv2.imread(path)
        if image is not None:
            pt1 = (50, 100)
            pt2 = (500, 600)
            # Draw a rectangle with color (0, 255, 0) (green in BGR format) and thickness 5
            cv2.rectangle(image, pt1, pt2, (0, 255, 0), 5)  # -1 means filled rectangle
            cv2.imshow("Rectangle", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: Could not load image")
    def drawCircle(self, path):
        image = cv2.imread(path)
        if image is not None:
            center = (410, 250)
            radius = 150
            # Draw a circle with color (0, 0, 255) (red in BGR format) and thickness 5
            cv2.circle(image, center, radius, (0, 0, 255), 5)#-1 means filled circle
            cv2.imshow("Circle", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: Could not load image")
    def drawPolygon(self, path): #Not use mostly by ai given
        image = cv2.imread(path)
        if image is not None:
            pts = [[100, 50], [200, 300], [400, 200], [500, 100]]
            pts = np.array(pts, np.int32)
            pts = pts.reshape((-1, 1, 2))
            # Draw a polygon with color (255, 0, 255) (purple in BGR format) and thickness 5
            cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 255), thickness=5)
            cv2.imshow("Polygon", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: Could not load image")
    def addText(self, path):
        image = cv2.imread(path)
        if image is not None:
            text = "Hello, OpenCV!"
            position = (50, 50)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (0, 255, 255)
            
            cv2.putText(image, text, position, font, font_scale, color, 2)
            cv2.imshow("Text", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: Could not load image")
    '''output = image.copy()
    cv2.line(output, (0,0), (300,300), (0,255,0), 3)
    cv2.rectangle(output, (50,50), (200,200), (255,0,0), 2)
    cv2.circle(output, (150,150), 50, (0,0,255), -1)
    cv2.putText(output, 'OpenCV', (10,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow('Drawing', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
    #video operations 
    def openCam(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret,frame=cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                print("failed to grab frame")
                break
            cv2.putText(frame, "Atharva", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xff==ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    def saveVideo(self):
        camera=cv2.VideoCapture(0)
        frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        recoder=cv2.VideoWriter("output.avi", codec, 20.0, (frame_width, frame_height))
        while True:
            success, image = camera.read()
            
            if not success:
                print("Failed to grab frame")
                break
            recoder.write(image)
            cv2.imshow("Recording Live", image)
            if cv2.waitKey(1) & 0xff==ord('q'):
                break

        camera.release()
        recoder.release()
        cv2.destroyAllWindows()
    
    #Image filtering processing 
    def blur(self, path):
        image = cv2.imread(path)
        if image is None:
            print("Error: Image not found.")
            return
        gaussian = cv2.GaussianBlur(image, (3, 3), 0)
        median = cv2.medianBlur(image, 3)

        cv2.imshow('Gaussian Blur', gaussian)
        cv2.imshow('Median Blur', median)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def sharpen(self, path):
        image = cv2.imread(path)
        if image is None:
            print("Error: Image not found.")
            return
        
        kernel = np.array([
            [0, -1, 0], 
            [-1, 5, -1], 
            [0, -1, 0]
            ])
        
        sharpened = cv2.filter2D(image, -1, kernel)

        cv2.imshow('Sharpened Image', sharpened)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def cannyEdge(self, path):
        image = cv2.imread(path)
        if image is None:
            print("Error: Image not found.")
            return
        edges = cv2.Canny(image, 50, 150)

        cv2.imshow('Canny Edges', edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def thresFunc(self, path):
        image=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        ret,thresh=cv2.threshold(image,120,255,cv2.THRESH_BINARY)
        cv2.imshow('Thresholded Image', thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def bitWiseOp(self, path):
        img1 = np.zeros((300, 300), dtype="uint8")
        img2 = np.zeros((300, 300), dtype="uint8")

        cv2.circle(img1, (150, 150), 100, 255, -1)
        cv2.rectangle(img2, (100, 100), (250, 250), 255, -1)

        bitwise_and = cv2.bitwise_and(img1, img2)
        bitwise_or = cv2.bitwise_or(img1, img2)
        bitwise_not = cv2.bitwise_not(img1)

        cv2.imshow("Circle", img1)
        cv2.imshow("Rectangle", img2)
        cv2.imshow("AND", bitwise_and)
        cv2.imshow("OR", bitwise_or)
        cv2.imshow("NOT", bitwise_not)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    def contourFunc(self,path):
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh=cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
        ###For shape detection
        """
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

            corners = len(approx)

            if corners == 3:
                shape_name = "Triangle"
                
            elif corners == 4:
                shape_name = "Rectangle"
            
            elif corners == 5:
                shape_name = "Pentagon"
            
            elif corners > 5:
                shape_name = "Circle"
            
            else:
                shape_name = "unknown"

            cv2.drawContours(img, [approx], 0, (0,255,0), 2)
            x = approx.ravel()[0]
            
            # [
            # [[100,200]],
            # [[150,250]],
            # [[120,270]],
            # ]
            # [100,200,150,250,120,270]
            
            y = approx.ravel()[1] - 10
            cv2.putText(img, shape_name, (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 2)

        """
        cv2.imshow("Contours", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #Face detection
    def detectFace(slef):
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        while True:
            ret,frame=cap.read()
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces=face_cascade.detectMultiScale(gray, 1.1, 5)
            
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            
            if not ret:
                print("failed to grab frame")
                break
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xff==ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows() 
    
    def smileEyeFace(self):
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
        while True:
            ret,frame=cap.read()
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces=face_cascade.detectMultiScale(gray, 1.1, 5)
            
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            
            roi_gray=gray[y:y+h,x:x+w]
            roi_color=frame[y:y+h,x:x+w]
            
            eye=eye_cascade.detectMultiScale(roi_gray,1.1,10)
            if len(eye)>0:
                cv2.putText(frame,"Eye Detected",(x,y-30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(225,0,0),2)
                
            smile=smile_cascade.detectMultiScale(roi_gray,1.1,10)
            if len(eye)>0:
                cv2.putText(frame,"Eye Detected",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(225,0,0),2)
                
                
            if not ret:
                print("failed to grab frame")
                break
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xff==ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows() 
        
obj1= ClassOpencv()
path="C:\\DS\\Atharva\\Face senti analysis\\learning process\\OpenCV\\Images\\AK.jpg"
### Image Operations
#obj1.showimg(path)
#obj1.saveimg(path)
#obj1.shape(path)
#obj1.toGray(path)
#obj1.rezied(path)
#obj1.crop(path)
#obj1.rotate(path)
#obj1.flip(path)

### Drawing Shapes
#obj1.drawLine(path)
#obj1.drawRectangle(path)
#obj1.drawCircle(path)
#obj1.drawPolygon(path)
#obj1.addText(path)

### Video Operations
#obj1.openCam()
#obj1.saveVideo()

### Image Filtering Processing
#obj1.blur(path)
#obj1.sharpen(path)
#obj1.cannyEdge(path)
#obj1.thresFunc(path)
#obj1.bitWiseOp(path)
#obj1.contourFunc(path)

###Face Detection
#obj1.detectFace()
obj1.smileEyeFace()