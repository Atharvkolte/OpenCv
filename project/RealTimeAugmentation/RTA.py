import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk,Scale

class VideoAugmentationApp:
    def __init__(self, Window):
        self.window = Window
        self.window.title("Video Augmentation Tool")
        self.cap=cv2.VideoCapture(0)
        self.aug_type=tk.StringVar(value="None")
        
        self.create_widgets()
        self.update()
        
    def create_widgets(self):
        self.video_label=ttk.Label(self.window)
        self.video_label.grid(row=0, column=0,columnspan=4)

        ttk.Label(self.window, text="Augmentation Type:").grid(row=1, column=0,padx=5, pady=5)

        self.aug_selection=ttk.Combobox(self.window, textvariable=self.aug_type, values=["None", "grayscale","sepia","sketch"],state="readonly")
        self.aug_selection.grid(row=1, column=1,padx=5, pady=5)
        
        self.brightness_scale = Scale(self.window, from_=0, to=100, orient=tk.HORIZONTAL, label="Brightness")
        self.brightness_scale.set(50)
        self.brightness_scale.grid(row=1, column=2, padx=5, pady=5)
        
        self.quit_button=ttk.Button(self.window, text="Quit", command=self.quit_app)
        self.quit_button.grid(row=1, column=3, padx=5, pady=5)
        
    def update(self):
        ret,frame=self.cap.read()
        if ret:
            frame=self.apply_augmentation(frame)
            frame=self.adjust_brightness(frame)
            self.display_frame(frame)
        self.window.after(10, self.update)
        
    def apply_augmentation(self, frame):
        aug_type=self.aug_type.get()
        if aug_type == "grayscale":
            frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame=cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif aug_type == "sepia":
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                      [0.349, 0.686, 0.168],
                                      [0.393, 0.769, 0.189]])
            frame= np.clip(frame, 0, 255)
        elif aug_type == "sketch":
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            inv_img = cv2.bitwise_not(gray_img)
            blur_img = cv2.GaussianBlur(inv_img, (21, 21), 0)   
            sketch_img = cv2.divide(gray_img, 255 - blur_img, scale=256)
            frame = cv2.cvtColor(sketch_img, cv2.COLOR_GRAY2BGR)
                    
        return frame
    
    def adjust_brightness(self, frame): 
        brightness = self.brightness_scale.get()
        frame = cv2.convertScaleAbs(frame, alpha=brightness/50)
        return frame
    
    def display_frame(self, frame):
        img=cv2.resize(frame, (640, 480))
        imgtk=cv2.imencode(".png", img)[1]
        imgtk = tk.PhotoImage(data=imgtk.tobytes())
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
    def quit_app(self):
        self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAugmentationApp(root)
    root.mainloop()
