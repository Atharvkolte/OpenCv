import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

class LiveGreyScaleFilter:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Grayscale Filter")
        self.run_filter=False
        self.setup_ui()
        self.cap = cv2.VideoCapture(0)
        
    def setup_ui(self):
        self.start_button=ttk.Button(self.root, text="Start Camera", command=self.start_filter)
        self.start_button.pack(pady=10)
        
        self.stop_button=ttk.Button(self.root, text="Stop Camera", command=self.stop_filter)
        self.stop_button.pack(pady=10)
        
    def start_filter(self):
        self.run_filter = True
        self.process_frames()
       
    def stop_filter(self):
        self.run_filter = False
        cv2.destroyAllWindows()
    
    def process_frames(self):
        if not self.run_filter:
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Grayscale Live Feed", gray_frame)
        cv2.waitKey(20)
        
        self.root.after(10, self.process_frames)
        
    def on_closing(self):
        self.run_filter = False
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()
        
if __name__ == "__main__":
    root=tk.Tk()
    app=LiveGreyScaleFilter(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()