import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

def rotate_document(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_not(binary)

    coord = np.column_stack(np.where(binary > 0))
    if len(coord) == 0:
        return image
    
    rect = cv2.minAreaRect(coord)
    angle = rect[2]
    if angle < -45:
        angle = -angle
    else:
        angle = 90 - angle
        
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def load_image():
    file_path = filedialog.askopenfilename(parent=app)
    if not file_path:
        return None
    image = cv2.imread(file_path)
    aligned_image = rotate_document(image)
    if aligned_image is not None:
        display_image(aligned_image)
    else:
        messagebox.showerror("Error", "Could not align the document.")

def display_image(image):
    cv_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv_img) 
    img = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor=tk.NW, image=img)
    canvas.image = img

app = tk.Tk()   
app.title("Document Alignment Tool")
app.geometry("800x600")

# Canvas goes on TOP
canvas = tk.Canvas(app, bg="white")
canvas.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

# Buttons stay at the BOTTOM
button_frame = tk.Frame(app)
button_frame.pack(side=tk.BOTTOM, fill=tk.X)

load_button = tk.Button(button_frame, text="Load Image", command=load_image)
load_button.pack(side=tk.LEFT, padx=10, pady=10)

app.mainloop()
