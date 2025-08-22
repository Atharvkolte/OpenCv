import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

# Dictoionary to store PIL img
images={"original":None, "sketch":None}


def open_file():
    filepath=filedialog.askopenfilename()
    if not filepath:
        return
    img=cv2.imread(filepath)
    display_image(img, original=True)
    sketch_img=convert_to_sketch(img)
    display_image(sketch_img, original=False)
    
def convert_to_sketch(img):
    gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_img=cv2.bitwise_not(gray_img)
    blurred_img=cv2.GaussianBlur(inverted_img, (21, 21), sigmaX=0,sigmaY=0)
    inverted_blur_img=cv2.bitwise_not(blurred_img)
    sketch_img=cv2.divide(gray_img, inverted_blur_img, scale=256.0)
    return sketch_img

def display_image(img, original=True):
    img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if original else img
    img_pil=Image.fromarray(img_rgb)
    img_tk=ImageTk.PhotoImage(img_pil)

    #store the PIL image in the dictionary
    if original:
        images["original"] = img_pil
    else:
        images["sketch"] = img_pil
        
    label=original_image_label if original else sketch_image_label
    label.config(image=img_tk)
    label.image=img_tk
    
    
def save_sketch():
    if images["sketch"] is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                   filetypes=[("PNG files", "*.png"),
                                                              ("JPEG files", "*.jpg"),
                                                              ("All files", "*.*")])
        if file_path:
            images["sketch"].save(file_path)
            messagebox.showinfo("Success", "Sketch image saved to {}".format(file_path))
        else:
            return
    else:
        messagebox.showwarning("Warning", "No sketch image to save.")   
        return
    
app=tk.Tk()
app.title("Image to Sketch Converter")
frame=tk.Frame(app)
frame.pack(padx=10, pady=10)  

original_image_label=tk.Label(frame, text="Original Image")
original_image_label.grid(row=0, column=0,padx=5, pady=5)
sketch_image_label=tk.Label(frame, text="Sketch Image")
sketch_image_label.grid(row=0, column=1,padx=5, pady=5)

btn_frame=tk.Frame(app)
btn_frame.pack(pady=10)

open_button=tk.Button(btn_frame, text="Open Image", command=open_file)
open_button.grid(row=0, column=0, padx=5)

save_button=tk.Button(btn_frame, text="Save Sketch", command=save_sketch)
save_button.grid(row=0, column=1, padx=5)

app.mainloop()