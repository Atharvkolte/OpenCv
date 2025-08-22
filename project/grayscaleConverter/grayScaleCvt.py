import cv2
import numpy as np
from tkinter import filedialog, messagebox, Tk, Label, Button
from tkinter import N,S, E, W

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        convert_to_grayscale(file_path)

def convert_to_grayscale(file_path):
    img = cv2.imread(file_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    display_image(gray_img)

def display_image(img):
    cv2.imshow("Grayscale Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def setup_ui(root):
    root.title("Grayscale Converter")
    root.geometry("300x200")
    root.resizable(False, False)
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)

    label = Label(root, text="Grayscale Converter", font=("Arial", 16))
    label.grid(row=0, column=0, padx=10, pady=10,sticky=N)
    
    open_button = Button(root, text="Open Image", command=open_file)
    open_button.grid(row=1, column=0, padx=10, pady=10, sticky=S+E+W)

if __name__=="__main__":
    root = Tk()
    setup_ui(root)
    root.mainloop()