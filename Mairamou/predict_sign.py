import tkinter as tk
from tkinter import filedialog, Label, Button
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
import cv2

# Load the trained model
MODEL_PATH = "best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (30, 30))  # Resize to match model input
    image = image.astype("float32") / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to classify the selected image
def classify_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        image = preprocess_image(file_path)
        prediction = model.predict(image)
        class_id = np.argmax(prediction)  # Get class with highest probability
        label.config(text=f"Predicted Class: {class_id}")
        
        # Display selected image
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

# Create the GUI
root = tk.Tk()
root.title("Traffic Sign Recognition")
root.geometry("400x400")

# Widgets
label = Label(root, text="Select an image for classification", font=("Arial", 12))
label.pack(pady=10)

image_label = Label(root)
image_label.pack()

button = Button(root, text="Select Image", command=classify_image)
button.pack(pady=10)

# Run the GUI
root.mainloop()
