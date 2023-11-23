import cv2
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import model_from_json

# Load the emotion detection model and weights
emotion_model = model_from_json(open("model_a1.json", "r").read())
emotion_model.load_weights("model_weights1.h5")

# Define the list of emotions
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Open a connection to the camera (you can change the index if you have multiple cameras)
cap = cv2.VideoCapture(0)

# Create a GUI window
root = tk.Tk()
root.title("Emotion Detection")

# Create a label to display the detected emotion
emotion_label = Label(root, font=("Helvetica", 16))
emotion_label.pack(pady=20)

# Create a canvas to display the video stream, set its size, and center it
canvas_width = 800
canvas_height = 600
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
canvas.pack()

# Function to resize and update the emotion label and video stream in real-time
def update_emotion_label_and_video():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (48, 48))  # Resize the frame to match the model's input shape
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    emotions_prob = emotion_model.predict(frame_gray[np.newaxis, :, :, np.newaxis])
    predicted_emotion = EMOTIONS_LIST[np.argmax(emotions_prob)]
    emotion_label.config(text="Detected Emotion: " + predicted_emotion)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (canvas_width, canvas_height))  # Resize the frame to fit the canvas
    photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
    canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo, anchor=tk.CENTER)
    canvas.photo = photo
    canvas.after(10, update_emotion_label_and_video)

# Start the real-time emotion detection function
update_emotion_label_and_video()

# Function to close the camera and the GUI window
def close_window():
    cap.release()
    root.destroy()

# Button to close the application
close_button = tk.Button(root, text="Close", command=close_window, font=("Helvetica", 14))
close_button.pack(pady=20)

# Run the GUI application
root.mainloop()
