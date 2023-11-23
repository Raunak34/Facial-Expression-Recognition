import tkinter as tk
import pyaudio
import wave
import os
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained emotion detection model
emotion_model = load_model('modelaudio_weights.h5')

# Define the emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Function to classify emotion from audio data
def classify_emotion(audio_data):
    # Preprocess and analyze the audio data (you will need to adapt this part)
    # Here, we assume 'audio_data' is preprocessed and represented as a feature vector.

    # Make a prediction using the emotion detection model
    prediction = emotion_model.predict(np.expand_dims(audio_data, axis=0))
    emotion_index = np.argmax(prediction)
    emotion = emotion_labels[emotion_index]

    return emotion

# Function to record audio
def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

    frames = []

    gui.title("Recording...")

    while True:
        try:
            data = stream.read(1024)
            frames.append(data)
        except KeyboardInterrupt:
            break

    gui.title("Emotion Detection")
    
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

    # Call the emotion classification function
    emotion = classify_emotion(audio_data)

    # Update the GUI with the detected emotion
    result_label.config(text="Detected Emotion: " + emotion)

# GUI setup
gui = tk.Tk()
gui.title("Emotion Detection")

record_button = tk.Button(gui, text="Record Audio", command=record_audio)
record_button.pack(pady=20)

result_label = tk.Label(gui, text="Detected Emotion: None")
result_label.pack()

gui.mainloop()
