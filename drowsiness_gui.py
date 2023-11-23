import tkinter as tk
from tkinter import filedialog
from tkinter import *
import cv2
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk

def DrowsinessModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Create the Tkinter window
top = tk.Tk()
top.geometry('800x600')
top.title('Drowsiness Detector')
top.configure(background='#CDCDCD')

# Create a label for displaying results
drowsiness_label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

# Load the Haar Cascade for face detection
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the drowsiness detection model
drowsiness_model = DrowsinessModel("drowsiness_model.json", "drowsiness_model_weights.h5")

# Function to detect drowsiness
def Detect(file_path):
    global Label_packed

    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image, 1.3, 5)
    
    try:
        for (x, y, w, h) in faces:
            # Extract the face region
            face = gray_image[y:y+h, x:x+w]
            
            # Resize the face image for drowsiness detection
            drowsiness_roi = cv2.resize(face, (48, 48))
            
            # Predict drowsiness
            drowsiness_pred = "Drowsy" if drowsiness_model.predict(drowsiness_roi[np.newaxis, :, :, np.newaxis])[0][0] > 0.5 else "Not Drowsy"
            
            print("Drowsiness State:", drowsiness_pred)
            
            drowsiness_label.configure(foreground="#011638", text="Drowsiness State: " + drowsiness_pred)
    
    except:
        drowsiness_label.configure(foreground="#011638", text="Unable to detect drowsiness")

# Function to show the Detect button
def show_Detect_button(file_path):
    detect_b = Button(top, text="Detect Drowsiness", command=lambda: Detect(file_path), padx=10, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.5, rely=0.85, anchor='center')

# Function to upload an image
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        drowsiness_label.configure(text='')
        show_Detect_button(file_path)
    except:
        pass

# Create the "Upload Image" button
upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=30)

# Place the labels and heading
sign_image.pack(side='top', expand='True')
drowsiness_label.pack(side='top', expand='True')
heading = Label(top, text='Drowsiness Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

# Run the Tkinter main loop
top.mainloop()
