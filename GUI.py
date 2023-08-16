import cv2
import PIL.Image, PIL.ImageTk
from cvzone.HandTrackingModule import HandDetector
from keras.utils import load_img, img_to_array
from keras.models import load_model
import numpy as np
import math
import tkinter as tk

# Define a function to capture frames from the camera
def show_frame():
    global cap, canvas, imgtk
    ret, frame = cap.read()
    if ret:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(img)
        imgtk = PIL.ImageTk.PhotoImage(image=img)
        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
    top.after(0, show_frame)

# Define a function to start the camera capture
def start_capture():
    global cap
    cap = cv2.VideoCapture(0)
    show_frame()

# Define a function to stop the camera capture
def stop_capture():
    global cap
    cap.release()

# Define a function to get the predicted class index
def get_predicted_class_index(res):
    model09AZ = load_model('model/model09AZ-OWNDS.h5')
    img2arr = img_to_array(res)
    imgexp = np.expand_dims(img2arr, axis=0)
    # Make the prediction
    predictions = model09AZ.predict(imgexp)
    # Get the predicted class index
    index = np.argmax(predictions)
    return index

# Define the main function
def run_detection():
    detector = HandDetector(detectionCon=0.8)
    while True:
        ret, frame = cap.read()
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        # Get the hand landmarks
        img = detector.findHands(frame)
        lmList, bboxInfo = detector.findPosition(img)
        if lmList:
            # Get the landmarks of the hand's index and middle fingers
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            # Calculate the distance between the two landmarks
            length = math.hypot(x2-x1, y2-y1)
            # Define the distance range to detect the gesture
            if length < 60:
                # Get the region of interest containing the gesture
                img = frame[bboxInfo[1]:bboxInfo[1]+bboxInfo[3],
                            bboxInfo[0]:bboxInfo[0]+bboxInfo[2]]
                img = cv2.resize(img, (224, 224))
                img = img/255.0
                # Get the predicted class index
                index = get_predicted_class_index(img)
                # Update the label
                label.config(text=index)
        # Display the video stream
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(img)
        imgtk = PIL.ImageTk.PhotoImage(image=img)
        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        top.update()

# Create the application main window
top = tk.Tk()
top.geometry("1100x600")

# Create the canvas
canvas = tk.Canvas(top, width=700, height=700)
canvas.pack()
canvas.place(relx=0.3, rely=0.6, anchor=tk.CENTER)

# Create the buttons
button1 = tk.Button(top, text="Start Detections", command=start_capture, width=30, height=3)
button1.pack()

button2 = tk.Button(top, text="Stop Detections", command=stop_capture, width=30, height=3)
button2.pack()
# Set the position of the buttons
button1.place(relx=0.8, rely=0.4, anchor=tk.CENTER)
button2.place(relx=0.8, rely=0.6, anchor=tk.CENTER)

label = tk.Label(top, text="", font=("Arial", 48))
label.pack()
label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

# Set the position of the label
# label.place(relx=0.8, rely=0.8, anchor=S)

#Entering the event main loop
top.mainloop()