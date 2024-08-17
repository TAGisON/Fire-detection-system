import numpy as np
import cv2
from keras.models import load_model
from BeepSound import beepsound
from CAMERA import camera, MobileCamera
import tkinter as tk

# Function to get screen width and height
def get_screen_size():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    return screen_width, screen_height

# Get screen size
screen_width, screen_height = get_screen_size()

width = 640
height = 480
threshold = 0.65  # Minimum threshold for classification

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Model file loaded
model = load_model('fire_detection_model.h5')
print("Model Loaded Successfully")

# Preprocess the camera image
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

count = 0
sound_file_path = '86502^alarm.wav'  # Update this path to your sound file

while True:
    imgOriginal = camera()  # or imgOriginal = MobileCamera()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preProcessing(img)
    img = img.reshape(1, 32, 32, 1)

    # Predicting
    predictions = model.predict(img)
    classIndex = int(np.argmax(predictions))
    probVal = np.amax(predictions)

    if probVal > threshold:
        cv2.putText(imgOriginal, f"{classIndex}   {probVal:.2f}",
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 0, 255), 1)
        if classIndex == 0:
            count += 1
        else:
            count = 0
        if count == 20:
            beepsound(sound_file_path)

    print(count)

    # Resize window to fit screen size
    resized_img = cv2.resize(imgOriginal, (min(screen_width, width), min(screen_height, height)))

    cv2.imshow("Original Image", resized_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
