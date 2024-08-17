from keras.models import load_model
import numpy as np
import cv2
import os



#model loaded
model = load_model('fire_detection_model.h5')
print("Model Loaded Successfully")


#preprocess the camera image
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img




def classify(img_file):
    print(img_file)
    img_name = img_file
    test_image=cv2.imread(img_name)
    img = np.asarray(test_image)
    img = cv2.resize(img, (32, 32))
    img = preProcessing(img)
    # cv2.imshow("Processsed Image", img)
    img = img.reshape(1, 32, 32, 1)
    # predicting
    classIndex = int(model.predict_classes(img))
    predictions = model.predict(img)
    probVal = np.amax(predictions)
    if classIndex==0:
        print("Fire + Probability : "+str(probVal))
    else:
        print("No Fire + Probability : "+str(probVal))

path = 'Testing'
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
   for file in f:
       files.append(os.path.join(r, file))

for f in files:
   classify(f)
   print('\n')


