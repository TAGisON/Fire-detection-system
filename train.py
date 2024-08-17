import numpy as np
import cv2
import os
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dropout,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


imagedimensions=(32,32,3)


#importing images
path="dataset"
mylist=os.listdir(path)
noofclasses=len(mylist)
print("Total number of classes detected : "+str(noofclasses))
images=[]
classno=[]
print("Importing classes.....")
time.sleep(2)
for x in range(0,noofclasses):
    mypiclist=os.listdir(path+"/"+str(x))
    for y in mypiclist:
        curimg=cv2.imread(path+"/"+str(x)+"/"+y)
        curimg=cv2.resize(curimg,(imagedimensions[0],imagedimensions[1]))
        images.append(curimg)
        classno.append(x)
    print(x,end=",")
print("\n")



#converting into array
images=np.array(images)
classno=np.array(classno)
print(images.shape)
print(classno.shape)




#spliting the data
testratio=0.2
validationratio=0.2
x_train,x_test,y_train,y_test=train_test_split(images,classno,test_size=testratio)
x_train,x_validation,y_train,y_validation=train_test_split(x_train,y_train,test_size=validationratio)






#plotting graph to know how many images are in each class
numofsamples=[]
for x in range(0,noofclasses):
    print(len(np.where(y_train==x)[0]))  #y contains the class ids so it will give the number of images
    numofsamples.append(len(np.where(y_train==x)[0]))
print(numofsamples)
plt.figure(figsize=(10,5))
plt.bar(range(0,noofclasses),numofsamples)
plt.title("Number of images for each classes")
plt.xlabel("Class ID")
plt.ylabel("number of Images")
plt.show()






#data preprocessing
def preprocessing(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img) #this will help to equalize the distribution of lighting in each images
    img=img/255  #changing 0 to 255 into 0 to 1
    return img
x_train=np.array(list(map(preprocessing,x_train)))#map will take each image from input and send it to preprocessing function
                                                 #The result from funtion is stored in list using list.
                                                 #that list is converted to array
                                                 #this same as done in line 31
x_test=np.array(list(map(preprocessing,x_test)))
x_validation=np.array(list(map(preprocessing,x_validation)))



#adding depth of 1...this will help the cnn to work proper
print(x_train.shape)
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
#print(x_train.shape)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_validation=x_validation.reshape(x_validation.shape[0],x_validation.shape[1],x_validation.shape[2],1)






#image augmenting
datagen=ImageDataGenerator(width_shift_range=0.1,
                           height_shift_range=0.1,
                           zoom_range=0.2,
                           shear_range=0.1,
                           rotation_range=10)
datagen.fit(x_train)




#label encoding
y_train=to_categorical(y_train,noofclasses)
y_test=to_categorical(y_test,noofclasses)
y_validation=to_categorical(y_validation,noofclasses)





#model defining
def mymodel():

    nooffilters=60
    sizeoffilter1=(5,5)
    sizeoffilter2 = (3,3)
    sizeofpool=(2,2)
    noofnode=500


    model=Sequential()
    model.add(Conv2D(nooffilters,sizeoffilter1,input_shape=(imagedimensions[0],
                                                             imagedimensions[1],
                                                             1),activation='relu'))
    model.add(Conv2D(nooffilters, sizeoffilter1,activation='relu'))
    model.add(MaxPooling2D(pool_size=sizeofpool))
    model.add(Conv2D(nooffilters//2, sizeoffilter2, activation='relu'))
    model.add(Conv2D(nooffilters//2, sizeoffilter2, activation='relu'))
    model.add(MaxPooling2D(pool_size=sizeofpool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noofnode,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noofclasses, activation='softmax'))
    model.compile(Adam(lr=0.001),loss='binary_crossentropy', metrics=['accuracy']) #lr means learning rate
    return model

model=mymodel()
print(model.summary())




#compiling the model
batchsize=32
epochs=100
stepsperepoch=len(x_train)//batchsize

history=model.fit_generator(datagen.flow(x_train,y_train,batch_size=batchsize), #batchsize means how many images at a time is taken to datagen for image augmenting
                    steps_per_epoch=stepsperepoch,
                    epochs=epochs,
                    validation_data=(x_validation,y_validation),
                    shuffle=1)


#plotting graphs
plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy Graph')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.savefig('Accuracy.png')

plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss Graph')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('Loss.png')
plt.show()
score=model.evaluate(x_test,y_test,verbose=0)
print("Test Score = ",score[0])
print("Test Accurancy = ",score[1])




#saving the model
model.save('fire_detection_model.h5')
print("Model Saved")