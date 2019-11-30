# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import os
#import cv2
import numpy as np
from PIL import Image
import random 

random.seed(204892)

def getData(w,h):
    
    files = "./training/Photos"
    training_label =[]
    training_data = []
    test_label =[]
    test_data = []
    names = []
    i = 0
    for file in os.listdir(files):
        currentLoc = os.path.join(files, file)           
        print(currentLoc)
        names.append(file)
        for f in os.listdir(currentLoc):
            if f.endswith(".jpg"):
                
                img = Image.open(os.path.join(currentLoc, f)).convert("LA").resize((w,h))
                imgArr = np.array(img)
                img.close()
                if  random.random() < .6:
                    training_data.append(imgArr)
                    training_label.append([i])
                else:
                    test_data.append(imgArr)
                    test_label.append([i])
                    
        i+=1
    training_data = np.array(training_data)
    training_label = np.array(training_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    return (training_data, training_label) , (test_data,test_label),names
    


widthImage =280
hieghtImage = 240
#(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
(train_images, train_labels), (test_images, test_labels),class_names  = getData(widthImage,hieghtImage)
print(len(train_images))
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

banana = np.array(Image.open("banana.jpg").convert("LA").resize((widthImage,hieghtImage))) / 255.0
banana = np.array([banana])


print(train_images[2])

channelImage = 2
modelShape = (hieghtImage,widthImage, channelImage)
if channelImage == 1:
    modelShape = (hieghtImage,widthImage,0)
model = models.Sequential()
#initail layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=modelShape))

#hidden layers
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#dense layers formating to output class
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(class_names), activation='softmax'))


#complie and training of model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=15, 
                    validation_data=(test_images, test_labels))


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
prediction = model.predict(banana)
print(class_names[np.argmax(prediction)])
print(np.unique(test_labels, return_counts=True))