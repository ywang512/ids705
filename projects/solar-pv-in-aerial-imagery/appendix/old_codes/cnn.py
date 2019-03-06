#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 18:40:05 2019

@author: azucenamv
"""
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout, BatchNormalization
import os
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('/Users/wang/Desktop/solar-pv-in-aerial-imagery/data2/training',
                                                 target_size = (101, 101),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('/Users/wang/Desktop/solar-pv-in-aerial-imagery/data2/testing',
                                            target_size = (101, 101),
                                            batch_size = 32,
                                            class_mode = 'binary')



# Initialising 
cnn_classifier = Sequential()

# 1st conv. layer
cnn_classifier.add(Conv2D(32, (3, 3), input_shape = (101, 101, 3), activation = 'relu'))
#cnn_classifier.add(MaxPooling2D(pool_size = (2, 2)))
cnn_classifier.add(BatchNormalization())
cnn_classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
cnn_classifier.add(MaxPooling2D(pool_size = (2, 2)))
cnn_classifier.add(BatchNormalization())
cnn_classifier.add(Conv2D(64, (3, 3), activation = 'relu')) 
cnn_classifier.add(BatchNormalization())
cnn_classifier.add(MaxPooling2D(pool_size = (2, 2)))
cnn_classifier.add(BatchNormalization())

# Flattening
cnn_classifier.add(Flatten())

# Full connection
cnn_classifier.add(Dense(units = 128, activation = 'relu'))
cnn_classifier.add(Dropout(0.25))
cnn_classifier.add(Dense(units = 2, activation = 'sigmoid'))

cnn_classifier.summary()

# Compiling the CNN
# Second attempt steps: 5000 and epochs: 5
# Validation steps = 2000
cnn_classifier.compile(optimizer = 'adam', 
                       loss = 'binary_crossentropy', 
                       metrics = ['accuracy'])
                   
cnn_classifier.fit_generator(training_set,
                             steps_per_epoch = 37,
                             epochs = 20,
                             validation_data = test_set,
                             validation_steps = 2000)



#Predictions
import numpy as np
from keras.preprocessing import image
import re
import pandas as pd 

proba = []
path = '/Users/wang/Desktop/solar-pv-in-aerial-imagery/data2/testing_nolabels/'
images=os.listdir(path)
for img in images:
    test_image = image.load_img(path+img, target_size = (101,101))
    test_image = image.img_to_array(test_image)/255
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn_classifier.predict(test_image)
    proba.append(cnn_classifier.predict_proba(test_image)[0][0])


# Submission
score = ["{:0.2f}".format(p) for p in proba]
#score = [round(p,2) for p in proba]
index = [re.sub(".tif","",img) for img in images]
submission_file = pd.DataFrame({'id':    index,
                                'score':  score})

submission_file.to_csv('/Users/wang/Desktop/solar-pv-in-aerial-imagery/data2/submission_yw.csv',
                           columns=['id','score'],
                           index=False)
