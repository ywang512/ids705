#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
import os
import cv2
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

train_datagen = ImageDataGenerator(
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

'''
training_set = train_datagen.flow_from_directory('/Users/wang/Desktop/solar-pv-in-aerial-imagery/data2/training',
                                                 target_size = (101, 101),
                                                 batch_size = 1,
                                                 class_mode = 'binary')


test_set = test_datagen.flow_from_directory('/Users/wang/Desktop/solar-pv-in-aerial-imagery/data2/testing',
                                            target_size = (101, 101),
                                            batch_size = 1,
                                            class_mode = 'binary')
'''

img = cv2.imread('/Users/wang/Desktop/solar-pv-in-aerial-imagery/data/training/4.tif')
cv2.imshow('image',img.reshape(101,101,3))
cv2.waitKey(0)
cv2.destroyAllWindows()
#print(img)

img = matplotlib.image.imread('/Users/wang/Desktop/solar-pv-in-aerial-imagery/data/training/4.tif')


fig, axs = plt.subplots(1,2)
ax = axs.flatten()
ax[0].imshow(img)
ax[1].imshow(img)
plt.show()




'''
train_datagen.fit(img.reshape(1,101,101,1))
count = 0
i = next(train_datagen.flow(img.reshape(1,101,101,1),  batch_size=1))[0]
print(i.reshape(101,101))
cv2.imshow('image', i.reshape(101,101))
cv2.waitKey(0)
cv2.destroyAllWindows()
count += 1
'''




