#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 18:40:05 2019

@author: azucenamv
"""
import numpy as np
import pandas as pd
import matplotlib.image as mpl
import math
import scipy.signal as ss
import cv2
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score, accuracy_score

#import keras.backend as K

DIR_image = "./data/training/"
DIR_label = "./data/training_labels/labels_training.csv"
DIR_test = "./data/testing/"

def load_data(dir_data, dir_labels, training=True):
    labels_pd = pd.read_csv(dir_labels)
    ids       = labels_pd.id.values
    data      = []
    for identifier in ids:
        fname     = dir_data + identifier.astype(str) + '.tif'
        image     = mpl.imread(fname)
        data.append(image)
    data = np.array(data) # Convert to Numpy array
    if training:
        labels = labels_pd.label.values
        return data, labels
    else:
        return data, ids

def Gaussian(sigma):
    alpha = math.ceil(3.5 * sigma)
    n = np.arange(-alpha, alpha+1)
    g = np.exp(-0.5 * n**2 / sigma**2)
    k = 1/np.sum(g)
    return k*g

def dGaussian(sigma):
    alpha = math.ceil(3.5 * sigma)
    n = np.arange(-alpha, alpha+1)
    d = n * np.exp(-0.5 * n**2 / sigma**2)
    k = -1/np.dot(n, d)
    return k*d

def gradient(img, sigma=2.5):
    '''img is a 2-dimensional numpy array representing a black-and-white image'''
    
    g = Gaussian(sigma)
    gx = np.expand_dims(g, 0)
    gy = np.expand_dims(g, 1)
    d = dGaussian(sigma)
    dx = np.expand_dims(d, 0)
    dy = np.expand_dims(d, 1)
    
    # derivative along x
    Ix = ss.convolve2d(ss.convolve2d(img, dx, mode = 'same'), gy, mode = 'same')

    # derivative along y
    Iy = ss.convolve2d(ss.convolve2d(img, gx, mode = 'same'), dy, mode = 'same')
    
    return (Ix, Iy)

def add_gradient(data, sigma):
  n = data.shape[0]
  returns = np.concatenate((data, np.zeros(n*101*101*1).reshape(n,101,101,1)), axis=-1)
  for i, img in enumerate(data):
    Ix, Iy = gradient(img[:,:,0], sigma)
    G = np.sqrt(Ix**2+Iy**2)
    returns[i,:,:,3] = G
  return returns

def add_RelLum(data):
  n = data.shape[0]
  returns = np.concatenate((data, np.zeros(n*101*101*1).reshape(n,101,101,1)), axis=-1)
  returns[:,:,:,-1] = 0.2126 * data[:,:,:,0] + 0.7152 * data[:,:,:,1] + 0.0722 * data[:,:,:,2]
  return returns


data, labels = load_data(DIR_image, DIR_label, training=True)
double_labels = to_categorical(labels, num_classes = 2)
#data = add_RelLum(data)
#data = add_gradient(data, 0.1)

shuffle_index = np.random.choice(1500, 1500, replace=False)
train_index = shuffle_index[:1350]
val_index = shuffle_index[1350:]
train_data, val_data = data[train_index], data[val_index]
train_label, val_label = double_labels[train_index], double_labels[val_index]


'''
model = Sequential()
#model.add(BatchNormalization())
model.add(Conv2D(32, (5, 5), input_shape = (101, 101, 4), activation = 'relu'))
#model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(Conv2D(64, (3, 3), activation = 'relu')) 
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 2, activation = 'sigmoid'))
'''

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = (101, 101, 3), activation = 'relu'))
model.add(Conv2D(32, (3, 3), activation = 'relu')) 
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(.25))

model.add(Conv2D(64, (3, 3), activation = 'relu')) 
model.add(Conv2D(64, (3, 3), activation = 'relu')) 
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(.25))

model.add(Conv2D(128, (1, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(128, (1, 1), activation = 'relu')) 
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(.25))

model.add(Flatten())

# Full connection
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))

model.summary()

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer = 'adam', 
                       loss = 'binary_crossentropy', 
                       metrics = ['accuracy'])


#==============================================================================


'''
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, 
                                              factor=0.5, min_lr=0.00001)
'''

datagen = ImageDataGenerator(rescale = 1./255, 
        samplewise_center=True,  # set each sample mean to 0
        samplewise_std_normalization=True,  # divide each input by its std
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        brightness_range=[0.5, 1.5],
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

datagen.fit(train_data)

batch_size = 32
epochs = 20

history = model.fit_generator(datagen.flow(train_data, train_label, batch_size=batch_size),
                              epochs = epochs, validation_data = (val_data,val_label),
                              verbose = 1, steps_per_epoch=train_data.shape[0] // batch_size)

#model.fit(train_data, train_label, batch_size=50, epochs=5)


loss, acc = model.evaluate(val_data, val_label)
print("Validation ACC:   ", acc * 100)

val_pre = model.predict(val_data)
auc = roc_auc_score(val_label[:,1], val_pre[:,1])
print("Validation AUC:   ", auc)




