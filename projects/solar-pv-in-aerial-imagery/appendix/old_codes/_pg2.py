import math
import scipy.signal as ss
import numpy as np
import pandas as pd
import matplotlib.image as mpl
import cv2
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
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

data, labels = load_data(DIR_image, DIR_label, training=True)

shuffle_index = np.random.choice(1500, 1500, replace=False)
train_index = shuffle_index[:1350]
val_index = shuffle_index[1350:]
train_data, val_data = data[train_index], data[val_index]
train_label, val_label = labels[train_index], labels[val_index]


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





