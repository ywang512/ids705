#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing the Keras libraries and packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import math
import scipy.signal as ss
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def preprocess_red(data, red=False):
	Nimg = data.shape[0]
	Npix = data.shape[1] * data.shape[2]
	Npixcolor = data.shape[1] * data.shape[2] * data.shape[3]
	if red == True:
		R = data[:,:,:,0]
		data = R.reshape(Nimg, Npix)
		return data
	elif red == False:
		return data.reshape(Nimg, Npixcolor)

def add_RelLum(data):
  n = data.shape[0]
  returns = np.concatenate((data, np.zeros(n*101*101*1).reshape(n,101,101,1)), axis=-1)
  returns[:,:,:,-1] = (0.2126 * data[:,:,:,0] + 0.7152 * data[:,:,:,1] + 0.0722 * data[:,:,:,2]) / 255.0
  #print(returns.shape)
  return returns

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

def gradient(img, sigma=0.1):
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

def add_gradient(data, sigma=0.1):
  n = data.shape[0]
  returns = np.concatenate((data, np.zeros(n*101*101*1).reshape(n,101,101,1)), axis=-1)
  for i, img in enumerate(data):
    Ix, Iy = gradient(img[:,:,0], sigma=sigma)
    G = np.sqrt(Ix**2+Iy**2)
    returns[i,:,:,3] = G
  #print(returns.shape)
  return returns



img = mpl.image.imread('/Users/wang/Desktop/solar-pv-in-aerial-imagery/data/training/4.tif').reshape(1,101,101,3)
img_g = add_gradient(img)
img_rl = add_RelLum(img)


fig, axs = plt.subplots(1,3,figsize=(12,4))
ax = axs.flatten()
ax[0].imshow(img.reshape(101,101,3))
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title("Original", fontsize=16)
ax[1].imshow(img_rl[0,:,:,3], cmap='gray')
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title("Relative Luminance", fontsize=16)
ax[2].imshow(img_g[0,:,:,3], cmap='gray')
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].set_title("Gradient", fontsize=16)
plt.show()



fig, axs = plt.subplots(2,2,figsize=(8,8))
ax = axs.flatten()
ax[0].imshow(img[0,:,:,0], cmap='gray')
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title("Red Channel", fontsize=16)
ax[1].imshow(img[0,:,:,1], cmap='gray')
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title("Green Channel", fontsize=16)
ax[2].imshow(img[0,:,:,2], cmap='gray')
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].set_title("Blue Channel", fontsize=16)
ax[3].imshow(img_rl[0,:,:,3], cmap='gray')
ax[3].set_xticks([])
ax[3].set_yticks([])
ax[3].set_title("Relative Luminance", fontsize=16)
plt.show()








