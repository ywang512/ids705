#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 11:33:39 2019

@author: azucenamv
"""

import pandas as pd
import os
import shutil

## This is where your unzip folders are (testing and training)
data_path = "/Users/azucenamv/Documents/MIDS/ML/Kaggle/Data2/"
data_train = "/Users/azucenamv/Documents/MIDS/ML/Kaggle/Data2/training/"

### Reading the 
labels = pd.read_csv(data_path+"labels_training.csv")

## Creating folders
os.mkdir(data_path+"Train")
os.mkdir(data_path+"Train/0")
os.mkdir(data_path+"Train/1")
os.mkdir(data_path+"Test")
os.mkdir(data_path+"Test/0")
os.mkdir(data_path+"Test/1")
os.rename(data_path+"testing",data_path+"Testing_nolabels")

### Splitting data into training and testing 
ltrain = labels.sample(n=1200)
ltest = labels.drop(ltrain.index)

### Moving images into folders
for ID,label in zip(ltrain.id,ltrain.label):
    if label == 1:
        shutil.move(data_train+str(ID)+".tif", data_path+"Train/1/"+str(ID)+".tif")
    if label == 0:
        shutil.move(data_train+str(ID)+".tif", data_path+"Train/0/"+str(ID)+".tif")
        
for ID,label in zip(ltest.id,ltest.label):
    if label == 1:
        shutil.move(data_train+str(ID)+".tif", data_path+"Test/1/"+str(ID)+".tif")
    if label == 0:
        shutil.move(data_train+str(ID)+".tif", data_path+"Test/0/"+str(ID)+".tif")

os.rmdir(data_path+"training")
       
