#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 18:26:45 2021

@author: sam
"""

import os
import sys
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from keras.preprocessing.image import load_img
from skimage.transform import resize
from keras.preprocessing.image import img_to_array
import tensorflow.keras.backend as K

#Read in paths for files
Basepath = sys.argv[1]
MaskPath = os.path.join(Basepath, 'fake_faces')
FacePath = os.path.join(Basepath, 'real')
#Decide if looking for distances to the same face or different faces after demasking
compSame = False

#Load model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(256, 256, 3))

#Read in picture files. Only care about order if looking for same face/mask base faces
_, _, Mfiles = next(os.walk(MaskPath))
print('final number of mask pictures: ' + str(len(Mfiles)))
_, _, Ffiles = next(os.walk(FacePath))
if(compSame):
    Mfiles = sorted(Mfiles)
    Ffiles = sorted(Ffiles)
print('final number of face pictures: ' + str(len(Ffiles)))

#Make batches of pictures
def makeBatch(path, files, start, size):
    AllImg = np.zeros((size, 256, 256, 3))
    for i in range(size):
        tempi = img_to_array(load_img(os.path.join(path, files[start + i])))
        AllImg[i] = tempi
    AllImg = K.constant(AllImg)
    return AllImg

batch = 100
imgs = len(Mfiles)
iters = imgs//batch

#Initialize the array of all encodings
eMaskAll = np.zeros((0, 2048))
eFaceAll = np.zeros((0, 2048))
#Get all the encodings of faces with and without masks
print('Getting Base Features')
for i in range(iters):
    print(i/iters)
    mB = makeBatch(MaskPath, Mfiles, batch*i, batch)
    fB = makeBatch(FacePath, Ffiles, batch*i, batch)
    encM = model.predict(mB)
    encF = model.predict(fB)
    eMaskAll = np.vstack((eMaskAll, encM))
    eFaceAll = np.vstack((eFaceAll, encF))
#Find the ``mask" feature
eMaskMean = np.sum(eMaskAll, axis = 0, keepdims = True)/imgs
eFaceMean = np.sum(eFaceAll, axis = 0, keepdims = True)/imgs
print(eMaskMean.shape)
print(eMaskAll.shape)

length = np.mean(np.sqrt(np.sum(eMaskAll*eMaskAll, axis = 1)))
print(length)

maskVec = eMaskMean - eFaceMean
maskVec = maskVec/(np.sqrt(np.dot(maskVec, maskVec.T)[0][0]))

#Get FID distances between face and masked encodings
print('Get Base FID Distance')
baseDist = np.mean(np.sum(np.abs(eMaskAll - eFaceAll), axis = 1))
baseDev = np.std(np.sum(np.abs(eMaskAll - eFaceAll), axis = 1))
print(baseDist)
print(baseDev)

#Get stripped face and mask encoding distances
print('Get Mask Stripped Distance')
newMask = eMaskAll - np.dot(np.dot(eMaskAll, maskVec.T), maskVec)
newFace = eFaceAll - np.dot(np.dot(eFaceAll, maskVec.T), maskVec)
newDist = np.mean(np.sum(np.abs(newMask - newFace), axis = 1))
newDev = np.std(np.sum(np.abs(newMask - newFace), axis = 1))
print(newDist)
print(newDev)

#Make cosine similarity distance
def CDist(A, B):
    m = np.mean(np.sum(A*B, axis = 1)/np.sqrt(np.sum(A*A, axis = 1)*np.sum(B*B, axis = 1)))
    d = np.std(np.sum(A*B, axis = 1)/np.sqrt(np.sum(A*A, axis = 1)*np.sum(B*B, axis = 1)))
    return m, d
#Get distances in cosine similarity
print(CDist(eMaskAll, eFaceAll))
print(CDist(newMask, newFace))