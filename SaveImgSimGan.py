#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 14:06:21 2021

@author: sam


"""

import sys
import time
import os
import shutil
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from itertools import groupby
from skimage.util import montage

from keras.layers import Dense, Reshape, Input, BatchNormalization, Concatenate, Activation, Add, Flatten
from keras.layers.convolutional import UpSampling2D, MaxPooling2D, Deconv2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, Nadam, Adamax
from keras import initializers
from keras import applications
from keras.utils import plot_model
from keras.preprocessing import image
from keras.preprocessing.image import save_img

import tensorflow as tf

##Code used to download a model and run it to save transformed images

#Locations of the base data and models
path = os.path.dirname('/home/sam/Documents/SimGan/')
data_dir_masks = os.path.join(path, 'face2maskTrain/masks')
data_dir_faces = os.path.join(path, 'face2maskTrain/faces')
cache_dir = os.path.join(path, 'cache')

#Name of the weights used for the generator model
weightName = 'refiner_model_pre_trained_200.h5'

#Define how the data will be passed into the generator and how many pictures to make
batch_size = 16
img_size = 128
total_pics = 300

#Define a dataset
pics = len(os.listdir(os.path.join(data_dir_masks, 'maskTrain')))
pics2 = len(os.listdir(os.path.join(data_dir_faces, 'faceTrain')))
basePath = os.path.join(path, 'outputs/gen0')

dataset = tf.keras.preprocessing.image_dataset_from_directory(data_dir_faces, batch_size = batch_size, labels = [0]*pics, image_size=[img_size, img_size])    
ds = iter(dataset)

#Find the dataset shape
for element in dataset.take(1):
    shapes = element[0].shape
    
img_width = shapes[2]
img_height = shapes[1]
channels = shapes[3]

#Define the generator model
def refiner_model(width = 55, height = 35, channels = 1):
    """
    The refiner network, Rθ, is a residual network (ResNet). It modifies the synthetic image on a pixel level, rather
    than holistically modifying the image content, preserving the global structure and annotations.
    
    :param input_image_tensor: Input tensor that corresponds to a synthetic image.
    :return: Output tensor that corresponds to a refined synthetic image.
    """
    #Generator model works in REsNet blocks which do not change the image lateral size
    def resnet_block(input_features, nb_features=64, kernel_size=3):
      y = Conv2D(nb_features, kernel_size=kernel_size, padding='same')(input_features)
      y = Activation('relu')(y)
      y = Conv2D(nb_features, kernel_size=kernel_size, padding='same')(y)
        
      y = Add()([y, input_features])
      y = Activation('relu')(y)
        
      return y
    
    #Define a convolution on the input to change the depth as needed for the
    #ResNet blocks
    input_layer = Input(shape=(height, width, channels))
    x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(input_layer)
    
    #Apply the ResNet blocks
    for _ in range(4):
        x = resnet_block(x)
        x = BatchNormalization()(x)
        
    #Reformat the processed image to be the same shape and amplitude as the input
    output_layer = Conv2D(channels, kernel_size=1, padding='same', activation='sigmoid')(x)
    output_layer = output_layer*255.0

    return Model(input_layer, output_layer, name='refiner')

#Make the generator
refiner = refiner_model(img_width, img_height, channels)
lr = 0.0
opt = tf.keras.optimizers.Adam(learning_rate=lr)
refiner.compile()

synthetic_img = Input(shape=(img_height, img_width, channels))
refined_output = refiner(synthetic_img)

#Load weights
pre_gen_path = os.path.join(cache_dir, weightName)

if os.path.isfile(pre_gen_path):
    refiner.load_weights(pre_gen_path)
    print('loaded pretrained refiner model weights')
else:
    print('no valid weights')

#Delete any files currently in the output area
folder = basePath
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

#Generate new pictures
runs = total_pics//batch_size
for j in range(0, runs):
    print(j/runs)
    input_batch = tf.convert_to_tensor(next(ds)[0], dtype=tf.float32)
    gen_batch = refiner.predict_on_batch(input_batch)
    for i in range(gen_batch.shape[0]):
        name = str(i + j*batch_size) + '.png'
        picPath = os.path.join(basePath, name)
        save_img(picPath, gen_batch[i])