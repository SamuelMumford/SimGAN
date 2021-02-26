#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 14:06:21 2021

@author: sam
"""

import sys
import time
import os
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

import tensorflow as tf

##SEE the CommandLinePreTrain.py FOR A FULL COMMENTED VERSION OF THIS CODE
##DESIGNED TO RUN ON A AWS SERVER.
##This was just code I used to get set up on my computer

print(tf.__version__)

path = os.path.dirname('/home/sam/Documents/SimGan/')
print(os.listdir(path))
data_dir_masks = os.path.join(path, 'face2maskTrain/masks')
print(os.listdir(data_dir_masks))
data_dir_faces = os.path.join(path, 'face2maskTrain/faces')
print(os.listdir(data_dir_faces))
cache_dir = os.path.join(path, 'cache')
print(os.listdir(cache_dir))

batch_size = 32
img_size = 128

lr = .001/(10)
Dlr = .001/10

pretTrainR = False
gen_pre_steps = 200
gen_log_interval = 10

preTrainD = True
disc_pre_steps = 200
disc_log_interval = 10

pics = len(os.listdir(os.path.join(data_dir_masks, 'maskTrain')))
pics2 = len(os.listdir(os.path.join(data_dir_faces, 'faceTrain')))

dataset = tf.keras.preprocessing.image_dataset_from_directory(data_dir_faces, batch_size = batch_size, labels = [0]*pics, image_size=[img_size, img_size])    
dataset_mask = tf.keras.preprocessing.image_dataset_from_directory(data_dir_masks, batch_size = batch_size, labels = [1]*pics2, image_size=[img_size, img_size])

for element in dataset.take(1):
    shapes = element[0].shape
    
img_width = shapes[2]
img_height = shapes[1]
channels = shapes[3]

global sr
sr = .0002/(255)

def self_regularisation_loss(y_true, y_pred):
    return tf.multiply(sr, tf.reduce_sum(tf.abs(y_pred - y_true)))

def local_adversarial_loss(y_true, y_pred):
    
    computed_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    output = tf.reduce_mean(computed_loss)
    
    return output

def refiner_model(width = 55, height = 35, channels = 1):
    """
    The refiner network, Rθ, is a residual network (ResNet). It modifies the synthetic image on a pixel level, rather
    than holistically modifying the image content, preserving the global structure and annotations.
    
    :param input_image_tensor: Input tensor that corresponds to a synthetic image.
    :return: Output tensor that corresponds to a refined synthetic image.
    """
    
    def resnet_block(input_features, nb_features=64, kernel_size=3):
      y = Conv2D(nb_features, kernel_size=kernel_size, padding='same')(input_features)
      y = Activation('relu')(y)
      y = Conv2D(nb_features, kernel_size=kernel_size, padding='same')(y)
        
      y = Add()([y, input_features])
      y = Activation('relu')(y)
        
      return y

    input_layer = Input(shape=(height, width, channels))
    # an input image of size w × h is convolved with 3 × 3 filters that output 64 feature maps
    x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(input_layer)

    for _ in range(4):
        x = resnet_block(x)
        x = BatchNormalization()(x)

    output_layer = Conv2D(channels, kernel_size=1, padding='same', activation='sigmoid')(x)
    output_layer = output_layer*255.0

    return Model(input_layer, output_layer, name='refiner')

def discriminator_model(width = 55, height = 35, channels = 1):
    input_layer = Input(shape=(height, width, channels))

    x = Conv2D(96, kernel_size=3, strides=2, padding='same', activation='relu')(input_layer)
    x = Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=3, strides=1, padding='same')(x)
    x = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = Conv2D(32, kernel_size=1, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=3, strides=1, padding='same')(x)
    x = Conv2D(4, kernel_size=1, strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    output_layer = Dense(2, activation = 'relu')(x)

    return Model(input_layer, output_layer, name='discriminator')

refiner = refiner_model(img_width, img_height, channels)
opt = tf.keras.optimizers.Adam(learning_rate=lr)
refiner.compile(loss=self_regularisation_loss, optimizer=opt)

#refiner.summary()
disc = discriminator_model(img_width, img_height, channels)
optD = tf.keras.optimizers.Adam(learning_rate=Dlr)
disc.compile(loss=local_adversarial_loss, optimizer=optD)

#disc.summary()

synthetic_img = Input(shape=(img_height, img_width, channels))
refined_output = refiner(synthetic_img)
discriminator_output = disc(refined_output)

combined_model = Model(inputs=synthetic_img, outputs=[refined_output, discriminator_output], name='combined')
#combined_model.summary()

def pretrain_gen(steps, log_interval, save_path, profiling=True):
    losses = []
    gen_loss = 0.
    if profiling:
        start = time.perf_counter()
    for i in range(steps):
        syn_imgs_batch = tf.convert_to_tensor(next(iter(dataset))[0], dtype=tf.float32)
        loss = refiner.train_on_batch(syn_imgs_batch, y=syn_imgs_batch)
        gen_loss += loss

        if (i+1) % log_interval == 0:
            print('pre-training generator step {}/{}: loss = {:.5f}'.format(i+1, steps, gen_loss / log_interval))
            losses.append(gen_loss / log_interval)
            gen_loss = 0.
        
    if profiling:
        duration = time.perf_counter() - start
        print('pre-training the refiner model for {} steps lasted = {:.2f} minutes = {:.2f} hours'.format(steps, duration/60., duration/3600.))
    
    refiner.save(save_path)
    
    return losses

pre_gen_path = os.path.join(cache_dir, 'refiner_model_pre_trained_200.h5')

if os.path.isfile(pre_gen_path):
    refiner.load_weights(pre_gen_path)
    print('loading pretrained refiner model weights')
if pretTrainR:
    losses = pretrain_gen(gen_pre_steps, gen_log_interval, pre_gen_path)


def pretrain_disc(steps, log_interval, save_path, profiling=True):
    for layer in disc.layers: layer.trainable = True
    for layer in refiner.layers: layer.trainable = False
    losses = []
    disc_loss = 0.
    real_loss = 0.
    sim_loss = 0.
    if profiling:
        start = time.perf_counter()
    for i in range(steps):
        real_batch = next(iter(dataset_mask))
        real_batch_img = tf.convert_to_tensor(real_batch[0], dtype=tf.float32)
        real_batch_lab = tf.one_hot(tf.convert_to_tensor(real_batch[1], dtype=tf.int32), 2)
        
        syn_batch = next(iter(dataset))
        syn_img_batch = refiner.predict_on_batch(tf.convert_to_tensor(syn_batch[0], dtype=tf.float32))
        syn_label_batch = tf.one_hot(tf.convert_to_tensor(syn_batch[1], dtype=tf.int32), 2)
        
        all_label = tf.concat([real_batch_lab, syn_label_batch], axis = 0)
        all_img = tf.concat([real_batch_img, syn_img_batch], axis = 0)
        
        indices = tf.range(start=0, limit=tf.shape(all_label)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
        
        sh_lab = tf.gather(all_label, shuffled_indices)
        sh_im = tf.gather(all_img, shuffled_indices)
        
#        cce = tf.keras.losses.CategoricalCrossentropy()
#        sim_loss += cce(syn_label_batch, tf.nn.softmax(disc.predict_on_batch(syn_img_batch)))
#        real_loss += cce(real_batch_lab, tf.nn.softmax(disc.predict_on_batch(real_batch_img)))
        
        temp_loss = disc.train_on_batch(sh_im, sh_lab)
        disc_loss += temp_loss

        if (i+1) % log_interval == 0:
            print('pre-training discriminator step {}/{}: loss = {:.5f}'.format(i+1, steps, disc_loss / log_interval))
            #print('refine loss ' + str(sim_loss/log_interval))
            #print('real loss ' + str(real_loss/log_interval))
            losses.append(disc_loss / log_interval)
            disc_loss = 0.
            real_loss = 0.
            sim_loss = 0.

    if profiling:
        duration = time.perf_counter() - start
        print('pre-training the discriminator model for {} steps lasted = {:.2f} minutes = {:.2f} hours'.format(steps, duration/60., duration/3600.))
    
    disc.save(save_path)
    
    return losses

pre_disc_path = os.path.join(cache_dir, 'disc_model_pre_trained_200.h5')

if os.path.isfile(pre_disc_path):
    print('loading pretrained discrim model weights')
    disc.load_weights(pre_disc_path)
if preTrainD:
    losses = pretrain_disc(disc_pre_steps, disc_log_interval, pre_disc_path)