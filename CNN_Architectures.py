'''
##__________________________________________________________________________
##                              CNN ARCHITECTURES
##
## This file defines the different CNNs Architectures I'm testing for this task
## Created on Feb 10, 2017
## author: MUNIR JOJO-VERGE
##
##__________________________________________________________________________
'''

'''
##__________________________________________________________________________
##  LIBRARIES
##__________________________________________________________________________
'''

import json
import os
import csv
import cv2
import sklearn
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import argparse

'''
##__________________________________________________________________________
##  KERAS LIBRARIES
##__________________________________________________________________________
'''
import keras
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import AveragePooling2D,Flatten, Dense, Lambda, Input, Dropout, SpatialDropout2D, merge, BatchNormalization
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2

np.random.seed(888)

from utils import get_cropped_shape


'''
##__________________________________________________________________________
##                    VGG16 - MODEL - ENTIRE MODEL (NO PRE-TRAINED)
## (Pseudo-VGG16  since I had to adjust to compesate for our imput shape
## Changes made:
## 1) All padding from Block 1 to Block 4 changed to "same"
## 2) Had to eliminate Block 5 since after all the MaxPoolings the input size became negative
##__________________________________________________________________________
'''
def Build_VGG16(CROP_WINDOW, INPUT_IMG_SHAPE):
    
    print("Building VGG16...")
    
    model = Sequential()
    
    # Crop
    model.add(Cropping2D(CROP_WINDOW, input_shape=INPUT_IMG_SHAPE, name="Crop"))
    
    # Normalize input.
    model.add(Lambda(lambda x: x/127.5 - 1., name="Normalize"))
    
    
    # Block 1    
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='elu', name='Block1_Cov1'))        
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='elu', name='Block1_Cov2'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name='Block1_Pool'))

    # Block 2
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='elu', name='Block2_Cov1'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='elu', name='Block2_Cov2'))
    model.add(MaxPooling2D((2,2), strides=(2,2), name='Block2_Pool'))
    
    # Block 3
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='elu', name='Block3_Cov1'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='elu', name='Block3_Cov2'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='elu', name='Block3_Cov3'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name='Block3_Pool'))

    # Block 4
    model.add(Convolution2D(512, 3, 3, border_mode='same', activation='elu', name='Block4_Cov1'))    
    model.add(Convolution2D(512, 3, 3, border_mode='same', activation='elu', name='Block4_Cov2'))    
    model.add(Convolution2D(512, 3, 3, border_mode='same', activation='elu', name='Block4_Cov3'))
    model.add(MaxPooling2D((2,2), strides=(2,2), name='Block4_Pool'))

    '''
    # Block 5
    model.add(Convolution2D(512, 3, 3, activation='elu', name='Block5_Cov1'))    
    model.add(Convolution2D(512, 3, 3, activation='elu', name='Block5_Cov2'))    
    model.add(Convolution2D(512, 3, 3, activation='elu', name='Block5_Cov3'))
    model.add(MaxPooling2D((2,2), strides=(2,2), name='Block5_Pool'))
    
    # Make sure pre trained layers from the VGG net don't change while training.
    for layer in model.layers:
        layer.trainable = False
        print(layer.input_shape)
    '''
    
    # Classification Block (Block 6)
    model.add(Flatten(name='Block6_Flatten'))
    model.add(Dropout(0.5, name='Block6_Dout1'))
    model.add(Dense(2048, activation='elu', name='Block6_Dense1'))
    model.add(Dropout(0.2, name='Block6_Dout2'))
    model.add(Dense(1024, activation='elu', name='Block6_Dense2'))
    model.add(Dropout(0.5, name='Block6_Dout3'))
    model.add(Dense(1, activation='linear', name='predictions'))
    

    return model

'''
##__________________________________________________________________________
##                    VGG16 - MODEL - (PRE-TRAINED)
## 
##__________________________________________________________________________
'''

def Build_VGG16_pretrained(CROP_WINDOW, INPUT_IMG_SHAPE):
    
    print("Building VGG16-pretrained...")    
    
    #Get back the convolutional part of a VGG network trained on ImageNet
    
    input_image = Input(shape = (80,80,3))
    base_model = VGG16(input_tensor=input_image, include_top=False)

    for layer in base_model.layers[:-3]:
        layer.trainable = False

    W_regularizer = l2(0.01)

    x = base_model.get_layer("block5_conv3").output
    x = AveragePooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(4096, activation="elu", W_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation="elu", W_regularizer=l2(0.01))(x)
    x = Dense(2048, activation="elu", W_regularizer=l2(0.01))(x)
    x = Dense(1, activation="linear")(x)
    
 

    return Model(input=input_image, output=x)
  

'''
##__________________________________________________________________________
##  COMMA AI - MODEL
##__________________________________________________________________________
'''
def Build_CommaAI(CROP_WINDOW, INPUT_IMG_SHAPE):
    
    print("Building CommaAI...")
    
    model = Sequential()
    
    # Crop
    model.add(Cropping2D(CROP_WINDOW, input_shape=INPUT_IMG_SHAPE, name="Crop"))
    
    # Normalize input.
    model.add(Lambda(lambda x: x/127.5 - 1., name="Normalize"))

    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", activation='elu', name='Conv-1'))    
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", activation='elu', name='Conv-2'))    
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same", activation='elu', name='Conv-3'))
    model.add(Flatten(name='Flt-1'))
    model.add(Dropout(0.2, name='Dout-1'))
    model.add(Activation('elu'))
    model.add(Dense(512))
    model.add(Dropout(0.5, name='Dout-2'))
    model.add(Activation('elu'))
    model.add(Dense(1, activation='linear', name='predictions'))

    
    return model


'''
##__________________________________________________________________________
##  NVIDIA - MODEL
##__________________________________________________________________________
'''
def Build_NVIDIA(CROP_WINDOW, INPUT_IMG_SHAPE, DROPOUT_RATE=0.0):
    
    print("Building NVIDIA...")
    
    model = Sequential()
    
    # Crop
    model.add(Cropping2D(CROP_WINDOW, input_shape=INPUT_IMG_SHAPE, name="Crop"))
    
    # Normalize input.
    model.add(Lambda(lambda x: x/127.5 - 1., name="Normalize"))
    
    #Convolution layers
    
    ## NOTE: using dropout tends to make all-zero predictions! 
    
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2), activation="elu", name="Conv_1"))
    #model.add(MaxPooling2D(name="MaxPool1"))
    # Dropout for regularization
    model.add(SpatialDropout2D(DROPOUT_RATE, name="Dropout1"))
    
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2), activation="elu", name="Conv_2"))
    #model.add(MaxPooling2D(name="MaxPool2"))
    # Dropout for regularization
    model.add(SpatialDropout2D(DROPOUT_RATE, name="Dropout2"))
    
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2), activation="elu", name="Conv_3"))
    #model.add(MaxPooling2D(name="MaxPool3"))
    # Dropout for regularization
    model.add(SpatialDropout2D(DROPOUT_RATE, name="Dropout3"))
       
    model.add(Convolution2D(64, 3, 3, border_mode="valid", subsample=(1,1), activation="elu", name="Conv_4"))
    #model.add(MaxPooling2D(name="MaxPool4"))
    # Dropout for regularization
    model.add(SpatialDropout2D(DROPOUT_RATE, name="Dropout4"))
    
    model.add(Convolution2D(64, 3, 3, border_mode="valid", subsample=(1,1), activation="elu", name="Conv_5"))
    #model.add(MaxPooling2D(name="MaxPool5"))        
    # Dropout for regularization
    model.add(SpatialDropout2D(DROPOUT_RATE, name="Dropout5"))

    # Flatten input in a non-trainable layer before feeding into
    # fully-connected layers.
    model.add(Flatten(name="Flatten"))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(1164, activation="elu", name="FC1"))
    #model.add(SpatialDropout2D(DROPOUT_RATE, name="Dropout6"))
    model.add(Dense(100, activation="elu", name="FC2"))
    #model.add(SpatialDropout2D(DROPOUT_RATE, name="Dropout7"))
    model.add(Dense(50, activation="elu", name="FC3"))
    #model.add(SpatialDropout2D(DROPOUT_RATE, name="Dropout8"))
    model.add(Dense(10, activation="elu", name="FC4"))
    model.add(Dropout(DROPOUT_RATE, name="Dropout9"))

    # Generate output (steering angles) with a single non-trainable node.   
    model.add(Dense(1, activation='linear', name='predictions'))
   
        
    return model
