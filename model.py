'''
Created on March 13, 2017

@author: MUNIR JOJO-VERGE

Rev-45
In this revision:
1) Since I'm relying heavily in data processing, I realized that it's much better to process the data on the fly and as needed instead
of processing it, storing it and retrieve it. This way I can also introduce a randon factor in the way I do things. For instance, if I want
to adjust brigtness randomly, it doesn't make sense to generate new images and store them. It's better to pre-process and feed the CNN
2) For the reason above, I'll go back to my data_gen2 in Rev 20 and earlier. This data generator does NOT use KERAS generator. It was built upon Udacity code-example. I will re-use it and generate the images to feed the ".flow" function.


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

from sklearn.model_selection import train_test_split
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot

'''
##__________________________________________________________________________
##  KERAS LIBRARIES
##__________________________________________________________________________
'''
import keras
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
#from keras.utils import plot_model

'''
##__________________________________________________________________________
##  MY LIBRARIES
##__________________________________________________________________________
'''
from CNN_Architectures import Build_VGG16, Build_VGG16_pretrained, Build_NVIDIA, Build_CommaAI
from utils import crop_image


'''
##__________________________________________________________________________
##  CONSTANT PARAMETERS
##__________________________________________________________________________
'''
# Based on our cameras: INPUT SHAPE.
IMG_WIDTH = 320 
IMG_HEIGHT = 160

GRAYSCALE = False
if GRAYSCALE:
    IMG_CH = 1 # n of channels
else:
    IMG_CH = 3 # n of channels

if keras.backend.image_dim_ordering() == 'tf':
    INPUT_IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CH)
else:
    INPUT_IMG_SHAPE = (IMG_CH, IMG_HEIGHT, IMG_WIDTH)


#For the different models we want to crop slighly different
CROP_WINDOW_VGG16 = ((40,40),(120,120)) ## Target size (80, 80, 3) This size is NOT random and is the result of multiple tests. Proves to be the best in terms of performance.
CROP_WINDOW_NVIDIA = ((74,20),(60,60)) # Our target image would be a 66x200.
CROP_WINDOW_COMMAAI = ((74,20),(60,60)) # Our target image would be a 66x220.

STEERING_CORRECTION = 0.2
STRAIGHT_STEERING_THRESHOLD = 0.1 # This means the steering between -x and +x it will be considered straight steering

BATCH_SIZE = 128
EPOCHS = 10
PATIENCE = 3
'''
##__________________________________________________________________________
##  CAPTURE AND ORGANIZATION OF DATA
##__________________________________________________________________________
'''

TRAINING_CSV_LOGS = [
                       'data/track1_main/driving_log.csv',
                       'data/track1_recovery/driving_log.csv',
                       'data/track1_reverse/driving_log.csv',
                       'data/track1_recovery_reverse/driving_log.csv',
                       'data/track2_main/driving_log.csv',
                       'data/track1_validation/driving_log.csv',
                       'data/track2_validation/driving_log.csv',
                       'data2/driving_log.csv' # Udacity data
                      ]

AUGMENTED_CSV_LOGS = ['data/Augmented/driving_log.csv']

'''
##__________________________________________________________________________
##  DATA AUGMENTATION AND DATA FEED (TO THE CNN) FUNCTIONS
##                      (GENERATORS)
##__________________________________________________________________________
'''
# randomily change the image brightness
def randomise_image_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # brightness - referenced Vivek Yadav post
    # https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.yh93soib0

    bv = .25 + np.random.uniform()
    hsv[::2] = hsv[::2]*bv

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# referenced Vivek Yadav post
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.yh93soib0
def warpAffine(image, steering):
    rows, cols, _ = image.shape
    transRange = 100
    numPixels = 10
    valPixels = 0.4
    transX = transRange * np.random.uniform() - transRange/2
    steering = steering + transX/transRange * 2 * valPixels
    transY = numPixels * np.random.uniform() - numPixels/2
    transMat = np.float32([[1, 0, transX], [0, 1, transY]])
    image = cv2.warpAffine(image, transMat, (cols, rows))
    return image, steering

##process camera image
def process_image(img_path, steering):      
    image = cv2.imread(img_path.strip())    
    image, steering = warpAffine(image, steering)
    image = randomise_image_brightness(image)

    return image, steering


def data_setup(training_csv_logs, AugmentedData_csv_log, AugmentData=False, use_AugmentedData=True, use_LeftRight_Cam=True):
    
    column_names = ['center', 'left', 'right',
                    'steering', 'throttle', 'brake', 'speed']
    
    # Put all the training data collected together (CSV files)
    training_data = pd.concat([pd.read_csv(path, names=column_names) for path in training_csv_logs])
    print('All training data put together')
    
    if use_LeftRight_Cam:
        print('Using Left and Right cameras')
        # Create DATA for the left camera images by subtracting the offset from the steering angle.
        leftCam_Data = training_data[['left', 'steering']].copy()
        leftCam_Data.loc[:, 'steering'] -= STEERING_CORRECTION

        # Create feature DATA for the right camera images by adding the offset to the steering angle.
        RightCam_Data = training_data[['right', 'steering']].copy()
        RightCam_Data.loc[:, 'steering'] += STEERING_CORRECTION
        
        img_paths = pd.concat([training_data.center,leftCam_Data.left,RightCam_Data.right]).str.strip()
        ang_values = pd.concat([training_data.steering,leftCam_Data.steering,RightCam_Data.steering])        
        
        print('Left and Right cameras images and steering angles put together with center images. Correction used:',STEERING_CORRECTION)
    else: # Use only the center camera
        print('Using only center camera')
        img_paths = pd.concat([training_data.center]).str.strip()
        ang_values = pd.concat([training_data.steering])
        print('Center camera data aquired')
        
    # No matter what camera we use we should now balance the data. If we used and adjust the steering of left/Right cameras
    # we might have even more un-balanced driving 
    if AugmentData:
        BalanceData(img_paths,ang_values) 
        
    if use_AugmentedData:        
        Augmented_data = pd.concat([pd.read_csv(path, names=column_names) for path in AugmentedData_csv_log])
        img_paths = pd.concat([img_paths,Augmented_data.center]).str.strip()
        ang_values = pd.concat([ang_values,Augmented_data.steering])
    
    #Split the Data 80% Train 20% Validation
    print('Spliting data')
    X_train, X_test, y_train, y_test = train_test_split(img_paths, ang_values, test_size=0.2, random_state=42)
    
    print (X_train.shape)
    
    return X_train, X_test, y_train, y_test

import csv
### __________________________
###       BALANCE/AUGMENT DATA
### __________________________
# To balance the data we collected we will start looking at how many Straight driving we have, how much turning to both siede
# and we will try to balance the data so we don't teach unevenly our CNN.
# Here's what I will do:
# 1) We know in advance that we have way more straight triving. We will count and use this figure as the target to balance.
# 2) We will count left and right turns..but we will count turns that involve a steering above xx (parameter). I'll start with 0.1
# 3) The difference between straight driving and left/right turns will tell me how much data I need to generate to "balance".
# 4) To generate Left Turn Data I will flip as many Right turn Data as I can that is qualified as "Good" turn i.e with a steering 
#    angle above xx (0.1 as above)
# 5) I will count the generated data and compare it with what needs to be generated. If we went flipped all the right turns and didn't             generate enough Left Data, we could: a) Go through all the left turns (above xx angle) and use and adjust the right camera image..making a             sharper left turn. We will go through until we generate enough data.
# 6) We will repeat steps 4 and 5 to generate the right data.
# THIS METHOD IS NOT, BY ALL MEANS, EFFICIENT OR PERFECT. THIS IS BRUT-FORCE AND INTUITIVE DATA GENERATION
def BalanceData(img_paths,ang_values): 
    
    NumSamples = len(ang_values)
    
    print('Augmenting/Balancing Data...')
    CSVRows = []    
    StraightDrivingSamples = 0
    LeftTurnDrivingSamples = 0
    RightTurnDrivingSamples = 0    
    #print(len(img_paths))
    for i in range(NumSamples-1): 
        steering = ang_values.iloc[i]
        if abs(steering) < STRAIGHT_STEERING_THRESHOLD:  # I will consider Straight Driving even slight steering
            StraightDrivingSamples += 1
        else:
            if steering < -STRAIGHT_STEERING_THRESHOLD:  # I will consider Right Driving 
                RightTurnDrivingSamples += 1
            else:
                LeftTurnDrivingSamples += 1
                
    Max =  max([StraightDrivingSamples,LeftTurnDrivingSamples,RightTurnDrivingSamples])
    LeftData_Needed = Max - LeftTurnDrivingSamples
    RightData_Needed = Max - RightTurnDrivingSamples
    StraightData_Needed = Max - StraightDrivingSamples
    
    print('Number Straight Samples:', StraightDrivingSamples)
    print('Number Straight Samples NEEDED:', StraightData_Needed)
    
    print('Number Left Samples:', LeftTurnDrivingSamples)
    print('Number Left Samples NEEDED:', LeftData_Needed)
    
    print('Number Right Samples:', RightTurnDrivingSamples)
    print('Number Right Samples NEEDED:', RightData_Needed)
    
    LeftData_Generated = 0
    RightData_Generated = 0
    StraightData_Generated = 0
    
    # Flip portion.
    
    # Left and Right Cameras' images flipping
    NeedMoreLeftData = LeftData_Needed > LeftData_Generated
    NeedMoreRightData = RightData_Needed > RightData_Generated
    print('Flipping images...')
    while (NeedMoreLeftData) or (NeedMoreRightData):       
        i = np.random.randint(NumSamples-1)
        steering = ang_values.iloc[i]
        Steering_Right = steering < -STRAIGHT_STEERING_THRESHOLD # Good Strong Steering
        Steering_Left = steering > STRAIGHT_STEERING_THRESHOLD
        if (NeedMoreLeftData and Steering_Right) or (NeedMoreRightData and Steering_Left):            
            # Load Image
            path = img_paths.iloc[i]
            image = cv2.imread(path.strip())
            # Flip Image
            image = cv2.flip(image, 1)

            #print ("image flipped " + str(i))
            newImgPath = './data/Augmented/IMG/'+str(i)+'.jpg' 
            cv2.imwrite(newImgPath,image)
            print(newImgPath)
            # Put all the training data together after adding the left and Right Cameras' adjusted angles
            CSVRows.append([newImgPath,'','',-steering])

            if (NeedMoreLeftData and Steering_Right):
                LeftData_Generated += 1
            else:
                RightData_Generated += 1

        NeedMoreLeftData = LeftData_Needed > LeftData_Generated
        NeedMoreRightData = RightData_Needed > RightData_Generated
    
    # To generate more straight driving data I can only think of duplicating what we already have. SOUNDS like a bad idea!!
    print('Generating Straight images...')
    NeedMoreStraightData = StraightData_Needed > StraightData_Generated
    while (NeedMoreStraightData):       
        i = np.random.randint(NumSamples-1)
        steering = ang_values.iloc[i]
        
        if abs(steering) < STRAIGHT_STEERING_THRESHOLD:            
            # Load Image
            path = img_paths.iloc[i]
            image = cv2.imread(path.strip())
            newImgPath = './data/Augmented/IMG/'+str(i)+'.jpg' 
            cv2.imwrite(newImgPath,image)
            print(newImgPath)
            # Put all the training data together after adding the left and Right Cameras' adjusted angles
            CSVRows.append([newImgPath,'','',steering])

            StraightData_Generated += 1
            
        NeedMoreStraightData = StraightData_Needed > StraightData_Generated
            
   
    my_df = pd.DataFrame(CSVRows)
    my_df.to_csv('./'+ AUGMENTED_CSV_LOGS[0], index=False, header=False)        
    
        
    print('Number Straight Samples Generated:', StraightData_Generated)
    print('TOTAL Number Straight Samples:', StraightDrivingSamples + StraightData_Generated)
    
    print('Number Left Samples Generated:', LeftData_Generated)
    print('TOTAL Number Left Samples:', LeftTurnDrivingSamples + LeftData_Generated)
          
    print('Number Right Samples Generated:', RightData_Generated)
    print('TOTAL Number Right Samples:', RightTurnDrivingSamples + RightData_Generated)
    
    print('TOTAL Number Samples:', StraightDrivingSamples + StraightData_Generated + LeftTurnDrivingSamples + LeftData_Generated + RightTurnDrivingSamples + RightData_Generated)



# create a training data generator for keras fit_model
def data_gen2(images_paths, steering_angles, CNN_Model_Name='NVIDIA', batch_size=64):
    
    data_count = len(images_paths)

    print("Datalog with %d rows." % (data_count))

    while True:  # need to keep generating data

        # initialise data extract
        features = []
        labels = []

        # create a random batch to return
        while len(features) < batch_size:            
            row = np.random.randint(data_count-1)
            img_path = images_paths.iloc[row]
            steering = steering_angles.iloc[row]
            ## Random image processing: jitter, brightness..
            image, steering = process_image(img_path, steering)
            
            if CNN_Model_Name == 'VGG16 - Pretrained':
                image = crop_image(image, CROP_WINDOW_VGG16)
                
            features.append(image)
            labels.append(steering)

        # yield the batch
        yield (np.array(features), np.array(labels))





'''
##__________________________________________________________________________
##  MAIN
##__________________________________________________________________________
'''

if __name__ == '__main__':
    
    #---------------------
    CNN_Model_Name = 'NVIDIA'
    #CNN_Model_Name = 'VGG16 - Pretrained'
    #CNN_Model_Name = 'VGG16'
    #CNN_Model_Name = 'COMMA-AI'
    #---------------------
    
    if CNN_Model_Name == 'NVIDIA':
        model = Build_NVIDIA(CROP_WINDOW_NVIDIA, INPUT_IMG_SHAPE)
    else:
        if CNN_Model_Name == 'VGG16':        
            model = Build_VGG16(CROP_WINDOW_VGG16, INPUT_IMG_SHAPE)
        else:            
            if CNN_Model_Name == 'VGG16 - Pretrained':        
                model = Build_VGG16_pretrained(CROP_WINDOW_VGG16, INPUT_IMG_SHAPE)
            else:
                model = Build_CommaAI(CROP_WINDOW_COMMAAI, INPUT_IMG_SHAPE)
            
    model.summary()      
    
    model.compile(optimizer="adam", loss="mse")
    
    
    # Persist trained model
    model_json = model.to_json()
    with open(CNN_Model_Name + '.json', 'w') as f:
        json.dump(model_json, f)
    
    #plot_model(model, to_file='model.png')
    SVG(model_to_dot(model).create(prog='dot', format='svg'))
    
    ##
    ## Set up the data: Put together all the CSV files, augment/balace data (if needed/wanted),...)
    ##
    print("Setting up data")
    
    X_train, X_test, y_train, y_test = data_setup(TRAINING_CSV_LOGS, AUGMENTED_CSV_LOGS, AugmentData=False, use_AugmentedData=True, use_LeftRight_Cam=True)

    
    checkpoint = ModelCheckpoint(CNN_Model_Name +'.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=PATIENCE, verbose=1, mode='auto')

    # Train the model
    history = model.fit_generator(data_gen2(X_train, y_train,CNN_Model_Name, batch_size=BATCH_SIZE),
                                  samples_per_epoch=len(X_train),
                                  validation_data=data_gen2(X_test, y_test, CNN_Model_Name,batch_size=BATCH_SIZE),
                                  nb_val_samples=len(X_test),
                                  nb_epoch=EPOCHS,
                                  callbacks=[checkpoint, early_stopping])

          
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(CNN_Model_Name + ' - Mean Square Error (MSE) loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'Validation set'], loc='upper right')
    plt.show()

    #out = model.predict(im)
    #print np.argmax(out)
