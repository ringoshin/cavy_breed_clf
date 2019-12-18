#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 00:19:17 2019

@author: ringoshin
"""

from lib.ml_common import (Vanilla_ML_Run, Vanilla_ML_Predict, 
                           Vanilla_ML_Run_CV)
from lib.nn_common import (Image_CNN_VGG, Image_CNN_VGG_Train,
                           Image__CNN_From_InceptionV3,
                           Image_CNN_From_InceptionV3_Train, 
                           Image_NN_Predict_One, Image_NN_Predict,
                           Image_NN_Plt_Acc, Image_NN_Plt_Loss, 
                           Image_NN_Plt_Training, Image_NN_Plt_Validation,
                           Save_Model_Data, Load_Model_Data)

import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

import tensorflow as tf
from keras.utils import to_categorical

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

# fix random seed for reproducibility
seed = 128
np.random.seed(seed)

new_test_path = 'images/new_test/'
new_test_list = os.listdir('images/new_test')

target_label = ['American', 'Abyssinian', 'Skinny']
class_label = ['American', 'Abyssinian', 'Skinny']

nrows = 150
ncolumns = 150
channels = 3
batch_size =64

def Read_and_Process_My_Images(list_of_images, X, y):
    global nrows, ncolumns, new_test_path
    
    for image in list_of_images:
        image=new_test_path+image
            
        #print(image)
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), \
                            (nrows, ncolumns), interpolation=cv2.INTER_CUBIC))
        y.append(image)
    return X, y


def Read_and_Process_One_Image(image_file):
    global nrows, ncolumns, new_test_path
    
    image=new_test_path+image_file
            
    #print(image)
    X = (cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), \
                            (nrows, ncolumns), interpolation=cv2.INTER_CUBIC))
    y = image
    return X, y


model, history, X_test, y_test = Load_Model_Data('inceptionV3_tuned_freeze-240_sgd_v1_epoch-100')

print()
print(" >> Predicting new test images")
plt.figure(figsize=(60,100))
num_images = len(new_test_list)
for i, image_file in enumerate(new_test_list, start=1):
    X_raw, y = Read_and_Process_One_Image(image_file)
    X = np.array(X_raw)
    y_pred_label, y_pred = Image_NN_Predict_One(model, X, target_label, batch_size=32, verbose=0)
    print(image_file, y_pred_label)
    plt.subplot(num_images, 1, i)
    plt.title(image_file + ' - ' + y_pred_label)
    imgplot = plt.imshow(mpimg.imread(new_test_path+image_file))
    
    