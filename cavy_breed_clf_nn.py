#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 02:27:36 2019

@author: ringoshin
"""

from lib.data_common import (target_label, Read_and_Process_Image, 
                             Load_and_Split)
from lib.ml_common import (Vanilla_ML_Run, Vanilla_ML_Predict, 
                           Vanilla_ML_Run_CV)
from lib.nn_common import (Image_CNN_VGG, Image_CNN_VGG_Train,
                           Image__CNN_From_InceptionV3,
                           Image_CNN_From_InceptionV3_Train, Image_NN_Predict,
                           Image_NN_Plt_Acc, Image_NN_Plt_Loss, 
                           Image_NN_Plt_Training, Image_NN_Plt_Validation,
                           Save_Model_Data, Load_Model_Data)

import numpy as np
import pandas as pd

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

# fix random seed for reproducibility
seed = 128
np.random.seed(seed)

# Image size for standardisation
image_shape = (150, 150)
input_shape = (-1, 150, 150, 3)

batch_size =64

# Load pre-split data from training, validation and test data sets
X_train, y_train = Load_and_Split('data/cavy_data_train.csv', image_shape)
X_val, y_val = Load_and_Split('data/cavy_data_val.csv', image_shape)
X_test, y_test = Load_and_Split('data/cavy_data_test.csv', image_shape)

# Keras
# CNN from scatch per VGG

model_0 = Image_CNN_VGG(3, input_shape=(150,150,3))
print(model_0.summary())

model_0, history_0 = Image_CNN_VGG_Train(model_0, 
                                     X_train.reshape((input_shape), 
                                     y_train,
                                     X_val.reshape((input_shape), 
                                     y_val, 
                                     epochs=10,  # was 20
                                     batch_size=batch_size, 
                                     verbose=0)

results_0 = {k: history_0.history[k][-1] for k in history_0.history.keys()}
print(results_0)

#clf_report_0, cf_matrix_0 = Image_NN_Predict(model_0, 
#                                           X_val.reshape((-1,150,150,3)), 
#                                           y_val, 
#                                           target_names=target_label,
#                                           batch_size=batch_size, 
#                                           verbose=1)

# Transfer learning from InceptionV3
model_1 = Image__CNN_From_InceptionV3(3, input_shape=(150,150,3))
model_1, history_1, history_2 = Image_CNN_From_InceptionV3_Train(model_1, 
                                     X_train.reshape((-1,150,150,3)), 
                                     y_train,
                                     X_val.reshape((-1,150,150,3)),
                                     y_val, 
                                     epochs=10,   # was 100 
                                     batch_size=batch_size, 
                                     verbose=2)

results_1 = {k: history_2.history[k][-1] for k in history_2.history.keys()}
print(results_1)

Image_NN_Plt_Training(history_1)
Image_NN_Plt_Acc(history_2)
#Image_NN_Plt_Loss(history_2)
Image_NN_Plt_Training(history_2)
Image_NN_Plt_Validation(history_2)


#clf_report_1, cf_matrix_1 = Image_NN_Predict(model_1,
#                                             X_test.reshape((-1,150,150,3)), 
#                                             y_test, 
#                                             target_names=target_label,
#                                             batch_size=batch_size, 
#                                             verbose=2)
    
#Save_Model_Data(model_1, history_2, X_test, y_test,
#                'inceptionV3_tuned_freeze-230_sgd_v1_epoch-100')

    
"""
X_test = X_val.reshape((-1,150,150,3))[:10]
test_datagen = ImageDataGenerator(rescale=1./255)

i = 0
text_labels=[]
plt.figure(figsize=(10,30))
for j, batch in enumerate(test_datagen.flow(X_test, batch_size=1)):
    pred = model.predict(batch)
    pred_pos = pred.argmax()
    text_labels.append(target_label[pred_pos])
    plt.subplot(5, 2, i+1)
    plt.title(text_labels[i])
    imgplot = plt.imshow(batch[0])
    i += 1
    if i%10==0:
        break

"""