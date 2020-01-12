"""
Created on Thu Aug  8 02:27:36 2019

@author: ringoshin
"""

from lib.data_common import (target_names, Read_and_Process_Image, 
                             Load_and_Split)
from lib.ml_common import (Vanilla_ML_Run, Vanilla_ML_Predict, 
                           Vanilla_ML_Run_CV)
from lib.nn_common import (Image_CNN_Multilayer, Image_CNN_Multilayer_Train,
                           Image_CNN_From_InceptionV3,
                           Image_CNN_From_InceptionV3_Train, Image_NN_Predict,
                           Image_NN_Plt_Acc, Image_NN_Plt_Loss, 
                           Image_NN_Plt_Training, Image_NN_Plt_Validation,
                           Save_NN_Model_Data, Load_NN_Model_Data)

import numpy as np
import pandas as pd

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

# fix random seed for reproducibility
seed = 128
np.random.seed(seed)

# Image size for standardisation
image_shape = (150, 150)
input_shape = (-1, 150, 150, 3)

batch_size = 64

# Load pre-split data from training, validation and test data sets
print(">>> Loading data")
X_train, y_train = Load_and_Split('data/cavy_data_train.csv', image_shape)
X_val, y_val = Load_and_Split('data/cavy_data_val.csv', image_shape)
X_test, y_test = Load_and_Split('data/cavy_data_test.csv', image_shape)


# Multilayer Image CNN
model_0 = Image_CNN_Multilayer(4, input_shape=(150,150,3))
print(model_0.summary())

model_0, history_0 = Image_CNN_Multilayer_Train(model_0, 
                                     X_train.reshape(input_shape), 
                                     y_train,
                                     X_val.reshape(input_shape), 
                                     y_val, 
                                     epochs=100,  # was 100
                                     batch_size=batch_size, 
                                     verbose=1)

results_0 = {k: history_0.history[k][-1] for k in history_0.history.keys()}
print(results_0)

clf_report_0, cf_matrix_0, y_pred_0, y_pred_bool_0 = Image_NN_Predict(model_0, 
                                                        X_val.reshape(input_shape), 
                                                        y_val, 
                                                        target_names=target_names,
                                                        batch_size=batch_size, 
                                                        verbose=2)


# Transfer learning from InceptionV3
model_1 = Image_CNN_From_InceptionV3(4, input_shape=(150,150,3))
model_1, history_1_1, history_1_2 = Image_CNN_From_InceptionV3_Train(model_1, 
                                        X_train.reshape((input_shape)), 
                                        y_train,
                                        X_val.reshape((input_shape)),
                                        y_val, 
                                        epochs=100,   # was 100 
                                        batch_size=batch_size, 
                                        verbose=1)

results_1 = {k: history_1_2.history[k][-1] for k in history_1_2.history.keys()}
print(results_1)

Image_NN_Plt_Training(history_1_1)
Image_NN_Plt_Acc(history_1_2)
Image_NN_Plt_Loss(history_1_2)
Image_NN_Plt_Training(history_1_2)
Image_NN_Plt_Validation(history_1_2)

clf_report_1, cf_matrix_1, y_pred_1, y_bool_1 = Image_NN_Predict(model_1,
                                                    X_val.reshape(input_shape), 
                                                    y_val, 
                                                    target_names=target_names,
                                                    batch_size=batch_size, 
                                                    verbose=2)
    

# Save the trained models
Save_NN_Model_Data(model_0, history_0, 'cnn_multilayer_v1_epoch-100')
Save_NN_Model_Data(model_1, history_1_2, 'inceptionV3_tuned_freeze-230_sgd_v1_epoch-100')

