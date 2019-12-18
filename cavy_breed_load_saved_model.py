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

new_test_path = 'pix/new_test/'
new_test_list = os.listdir('pix/new_test')

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


#model, history, X_test, y_test = Load_Model_Data('inceptionV3_tuned_freeze-230_sgd_v1_epoch-100')
#clf_report, cf_matrix, y_pred = Image_NN_Predict(model,
#                                         X_test.reshape((-1,150,150,3)), 
#                                         y_test, 
#                                         target_names=target_label,
#                                         batch_size=batch_size, 
#                                         verbose=2)


#results = {k: history.history[k][-1] for k in history.history.keys()}
#print(results)
#Image_NN_Plt_Acc(history1)
#Image_NN_Plt_Training(history1)


model, history, X_test, y_test = Load_Model_Data('inceptionV3_tuned_freeze-240_sgd_v1_epoch-100')
clf_report, cf_matrix, y_pred = Image_NN_Predict(model,
                                         X_test.reshape((-1,150,150,3)), 
                                         y_test, 
                                         target_names=target_label,
                                         batch_size=batch_size, 
                                         verbose=2)
#Image_NN_Plt_Acc(history)
#Image_NN_Plt_Training(history)

#y_test = to_categorical(y_test)


#model3, history3, X_test3, y_test3 = Load_Model_Data('inceptionV3_tuned_freeze-249_sgd_v1_epoch-100')
#clf_report, cf_matrix, y_pred3 = Image_NN_Predict(model3,
#                                         X_test3.reshape((-1,150,150,3)), 
#                                         y_test3, 
#                                         target_names=target_label,
#                                         batch_size=batch_size, 
#                                         verbose=2)

#Image_NN_Plt_Acc(history3)
#Image_NN_Plt_Training(history3)

#y_test3 = to_categorical(y_test3)



import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

y_test = to_categorical(y_test)

n_classes = 3

# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['red', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of {0} (area = {1:0.2f})'
             ''.format(class_label[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontweight='bold')
plt.ylabel('True Positive Rate',fontweight='bold')
plt.title('Extension of Receiver Operating Characteristic to Multi-class',fontweight='bold')
plt.legend(loc="lower right")
plt.show()


# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['red', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of {0} (area = {1:0.2f})'
             ''.format(class_label[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate',fontweight='bold')
plt.ylabel('True Positive Rate',fontweight='bold')
plt.title('Extension of Receiver Operating Characteristic to Multi-class',fontweight='bold')
plt.legend(loc="lower right")
plt.show()


print()
print(" >> Predicting new test images")
plt.figure(figsize=(60,100))
num_images = len(new_test_list)
for i, image_file in enumerate(new_test_list, start=1):
    X_raw, y = Read_and_Process_One_Image(image_file)
    X = np.array(X_raw)
    y_pred_label, y_pred = Image_NN_Predict_One(model, X, target_label, batch_size=32, verbose=0)
    plt.subplot(num_images, 1, i)
    plt.title(image_file + ' - ' + y_pred_label)
    imgplot = plt.imshow(mpimg.imread(new_test_path+image_file))
    
    