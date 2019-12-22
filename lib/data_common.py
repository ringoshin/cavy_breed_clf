#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 9 Aug 2019

@author: ringoshin
"""

import cv2
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelBinarizer, LabelEncoder


target_names = ['Abyssinian', 'American', "Silkie", 'Skinny']


def Read_and_Process_Image(image_path, image_shape):
    image_resize = cv2.resize(cv2.imread(image_path, cv2.IMREAD_COLOR), \
                              image_shape, 
                              interpolation=cv2.INTER_CUBIC)
    return image_resize.flatten()


def Load_and_Split(dataset_path, image_shape):
    lb = LabelBinarizer()
    df = pd.read_csv(dataset_path)
    df['image'] = df['image_path'].map(lambda x:
                            Read_and_Process_Image(x, image_shape))
    X = np.array([image for image in df['image']])
    y = lb.fit_transform(df['breed'])
    return X, y

                         
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    
    X_train, y_train = Load_and_Split('data/cavy_data_train.csv', (150,150))
    X_val, y_val = Load_and_Split('data/cavy_data_val.csv', (150,150))
    X_test, y_test = Load_and_Split('data/cavy_data_test.csv', (150,150))
    plt.figure(figsize=(5,15))
    plt.subplot(3, 1, 1)
    _ = plt.title('train: ' + str(np.argmax(y_train[0])))
    _ = plt.imshow(X_train[0].reshape(150,150,3))
    plt.subplot(3, 1, 2)
    _ = plt.title('val: ' + str(np.argmax(y_val[0])))
    _ = plt.imshow(X_val[0].reshape(150,150,3))
    plt.subplot(3, 1, 3)
    _ = plt.title('test: ' + str(np.argmax(y_test[0])))
    _ = plt.imshow(X_test[0].reshape(150,150,3))
    pass
