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


target_label = ['Abyssinian', 'American', 'Skinny']


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
    X_test, y_test = Load_and_Split('data/cavy_data_test.csv', (150,150))
    pass
