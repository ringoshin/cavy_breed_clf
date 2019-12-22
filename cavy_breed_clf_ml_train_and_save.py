#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 8 Aug 2019

@author: ringoshin
"""

from lib.data_common import (target_names, Read_and_Process_Image, 
                             Load_and_Split)
from lib.ml_common import (Vanilla_ML_Run, Vanilla_ML_Predict, 
                           Vanilla_ML_Run_CV, Show_Confusion_Matrix,
                           Save_Model_Data)

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier


# Fix random seed for reproducibility
seed = 128
np.random.seed(seed)

# Image size for standardisation
image_shape = (150, 150)

# Load pre-split data from training, validation and test data sets
print(">> Loading data")
X_train, y_train = Load_and_Split('data/cavy_data_train.csv', image_shape)
X_val, y_val = Load_and_Split('data/cavy_data_val.csv', image_shape)
X_test, y_test = Load_and_Split('data/cavy_data_test.csv', image_shape)

# List of selected models that will learn to predict each class against the others
clf_list = {'dummy': OneVsRestClassifier(DummyClassifier(), n_jobs=-1),
            'log reg': OneVsRestClassifier(LogisticRegression(multi_class='ovr', n_jobs=-1), n_jobs=-1),
#            'knn': KNeighborsClassifier(n_neighbors=5),
            'naive bayes': OneVsRestClassifier(GaussianNB(), n_jobs=-1),
#            'svm': LinearSVC(),
            'random forest': OneVsRestClassifier(RandomForestClassifier(n_estimators=100), n_jobs=-1)
            }

# Train and predict using vanilla traditional models to set a baseline 
# for latter deep learning models
clf_list = Vanilla_ML_Run(clf_list, X_train, y_train)
predict_list = Vanilla_ML_Predict(clf_list, X_val, y_val, target_names)

#[print(predict_list[k][0]) for k in predict_list.keys()]
#[print(predict_list[k][1]) for k in predict_list.keys()]

# A peek at the confusion matrices for an overview of the model 
# performances
Show_Confusion_Matrix(predict_list['dummy'][2], target_names, 
                      clf_name="Dummy")
Show_Confusion_Matrix(predict_list['log reg'][2], target_names,
                      clf_name="Logistic Regression")
Show_Confusion_Matrix(predict_list['naive bayes'][2], target_names,
                      clf_name="Naive Bayes")
Show_Confusion_Matrix(predict_list['random forest'][2], target_names,
                      clf_name="Random Forest")

# Cross validation runs for reduced list of models with good performances
#X_train_val = np.concatenate([X_train, X_val])
#y_train_val = np.concatenate([y_train, y_val])
#scores = Vanilla_ML_Run_CV(clf_list, X_train_val, y_train_val, n_splits=3)

# Save all models for future multiple prediction runs
# Save_Model_Data(clf_list['dummy'], 'ovr_dummy_v2')
# Save_Model_Data(clf_list['log reg'], 'ovr_logistic_regression_ovr_v2')
# Save_Model_Data(clf_list['naive bayes'], 'ovr_naive_bayes_v2')
# Save_Model_Data(clf_list['random forest'], 'ovr_random_foreset_n_est-100_v2')
