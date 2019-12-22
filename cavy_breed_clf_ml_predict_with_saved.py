"""
Created on Sat Aug 17 00:19:17 2019

@author: ringoshin
"""

from lib.data_common import (target_names, Read_and_Process_Image, 
                             Load_and_Split)

from lib.ml_common import (Predict_and_Report, Vanilla_ML_Predict,
                           Show_Confusion_Matrix, Load_Model_Data,
                           Plot_Precision_Recall_Curve, Plot_ROC_Curve,
                           Compare_Multiple_ROC_Curves)

import numpy as np
import pandas as pd

# fix random seed for reproducibility
seed = 128
np.random.seed(seed)

# Image size for standardisation
image_shape = (150, 150)

# Load pre-split data from training, validation and test data sets
print(">> Loading data")
X_train, y_train = Load_and_Split('data/cavy_data_train.csv', image_shape)
X_val, y_val = Load_and_Split('data/cavy_data_val.csv', image_shape)
X_test, y_test = Load_and_Split('data/cavy_data_test.csv', image_shape)


# One-vs-rest model list
ovr_clf_list = {"OVR Dummy": "ovr_dummy_v2",
                "OVR Logistic Regression": 'ovr_logistic_regression_ovr_v2', 
                "OVR Naive Bayes": 'ovr_naive_bayes_v2',
                "OVR Random Forest": 'ovr_random_foreset_n_est-100_v2', 
           }

# Measure selected models via: Classification Report
#                              Confusion Matrix
#                              Precision vs Recall Curve
#                              ROC Curve and ROC_AUC
y_test_dict = dict()
y_score_dict = dict()

for clf_name, clf_fname in ovr_clf_list.items():
    clf = Load_Model_Data(clf_fname)
    
    clf_acc, clf_report, cf_matrix = Predict_and_Report(clf, X_test, y_test, 
                                                        target_names)
    print()
    print("{}'s accuracy: {:.2%}".format(clf_name, clf_acc))
    print("  > Classification report:")
    print(clf_report)
    Show_Confusion_Matrix(cf_matrix, target_names, clf_name=clf_name)

    y_score = clf.predict_proba(X_test)
    recall, precision = Plot_Precision_Recall_Curve(y_test, y_score, 
                                       target_names, clf_name=clf_name)
    fpr, tpr, roc_auc = Plot_ROC_Curve(y_test, y_score, 
                                       target_names, clf_name=clf_name)
    y_test_dict[clf_name] = y_test
    y_score_dict[clf_name] = y_score

# Compare ROC curves and AUC scores between different models
Compare_Multiple_ROC_Curves(y_test_dict, y_score_dict)
#Compare_Multiple_ROC_Curves(y_test_dict, y_score_dict, zoom_level=0.5)

# Deep dive at selected model
clf_name = 'OVR Random Forest'
y_test = y_test_dict[clf_name]
y_score = y_score_dict[clf_name] 
_ = Plot_ROC_Curve(y_test, y_score, target_names, clf_name=clf_name)
_ = Plot_ROC_Curve(y_test, y_score, target_names, 
                   clf_name=clf_name, zoom_level=0.6)
