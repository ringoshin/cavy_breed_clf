"""
Created on Thu Aug  8 02:27:36 2019

@author: ringoshin
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from itertools import cycle
from scipy import interp

from sklearn.metrics import (classification_report, confusion_matrix, 
                             precision_recall_curve, average_precision_score,
                             roc_curve, auc, roc_auc_score, f1_score, accuracy_score)
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     cross_validate, StratifiedKFold, KFold)
from sklearn.utils import shuffle


def Vanilla_ML_Run(clf_list, X_train, y_train):
    print(">> Starting %d vanilla ML runs" %(len(clf_list)))
    print()
    
    new_clf_list = {}
    for clf_name, clf in clf_list.items():
        print(" > Training", clf_name)
        clf.fit(X_train, y_train)
        new_clf_list[clf_name] = clf
    
    print()
    return new_clf_list


def Vanilla_ML_Run_CV(clf_list, X, y, n_splits=5):
    print(">> Starting %d vanilla ML runs" %(len(clf_list)))
    print()
    
    for clf_name, clf in clf_list.items():
        print(" > Cross-Validation of {} with {} folds".format(clf_name, n_splits))
        skf = StratifiedKFold(n_splits=n_splits)
        scores = cross_validate(clf, X, y, cv=skf,
                                 scoring=('balanced_accuracy', 'f1_macro', 'f1_weighted',
                                          'recall_macro', 'recall_weighted'))
    
    for score_name, score_value in scores.items():
        print(" >", score_name)
        print("   >", score_value)
    print()    
    return scores


def Predict_and_Report(clf, X_val, y_val, target_names):
    y_pred = clf.predict(X_val)
    clf_acc = accuracy_score(y_val, y_pred)
    clf_report = classification_report(y_val, y_pred, target_names=target_names)
    cf_matrix = confusion_matrix(y_val.argmax(axis=1), y_pred.argmax(axis=1))
    #print(clf_report)
    #print(cf_matrix)
    #print()
    return (clf_acc, clf_report, cf_matrix)


def Vanilla_ML_Predict(clf_list, X_val, y_val, target_names):
    print(">> Starting %d vanilla ML predictions" %(len(clf_list)))
    print()
    
    predict_list = {}
    for clf_name, clf in clf_list.items():
        print(" > Predicting for", clf_name)
        predict_list[clf_name] = Predict_and_Report(clf, X_val, y_val, target_names)
        
    print()
    return predict_list
   

def Show_Confusion_Matrix(cf_matrix, target_names, clf_name="Model's"):
    """ Print confusion matrix for specified classifier
    """
    plt.figure(dpi=150)
    sns.heatmap(cf_matrix, cmap=plt.cm.Blues, annot=True, square=True,
            xticklabels=target_names, yticklabels=target_names)

    plt.xlabel('Predicted breeds')
    plt.ylabel('Actual breeds')
    plt.title('{} Confusion Matrix'.format(clf_name))
 

def Plot_Precision_Recall_Curve(y_test, y_score, target_names, clf_name="Model's", zoom_level=1.0):
    """ Plot precision recall curve for each class onto one chart
    """

    # Zoom in view of the upper right corner, if zoom_level is set
    plt.figure(dpi=150)
    plt.xlim(1-zoom_level, 1.0)
    plt.ylim(1-zoom_level, 1.05)

    precision = dict()
    recall = dict()
    ap_score = dict()
        
    # Plots Precision-Recall curve for each class
    for i, label in enumerate(target_names):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        ap_score[i] = average_precision_score(y_test[:, i], y_score[:, i])
        plt.plot(recall[i], precision[i], lw=2, label='{} (area: {:.2f})'.format(label, ap_score[i]))

    #
    # Compute micro-average precision-recall curve and AP score (area)
    #
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
    ap_score["micro"] = average_precision_score(y_test, y_score, average="micro")

    # Plot both micro-average precision recall curves
    plt.plot(precision["micro"], recall["micro"],
            label='Micro-average (area: {0:0.2f})'
                ''.format(ap_score["micro"]),
                linestyle=':', linewidth=4)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.title("{} Precision-Recall Curves".format(clf_name))
    plt.show()
    return recall, precision


def Compare_Precision_Recall_Curves(y_test, y_score, zoom_level=1.0):
    """ Plot multiple PR curves from different models for comparison
    """
    plt.figure(dpi=150)
    # Zoom in view of the upper left corner, if zoom_level is set
    plt.xlim(1-zoom_level, 1.0)
    plt.ylim(1-zoom_level, 1.05)

    for clf_name in y_score.keys():
        precision, recall, _ = precision_recall_curve(y_test.ravel(), y_score[clf_name].ravel())
        ap_score = average_precision_score(y_test, y_score[clf_name])
        plt.plot(precision, recall, label="{} (area: {:.2f})".format(clf_name, ap_score))

    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title("Precision-Recall Curves of different Models")
    plt.legend(loc='best')
    plt.show()


def Plot_ROC_Curve(y_test, y_score, target_names, clf_name="Model's", zoom_level=1.0):
    """ Plot ROC curve for each class onto one chart
    """
    plt.figure(dpi=150)

    # Zoom in view of the upper left corner, if zoom_level is set
    plt.xlim(0, zoom_level)
    plt.ylim(1-zoom_level, 1.05)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Plot ROC curves for each class
    for i, label in enumerate(target_names):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, 
                 label='{} (AUC: {:.2f})'.format(label, roc_auc[i]))
    #
    # Compute micro-average ROC curve and ROC area
    #
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    #
    # Compute macro-average ROC curve and ROC area
    #

    # First aggregate all false positive rates
    n_classes = y_test.shape[1]
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

    # Plot both micro and macro average ROC curves
    plt.plot(fpr["micro"], tpr["micro"],
            label='Micro-average (AUC: {0:0.2f})'
                ''.format(roc_auc["micro"]),
                linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='Macro-average (AUC: {0:0.2f})'
                ''.format(roc_auc["macro"]),
                linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="best")
    plt.title("{} ROC Curves: Breeds vs Averages".format(clf_name))
    plt.show()
    
    return fpr, tpr, roc_auc


def Compare_Multiple_ROC_Curves(y_test, y_score, zoom_level=1.0):
    """ Plot multiple ROC curves from different models for comparison
    """
    plt.figure(dpi=150)
    # Zoom in view of the upper left corner, if zoom_level is set
    plt.xlim(0, zoom_level)
    plt.ylim(1-zoom_level, 1.05)

    for clf_name in y_score.keys():
        fpr, tpr, _ = roc_curve(y_test.ravel(), y_score[clf_name].ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label="{} (AUC: {:.2f})".format(clf_name, roc_auc))

    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title("ROC Curves of different Models")
    plt.legend(loc='best')
    plt.show()


def Save_Model_Data(model, model_name, history=''):
    with open('models/' + model_name + '_model.pkl', 'wb') as f:
            pickle.dump(model, f)
            
    if history:
        with open('models/' + model_name + '_history.pkl', 'wb') as f:
            pickle.dump(history, f)        
            
        
def Load_Model_Data(model_name, neural_network=False):
    model = pickle.load(open('models/' + model_name + '_model.pkl', 'rb'))
    
    if neural_network:
        history = pickle.load(open('models/' + model_name + '_history.pkl', 'rb'))
        return model, history
    else:
        return model



if __name__ == '__main__':
    from lib.data_common import (target_names, Load_and_Split)
    from sklearn.linear_model import LogisticRegression

    X_train, y_train = Load_and_Split('data/cavy_data_train.csv', (150,150))
    X_test, y_test = Load_and_Split('data/cavy_data_test.csv', (150,150))

#    clf_list = {'log reg': LogisticRegression(multi_class='ovr', n_jobs=-1)}
    clf_list = {'log reg': OneVsRestClassifier(LogisticRegression(multi_class='ovr', n_jobs=-1), n_jobs=-1)}
    
    new_clf_list = Vanilla_ML_Run(clf_list, X_train, y_train)
 
    pass
