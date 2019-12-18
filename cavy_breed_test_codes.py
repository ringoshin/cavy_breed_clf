from lib.data_common import (target_label, Read_and_Process_Image, 
                             Load_and_Split)    
                             
import cv2
import numpy as np
import pandas as pd
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

from sklearn.metrics import (classification_report, f1_score, confusion_matrix)
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     cross_validate, StratifiedKFold, KFold)
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


def Show_Sample_Images(image_list):
    for image_file in image_list:
        img = mpimg.imread(image_file)
        _ = plt.imshow(img)
        _ = plt.title(image_file)
        plt.show()


def Check_Class_Balance(y):
    ax = sns.countplot(y)
    ax.set_title("Label Count")
    plt.show()


# fix random seed for reproducibility
seed = 128
np.random.seed(seed)
image_shape = (150, 150)

target_names = ['Abyssinian', 'American', 'Skinny']
le = LabelEncoder()


dataset_path = 'data/cavy_data_raw.csv'
df_raw = pd.read_csv(dataset_path)
Check_Class_Balance(df_raw['breed'])

df_sample = df_raw['image_path'].sample(n=5, random_state=5)
Show_Sample_Images(df_sample)

sys.exit()


dataset_path = 'data/cavy_data_val.csv'
df_val = pd.read_csv(dataset_path)
df_val['image'] = df_val['image_path'].map(lambda x:
                        Read_and_Process_Image(x, image_shape))
X_val = np.array([image for image in df_val['image']])
y_val = le.fit_transform(df_val['breed'])

print(" > Preprocessing...")
dataset_path = 'data/cavy_data_train.csv'
df_train = pd.read_csv(dataset_path)
df_train['image'] = df_train['image_path'].map(lambda x:
                        Read_and_Process_Image(x, image_shape))
X_train = np.array([image for image in df_train['image']])
y_train = le.fit_transform(df_train['breed'])

clf = LogisticRegression(multi_class='ovr', n_jobs=-1)

print(" > Training...")
clf.fit(X_train, y_train)

print(" > Predicting...")
y_pred = clf.predict(X_val)
clf_report = classification_report(y_val, y_pred, target_names=target_names)
cf_matrix = confusion_matrix(y_val, y_pred)
print(clf_report)
print(cf_matrix)
print()


# Print confusion matrix for specified classifier
clf_name = "Logistic Regression"

plt.figure(dpi=150)
sns.heatmap(cf_matrix, cmap=plt.cm.Blues, annot=True, square=True,
        xticklabels=target_names, yticklabels=target_names)

plt.xlabel('Predicted breeds')
plt.ylabel('Actual breeds')
plt.title('{} Confusion Matrix'.format(clf_name))


