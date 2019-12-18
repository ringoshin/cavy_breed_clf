"""
Created on Sun 8 Aug 2019

@author: ringoshin

One-time removal of underscores from some of the image extensions
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split


def Save_Data(X, y, filename):
    """
    Saving dataset variants raw, training, test, validation)
    """
    df = pd.DataFrame({'image_path': X, 'breed': y})
    return df.to_csv(filename, index=False)


seed = 42
image_folders = ['American', 'Abyssinian', 'Skinny']
image_path = 'images'
X, y = [], []

print(" > Creating datasets from images...")
for folder_name in image_folders:
    image_list_path = os.path.join(image_path, folder_name)
    for path, dirs, files in os.walk(image_list_path):
        for filename in files:
            X.append(os.path.join(path,filename))
            y.append(folder_name)

# Split for the test dataset
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, 
                                                            test_size=0.2, 
                                                            shuffle=True, 
                                                            stratify= y, 
                                                            random_state=seed)

# Split for the training and validation datasets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, 
                                                  y_train_val, 
                                                  test_size=0.2, 
                                                  shuffle=True, 
                                                  stratify= y_train_val, 
                                                  random_state=seed)

Save_Data(X, y, 'data/cavy_data_raw.csv')
Save_Data(X_train, y_train, 'data/cavy_data_train.csv')
Save_Data(X_val, y_val, 'data/cavy_data_val.csv')
Save_Data(X_test, y_test, 'data/cavy_data_test.csv')

print(" > done!")