#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 02:27:06 2019

@author: ringoshin
"""

import numpy as np

from keras import models
from keras.models import Model, Sequential
from keras import layers
from keras.layers import (Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, 
                          Flatten, Dropout)
from keras import optimizers
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.wrappers.scikit_learn import KerasClassifier

from keras.callbacks import EarlyStopping

from sklearn.metrics import (classification_report, f1_score, confusion_matrix)

import matplotlib.pyplot as plt

import pickle


def Create_Data_Generator(X_train, y_train, X_val, y_val, batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,)
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow(X_train, y_train, 
                                         batch_size=batch_size)
    val_generator = val_datagen.flow(X_val, y_val, 
                                     batch_size=batch_size, shuffle=False)
    
    return train_generator, val_generator


def Image_CNN_VGG(target_num, input_shape=(150,150,3)):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(target_num, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics=['categorical_accuracy'])
    return model


def Image_CNN_VGG_Train(model, X_train, y_train, X_val, y_val, epochs=10, 
                        batch_size=32, verbose=0):
    print(">> Training cnn based on vgg")
    ntrain = len(X_train)
    nval = len(X_val)
    train_generator, val_generator = Create_Data_Generator(X_train, y_train, 
                                                           X_val, y_val, 
                                                           batch_size=batch_size)    
    history = model.fit_generator(train_generator, 
                                  steps_per_epoch=ntrain // batch_size,
                                  epochs=epochs,
                                  validation_data=val_generator,
                                  validation_steps=nval // batch_size,
                                  verbose=verbose)
    print()
    return model, history


def Image__CNN_From_InceptionV3(target_num, input_shape=(150,150,3)):
    # create the base pre-trained model
    base_model = InceptionV3(input_shape=input_shape, weights='imagenet', include_top=False)
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(target_num, activation='softmax')(x)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
        
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
                  metrics=['categorical_accuracy'])
    return model


def Image_CNN_From_InceptionV3_Train(model, X_train, y_train, X_val, y_val, 
                                     epochs=10, batch_size=32, verbose=0):
    print(">>> Training with pre-trained Inception v3")
    print(" >> Training model with new data")
    
    early_stopping_monitor_1 = EarlyStopping(patience=9)
    early_stopping_monitor_2 = EarlyStopping(patience=15)
    
    ntrain = len(X_train)
    nval = len(X_val)
    train_generator, val_generator = Create_Data_Generator(X_train, y_train, 
                                                           X_val, y_val, 
                                                           batch_size=batch_size)   
    
    # train the model on the new data for a few epochs
    history_1_verbose = 2 if verbose==1 else verbose  # Go for quieter verbosity
    history_1 = model.fit_generator(train_generator, 
                                   steps_per_epoch=ntrain // batch_size,
                                   epochs=epochs,
                                   validation_data=val_generator,
                                   validation_steps=nval // batch_size, 
                                   verbose=history_1_verbose,
                                   callbacks=[early_stopping_monitor_1])
#                                   verbose=0,
#                                   callbacks=[TQDM_Callback, early_stopping_monitor])
    
    print()
    
    # freeze the bottom N layers and train the remaining top layers
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:230]:
        layer.trainable = False
    for layer in model.layers[230:]:
        layer.trainable = True
    
    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    learning_rate = 0.1
    decay_rate = learning_rate / epochs
    momentum = 0.8
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

#    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), 
    model.compile(optimizer=sgd, 
                  loss='categorical_crossentropy', 
                  metrics=['categorical_accuracy'])
    
    print(" >> Re-training model to fine-tune")
    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    
    history_2 = model.fit_generator(train_generator, 
                                   steps_per_epoch=ntrain // batch_size,
                                   epochs=epochs,
                                   validation_data=val_generator,
                                   validation_steps=nval // batch_size, 
                                   verbose=verbose)
#                                   callbacks=[early_stopping_monitor_2])
#                                   verbose=0,
#                                   callbacks=[TQDM_Callback])
    
    print()    
    return model, history_1, history_2


def Image_NN_Predict(model, X_val, y_val, target_names, batch_size=32, verbose=1):
    print(">> Predicting on neural network")
    X_rescale = X_val/255.
    y_pred = model.predict(X_rescale, batch_size=batch_size, verbose=verbose)
    y_bool = np.argmax(y_pred, axis=1)
    clf_report = classification_report(y_val, y_bool, target_names=target_names)
    cf_matrix = confusion_matrix(y_val, y_bool)
    print(clf_report)
    print(cf_matrix)
    print()
    return clf_report, cf_matrix, y_pred


def Image_NN_Predict_One(model, X_val, target_names, batch_size=32, verbose=1):
    #print(">> Predicting one image on neural network")
    X_rescale = X_val/255.
    y_pred = model.predict(X_rescale.reshape((-1,150,150,3)), batch_size=batch_size, verbose=verbose)
    y_bool = int(np.argmax(y_pred, axis=1))
    #print(" > Predicted >", target_names[y_bool])
    #print(" > all proba >", y_pred)
    #print()
    return target_names[y_bool], y_pred
    

def Image_NN_Plt_Acc(history):
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    epochs = range(len(acc))

    # Plot the accuracy curves
    plt.figure(figsize=(6,6))
    plt.plot(epochs, acc,'b', linestyle='-', label='train')
    plt.plot(epochs, val_acc,'r', linestyle='-', label='val')
    plt.title('Training/Validation Categorical Accuracy')
    plt.legend()


def Image_NN_Plt_Loss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))

    # Plot the accuracy curves
    plt.figure(figsize=(6,6))
    plt.plot(epochs, loss,'b', linestyle='-', label='train')
    plt.plot(epochs, val_loss,'r', linestyle='-', label='val')
    plt.title('Training/Validation Loss')
    plt.legend()


def Image_NN_Plt_Training(history):
    acc = history.history['categorical_accuracy']
    loss = history.history['loss']
    epochs = range(len(acc))

    # Plot the accuracy curves
    plt.figure(figsize=(6,6))
    plt.plot(epochs, acc,'b', linestyle='-', label='accuracy')
    plt.plot(epochs, loss,'r', linestyle='-', label='loss')
    plt.title('Training Accuracy/Loss per Epoch')
    plt.legend()
    
    
def Image_NN_Plt_Validation(history):
    val_acc = history.history['val_categorical_accuracy']
    val_loss = history.history['val_loss']
    epochs = range(len(val_acc))

    # Plot the accuracy curves
    plt.figure(figsize=(6,6))
    plt.plot(epochs, val_acc,'b', linestyle='-', label='accuracy')
    plt.plot(epochs, val_loss,'r', linestyle='-', label='loss')
    plt.title('Validation Accuracy/Loss per Epoch')
    plt.legend()
    
    
def Save_Model_Data(model, history, X_test, y_test, model_name):
    with open('models/' + model_name + '_model.pkl', 'wb') as f:
            pickle.dump(model, f)
    
    with open('models/' + model_name + '_history.pkl', 'wb') as f:
            pickle.dump(history, f)        
            
    with open('models/' + model_name + '_x_test_data.csv', 'w') as f:
        np.savetxt(f, X_test)
    
    with open('models/' + model_name + '_y_test_data.csv', 'w') as f:
        np.savetxt(f, y_test)
        
        
def Load_Model_Data(model_name):
    model = pickle.load(open('models/' + model_name + '_model.pkl', 'rb'))
    history = pickle.load(open('models/' + model_name + '_history.pkl', 'rb'))
    X_test = np.loadtxt('models/' + model_name + '_x_test_data.csv')
    y_test = np.loadtxt('models/' + model_name + '_y_test_data.csv')
    
    return model, history, X_test, y_test
    
        
if __name__ == '__main__':
    pass
