#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 23:48:57 2020

@author: oguzguzel
"""

from tensorflow import keras

import pandas as pd
import numpy as np

# Load the data and shuffle each time
train_df = pd.read_csv('./data/train.csv')
one_hot_color = pd.get_dummies(train_df.color).values
one_hot_marker = pd.get_dummies(train_df.marker).values

labels = np.concatenate((one_hot_color, one_hot_marker), axis=1)

# Building model around the data: Define an input layer of 2 neurons, a hidden layer of 4 neurons which has "reLU" activation and an output layer of 2 neurons
model = keras.Sequential([
	keras.layers.Dense(64, input_shape=(2,), activation='relu'),
    keras.layers.Dense(64, activation='relu'),
	keras.layers.Dense(9, activation='sigmoid')])

# Using "adam" optimizer, BinaryCrossentropy lets us to have multiple things to be 1, meaning that we have more than one label
model.compile(optimizer='adam', 
	          loss=keras.losses.BinaryCrossentropy(from_logits=False),
	          metrics=['accuracy'])

x = np.column_stack((train_df.x.values, train_df.y.values))

# np.random.RandomState(seed=42).shuffle(x)
# np.random.RandomState(seed=42).shuffle(labels)


model.fit(x, labels, batch_size=32, epochs=20)

test_df = pd.read_csv('./data/test.csv')
one_hot_color_test = pd.get_dummies(test_df.color).values
one_hot_marker_test = pd.get_dummies(test_df.marker).values

test_labels = np.concatenate((one_hot_color_test, one_hot_marker_test), axis =1)

test_x = np.column_stack((test_df.x.values, test_df.y.values))


# Test the model
print("EVALUATION")
model.evaluate(test_x, test_df.color.values)

print(np.round(model.predict(np.array([[0,3]]))))
