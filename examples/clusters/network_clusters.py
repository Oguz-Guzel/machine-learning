#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 23:48:57 2020

@author: oguzguzel
"""

import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np

# Load the data and shuffle each time
train_df = pd.read_csv('./data/train.csv')
color_dict = {'red': 0, 'blue': 1, 'green': 2, 'teal': 3, 'orange': 4, 'purple': 5}
train_df['color'] = train_df.color.apply(lambda x: color_dict[x])
np.random.shuffle(train_df.values)

print(train_df.head())
print(train_df.color.unique())

# Building model around the data: Define an input layer of 2 neurons, a hidden layer of 4 neurons which has "reLU" activation and an output layer of 2 neurons
model = keras.Sequential([
	keras.layers.Dense(32, input_shape=(2,), activation='relu'),
    keras.layers.Dense(32,  activation='relu'),
    keras.layers.Dropout(0.2),
	keras.layers.Dense(6, activation='sigmoid')])

# Using "adam" optimizer 
model.compile(optimizer='adam', 
	          loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	          metrics=['accuracy'])

x = np.column_stack((train_df.x.values, train_df.y.values))

model.fit(x, train_df.color.values, batch_size=4, epochs=10)

test_df = pd.read_csv('./data/test.csv')
test_x = np.column_stack((test_df.x.values, test_df.y.values))

# Test the model
print("EVALUATION")
test_df['color'] = test_df.color.apply(lambda x: color_dict[x])
model.evaluate(test_x, test_df.color.values)

print(np.round(model.predict(np.array([[0,1]]))))

