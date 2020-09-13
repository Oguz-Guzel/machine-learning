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
np.random.shuffle(train_df.values)


# Building model around the data: Define an input layer of 2 neurons, a hidden layer of 4 neurons which has "reLU" activation and an output layer of 2 neurons
model = keras.Sequential([
	keras.layers.Dense(4, input_shape=(2,), activation='relu'),
	keras.layers.Dense(2, activation='sigmoid')])

# Using "adam" optimizer 
model.compile(optimizer='adam', 
	          loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	          metrics=['accuracy'])

x = np.column_stack((train_df.x.values, train_df.y.values))

model.fit(x, train_df.color.values, batch_size=4, epochs=3)

test_df = pd.read_csv('./data/test.csv')
test_x = np.column_stack((test_df.x.values, test_df.y.values))

print("EVALUATION")
# Test the model
model.evaluate(test_x, test_df.color.values)









