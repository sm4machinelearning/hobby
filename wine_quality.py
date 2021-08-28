# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 11:08:44 2021

@author: C64990
"""

'''
Objective:
In this lab, you will use the Keras Sequential API to create a classification model. 
You will learn how to use the tf.data API for creating input pipelines and 
use feature columns to prepare the data to be consumed by a neural network. 

Lab Scope:
This lab does not cover how to make predictions on the model or deploy it to 
Cloud AI Platform.

Learning objectives:
    Apply techniques to clean and inspect data.
    Split dataset into training, validation and test datasets.
    Use the tf.data.Dataset to create an input pipeline.
    Use feature columns to prepare the data to be tained by a neural network.
    Define, compile and train a model using the Keras Sequential API.

In a classification problem, we aim to select the output from a limited 
set of discrete values, like a category or a class. Contrast this with a 
regression problem, where we aim to predict a value from a continuos range 
of values.

This notebook uses the Wine Production Quality Dataset and builds a model to 
predict the production quality of wine given a set of attributes such as 
its citric acidity, density, and others. 

To do this, we'll provide the model with examples of different wines produced, 
that received a rating from an evaluator. The ratings are provided by the 
numbers 0 - 10 (0 being of very poor quality and 10 being of great quality). 
We will then try and use this model to predict the rate a new wine will 
receive by infering towards the trained model.

Since we are learning how to use the Tensorflow 2.x API, this example uses the 
tf.keras API. Please see this guide for details.

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path

datafile = r'C:\Users\C64990\hobby\stock_data\winequality-white.csv'
data = pd.read_csv(datafile, sep=';')
#data = data[:50]

for col in data.columns:
    if ' ' in col:
        data = data.rename(columns={col : col.replace(' ', '_')})

print (data.describe().T.to_string())

import seaborn as sns
sns.pairplot(data[["quality", "citric_acid", "residual_sugar", "alcohol"]], diag_kind="kde")


'''
--- Some considerations ---
Did you notice anything when looking at the stats table?
One useful piece of information we can get from those are, for example, 
min and max values. 

This allows us to understand ranges in which these features fall in.
Based on the description of the dataset and the task we are trying to achieve, 
do you see any issues with the examples we have available to train on?

Did you notice that the ratings on the dataset range from 3 to 9? In this 
dataset, there is no wine rated with a 10 or a 0 - 2 rating. This will likely 
produce a poor model that is not able to generalize well to examples of 
fantastic tasting wine (nor to the ones that taste pretty bad!). 
One way to fix this is to make sure your dataset represents all possible 
classes well. 

Another analysis, that we do not do on this exercise, is check 
if the data is balanced. Having a balanced dataset produces fair model, 
and that is always a good thing!

Split the data into train, validation and test¶
Now split the dataset into a training, validation, and test set.
Test sets are used for a final evaluation of the trained model.
There are more sophisticated ways to make sure that your splitting 
methods are repeatable. Ideally, the sets would always be the same 
after splitting to avoid randomic results, which makes experimentation difficult.

'''


from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.2, random_state=100)
train, val = train_test_split(train, test_size=0.2, random_state=100)


'''

Use the tf.data.Dataset
The tf.data.Dataset allows for writing descriptive and efficient input pipelines. 

Dataset usage follows a common pattern:
    Create a source dataset from your input data.
    Apply dataset transformations to preprocess the data.
    Iterate over the dataset and process the elements.
    Iteration happens in a streaming fashion, so the full dataset 
    does not need to fit into memory.

The df_to_dataset method below creates a dataset object from a pandas dataframe.

'''


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def df_to_dataset(dataframe, epochs=10, shuffle=True, batch_size=5):
    
    dataframe = dataframe.copy()
    
    #extracting the column which contains the training label
    labels = tf.keras.utils.to_categorical(dataframe.pop('quality'), 
                                           num_classes=11) 
    
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), 
                                             labels))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.repeat(epochs).batch(batch_size)
        
    return ds


train_ds = df_to_dataset(train)
val_ds = df_to_dataset(val)#, shuffle=False)
test_ds = df_to_dataset(test)#, shuffle=False)


for feature_batch, label_batch in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of citric acid:', feature_batch['citric_acid'])
    print('A batch of quality:', label_batch )



'''

Create feature columns
TensorFlow provides many types of feature columns. In this exercise, 
all the feature columns are of type numeric. If there were any text or 
categorical values, transformations would need to take place to make 
the input all numeric.

However, you often don't want to feed a number directly into the model, 
but instead split its value into different categories based on numerical 
ranges. To do this, use the bucketized_column method of feature columns. 
This allows for the network to represent discretized dense input bucketed 
by boundaries. Feature columns are the object type used to create 
feature layers, which we will feed to the Keras model.
'''


def categorize(data):
    desc = data.describe()
    minv, maxv = desc['min'], desc['max']
    meanv, stdv = desc['mean'], desc['std']
    minv, maxv = minv - stdv, maxv + stdv
    rang = list(np.sort(np.linspace(minv, maxv, 7)))
    return rang

from tensorflow import feature_column
feature_columns = {}
data_feat = data.drop(columns=['quality'])
for colname in data_feat.columns:
    bounds = categorize(data_feat[colname])
    col = tf.feature_column.numeric_column(colname)
    feature_columns['bucketized_' + str(colname)] = tf.feature_column.bucketized_column(col, boundaries=bounds)



'''
Define, compile and train the Keras model
We will be using the Keras Sequential API to create the logistic regression 
model for the classification of the wine quality.
The model will be composed of the input layer (feature_layer created above), 
a single dense layer with two neural nodes, and the output layer, which will 
allow the model to predict the rating (1 - 10) of each instance being inferred.
When compiling the model, we define a loss function, an optimizer and which 
metrics to use to evaluate the model. CategoricalCrossentropy is a type of 
loss used in classification tasks. Losses are a mathematical way of measuring 
how wrong the model predictions are.
Optimizers tie together the loss function and model parameters by updating the 
model in response to the output of the loss function. In simpler terms, 
optimizers shape and mold your model into its most accurate possible form by 
playing with the weights. The loss function is the guide to the terrain, 
telling the optimizer when it’s moving in the right or wrong direction. 
We will use Adam as our optimizer for this exercise. Adam is an optimization 
algorithm that can be used instead of the classical stochastic gradient 
descent procedure to update network weights iterative based in training data.
There are many types of optimizers one can chose from. Ideally, when creating 
an ML model, try and identify an optimizer that has been empirically 
adopted on similar tasks.
'''


# Create a feature layer from the feature columns
feature_layer = tf.keras.layers.DenseFeatures(list(feature_columns.values()))


model = tf.keras.Sequential([feature_layer, 
                             layers.Dense(8, activation='relu'),
                             layers.Dense(8, activation='relu'),
                             layers.Dense(11, activation='softmax')])


model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy', 'mse'])

history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=5)

from matplotlib import pyplot
pyplot.plot(history.history['mse'])
pyplot.title('MSE')
pyplot.show()
pyplot.clf()
pyplot.plot(history.history['accuracy'])
pyplot.title('Acuuracy')
pyplot.show()
pyplot.clf()


'''
Conclusion
This notebook introduced a few concepts to handle a classification problem 
with Keras Sequential API.
We looked at some techniques to clean and inspect data.
We split the dataset into training, validation and test datasets.
We used the tf.data.Dataset to create an input pipeline.
We went over some basics on loss and optimizers.
We covered the steps to define, compile and train a model using the Keras 
Sequential API.
'''



