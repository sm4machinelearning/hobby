# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 22:17:06 2020

@author: C64990
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os


apdata = '../stock_data/AAPL.csv'
data = pd.read_csv(apdata)
data = data.dropna()

# edit column names
data.columns = [x.lower() for x in data.columns]
data = data.rename(columns = {'adj close':'adj_close'})

# edit date column to make it datetime
data['date'] = pd.to_datetime(data['date'])
data['date'] = data['date'].dt.date
data = data.sort_values('date', ascending=True)
data.reset_index(inplace=True, drop=True)

# make year column
data['year'] = [date[:4] for date in data['date'].astype(str)]

# Part 1
# plot average every year
avg_close = data.groupby('year')['close'].agg('mean')
avg_close = avg_close.sort_index()
avg_close.plot()
plt.show()

# Part 2
# plot percentage increase every year
data['close_prev'] = data['close'].shift(periods=1)
data['close_prev'] = data['close_prev'].fillna(0)
data['perc_inc'] = ((data['close'] - data['close_prev']) / data['close_prev']) * 100
data = data.replace([np.inf, -np.inf], 0)
#data = data.set_index('date')
data['perc_inc'].plot()
plt.show()

# Part 3
# trend of previous 3 days
data['trend'] = 0
for i in range(3, data.shape[0]):
    day1 = data.loc[i-3, 'perc_inc']
    day2 = data.loc[i-2, 'perc_inc']
    day3 = data.loc[i-1, 'perc_inc']
    if (day1 > 0) & (day2 > 0) & (day3 > 0):
        data.loc[i, 'trend'] = 1
    else:
        data.loc[i, 'trend'] = -1
        
trend_close = data.groupby('year')['trend'].agg('mean')
trend_close = trend_close.sort_index()
trend_close.plot()
plt.show()


# Part 4
# make linear regression model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



def train_test_split(data, partition_size=None, test_year=None):
    if partition_size:    
        rows = data.shape[0]
        rows_test = math.floor(rows * partition_size)
        rows_train = rows - rows_test
        data_train = data[:rows_train]
        data_test = data[rows_train:]
    
    if test_year:
        print (test_year)
        data_train = data.loc[data['year'] != test_year]
        data_test = data.loc[data['year'] == test_year]
    
    return data_train, data_test



def evaluate(y_true, y_pred):
    l = len(y_true)
    df = pd.DataFrame(columns=['y_true', 'y_pred', '5up', '5down'], index=np.arange(l))
    df['y_true'] = y_true
    df['y_pred'] = y_pred
    
    up = df.loc[df['y_pred'] > df['y_true']].shape[0]
    down = df.loc[df['y_pred'] < df['y_true']].shape[0]
    fine = df.loc[df['y_pred'] == df['y_true']].shape[0]

    up = np.round((up/l) * 100, 0)
    down = np.round((down/l) * 100, 0)
    fine = np.round((fine/l) * 100, 0)

    return up, down, fine



test_size = 0.30
data_train, data_test = train_test_split(data, partition_size = test_size)

#test_year = '2020'
#data_train, data_test = train_test_split(data, test_year = test_year)

X_train = data_train[['trend', 'close_prev']].to_numpy()
X_train = np.reshape(X_train, (-1, 2))
X_test = data_test[['trend', 'close_prev']].to_numpy() 
X_test = np.reshape(X_test, (-1, 2))

y_train = data_train['close'].to_numpy()
y_test = data_test['close'].to_numpy()

linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X_train, y_train)  # perform linear regression
y_pred = linear_regressor.predict(X_test)  # make predictions
print ('mse = ', mean_squared_error(y_test, y_pred))

up, down, fine = evaluate(y_test, y_pred)
print ('up percentage = ', up)
print ('down percentage = ', down)
print ('correct percentage = ', fine)

plt.plot(y_test, color='blue')
plt.plot(y_pred, color='red')
plt.show()











