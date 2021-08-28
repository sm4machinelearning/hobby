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
import tensorflow as tf
import keras
import datetime
import statsmodels as st

datafile = '../stock_data/AAPL.csv'
data = pd.read_csv(datafile)
data.columns = [x.lower() for x in data.columns]
data = data.rename(columns={'adj close' : 'adj_close',
                            'date' : 'initdate'})
data['initdate'] = pd.to_datetime(data['initdate'])

'''
data['date'] = data['initdate'].dt.date
data = data.sort_values('date', ascending=True)
data.reset_index(inplace=True, drop=True)

data['year'] = [date[:4] for date in data['date'].astype(str)]
data['weeknum'] = data['initdate'].dt.week
data['year_week'] = data['year'].astype(str) + '_' + data['weeknum'].astype(str)
print (data)

def mix_sort(data, column, delimiter='_'):
    data[['year','week']] = data['year_week'].str.split('_', 1, expand=True)
    data['year'] = data['year'].astype(int)
    data['week'] = data['week'].astype(int)
    data = data.sort_values(by=['year','week'])
    data = data.drop(columns=['year','week'])
    return data

weekly_mean = data.groupby('year_week')['close'].agg('mean')
weekly_mean = weekly_mean.reset_index()
weekly_mean = mix_sort(weekly_mean, 'year_week')

weekly_mean.plot()
'''

data = data.set_index('initdate')
dataw = data.resample('w').mean()
dataw = dataw[['close']]
dataw['weekly_ret'] = np.log(dataw['close']).diff()
dataw['weekly_ret'].plot()
dataw = dataw[['weekly_ret']]
dataw = dataw.dropna()

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

rolmean = dataw.rolling(20).mean()
rolstd = dataw.rolling(20).std()

plt.figure(figsize=(12, 6))
orig = plt.plot(dataw, color='blue', label='Original')
nean = plt.plot(rolmean, color='red', label='Rolling mean')
std = plt.plot(rolstd, color='black', label='Rolling std deviation')
plt.title('Rolling mean and standard deviation')
plt.legend()
plt.show(block=False)


dftest = sm.tsa.adfuller(dataw['weekly_ret'], autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value ({0})'.format(key)] = value
print (dfoutput)



# the autocorrelation chart provides just the correlation at increasing lags
from statsmodels.graphics.tsaplots import plot_acf
fig, ax = plt.subplots(figsize=(12,5))
plot_acf(dataw.values, lags=10, ax=ax)
plt.show()


from statsmodels.graphics.tsaplots import plot_pacf
fig, ax = plt.subplots(figsize=(12,5))
plot_pacf(dataw.values, lags=10, ax=ax)
plt.show()




# Notice that you have to use udiff - the differenced data rather than the original data. 
from statsmodels.tsa.arima_model import ARMA
ar1 = ARMA(tuple(dataw.values), (3, 1)).fit()
ar1.summary()

plt.figure(figsize=(12, 8))
plt.plot(dataw.values, color='blue')
preds = ar1.fittedvalues
plt.plot(preds, color='red')
plt.show()


steps = 2
forecast = ar1.forecast(steps=steps)[0]
plt.figure(figsize=(12, 8))
plt.plot(dataw.values, color='blue')
preds = ar1.fittedvalues
plt.plot(preds, color='red')
plt.plot(pd.DataFrame(np.array([preds[-1],forecast[0]]).T,index=range(len(dataw.values)+1, len(dataw.values)+3)), color='green')
plt.plot(pd.DataFrame(forecast,index=range(len(dataw.values)+1, len(dataw.values)+1+steps)), color='green')
plt.title('Display the predictions with the ARIMA model')
plt.show()
