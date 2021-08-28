# -*- cdaoding: utf-8 -*-
"""
Created on Sat Jun  6 22:41:43 2020

@author: C64990
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('snp500data.csv')
data['Year'] = [date[:4] for date in data['Date']]
data['Mean'] = (data['High'] + data['Low']) / 2

#data = data.loc[data['Year'] > '1990']
#plt.plot(data['Open'])
#plt.plot(data['Close'])
#plt.plot(data['High'])
#plt.plot(data['Low'])
#plt.plot(data['Mean'])


#datay = data.groupby(by='Year')['Volume'].agg(['sum','mean'])
#datay = datay[datay.index > '1990']
#plt.plot(datay['mean'])
#plt.plot(datay['sum'])
#plt.grid(True)

datavg = data.groupby(by='Year')['Mean'].agg(['sum', 'mean'])
datavg = datavg[datavg.index > '2010']
plt.plot(datavg['mean'])
#plt.plot(datavg['sum'])
plt.grid(True)

