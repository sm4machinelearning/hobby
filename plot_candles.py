# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 15:27:51 2021

@author: C64990
"""

import numpy as np
import pandas as pd
import mplfinance as mpf
import os.path

def ema(values, period):
    values = np.array(values)
    return pd.ewma(values, span=period)[-1]

path = r'C:\Users\C64990\hobby\stock_data'
datafile = os.path.join(path, 'AAPL.csv')
data = pd.read_csv(datafile)
data['Date'] = pd.to_datetime(data['Date'])
data = data.dropna()
data = data.set_index('Date')
data = data.rename(columns={'Adj Close':'Adj_Close'})
#data['observation'] = data[['Open', 'Close', 'High', 'Low']].mean()
data['observation'] = data['Close']

from candles import *
plot_candles(data, 'observation')
plot_keltner(data, 'observation', 21)
plot_bollinger(data, 'observation', 20)



