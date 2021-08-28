# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 05:58:45 2021

@author: C64990
"""
import datetime
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'
#pio.renderers.default = 'svg'


def plot_candles(data, variable):
    #swins = [50, 100, 200]
    #ewins = [8, 21, 34, 55, 89]
    
    swins = [50, 100]
    for win in swins:
        data['sma_' + str(win) + '_' + str(variable)] = data[variable].rolling(window=win).mean()
    
    ewins = [8, 21, 34]
    for win in ewins:
        data['ema_' + str(win) + '_' + str(variable)] = data[variable].ewm(span=win,adjust=False).mean()
    
    colordict = {}
    colordict['sma_50_' + str(variable)] = 'rgb(101,67,33)'
    colordict['sma_100_' + str(variable)] = 'rgb(134,136,138)'
    colordict['sma_200_' + str(variable)] = 'rgb(0,0,0)'
    colordict['ema_8_' + str(variable)] = 'rgb(0,0,255)'
    colordict['ema_21_' + str(variable)] = 'rgb(0,255,0)'
    colordict['ema_34_' + str(variable)] = 'rgb(255,255,0)'
    colordict['ema_55_' + str(variable)] = 'rgb(255,127,0)'
    colordict['ema_89_' + str(variable)] = 'rgb(255,0,0)'

    today = datetime.date.today()
    dd = datetime.timedelta(days=120)
    prev_date = today - dd

    data = data.loc[data.index > str(prev_date)]
    data['weekday'] = data.index.dayofweek
    mondays = data.loc[data['weekday'] == 0].index
    fridays = data.loc[data['weekday'] == 4].index
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close']))
    for win in swins:
        fig.add_trace(go.Scatter(x=data.index,
                                 y=data['sma_' + str(win) + '_' + str(variable)],
                                 name='sma_' + str(win) + '_' + str(variable),
                                 line=dict(color=colordict['sma_' + str(win) + '_' + str(variable)], width=3)))
    
    for win in ewins:
        fig.add_trace(go.Scatter(x=data.index,
                                 y=data['ema_' + str(win) + '_' + str(variable)],
                                 name='ema_' + str(win) + '_' + str(variable),
                                 line=dict(color=colordict['ema_' + str(win) + '_' + str(variable)], width=3)))
    
    fig.show()
    
    
def plot_keltner(data, variable, num_periods):
    
    data['ema_' + str(num_periods) + '_' + str(variable)] = data[variable].ewm(span = num_periods, adjust = False).mean()
    data['tr1'] = data['High'] - data['Low']
    data['tr2'] = data['High'] - data['Close'].shift(1)
    data['tr3'] = data['Low'] - data['Close'].shift(1)
    data['tr'] = data[['tr1','tr2','tr3']].max(axis=1)
    data['atr'] = data['tr'].rolling(num_periods).mean()
    data['atr'].iloc[num_periods:] = 0
    for i in range(num_periods, data.shape[0]):
        data['atr'].iloc[i] = ((data['atr'].iloc[i-1] * (num_periods - 1)) + data['tr'].iloc[i]) / num_periods
    
    data['ema_p1'] = data['ema_' + str(num_periods) + '_' + str(variable)] + data['atr']
    data['ema_m1'] = data['ema_' + str(num_periods) + '_' + str(variable)] - data['atr']

    data['ema_p2'] = data['ema_' + str(num_periods) + '_' + str(variable)] + (2 * data['atr'])
    data['ema_m2'] = data['ema_' + str(num_periods) + '_' + str(variable)] - (2 * data['atr'])

    data['ema_p3'] = data['ema_' + str(num_periods) + '_' + str(variable)] + (3 * data['atr'])
    data['ema_m3'] = data['ema_' + str(num_periods) + '_' + str(variable)] - (3 * data['atr'])

    colordict = {}
    colordict['ema_' + str(num_periods) + '_' + str(variable)] = 'rgb(0,0,0)'
    colordict['ema_p1'] = 'rgb(0,0,255)'
    colordict['ema_m1'] = 'rgb(0,0,255)'
    colordict['ema_p2'] = 'rgb(0,255,0)'
    colordict['ema_m2'] = 'rgb(0,255,0)'
    colordict['ema_p3'] = 'rgb(255,0,0)'
    colordict['ema_m3'] = 'rgb(255,0,0)'

    today = datetime.date.today()
    dd = datetime.timedelta(days=120)
    prev_date = today - dd

    data = data.loc[data.index > str(prev_date)]
    data['weekday'] = data.index.dayofweek
    mondays = data.loc[data['weekday'] == 0].index
    fridays = data.loc[data['weekday'] == 4].index
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close']))

    fig.add_trace(go.Scatter(x=data.index,
                             y=data['ema_' + str(num_periods) + '_' + str(variable)],
                             name='ema_' + str(num_periods) + '_' + str(variable),
                             line=dict(color=colordict['ema_' + str(num_periods) + '_' + str(variable)], width=1, dash='dash')))
    
    ems = ['ema_p1','ema_m1','ema_p2','ema_m2','ema_p3','ema_m3']
    for em in ems:
        fig.add_trace(go.Scatter(x=data.index,
                                 y=data[em],
                                 name=em,
                                 line=dict(color=colordict[em], width=1)))
        
    fig.show()
    

def plot_bollinger(data, variable, num_periods):
    sma = data[variable].rolling(window=num_periods).mean()
    std = data[variable].rolling(window=num_periods).std()
    
    data['sma_' + str(num_periods) + '_' + str(variable)] = sma
    data['std_' + str(num_periods) + '_' + str(variable)] = std
    
    data['sma_bband_p1'] = sma + std
    data['sma_bband_m1'] = sma - std
    
    data['sma_bband_p2'] = sma + 2*std
    data['sma_bband_m2'] = sma - 2*std
    
    data['sma_bband_p3'] = sma + 3*std
    data['sma_bband_m3'] = sma - 3*std
    
    colordict = {}
    colordict['sma_' + str(num_periods) + '_' + str(variable)] = 'rgb(0,0,0)'
    colordict['sma_bband_p1'] = 'rgb(0,0,255)'
    colordict['sma_bband_m1'] = 'rgb(0,0,255)'
    colordict['sma_bband_p2'] = 'rgb(0,255,0)'
    colordict['sma_bband_m2'] = 'rgb(0,255,0)'
    colordict['sma_bband_p3'] = 'rgb(255,0,0)'
    colordict['sma_bband_m3'] = 'rgb(255,0,0)'

    today = datetime.date.today()
    dd = datetime.timedelta(days=120)
    prev_date = today - dd

    data = data.loc[data.index > str(prev_date)]
    data['weekday'] = data.index.dayofweek
    mondays = data.loc[data['weekday'] == 0].index
    fridays = data.loc[data['weekday'] == 4].index
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close']))

    fig.add_trace(go.Scatter(x=data.index,
                             y=data['sma_' + str(num_periods) + '_' + str(variable)],
                             name='sma_' + str(num_periods) + '_' + str(variable),
                             line=dict(color=colordict['sma_' + str(num_periods) + '_' + str(variable)], width=1, dash='dash')))
    
    bands = ['sma_bband_p1', 'sma_bband_m1', 'sma_bband_p2', 'sma_bband_m2', 
           'sma_bband_p3', 'sma_bband_m3']
    for band in bands:
        fig.add_trace(go.Scatter(x=data.index,
                                 y=data[band],
                                 name=band,
                                 line=dict(color=colordict[band], width=1)))
    fig.show()
    
    
    





    
    
        
'''
mpf.plot(data,type='candle',mav=(3,6,9),volume=True,show_nontrading=True)
mpf.plot(data,type='renko')
data.columns = [x.lower() for x in data.columns]
data = data.rename(columns={'adj close' : 'adj_close',
                            'date' : 'initdate'})
data['initdate'] = pd.to_datetime(data['initdate'])
data['date'] = data['initdate'].dt.date
data = data.sort_values('date', ascending=True)
data.reset_index(inplace=True, drop=True)
print (data)
import pandas as pd
import numpy as np
def ema(values, period):
    values = np.array(values)
    return pd.ewma(values, span=period)[-1]
values = [9, 5, 10, 16, 5]
period = 5
'''