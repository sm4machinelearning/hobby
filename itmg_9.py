# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 22:17:06 2020

@author: C64990
"""
import scipy
import sklearn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os


""""""""""""""""""""
# Part 1
""""""""""""""""""""
#from backtester.dataSource.csv_data_source import CsvDataSource
datafile = '../stock_data/MQK_train.csv'
data = pd.read_csv(datafile)
data = data.rename(columns={'Unnamed: 0':'ticker'})
data_train = data.set_index('ticker')

datafile = '../stock_data/MQK_valid.csv'
data = pd.read_csv(datafile)
data = data.rename(columns={'Unnamed: 0':'ticker'})
data_valid = data.set_index('ticker')

datafile = '../stock_data/MQK_test.csv'
data = pd.read_csv(datafile)
data = data.rename(columns={'Unnamed: 0':'ticker'})
data_test = data.set_index('ticker')

""""""""""""""""""""
# Part 2
#Loading our data
""""""""""""""""""""
def loadData(data):
    data['Stock Price'] =  data['stockTopBidPrice'] +\
                           data['stockTopAskPrice'] / 2.0
    data['Future Price'] = data['futureTopBidPrice'] +\
                           data['futureTopAskPrice'] / 2.0
    # basis after 5 min is the target now
    data['Y(Target)'] = data['basis'].shift(-5) 
    del data['benchmark_score']
    del data['FairValue']
    return data

data_train = loadData(data_train)

""""""""""""""""""""
# Part 3
""""""""""""""""""""
def prepareData(data, period):
    data['Y(Target)'] = data['basis'].rolling(period).mean().shift(-period)
    if 'FairValue' in data.columns:
        del data['FairValue']
    data.dropna(inplace=True)
    return data

period = 5
data_train = prepareData(data_train, period)
data_valid = prepareData(data_valid, period)
data_test = prepareData(data_test, period)
backtestdata = data_test.copy()

""""""""""""""""""""
# Part 4
""""""""""""""""""""
def difference(dataDf, period):
    return dataDf.sub(dataDf.shift(period), fill_value=0)

def ewm(dataDf, halflife):
    return dataDf.ewm(halflife=halflife,ignore_na=False,min_periods=0,adjust=True).mean()

def rsi(data, period):
    data_upside = data.sub(data.shift(1), fill_value=0)
    data_downside = data_upside.copy()
    data_downside[data_upside > 0] = 0
    data_upside[data_upside < 0] = 0
    avg_upside = data_upside.rolling(period).mean()
    avg_downside = - data_downside.rolling(period).mean()
    rsi = 100 - (100 * avg_downside / (avg_downside + avg_upside))
    rsi[avg_downside == 0] = 100
    rsi[(avg_downside == 0) & (avg_upside == 0)] = 0

    return rsi

def create_features(data):
    cols = ['emabasis3','emabasis5','emabasis2','emabasis7','emabasis10','emabasis4',
            'rsi15','rsi10','rsi5',
            'mom1','mom10','mom3','mom5']
    
    basis_X = pd.DataFrame(index = data.index, columns =  cols)
    
    basis_X['mom1'] = difference(data['basis'],2)
    basis_X['mom3'] = difference(data['basis'],4)
    basis_X['mom5'] = difference(data['basis'],6)
    basis_X['mom10'] = difference(data['basis'],11)

    basis_X['rsi15'] = rsi(data['basis'],15)
    basis_X['rsi10'] = rsi(data['basis'],10)
    basis_X['rsi5'] = rsi(data['basis'],5)
    
    basis_X['emabasis2'] = ewm(data['basis'],2)
    basis_X['emabasis3'] = ewm(data['basis'],3)
    basis_X['emabasis4'] = ewm(data['basis'],4)
    basis_X['emabasis5'] = ewm(data['basis'],5)
    basis_X['emabasis7'] = ewm(data['basis'],7)
    basis_X['emabasis10'] = ewm(data['basis'],10)

    basis_X['basis'] = data['basis']
    basis_X['vwapbasis'] = data['stockVWAP']-data['futureVWAP']
    
    basis_X['swidth'] = data['stockTopAskPrice']-data['stockTopBidPrice']
    basis_X['fwidth'] = data['futureTopAskPrice']-data['futureTopBidPrice']
    
    basis_X['btopask'] = data['stockTopAskPrice']-data['futureTopAskPrice']
    basis_X['btopbid'] =data['stockTopBidPrice']-data['futureTopBidPrice']
    basis_X['bavgask'] = data['stockAverageAskPrice']-data['futureAverageAskPrice']
    basis_X['bavgbid'] = data['stockAverageBidPrice']-data['futureAverageBidPrice']
    basis_X['bnextask'] = data['stockNextAskPrice']-data['futureNextAskPrice']
    basis_X['bnextbid'] = data['stockNextBidPrice']-data['futureNextBidPrice']
    basis_X['topaskvolratio'] = data['stockTopAskVol']/data['futureTopAskVol']
    basis_X['topbidvolratio'] = data['stockTopBidVol']/data['futureTopBidVol']
    basis_X['totalaskvolratio'] = data['stockTotalAskVol']/data['futureTotalAskVol']
    basis_X['totalbidvolratio'] = data['stockTotalBidVol']/data['futureTotalBidVol']
    basis_X['nextbidvolratio'] = data['stockNextBidVol']/data['futureNextBidVol']
    basis_X['nextaskvolratio'] = data['stockNextAskVol']-data['futureNextAskVol']
    
    basis_X['emabasisdi4'] = basis_X['emabasis7'] - basis_X['emabasis5'] + basis_X['emabasis2']
    basis_X['emabasisdi7'] = basis_X['emabasis7'] - basis_X['emabasis5']+ basis_X['emabasis3']
    basis_X['emabasisdi1'] = basis_X['emabasis10'] - basis_X['emabasis5'] + basis_X['emabasis3']
    basis_X['emabasisdi3'] = basis_X['emabasis10'] - basis_X['emabasis3']+ basis_X['emabasis5']
    basis_X['emabasisdi5'] = basis_X['emabasis7']- basis_X['emabasis5'] + data['basis']
    basis_X['emabasisdi'] = basis_X['emabasis5'] - basis_X['emabasis3'] + data['basis']
    basis_X['emabasisdi6'] = basis_X['emabasis7'] - basis_X['emabasis3']+ data['basis']
    basis_X['emabasisdi2'] = basis_X['emabasis10'] - basis_X['emabasis5']+ data['basis']
    basis_X['emabasisdi3'] = basis_X['emabasis10'] - basis_X['emabasis3']+ basis_X['emabasis5']
    
    basis_X = basis_X.fillna(0)
    
    basis_y = data['Y(Target)']
    basis_y.dropna(inplace=True)
    
    print("Any null data in y: %s, X: %s"%(basis_y.isnull().values.any(), basis_X.isnull().values.any()))
    print("Length y: %s, X: %s"%(len(basis_y.index), len(basis_X.index)))
    
    return basis_X, basis_y

basis_X_train, basis_y_train = create_features(data_train)
basis_X_test, basis_y_test = create_features(data_valid)


""""""""""""""""""""
# Part 5
# Linear regression
""""""""""""""""""""
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

def linear_regression(basis_X_train, basis_y_train, basis_X_test,basis_y_test):
    
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(basis_X_train, basis_y_train)
    # Make predictions using the testing set
    basis_y_pred = regr.predict(basis_X_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(basis_y_test, basis_y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(basis_y_test, basis_y_pred))

    # Plot outputs
    plt.scatter(basis_y_pred, basis_y_test,  color='black')
    plt.plot(basis_y_test, basis_y_test, color='blue', linewidth=3)

    plt.xlabel('Y(actual)')
    plt.ylabel('Y(Predicted)')

    plt.show()
    
    return regr, basis_y_pred


_, basis_y_pred = linear_regression(basis_X_train, basis_y_train, basis_X_test,basis_y_test)

""""""""""""""""""""
# Part 6
# linear regression with normalized basis
""""""""""""""""""""
def normalize(basis_X, basis_y, period):
    basis_X_norm = (basis_X - basis_X.rolling(period).mean())/basis_X.rolling(period).std()
    basis_X_norm.dropna(inplace=True)
    basis_y_norm = (basis_y - basis_X['basis'].rolling(period).mean())/basis_X['basis'].rolling(period).std()
    basis_y_norm = basis_y_norm[basis_X_norm.index]
    
    return basis_X_norm, basis_y_norm

norm_period = 375

basis_X_norm_train, basis_y_norm_train = normalize(basis_X_train, 
                                                   basis_y_train, 
                                                   norm_period)

basis_X_norm_test, basis_y_norm_test = normalize(basis_X_test, 
                                                 basis_y_test, 
                                                 norm_period)

regr_norm, basis_y_pred = linear_regression(basis_X_norm_train, 
                                            basis_y_norm_train, 
                                            basis_X_norm_test, 
                                            basis_y_norm_test)


basis_y_pred = basis_y_pred * basis_X_test['basis'].rolling(period).std()[basis_y_norm_test.index] + basis_X_test['basis'].rolling(period).mean()[basis_y_norm_test.index]

print("Mean squared error: %.2f"
      % mean_squared_error(basis_y_test[basis_y_norm_test.index], basis_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(basis_y_test[basis_y_norm_test.index], basis_y_pred))

# Plot outputs
plt.scatter(basis_y_pred, basis_y_test[basis_y_norm_test.index],  color='black')
plt.plot(basis_y_test, basis_y_test, color='blue', linewidth=3)

plt.xlabel('Y(actual)')
plt.ylabel('Y(Predicted)')


for i in range(len(basis_X_train.columns)):
    print('%.4f, %s'%(regr_norm.coef_[i], basis_X_train.columns[i]))


""""""""""""""""""""
# Part 7
# observe predictive features by seeing correlation
""""""""""""""""""""
import seaborn

c = basis_X_train.corr()
plt.figure(figsize=(10,10))
seaborn.heatmap(c, cmap='RdYlGn_r', mask = (np.abs(c) <= 0.8))
plt.show()


""""""""""""""""""""
# Part 8
# create features again and observe prediction using linear regression
""""""""""""""""""""

def create_features_again(data):
    basis_X = pd.DataFrame(index = data.index, columns =  [])
    
    basis_X['mom10'] = difference(data['basis'],11)
    
    basis_X['emabasis2'] = ewm(data['basis'],2)
    basis_X['emabasis5'] = ewm(data['basis'],5)
    basis_X['emabasis10'] = ewm(data['basis'],10)

    basis_X['basis'] = data['basis']

    basis_X['totalaskvolratio'] = (data['stockTotalAskVol']-data['futureTotalAskVol'])/100000
    basis_X['totalbidvolratio'] = (data['stockTotalBidVol']-data['futureTotalBidVol'])/100000
    
    basis_X = basis_X.fillna(0)
    
    basis_y = data['Y(Target)']
    basis_y.dropna(inplace=True)
    
    print("Any null data in y: %s, X: %s"%(basis_y.isnull().values.any(), basis_X.isnull().values.any()))
    print("Length y: %s, X: %s"%(len(basis_y.index), len(basis_X.index)))
    
    return basis_X, basis_y

basis_X_train, basis_y_train = create_features_again(data_train)
basis_X_test, basis_y_test = create_features_again(data_valid)

norm_period = 375

basis_X_norm_train, basis_y_norm_train = normalize(basis_X_train, 
                                                   basis_y_train, 
                                                   norm_period)

basis_X_norm_test, basis_y_norm_test = normalize(basis_X_test,
                                                 basis_y_test, 
                                                 norm_period)

regr_norm, basis_y_pred = linear_regression(basis_X_norm_train,
                                            basis_y_norm_train,
                                            basis_X_norm_test,
                                            basis_y_norm_test)


basis_y_pred = basis_y_pred * basis_X_test['basis'].rolling(period).std()[basis_y_norm_test.index] + basis_X_test['basis'].rolling(period).mean()[basis_y_norm_test.index]

print("Mean squared error: %.2f"
      % mean_squared_error(basis_y_test[basis_y_norm_test.index], basis_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(basis_y_test[basis_y_norm_test.index], basis_y_pred))

# Plot outputs
plt.scatter(basis_y_pred, basis_y_test[basis_y_norm_test.index],  color='black')
plt.plot(basis_y_test, basis_y_test, color='blue', linewidth=3)

plt.xlabel('Y(actual)')
plt.ylabel('Y(Predicted)')

plt.show()

for i in range(len(basis_X_train.columns)):
    print(regr_norm.coef_[i], basis_X_train.columns[i])
    
_, basis_y_pred = linear_regression(basis_X_train, basis_y_train, basis_X_test,basis_y_test)
basis_y_regr = basis_y_pred.copy()

linreg_coeff = _.coef_

'''
""""""""""""""""""""
# Part 9
# Knearest neighbors method
""""""""""""""""""""
from sklearn import neighbors
n_neighbors = 5

model = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
model.fit(basis_X_train, basis_y_train)
basis_y_pred = model.predict(basis_X_test)
print (model)
knn_coeff = model.get_params(deep=True)

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(basis_y_test, basis_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(basis_y_test, basis_y_pred))

# Plot outputs
plt.scatter(basis_y_pred, basis_y_test,  color='black')
plt.plot(basis_y_test, basis_y_test, color='blue', linewidth=3)

plt.xlabel('Y(actual)')
plt.ylabel('Y(Predicted)')

plt.show()

basis_y_knn = basis_y_pred.copy()


""""""""""""""""""""
# Part 10
# support vector machines
""""""""""""""""""""
from sklearn.svm import SVR

model = SVR(kernel='rbf', C=1e3, gamma=0.1)

model.fit(basis_X_train, basis_y_train)
basis_y_pred = model.predict(basis_X_test)
svr_coeff = model.coef_

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(basis_y_test, basis_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(basis_y_test, basis_y_pred))

# Plot outputs
plt.scatter(basis_y_pred, basis_y_test,  color='black')
plt.plot(basis_y_test, basis_y_test, color='blue', linewidth=3)


plt.xlabel('Y(actual)')
plt.ylabel('Y(Predicted)')

plt.show()

basis_y_svr = basis_y_pred.copy()


""""""""""""""""""""
# Part 11
# Decision Trees
""""""""""""""""""""
from sklearn import ensemble
model=ensemble.ExtraTreesRegressor()
model.fit(basis_X_train, basis_y_train)
basis_y_pred = model.predict(basis_X_test)
dtree_coeff = model.coef_

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(basis_y_test, basis_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(basis_y_test, basis_y_pred))

# Plot outputs
plt.scatter(basis_y_pred, basis_y_test,  color='black')
plt.plot(basis_y_test, basis_y_test, color='blue', linewidth=3)


plt.xlabel('Y(actual)')
plt.ylabel('Y(Predicted)')

plt.show()

basis_y_trees = basis_y_pred.copy()



""""""""""""""""""""
# Part 12
# ensembel method
""""""""""""""""""""
basis_y_pred_ensemble = (basis_y_trees + basis_y_svr + +basis_y_knn + basis_y_regr)/4
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(basis_y_test, basis_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(basis_y_test, basis_y_pred))

# Plot outputs
plt.scatter(basis_y_pred, basis_y_test,  color='black')
plt.plot(basis_y_test, basis_y_test, color='blue', linewidth=3)

plt.xlabel('Y(actual)')
plt.ylabel('Y(Predicted)')

plt.show()
'''

""""""""""""""""""""
# Part 13
# use backtester
""""""""""""""""""""
basis_X_test, basis_y_test = create_features_again(backtestdata)
y_pred = pd.DataFrame(index=basis_y_test.index, columns=['y_true'])
y_pred['y_true'] = basis_y_test.values

param_dict = {'basis_y_regr': linreg_coeff,}
#              'basis_y_knn': knn_coeff,
#              'basis_y_svr': svr_coeff,
#              'basis_y_trees': dtree_coeff}
model_types = ['basis_y_regr']#, 'basis_y_knn', 'basis_y_svr', 'basis_y_trees']

for model in model_types:
    test_eval = basis_X_test.copy()
    params = param_dict[model]
    variables = ['mom10', 'emabasis2', 'emabasis5', 'emabasis10', 
                 'basis', 'totalaskvolratio', 'totalbidvolratio']
    for i in range(len(variables)):
        test_eval[variables[i]] = test_eval[variables[i]] * params[i]
    y_pred[model] = test_eval.sum(axis=1)






#from smbktr import *
#
#if updateCheck():
#        print('Your version of the auquan toolbox package is old.\
#              Please update by running the following command:')
#        print('pip install -U auquan_toolbox')
#else:
#    tsParams = MyTradingParams(basis_X_test)
##     import pdb;pdb.set_trace()
#    tradingSystem = TradingSystem(tsParams)
#    
#    results = tradingSystem.startTrading(onlyAnalyze=False,
#                                         shouldPlot=False,
#                                         makeInstrumentCsvs=False)
#    
##    Set onlyAnalyze to True to quickly generate csv files with all the features
##    Set onlyAnalyze to False to run a full backtest
##    Set makeInstrumentCsvs to False to not make instrument specific csvs in runLogs. 
##    This improves the performance BY A LOT
#    
#print (results)

