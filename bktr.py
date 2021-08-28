# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 04:53:32 2021

@author: C64990
"""

from backtester.trading_system import TradingSystem
from backtester.trading_system_parameters import TradingSystemParameters
from backtester.features.feature import Feature
from backtester.executionSystem.simple_execution_system_fairvalue import SimpleExecutionSystemWithFairValue
from backtester.orderPlacer.backtesting_order_placer import BacktestingOrderPlacer
from backtester.version import updateCheck
from backtester.constants import *
from backtester.timeRule.us_time_rule import USTimeRule
from backtester.logger import *

class MyTradingParams(TradingSystemParameters):
    def __init__(self, backtest_data):
        super(MyTradingParams, self).__init__()
        self.count = 0 
        self.params = {}
        self.start = '2017/01/06'
        self.end = '2017/01/10'
        self.instrumentIds = ['MQK']
        self.backtestdata = backtest_data

    '''
    Returns an instance of class DataParser. Source of data for instruments
    '''

    def getDataParser(self):
        z = self.backtestdata
        return z

        #dataSetId = 'trainingData3'
        #downloadUrl = 'https://github.com/Auquan/auquan-historical-data/raw/master/qq2Data'
        
#        z= CsvDataSource(cachedFolderName='historicalData/',
#                             dataSetId=dataSetId,
#                             instrumentIds=self.instrumentIds,
#                             downloadUrl = downloadUrl,
#                             timeKey = '',
#                             timeStringFormat = '%Y-%m-%d %H:%M:%S',
#                             startDateStr=self.start,
#                             endDateStr=self.end,
#                             liveUpdates=True,
#                             pad=True)




    def getTimeRuleForUpdates(self):
        return USTimeRule(startDate = self.start,
                        endDate = self.end,
                        startTime='9:30',
                        endTime='15:30',
                        frequency='M', sample='1')

    
    '''
    Return starting capital
    '''
    def getStartingCapital(self):
        return 10000
    
    '''
    This is a way to use any custom features you might have made.
    '''

    def getCustomFeatures(self):
        return {'prediction': TrainingPredictionFeature}

    def getInstrumentFeatureConfigDicts(self):

        predictionDict = {'featureKey': 'prediction',
                                'featureId': 'prediction',
                                'params': {}}

        # ADD RELEVANT FEATURES HERE
        expma5dic = {'featureKey': 'emabasis5',
                 'featureId': 'exponential_moving_average',
                 'params': {'period': 5,
                              'featureName': 'basis'}}
        expma10dic = {'featureKey': 'emabasis10',
                 'featureId': 'exponential_moving_average',
                 'params': {'period': 10,
                              'featureName': 'basis'}}                     
        expma2dic = {'featureKey': 'emabasis2',
                 'featureId': 'exponential_moving_average',
                 'params': {'period': 2,
                              'featureName': 'basis'}}
        mom10dic = {'featureKey': 'mom10',
                 'featureId': 'difference',
                 'params': {'period': 10,
                              'featureName': 'basis'}}
        scoreDict = {'featureKey': 'score',
                     'featureId': 'score_fv',
                     'params': {'predictionKey': 'prediction',
                                'price': 'basis'}}
        return {INSTRUMENT_TYPE_STOCK: [expma5dic,expma2dic,expma10dic,mom10dic,
                                        predictionDict, scoreDict]}

    '''
    Returns an array of market feature config dictionaries
    '''

    def getMarketFeatureConfigDicts(self):
    # ADD RELEVANT FEATURES HERE
        scoreDict = {'featureKey': 'score',
                     'featureId': 'score_fv',
                     'params': {'featureName': self.getPriceFeatureKey(),
                                'instrument_score_feature': 'score'}}
        return [scoreDict]


    '''
    A function that returns your predicted value based on your heuristics.
    '''

    def getPrediction(self, time, updateNum, instrumentManager):

        predictions = pd.Series(0.0, index = self.instrumentIds)

        # holder for all the instrument features
        lbInstF = instrumentManager.getLookbackInstrumentFeatures()

        ### TODO : FILL THIS FUNCTION TO RETURN A BUY (1) or SELL (0) prediction for each stock
        ### USE TEMPLATE BELOW AS EXAMPLE

        # dataframe for a historical instrument feature (mom10 in this case). The index is the timestamps
        # of upto lookback data points. The columns of this dataframe are the stock symbols/instrumentIds.

        # Get the last row of the dataframe, the most recent datapoint
        mom10 = lbInstF.getFeatureDf('mom10').iloc[-1]
        emabasis2 = lbInstF.getFeatureDf('emabasis2').iloc[-1]
        emabasis5 = lbInstF.getFeatureDf('emabasis5').iloc[-1]
        emabasis10 = lbInstF.getFeatureDf('emabasis10').iloc[-1] 
        basis = lbInstF.getFeatureDf('basis').iloc[-1]
        totalaskvol = (lbInstF.getFeatureDf('stockTotalAskVol').iloc[-1] - lbInstF.getFeatureDf('futureTotalAskVol').iloc[-1])/100000.0
        totalbidvol = (lbInstF.getFeatureDf('stockTotalBidVol').iloc[-1] - lbInstF.getFeatureDf('futureTotalBidVol').iloc[-1])/100000.0
        
        coeff = [ 0.03249183, 0.49675487, -0.22289464, 0.2025182, 0.5080227, -0.21557005, 0.17128488]

        predictions['MQK'] = coeff[0] * mom10['MQK'] + coeff[1] * emabasis2['MQK'] +\
                      coeff[2] * emabasis5['MQK'] + coeff[3] * emabasis10['MQK'] +\
                      coeff[4] * basis['MQK'] + coeff[5] * totalaskvol['MQK']+\
                      coeff[6] * totalbidvol['MQK']
                    
        predictions.fillna(emabasis5,inplace=True)
        
        print('Current basis: %.3f, predicted basis: %.3f'%(basis['MQK'], predictions['MQK']))
        if updateNum>1:
            print('Current position: %.0f'%lbInstF.getFeatureDf('position').iloc[-1]['MQK'])

        return predictions

    '''
    Here we convert prediction to intended positions for different instruments.
    '''

    def getExecutionSystem(self):
        return SimpleExecutionSystemWithFairValue(enter_threshold_deviation=0.5, exit_threshold_deviation=0.2, 
                                                longLimit=250, shortLimit=250, capitalUsageLimit=0.05, 
                                                enterlotSize=10, exitlotSize=10, 
                                                limitType='L', price=self.getPriceFeatureKey())

    '''
    For Backtesting, we use the BacktestingOrderPlacer, which places the order which we want, 
    and automatically confirms it too.
    '''

    def getOrderPlacer(self):
        return BacktestingOrderPlacer()

    '''
    Returns the amount of lookback data you want for your calculations.
    '''

    def getLookbackSize(self):
        return 90


    def getPriceFeatureKey(self):
        return 'basis'


    def getMetricsToLogRealtime(self):
        # Everything will be logged if left as is
        return {
            'market': None,
            'instruments': None
        }


class TrainingPredictionFeature(Feature):

    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        t = MyTradingParams()
        return t.getPrediction(time, updateNum, instrumentManager)
