# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 20:09:03 2021

@author: C64990
"""

class create_features:
    
    def __init__(self, data):
        self.data = data
        
    def difference(self, var, period):
        dataDf = self.data[var]
        return dataDf.sub(dataDf.shift(period), 
                          fill_value=0)

    def ewm(self, var, halflife):
        dataDf = self.data[var]
        return dataDf.ewm(halflife=halflife,
                          ignore_na=False,
                          min_periods=0,
                          adjust=True).mean()

    def rsi(self, var, period):
        dataDf = self.data[var]
        data_upside = dataDf.sub(dataDf.shift(1), fill_value=0)
        data_downside = data_upside.copy()
        data_downside[data_upside > 0] = 0
        data_upside[data_upside < 0] = 0
        avg_upside = data_upside.rolling(period).mean()
        avg_downside = - data_downside.rolling(period).mean()
        rsi = 100 - (100 * avg_downside / (avg_downside + avg_upside))
        rsi[avg_downside == 0] = 100
        rsi[(avg_downside == 0) & (avg_upside == 0)] = 0
        return rsi
    


from features_class import *
create_feat = create_features(data)

def create_features(data):
    cols = ['emabasis2', 'emabasis3', 'emabasis4', 
            'emabasis5', 'emabasis7', 'emabasis10', 
            'rsi5', 'rsi10', 'rsi15',
            'mom1', 'mom3','mom5', 'mom10']
    basis_X = pd.DataFrame(index = data.index, columns = cols)
    
    basis_X['mom1'] = create_feat.difference('basis', 2)
    basis_X['mom3'] = create_feat.difference('basis', 4)
    basis_X['mom5'] = create_feat.difference('basis', 6)
    basis_X['mom10'] = create_feat.difference('basis', 11)

    basis_X['rsi15'] = create_feat.rsi('basis', 15)
    basis_X['rsi10'] = create_feat.rsi('basis', 10)
    basis_X['rsi5'] = create_feat.rsi('basis', 5)
    
    basis_X['emabasis2'] = create_feat.ewm('basis', 2)
    basis_X['emabasis3'] = create_feat.ewm('basis', 3)
    basis_X['emabasis4'] = create_feat.ewm('basis', 4)
    basis_X['emabasis5'] = create_feat.ewm('basis', 5)
    basis_X['emabasis7'] = create_feat.ewm('basis', 7)
    basis_X['emabasis10'] = create_feat.ewm('basis', 10)

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

basis_X_train, basis_y_train = create_features(data)
basis_X_test, basis_y_test = create_features(data_valid)