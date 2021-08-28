# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 11:29:52 2020

@author: Casper Witlox, Adnaan Willson and Gerben van der Schaaff

FIRST I DO IT FOR DECISION TREE, LATER TRY IT FOR MIRCO
"""

import math
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from MIRCO import MIRCO

"LOAD DATA HERE"
df = pd.read_csv('german.data-numeric.csv',sep=',', header=None)
n_rows = df.shape[0]
n_col = df.shape[1]

y_total = np.array(df[24]) 
X_total = np.array(df.iloc[:,0:24])

kf = KFold(n_splits=5)

for train_index, test_index in kf.split(X_total):
    X_train, X_test = X_total[train_index], X_total[test_index]
    y_train, y_test = y_total[train_index], y_total[test_index]

X_train_df = pd.DataFrame(X_train)

"SET HYPERPARAMETERS HERE"
hypsDT = [8]         
hypsRF = [5]          
n_ubounds = 4
n_lbounds = 3           

"Random Forest needed to perform MIRCO"         
RF_classifier = RandomForestClassifier(n_estimators = hypsRF[0],max_depth = hypsDT[0], max_features = "sqrt") #KEYERROR when max_depth = hyps[i]
RF_fit = RF_classifier.fit(X_train,y_train)

"Perform MIRCO here"
MIRCO_classifier = MIRCO(RF_fit)                    
MIRCO_fit = MIRCO_classifier.fit(X_train,y_train)

ubounds_long, lbounds_long = MIRCO_fit.exportRules()
m_ubounds, n_ubounds = ubounds_long.shape
m_lbounds, n_lbounds = lbounds_long.shape

max_rowindexU = 0
max_colindexU = 0                                  
max_rowindexL = 0
max_colindexL = 0

for i in range(0,m_ubounds): 
    for j in range(0,n_ubounds):
        number = not math.isnan(ubounds_long[i,j].item())
        if number:
            max_rowindexU = i
            max_colindexU = j

for i in range(0,m_lbounds):
    for j in range(0,n_lbounds):
        number = not math.isnan(lbounds_long[i,j].item())
        if number:
            max_rowindexL = i
            max_colindexL = j

ubounds = ubounds_long[0:max_rowindexU+1,0:max_colindexU+1]
lbounds = lbounds_long[0:max_rowindexL+1,0:max_colindexL+1]

"""Assign rownames and column names for each Clause and Rule"""
m_ubounds_short, n_ubounds_short = ubounds.shape
m_lbounds_short, n_lbounds_short = lbounds.shape

rownamesU = ['']*m_ubounds_short
colnamesU = ['']*n_ubounds_short
rownamesL = ['']*m_lbounds_short
colnamesL = ['']*n_lbounds_short

for i in range(0,m_ubounds_short):
    for j in range(0,n_ubounds_short):
         rownamesU[i] = 'Clause ' + str(i)
         colnamesU[j] = 'Rule ' + str(j)     

for i in range(0,m_lbounds_short):
    for j in range(0,n_lbounds_short):
         rownamesL[i] = 'Clause ' + str(i)
         colnamesL[j] = 'Rule ' + str(j) 

ubounds = pd.DataFrame(ubounds,index=rownamesU,columns=colnamesU)
lbounds = pd.DataFrame(lbounds,index=rownamesL,columns=colnamesL)

"MAKE OUTPUT FOR EXCEL"
writer = pd.ExcelWriter(r'\\solon.prd\files\P\Global\Users\C78579\UserData\Documents\Machine Learning in Finance\Week 3\Assignment (verplicht)\Output_Step4.xlsx', engine='xlsxwriter')
ubounds.to_excel (writer,sheet_name='Upper Bounds',index = True, header=True)
lbounds.to_excel (writer,sheet_name='Lower Bounds',index = True, header=True)
writer.save()
        