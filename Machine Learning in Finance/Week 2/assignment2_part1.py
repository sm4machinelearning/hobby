# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 13:36:48 2020

@author: Casper Witlox (426233)
"""

"perform assignment 2"
import pandas as pd
from sklearn.linear_model import Ridge

#adjust data
df = pd.read_csv('trainingSet.csv')
outcome = df["Outcome"]
del df["Outcome"]
X = df

#declare ridge variables
lamba = 100

"develop prediction model using Ridge regression on diabetes"
#using ridge regression function
rr = Ridge(lamba)
rr.fit(X, outcome)
w_ridge = rr.coef_

