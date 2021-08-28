# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:31:24 2020

@author: Casper Witlox (426233)
"""

"perform assignment 2"
import pandas as pd
from sklearn.linear_model import Lasso

#adjust data
df = pd.read_csv('trainingSet.csv')
outcome = df["Outcome"]
del df["Outcome"]
X = df

#declare LASSO variables
lamba = 0.1

"develop prediction model using LASSO regression on diabetes"
#using LASSO regression function
Lr = Lasso(lamba)
Lr.fit(X, outcome)
w_LASSO = Lr.coef_
