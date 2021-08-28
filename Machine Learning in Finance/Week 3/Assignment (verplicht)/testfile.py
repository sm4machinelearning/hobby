# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:02:38 2020

@author: C78579
"""

import numpy as np
from sklearn.model_selection import KFold

X = ["a", "b", "c", "d"]
kf = KFold(n_splits=2)
for train, test in kf.split(X):
    print("%s %s" % (train, test))