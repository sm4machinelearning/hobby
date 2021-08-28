# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 06:40:44 2021

@author: C64990
"""

from scipy.stats import norm
import numpy as np


delta, dt = 0.25, 0.1
x0 = 0
n = 20
arrx = np.zeros((n))
arrx[0] = x0

for k in range(1, n):
    arrx[k] = arrx[k-1] + norm.rvs(scale=(delta**2)*dt)
    
print (arrx)