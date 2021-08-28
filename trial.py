# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 19:24:18 2021

@author: C64990
"""

import numpy as np
import pandas as pd
a = np.arange(5)
b = np.arange(5) * 2
df = pd.DataFrame(columns=['a','b'])
df['a'] = a
df['b'] = b
df['tr2'] = df['a'] - df['b'].shift(1)
df['roll'] = df['a'].rolling(2).mean()
df['roll'] = ((df['roll'].shift(1) * (3- 1)) + df['roll']) / 3
print (df)



