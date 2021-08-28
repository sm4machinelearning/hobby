# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 19:58:18 2021

@author: C64990
"""

import numpy as np
from itertools import permutations  
import pandas as pd


arr = ['a','b','c','my']
allp = list(permutations(arr))
allps = {}
for p in range(len(allp)):
    allps['s' + str(p)] = allp[p]
    allps['s' + str(p)] = allp[p]
    allps['s' + str(p)] = allp[p]

a,b,c,my = 0.3,0.2,0.5,0.17
arr = [a,b,c,my]

d = {'a': a, 'b': b, 'c': c, 'my':my}
d = [key for (key, value) in sorted(d.items(), key=lambda key_value: key_value[1])]


#>>> sorted(d.items(), key=lambda key_value: key_value[1])
#[('a', 1), ('c', 2), ('b', 3)]

def determine_state(val):
    if val == 0:
        state = 's1'
    elif val > 0 and val < 0.1:
        state = 's2'
    elif val == 0.1:
        state = 's3'
    elif val > 0.1 and val < 0.2:
        state = 's4'
    elif val == 0.2:
        state = 's5'
    elif val > 0.2 and val < 0.3:
        state = 's6'
    elif val == 0.3:
        state = 's7'
    elif val > 0.3 and val < 0.4:
        state = 's8'
    elif val == 0.4:
        state = 's9'
    elif val > 0.4 and val < 0.5:
        state = 's10'
    elif val == 0.5:
        state = 's11'
    elif val > 0.5 and val < 0.6:
        state = 's12'
    elif val == 0.6:
        state = 's13'
    elif val > 0.6 and val < 0.7:
        state = 's14'
    elif val == 0.7:
        state = 's15'
    elif val > 0.7 and val < 0.8:
        state = 's16'
    elif val == 0.8:
        state = 's17'
    elif val > 0.8 and val < 0.9:
        state = 's18'
    elif val == 0.9:
        state = 's19'
    elif val > 0.9 and val < 1.0:
        state = 's20'
    elif val == 1.0:
        state = 's21'
    return state
    

arr = np.arange(1, 22)
arr = ['s' + str(s) for s in arr]
from itertools import combinations
allcombinations = list(combinations(arr, 3))
allcombinations = [list(x) for x in allcombinations]
allstates = ['_'.join(x) for x in allcombinations]
#allstates = []
#for ar in arr:
#    for comb in allcombinations:
#        comb = comb + [ar]
#        allstates.append(comb)
#allstates = np.asarray(allstates)
allstates = np.asarray(allcombinations)
mat = pd.DataFrame(index=allstates, columns=arr)
print (mat)




s = 's1_s2_s3'
print (s.split('_'))


# importing pandas as pd 
import pandas as pd 
  
# Creating the dataframe  
df = pd.DataFrame({"A":[4, 5, 2, None], 
                   "B":[11, 2, None, 8],  
                   "C":[1, 8, 66, 4]}) 
  
# Skipna = True will skip all the Na values 
# find maximum along column axis 
df.idxmax(skipna = True) 






