# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 22:34:26 2021

@author: C64990
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nx, ny = (8, 8)
x = np.arange(nx)
y = np.arange(ny)
xgrid, ygrid = np.meshgrid(x, y)

init_state_w = {'we1':'00',
                'we2':'70',
                'wh1':'10',
                'wh2':'60',
                'wc1':'20',
                'wc2':'50',
                'wq':'30',
                'wk':'40',
                'ww1':'01',
                'ww2':'11',
                'ww3':'21',
                'ww4':'31',
                'ww5':'41',
                'ww6':'51',
                'ww7':'61',
                'ww8':'71',
                }

init_state_b = {'be1':'07',
                'be2':'77',
                'bh1':'17',
                'bh2':'67',
                'bc1':'27',
                'bc2':'57',
                'bq':'47',
                'bk':'37',
                'bw1':'06',
                'bw2':'16',
                'bw3':'26',
                'bw4':'36',
                'bw5':'46',
                'bw6':'56',
                'bw7':'66',
                'bw8':'76',
                }

we1 = ['10','20','30','40','50','60','70',
       '-10','-20','-30','-40','-50','-60','-70',
       '01','02','03','04','05','06','07',
       '0-1','0-2','0-3','0-4','0-5','0-6','0-7']

we2 = ['10','20','30','40','50','60','70',
       '-10','-20','-30','-40','-50','-60','-70',
       '01','02','03','04','05','06','07',
       '0-1','0-2','0-3','0-4','0-5','0-6','0-7']

be1 = ['10','20','30','40','50','60','70',
       '-10','-20','-30','-40','-50','-60','-70',
       '01','02','03','04','05','06','07',
       '0-1','0-2','0-3','0-4','0-5','0-6','0-7']

be2 = ['10','20','30','40','50','60','70',
       '-10','-20','-30','-40','-50','-60','-70',
       '01','02','03','04','05','06','07',
       '0-1','0-2','0-3','0-4','0-5','0-6','0-7']


def get_key(my_dict, val):
    for key, value in my_dict.items():
         if val == value:
             return key

fig, ax = plt.subplots(figsize=(8, 8))
for x in xgrid[0,:]:
    for y in ygrid[:,0]:
        val = str(x) + str(y)
        if val in init_state_w.values():
            ax.text(x/8, y/8, str(get_key(init_state_w, val)), color='k')
        if val in init_state_b.values():
            ax.text(x/8, y/8, str(get_key(init_state_b, val)), color='k')
    
    
    

