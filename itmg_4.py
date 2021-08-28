# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 22:17:06 2020

@author: C64990
"""

tuple = ('left', 44.44, 'temp', 99.99)
print(tuple[0])

print('{0:-2%}'.format(1.0/3))


import yfinance as yf

msft = yf.Ticker("MSFT")
print(msft)
"""
returns
<yfinance.Ticker object at 0x1a1715e898>
"""

# get stock info
print (msft.info)
print (msft.history(period="max"))



























