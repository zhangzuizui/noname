#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:22:01 2017

@author: zzzzzui
"""
#%matplotlib inline

import pandas as pd
import numpy as np
from pylab import *
'''
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
'''


path = '/home/zzzzzui/Documents/something/TimeSeries'
fileName = 'AirPassengers.csv'
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')


data = pd.read_csv(path + '/' + fileName, parse_dates=['Month'], 
                   index_col='Month', date_parser=dateparse)

ts = data['#Passengers']

rolmean = pd.rolling_mean(ts, window=12)
rolstd = pd.rolling_std(ts, window=12)

orig = plot(ts, color='blue', label='Original')
mean = plot(rolmean, color='orange', label='Rolling Mean')
std = plot(rolstd, color='black', label='Rolling Std')
title('hhhhhhhhhhh')

from statsmodels.tsa.stattools import adfuller

adfuller()