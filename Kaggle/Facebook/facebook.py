# -*- coding: utf-8 -*-
"""
Created on Mon May 11 21:43:43 2015

@author: Jared
"""

import csv as csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as p
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm, cross_validation
from sklearn.metrics import mean_squared_error
import sklearn.metrics as skm
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
#from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from datetime import datetime
import timeit

#%%
tic0=timeit.default_timer()
train_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Facebook/train.csv', header=0)
test_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Facebook/test.csv', header=0)
bids_df = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Facebook/bids.csv', header=0)
toc=timeit.default_timer()
print('Load Time',toc - tic0)
#%%
df = pd.concat([train_df,test_df], ignore_index=True)
unique_bidders = train_df['bidder_id'].unique()
small_df = bids_df.head(5)

#bid_id_dict = df['bidder_id'].to_dict()
#bid_id_dict = df.set_index('bidder_id').to_dict()
bid_id_dict = dict(zip(df.bidder_id, df.index))
#%%
tic=timeit.default_timer()
bids_df['bidder_id'] = bids_df['bidder_id'].map(lambda x: bid_id_dict[x])
toc=timeit.default_timer()
print('Load Time',toc - tic)
#%%
temp_df = pd.DataFrame({'auction': bids_df.auction.unique(),
                        'auction_id':range(len(bids_df.auction.unique()))})
bids_df = bids_df.reset_index().merge(temp_df, on='auction', how='left').set_index('index')
temp_df = pd.DataFrame({'merchandise': bids_df.merchandise.unique(),
                        'merchandise_id':range(len(bids_df.merchandise.unique()))})
bids_df = bids_df.reset_index().merge(temp_df, on='merchandise', how='left').set_index('index')
temp_df = pd.DataFrame({'device': bids_df.device.unique(),
                        'device_id':range(len(bids_df.device.unique()))})
bids_df = bids_df.reset_index().merge(temp_df, on='device', how='left').set_index('index')
temp_df = pd.DataFrame({'url': bids_df.url.unique(),
                        'url_id':range(len(bids_df.url.unique()))})
bids_df = bids_df.reset_index().merge(temp_df, on='url', how='left').set_index('index')
temp_df = pd.DataFrame({'country': bids_df.country.unique(),
                        'country_id':range(len(bids_df.country.unique()))})
bids_df = bids_df.reset_index().merge(temp_df, on='country', how='left').set_index('index')
temp_df = pd.DataFrame({'ip': bids_df.ip.unique(),
                        'ip_id':range(len(bids_df.ip.unique()))})
bids_df = bids_df.reset_index().merge(temp_df, on='ip', how='left').set_index('index')
toc=timeit.default_timer()
#%%
bids2_df = bids_df.drop(['auction','merchandise','device','url','ip','country'],axis=1)

#%%
#bids2_df.to_csv('/Users/Jared/DataAnalysis/Kaggle/Facebook/bids2.csv',index=False)
print('Total Time',toc - tic0)