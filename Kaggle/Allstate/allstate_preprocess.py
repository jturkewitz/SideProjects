#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 15:08:07 2016

@author: Jared
"""

#%%
import pandas as pd
import numpy as np
import random

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn import cross_validation
    
#import xgboost as xgb
import operator
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
import timeit

warnings.filterwarnings("ignore")

tic0=timeit.default_timer()
pd.options.mode.chained_assignment = None  # default='warn'
#%%
tic=timeit.default_timer()
train_orig = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/train.csv', header=0)
test_orig = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/test.csv', header=0)
toc=timeit.default_timer()
print('Load Time',toc - tic)
#%%
is_sub_run = False
#is_sub_run = True
#%%
train_small = train_orig.sample(frac=0.01)

#%%
random_seed = 5
random.seed(random_seed)
np.random.seed(random_seed)

tic=timeit.default_timer()
combined = pd.concat([train_orig, test_orig], axis=0,ignore_index=True)

combined.fillna(-1,inplace=True)

toc=timeit.default_timer()
print('Merging Time',toc - tic)


combined_small = combined.sample(frac = 0.1)



#%%
def map_by_ab(x):
    if x == 'A':
        return 0
    elif x == 'B':
        return 1
    else:
        return x
for i in range(1,73):
    name = 'cat' + str(i)
    new_name = 'c' + str(i)
    combined[new_name] = combined[name].map(lambda x: map_by_ab(x)).astype(int)


#%%

def convert_strings_to_ints(input_df,col_name,output_col_name):
    input_df.sort_values(col_name,inplace=True)
    labels, levels = pd.factorize(input_df[col_name])
    input_df[output_col_name] = labels
    input_df[output_col_name] = input_df[output_col_name].astype(int)
    output_dict = dict(zip(input_df[col_name],input_df[output_col_name]))
    return (output_dict,input_df)
#%%
for i in range(73,117):
    name = 'cat' + str(i)
    new_name = 'c' + str(i)
    (output_dict,combined) = convert_strings_to_ints(combined,name,new_name)
    
#%%
cat_list = [name for name in combined.columns if name.startswith('cat')]
            
combined.drop(cat_list,axis=1,inplace=True)
#%%
cols = combined.columns.tolist()

cols.remove('id')
cols.remove('loss')

cols = ['id','loss'] + cols
#%%
combined = combined[cols]
#%%
combined.sort_values(by='id',inplace=True)

combined['id'] = combined['id'].astype(int)
#%%
combined['loss'] = combined['loss'].astype(float)

#%%
cont_list = [name for name in combined.columns if name.startswith('cont')]
for name in cont_list:
    combined[name] = combined[name].astype(float)
#%%
combined.to_csv('combined_ints.csv',index=False,float_format='%.15g')
#%%
combined_2 = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/combined_ints.csv', header=0)
#%%