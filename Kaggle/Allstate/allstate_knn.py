#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 21:39:49 2016

@author: Jared
"""

#%%
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import random
import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator
import pylab as p



with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn import cross_validation
    import xgboost as xgb

#import xgboost as xgb
import operator
from sklearn.neighbors.regression import KNeighborsRegressor, check_array, _get_weights

import timeit
import scipy.stats as stats

#from allstate import plot_feature_loss

tic0=timeit.default_timer()
pd.options.mode.chained_assignment = None  # default='warn'
#%%
base_features = ['cont1', 'cont10', 'cont11', 'cont12', 'cont13',
       'cont14', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7',
       'cont8', 'cont9', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8',
       'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18',
       'c19', 'c20', 'c21', 'c22', 'c23', 'c24', 'c25', 'c26', 'c27',
       'c28', 'c29', 'c30', 'c31', 'c32', 'c33', 'c34', 'c35', 'c36',
       'c37', 'c38', 'c39', 'c40', 'c41', 'c42', 'c43', 'c44', 'c45',
       'c46', 'c47', 'c48', 'c49', 'c50', 'c51', 'c52', 'c53', 'c54',
       'c55', 'c56', 'c57', 'c58', 'c59', 'c60', 'c61', 'c62', 'c63',
       'c64', 'c65', 'c66', 'c67', 'c68', 'c69', 'c70', 'c71', 'c72',
       'c73', 'c74', 'c75', 'c76', 'c77', 'c78', 'c79', 'c80', 'c81',
       'c82', 'c83', 'c84', 'c85', 'c86', 'c87', 'c88', 'c89', 'c90',
       'c91', 'c92', 'c93', 'c94', 'c95', 'c96', 'c97', 'c98', 'c99',
       'c100', 'c101', 'c102', 'c103', 'c104', 'c105', 'c106', 'c107',
       'c108', 'c109', 'c110', 'c111', 'c112', 'c113', 'c114', 'c115',
       'c116']
cont_features = ['cont1', 'cont10', 'cont11', 'cont12', 'cont13',
       'cont14', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7',
       'cont8', 'cont9']
c1_to_72 = []
for i in range(1,73):
    c1_to_72.append('c'+str(i))
c1_to_116 = []
for i in range(1,117):
    c1_to_116.append('c'+str(i))
#%%
random_seed = 5
random.seed(random_seed)
np.random.seed(random_seed)

tic=timeit.default_timer()
combined = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/combined_ints.csv', header=0)
toc=timeit.default_timer()
print('Load Time',toc - tic)

combined['loss_orig'] = combined['loss']
#%%
is_sub_run = False
#is_sub_run = True
if (is_sub_run):
    train = combined.loc[combined['loss'] != -1 ]
    test = combined.loc[combined['loss'] == -1 ]
else:
    train = combined.loc[(combined['loss'] != -1) & (combined['id'] > 200000)]
    test = combined.loc[(combined['loss'] != -1) & (combined['id'] <= 200000)]
    train = train.sample(frac = 0.1, random_state = 5)
    test = test.sample(frac = 0.05, random_state = 5)
#%%
class MedianKNeighborsRegressor(KNeighborsRegressor):
    def predict(self, X):
        X = check_array(X, accept_sparse='csr')

        neigh_dist, neigh_ind = self.kneighbors(X)

        weights = _get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        ######## Begin modification
        if weights is None:
            y_pred = np.median(_y[neigh_ind], axis=1)
        else:
            # y_pred = weighted_median(_y[neigh_ind], weights, axis=1)
            raise NotImplementedError("weighted median")
        ######### End modification

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred
#%%
def fit_knn_model(train, test, params_scale = {}, features = [], num_neighbors = 10,
                  random_seed = 5, calculate_mae = True
                  ):

    tic=timeit.default_timer()
    random.seed(random_seed)
    np.random.seed(random_seed)

    train_tmp = train.copy()
    test_tmp = test.copy()
    for key in params_scale:
        if key not in train_tmp.columns:
            print (key,'not_found')
            continue
        else:
            train_tmp[key] = train_tmp[key] * params_scale[key]
            test_tmp[key] = test_tmp[key] * params_scale[key]

    train_data = train_tmp[features].values
    train_loss = train_tmp['loss'].astype(float).values
    test_data = test_tmp[features].values

    columns = ['pred']
    def weight_func(distances):
        return (distances ** -1)
#    knn_reg = KNeighborsRegressor(n_neighbors=num_neighbors, weights=weight_func,
#    knn_reg = KNeighborsRegressor(n_neighbors=num_neighbors, weights=None,
    knn_reg = MedianKNeighborsRegressor(n_neighbors=num_neighbors, weights=None,
#                           metric='manhattan',n_jobs = 2)
#                           metric='euclidean',n_jobs = 2)
                           metric='hamming',n_jobs = 2)
    knn_reg.fit(train_data,train_loss)
    y_pred = knn_reg.predict(test_data)
    result_knn = pd.DataFrame(index=test.id, columns=columns,data=y_pred)
    if(is_sub_run):
        print('creating xgb output')
    else:
        if(calculate_mae):
            result_knn.reset_index('id',inplace=True)
            result_knn = pd.merge(result_knn,test[['id','loss']],left_on = ['id'],
                                   right_on = ['id'],how='left')
            result_knn['error_0'] = result_knn['loss'] - result_knn['pred']
            result_knn['error'] = np.abs(result_knn['loss'] - result_knn['pred'])
            print('mae',round(result_knn['error'].mean(),5))
    toc=timeit.default_timer()
    print('xgb Time',toc - tic)
    return result_knn

#%%

knn_features = []
#knn_features += base_features
#knn_features += cont_features
knn_features += c1_to_72
#knn_features += c1_to_116
#knn_features += ['c110']

#params = {'cont1':10,'cont2':100,'cont3':10,'cont4':10,'cont5':10,'cont6':10,
#          'cont7':100,'cont8':0,'cont9':100,'cont10':10,'cont11':10,'cont12':0,
#          'cont13':100,'cont14':100}
params = {}
result_knn = fit_knn_model(train,test,params,knn_features,num_neighbors = 100)

#%%
toc=timeit.default_timer()
print('Total Time',toc - tic0)