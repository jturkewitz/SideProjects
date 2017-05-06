#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 23:02:05 2016

@author: Jared
"""

#%%
#%%
import pandas as pd
import numpy as np
import random
import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator
import pylab as p
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cross_validation import KFold
from statsmodels.regression.quantile_regression import QuantReg
from scipy.optimize import minimize


import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn import cross_validation
    import xgboost as xgb

#import xgboost as xgb
import operator
import timeit
import scipy.stats as stats

warnings.filterwarnings("ignore")

tic0=timeit.default_timer()
pd.options.mode.chained_assignment = None  # default='warn'


#%%

#%%
random_seed = 5
random.seed(random_seed)
np.random.seed(random_seed)

tic=timeit.default_timer()
combined = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/combined_ints.csv', header=0)
res_xgb_1 = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/All_Preds/all_preds_xgb_1.csv', header=0)
res_xgb_2 = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/All_Preds/all_preds_xgb_2.csv', header=0)
res_xgb_3 = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/All_Preds/all_preds_xgb_3.csv', header=0)
res_xgb_4 = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/All_Preds/all_preds_xgb_4.csv', header=0)
res_xgb_5 = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/All_Preds/all_preds_xgb_5.csv', header=0)

res_nn_1 = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/All_Preds/all_preds_nn_1.csv', header=0)
res_nn_2 = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/All_Preds/all_preds_nn_2.csv', header=0)
res_nn_3 = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/All_Preds/all_preds_nn_3.csv', header=0)
res_nn_4 = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/All_Preds/all_preds_nn_4.csv', header=0)

res_stacked = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/All_Preds/all_preds_stacked.csv', header=0)

toc=timeit.default_timer()
print('Load Time',toc - tic)

combined['loss_orig'] = combined['loss']
combined_2 = combined[['id']].copy()
#%%
res_xgb_1.rename(columns={'pred':'pred_xgb_1'},inplace=True)
res_xgb_2.rename(columns={'pred':'pred_xgb_2'},inplace=True)
res_xgb_3.rename(columns={'pred':'pred_xgb_3'},inplace=True)
res_xgb_4.rename(columns={'pred':'pred_xgb_4'},inplace=True)
res_xgb_5.rename(columns={'pred':'pred_xgb_5'},inplace=True)

res_nn_1.rename(columns={'pred':'pred_nn_1'},inplace=True)
res_nn_2.rename(columns={'pred':'pred_nn_2'},inplace=True)
res_nn_3.rename(columns={'pred':'pred_nn_3'},inplace=True)
res_nn_4.rename(columns={'pred':'pred_nn_4'},inplace=True)

res_stacked.rename(columns={'pred':'pred_stacked'},inplace=True)

#%%

DFS = [res_xgb_1,res_xgb_2,res_xgb_3,res_xgb_4,res_xgb_5,
       res_nn_1,res_nn_2,res_nn_3,res_nn_4, res_stacked]
for df in DFS:
    combined_2 = pd.merge(combined_2,df,left_on = ['id'],
                       right_on = ['id'],how='left')
del DFS
combined = pd.merge(combined,combined_2,left_on = ['id'],
                       right_on = ['id'],how='left')
#%%
combined_small = combined.sample(frac = 0.01)
#%%
pred_features = ['pred_xgb_1', 'pred_xgb_2', 'pred_xgb_3',
       'pred_xgb_4', 'pred_xgb_5', 'pred_nn_1', 'pred_nn_2', 'pred_nn_3',
       'pred_nn_4']

combined['max_pred'] = combined[pred_features].max(axis=1)
combined['min_pred'] = combined[pred_features].min(axis=1)
combined['std_pred'] = combined[pred_features].std(axis=1)
combined['sum_pred'] = combined[pred_features].sum(axis=1)
combined['range_pred'] = combined['max_pred'] - combined['min_pred']
#%%
combined['const'] = 1
#%%
combined['low_range'] = 0
low_range = combined['range_pred'] <= 50
combined['low_range'][low_range] = 1

medium_low_range = (combined['range_pred'] > 50) & (combined['range_pred'] <= 200)
combined['medium_low_range'] = 0
combined['medium_low_range'][medium_low_range] = 1

medium_high_range = (combined['range_pred'] > 200) & (combined['range_pred'] <= 500)
combined['medium_high_range'] = 0
combined['medium_high_range'][medium_high_range] = 1


high_range = (combined['range_pred'] > 500) & (combined['range_pred'] <= 3000)
combined['high_range'] = 0
combined['high_range'][high_range] = 1

very_high_range = (combined['range_pred'] > 3000)
combined['very_high_range'] = 0
combined['very_high_range'][very_high_range] = 1
#%%
#is_sub_run = False
is_sub_run = True
if (is_sub_run):
    train = combined.loc[combined['loss'] != -1 ]
    test = combined.loc[combined['loss'] == -1 ]
else:
    train = combined.loc[(combined['loss'] != -1) & (combined['id'] > 200000)]
    test = combined.loc[(combined['loss'] != -1) & (combined['id'] <= 200000)]
#    train = combined.loc[(combined['loss'] != -1) & ~((combined['id'] > 200000) & (combined['id'] <= 400000))]
#    test = combined.loc[(combined['loss'] != -1) & ((combined['id'] > 200000) & (combined['id'] <= 400000))]
#%%
#nn_1_train = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/Train_Results_Id_2e5/train_nn_oob_1.csv', header=0)
#nn_1_test = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/Submissions/test_keras_sub_5.csv', header=0)
#nn_1_test.rename(columns={'loss':'pred'},inplace=True)
#res_nn_1 = pd.concat([nn_1_test,nn_1_train],ignore_index=True)
#res_nn_1.index = res_nn_1['id']
#res_nn_1.to_csv('all_preds_nn_1.csv', index = False)
#
#nn_2_train = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/Train_Results_Id_2e5/train_nn_oob_2.csv', header=0)
#nn_2_test = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/Submissions/test_keras_sub_6.csv', header=0)
#nn_2_test.rename(columns={'loss':'pred'},inplace=True)
#res_nn_2 = pd.concat([nn_2_test,nn_2_train],ignore_index=True)
#res_nn_2.index = res_nn_2['id']
#res_nn_2.to_csv('all_preds_nn_2.csv', index = False)
#
#nn_3_train = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/Train_Results_Id_2e5/train_nn_oob_3.csv', header=0)
#nn_3_test = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/Submissions/test_keras_sub_7.csv', header=0)
#nn_3_test.rename(columns={'loss':'pred'},inplace=True)
#res_nn_3 = pd.concat([nn_3_test,nn_3_train],ignore_index=True)
#res_nn_3.index = res_nn_3['id']
#res_nn_3.to_csv('all_preds_nn_3.csv', index = False)
#
#nn_4_train = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/Train_Results_Id_2e5/train_nn_oob_4.csv', header=0)
#nn_4_test = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/Submissions/test_keras_sub_8.csv', header=0)
#nn_4_test.rename(columns={'loss':'pred'},inplace=True)
#res_nn_4 = pd.concat([nn_1_test,nn_4_train],ignore_index=True)
#res_nn_4.index = res_nn_4['id']
#res_nn_4.to_csv('all_preds_nn_4.csv', index = False)

#%%
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

def print_mean_loss(input_df,col_name):
    for value in input_df[col_name].unique():
        cut_df = input_df.loc[input_df[col_name] == value]
        print(col_name,value)
        print('number',len(cut_df))
        print(cut_df['loss'].mean())
def apply_dict(common_dict,x,def_val=0):
    try:
        return common_dict[x]
    except KeyError:
        return def_val

def logcosh_obj(preds, dtrain):
    labels = dtrain.get_label()
    grad = np.tanh(preds - labels)
    hess = 1.0 - grad*grad
    return grad, hess
def get_mae(df,col):
    result_1 = df.copy()
    result_1['error'] = np.abs(result_1['loss_orig'] - result_1[col])
    print('mae',round(result_1['error'].mean(),5))
#%%
def fit_xgb_model(train, test, params, xgb_features, num_rounds = 10, num_rounds_es = 200000,
                  do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
                  random_seed = 5, calculate_mae = True, use_log_transform = False,
                  use_custom_obj = False, obj = logcosh_obj, use_weights = False, log_const = 200,
                  use_power_transform = False, power_constant = 0.5, power_shift = 0,
                  ):

    tic=timeit.default_timer()
    random.seed(random_seed)
    np.random.seed(random_seed)

    if use_log_transform:
        train['loss'] = np.log(train['loss_orig'] + log_const)
        if not is_sub_run:
            test['loss'] = np.log(test['loss_orig'] + log_const)
    elif use_power_transform:
        train['loss'] = (train['loss_orig'] + power_shift) ** power_constant
        if not is_sub_run:
            test['loss'] = (test['loss_orig'] + power_shift) ** power_constant
    else:
        train['loss'] = train['loss_orig']
        test['loss'] = test['loss_orig']

    X_train, X_watch = cross_validation.train_test_split(train, test_size=0.2,random_state=random_seed)
    train_data = X_train[xgb_features].values
    train_data_full = train[xgb_features].values
    watch_data = X_watch[xgb_features].values
    train_loss = X_train['loss'].astype(float).values
    train_loss_full = train['loss'].astype(float).values
    watch_loss = X_watch['loss'].astype(float).values
    test_data = test[xgb_features].values

    if(use_weights):
        dtrain = xgb.DMatrix(train_data, train_loss,weight = train['weights'].values)
        dtrain_full = xgb.DMatrix(train_data_full, train_loss_full, weight = X_train['weights'].values)
    else:
        dtrain = xgb.DMatrix(train_data, train_loss)
        dtrain_full = xgb.DMatrix(train_data_full, train_loss_full)

    dwatch = xgb.DMatrix(watch_data, watch_loss)
    dtest = xgb.DMatrix(test_data)
    watchlist = [(dtrain, 'train'),(dwatch, 'watch')]
    if use_custom_obj:
        if use_early_stopping:
            xgb_classifier = xgb.train(params, dtrain, num_boost_round=num_rounds_es, evals=watchlist,
                                early_stopping_rounds=100, verbose_eval=50,obj = obj)
            y_pred = xgb_classifier.predict(dtest,ntree_limit=xgb_classifier.best_iteration)
        else:
            xgb_classifier = xgb.train(params, dtrain_full, num_boost_round=num_rounds, evals=[(dtrain_full,'train')],
                                verbose_eval=50, obj = obj)
            y_pred = xgb_classifier.predict(dtest)
    else:
        if use_early_stopping:
            xgb_classifier = xgb.train(params, dtrain, num_boost_round=num_rounds_es, evals=watchlist,
                                early_stopping_rounds=100, verbose_eval=50)
            y_pred = xgb_classifier.predict(dtest,ntree_limit=xgb_classifier.best_iteration)
        else:
            xgb_classifier = xgb.train(params, dtrain_full, num_boost_round=num_rounds, evals=[(dtrain_full,'train')],
                                verbose_eval=50)
            y_pred = xgb_classifier.predict(dtest)


    if(print_feature_imp):
        create_feature_map(xgb_features)
        imp_dict = xgb_classifier.get_fscore(fmap='xgb.fmap')
        imp_dict = sorted(imp_dict.items(), key=operator.itemgetter(1),reverse=True)
        print('{0:<20} {1:>5}'.format('Feature','Imp'))
        print("--------------------------------------")
        num_to_print = 40
        num_printed = 0
        for i in imp_dict:
            num_printed = num_printed + 1
            if (num_printed > num_to_print):
                continue
            print ('{0:20} {1:5.0f}'.format(i[0], i[1]))
    columns = ['pred']

    result_xgb_df = pd.DataFrame(index=test.id, columns=columns,data=y_pred)
    if use_log_transform:
        result_xgb_df['pred'] = np.exp(result_xgb_df['pred']) - log_const
    elif use_power_transform:
        result_xgb_df['pred'] = (result_xgb_df['pred'] ** (1.0 / power_constant)) - power_shift
    else:
        pass
    result_xgb_df['pred'] = np.abs(result_xgb_df['pred'])
    result_xgb_df.reset_index('id',inplace=True)
    if(is_sub_run):
        print('creating xgb output')
        result_xgb_df.index = result_xgb_df['id']
    else:
        if(calculate_mae):

#            result_xgb_df = pd.merge(result_xgb_df,test[['id','loss'] + xgb_features],left_on = ['id'],
            result_xgb_df = pd.merge(result_xgb_df,test[['id','loss']],left_on = ['id'],
                                   right_on = ['id'],how='left')
            if use_log_transform:
                result_xgb_df['loss'] = np.exp(result_xgb_df['loss']) - log_const
            elif use_power_transform:
                result_xgb_df['loss'] = (result_xgb_df['loss'] ** (1.0 / power_constant)) - power_shift
            else:
                pass
#                result_xgb_df['loss'] = result_xgb_df['loss']

            result_xgb_df['error_0'] = result_xgb_df['loss'] - result_xgb_df['pred']
            result_xgb_df['error'] = np.abs(result_xgb_df['loss'] - result_xgb_df['pred'])
            cols = result_xgb_df.columns.tolist()
            cols.remove('id')
            cols.remove('loss')
            cols.remove('error')
            result_xgb_df = result_xgb_df[['id','loss','error'] + cols]
            print('mae',round(result_xgb_df['error'].mean(),5))
    toc=timeit.default_timer()
    print('xgb Time',toc - tic)
    return result_xgb_df

#%%
nfolds = 5
if nfolds > 1:
    folds = KFold(len(train), n_folds = nfolds, shuffle = True, random_state = 111)
else:
    folds = [(slice(None, None),slice(None,None))]
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
#%%

xgb_features_stack_1 = []
xgb_features_stack_1 += pred_features
xgb_features_stack_1 += ['max_pred','min_pred','std_pred','sum_pred','range_pred']
#xgb_features_stack_1 += base_features

#xgb_features_1 += ['sum_c1to72']

params_stack_1 = {'learning_rate': 0.005,
              'subsample': 0.4,
#              'reg_alpha': 2,
#              'lambda': 0.9,
#              'gamma': 1.5,
              'base_score':5,
              'seed': 6,
              'colsample_bytree': 0.5,
#              'n_estimators': 100,
              'objective': 'reg:linear',
              'eval_metric':'mae',
#              'min_child_weight': 2,
#              'eval_metric':'rmse',
              'max_depth': 2,
              }

#result_xgb_stack_1 = fit_xgb_model(train,test,params_stack_1, xgb_features_stack_1,use_early_stopping = True,
#                              print_feature_imp = True, use_log_transform = True, random_seed = 6)
#num_rounds_stack_1 = 1191
#if is_sub_run:
#    num_rounds_stack_1 /= (0.8 * 0.66)
#else:
#    num_rounds_stack_1 /= (0.8)
#num_rounds_stack_1 = int(num_rounds_stack_1)
#result_xgb_stack_1 = fit_xgb_model(train,test,params_stack_1,xgb_features_stack_1,
#                              num_rounds = num_rounds_stack_1, use_log_transform = True,
#                              use_early_stopping = False,random_seed = 6)

#result_xgb_samp_1 = result_xgb_1.sample(frac = 0.1,random_state = 3)
#%%
#%%
xgb_features_stack_2 = []
xgb_features_stack_2 += pred_features
#xgb_features_stack_2 += ['max_pred','min_pred','std_pred','sum_pred','range_pred']

#xgb_features_stack_2 += base_features
#xgb_features_stack_2 += ['sum_c1to72']

params_stack_2 = {'learning_rate': 0.005,
              'subsample': 0.99,
#              'reg_alpha': 0.5,
#              'lambda': 0.99,
#              'gamma': 1000000,
              'seed': 7,
              'colsample_bytree': 0.5,
#              'n_estimators': 100,
              'objective': 'reg:linear',
              'eval_metric':'mae',
              'base_score':2000,
#              'min_child_weight': 2,
#              'max_delta_step': 200,
#              'eval_metric':'rmse',
              'max_depth': 2,
              }

fair_constant = 60
def fair_obj(preds, dtrain):
    labels = dtrain.get_label()
    x = (preds - labels)
    den = abs(x) + fair_constant
    grad = fair_constant * x / (den)
    hess = fair_constant * fair_constant / (den * den)
    return grad, hess


#result_xgb_stack_2 = fit_xgb_model(train,test,params_stack_2, xgb_features_stack_2,use_early_stopping = True,
#                              print_feature_imp = True, random_seed = 6, use_custom_obj = True, obj = fair_obj)
#num_rounds_stack_2 = 2139
#if is_sub_run:
#    num_rounds_stack_2 /= (0.8 * 0.66)
#else:
#    num_rounds_stack_2 /= (0.8)
#num_rounds_stack_2 = int(num_rounds_stack_2)
#result_xgb_stack_2 = fit_xgb_model(train,test,params_stack_2,xgb_features_stack_2,
#                              num_rounds = num_rounds_stack_2,use_early_stopping = False,
#                              random_seed = 6, use_custom_obj = True, obj = fair_obj)
#
#DF_LIST_2 = []
#for (inTr, inTe) in folds:
#    xtr = train.iloc[inTr].copy()
#    xte = train.iloc[inTe].copy()
#
#    num_rounds_stack_2 = 2139
#    if is_sub_run:
#        num_rounds_stack_2 /= (0.66)
#    else:
#        pass
#    num_rounds_stack_2 = int(num_rounds_stack_2)
#    result_xgb_temp = fit_xgb_model(xtr,xte,params_stack_2,xgb_features_stack_2,
#                                   num_rounds = num_rounds_stack_2,use_early_stopping = False,
#                                   random_seed = 6, use_custom_obj = True, obj = fair_obj)
#    DF_LIST_2.append(result_xgb_temp)
#res_oob_xgb_2 = pd.concat(DF_LIST_2,ignore_index=True)
#res_oob_xgb_2.index = res_oob_xgb_2['id']
#res_xgb_2_stack = pd.concat([result_xgb_stack_2,res_oob_xgb_2],ignore_index=True)
#res_xgb_2_stack.index = res_xgb_2_stack['id']
#del DF_LIST_2

#%%

reg_features = []
reg_features += ['const']
reg_features += pred_features

#%%

#pred_features = ['pred_xgb_1', 'pred_xgb_2', 'pred_xgb_3',
#       'pred_xgb_4', 'pred_xgb_5', 'pred_nn_1', 'pred_nn_2', 'pred_nn_3',
#       'pred_nn_4']

tic=timeit.default_timer()

def func_to_minimize(x):

    ans = np.sum(np.abs((x[0] * train['pred_xgb_1'] + x[1] * train['pred_xgb_2'] +
                        x[2] * train['pred_xgb_3'] + x[3] * train['pred_xgb_4'] +
                        x[4] * train['pred_xgb_5'] + x[5] * train['pred_nn_1'] +
                        x[6] * train['pred_nn_2'] + x[7] * train['pred_nn_3'] +
                        x[8] * train['pred_nn_4'] + x[9] * train['pred_stacked']) / np.sum(x[0:10])
                        + x[10] * train['low_range']
                        + x[11] * train['medium_low_range']
                        + x[12] * train['medium_high_range']
                        + x[13] * train['high_range']
                        + x[14] * train['very_high_range']
                        + x[15] * train['const']
                        - train['loss_orig']))
    return ans


x0 = [0.5, 1, 0.1, 1, 0.1, 0.1, 0.5, 0.5, 0.1, 1, -50, -40, -30, 100, 1000, 0]
x_bounds = [(-1,5), (-1,5), (-1,5), (-1,5), (-1,5), (-1,5), (-1,5), (-1,5),
                 (-1,5), (-1,5), (-100,100), (-100,100), (-100,100), (-200,200), (-1000,5000), (-20,20)]
#x0 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
#x0 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 10]
#min_array = minimize(func_to_minimize, x0, bounds = x_bounds,method='SLSQP',options = {'maxiter': 1000})
#min_array = minimize(func_to_minimize, x0,  bounds = x_bounds, method='TNC', options =  {'maxiter':500})
#min_array = minimize(func_to_minimize, x0,  method='TNC')
#min_array = minimize(func_to_minimize, x0, method='SLSQP',options = {'maxiter': 500})
#min_array = minimize(func_to_minimize, x0,  method='TNC', options =  {'maxiter':500})
min_array = minimize(func_to_minimize, x0,  method='TNC')

x = min_array.x

res_ens = test[['id','loss_orig']].copy()
res_ens['pred'] = ((x[0] * test['pred_xgb_1'] + x[1] * test['pred_xgb_2'] +
                        x[2] * test['pred_xgb_3'] + x[3] * test['pred_xgb_4'] +
                        x[4] * test['pred_xgb_5'] + x[5] * test['pred_nn_1'] +
                        x[6] * test['pred_nn_2'] + x[7] * test['pred_nn_3'] +
                        x[8] * test['pred_nn_4'] + x[9] * test['pred_stacked']) / np.sum(x[0:10])
                        + x[10] * test['low_range']
                        + x[11] * test['medium_low_range']
                        + x[12] * test['medium_high_range']
                        + x[13] * test['high_range']
                        + x[14] * test['very_high_range']
                        + x[15] * test['const']
                        )
print(np.mean(np.abs(res_ens['pred'] - test['loss_orig'])))

res_ens['pred'] = res_ens['pred'].map(lambda x: 10 if x < 0 else x)
toc=timeit.default_timer()
print('Minimize Time',toc - tic)
#%%

#mod = QuantReg(train['loss_orig'].values, train[reg_features].values)
#res = mod.fit(q=.5)
#res_quant_reg = mod.predict(test[reg_features].values)
#print(res.summary())


#%%
##result_xgb_ens = result_xgb_4.copy()
#if(is_sub_run):
#    print('is a submission run')
#else:
##    result_xgb_ens['id'] = result_xgb_1['id']
##    result_xgb_ens = pd.merge(result_xgb_ens,test[['id','loss_orig']],left_on = ['id'],
##                           right_on = ['id'],how='left')
#    res_ens['error'] = np.abs(res_ens['loss_orig'] - res_ens['pred'])
#    res_ens['error_0'] = res_ens['loss_orig'] - res_ens['pred']
#    print('mae',res_ens.error.mean())

#%%
if(is_sub_run):
#    submission = result_xgb_stack_2.copy()
    submission = res_ens.copy()
#    submission.reset_index('id',inplace=True)
    submission.rename(columns = {'pred':'loss'},inplace=True)
    submission = submission[['id','loss']]
    submission.to_csv('allstate_sub_stacked.csv', index=False)
    print('stacked submission created')
#%%
toc=timeit.default_timer()
print('Total Time',toc - tic0)