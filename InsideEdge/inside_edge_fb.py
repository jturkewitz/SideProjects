#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:44:07 2016

@author: Jared
"""

#%%
# Import Packages
get_ipython().magic('matplotlib inline')

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

from sklearn import cross_validation
from sklearn.grid_search import RandomizedSearchCV
from sklearn import metrics as skm

from scipy.stats import uniform as sp_rand
from scipy.stats import randint as sp_randint

import xgboost as xgb
from sklearn.cross_validation import KFold

import sys
import operator
import random
import timeit

sys.path.append("../modules")

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import pylab as p
#%%
tic0=timeit.default_timer()
#%%
#tic=timeit.default_timer()
#xls = pd.ExcelFile('MLB_Challenge_Data updated 10-21-16.xlsx')
#toc = timeit.default_timer()
#
#dataset = xls.parse('MLB_Challenge_Data updated 10-2')
#
#print('Load Time',toc - tic)
##Make it into a csv for faster load times
##only need to do this step once
#dataset.to_csv('mlb_data.csv', index=False)

#%%
#NOTE MAY BE LEAK IN RECORD NUMBER
#%%
tic=timeit.default_timer()
#dataset_old = pd.read_csv('mlb_data.csv',header=0)
dataset = pd.read_csv('MLB_Challenge_Data 2015 upd 11-2-16.csv',header=0)
toc=timeit.default_timer()
print('Load Time',toc - tic)
#%%
dataset.fillna(-1000,inplace=True)


#%%
dataset.rename(columns={'Actual_PTS':'points','RECORDNUM':'id'},inplace=True)
#dataset.columns = [x.lower() for x in dataset.columns]
dataset.columns = [x.replace (" ", "_") for x in dataset.columns]
dataset.columns = [x.replace ("-", "_") for x in dataset.columns]
#%%
object_cols = []
object_hash_cols = []
for feature in dataset.columns:
    if dataset[feature].dtype.name == 'object':
        object_cols.append(feature)
        feature_name_hash = feature + str('_hash')
        object_hash_cols.append(feature_name_hash)
        dataset[feature_name_hash] = pd.factorize(dataset[feature])[0]
#%%

#is_sub_run = False
is_sub_run = True
random_seed = 4
if (is_sub_run):
    train = dataset.loc[dataset['points'] != -1000 ]
    test = dataset.loc[dataset['points'] == -1000 ]
else:
    train, test = cross_validation.train_test_split(dataset.loc[dataset['points'] != -1000], test_size = 0.3, random_state = random_seed)
#    train = dataset[(dataset['Week'] <= 19) & (dataset['points'] != -1000)].copy()
#    test = dataset[(dataset['Week'] > 19) & (dataset['points'] != -1000)].copy()

nfolds = 5
if nfolds > 1:
    folds = KFold(len(train), n_folds = nfolds, shuffle = True, random_state = 111)
else:
    folds = [(slice(None, None),slice(None,None))]
#%%
def get_mse(df,col = 'pred'):
    print('mse',np.sqrt(np.square(df[col] - df['points']).mean()))
#%%
#np.sqrt(np.square(test.points - test.points.mean()).mean())
#%%
#testing leakage
#if is_sub_run:
#    test['order_by_week'] = 1
#    test['order_by_week'] = test.groupby(['Week'])['order_by_week'].cumsum()
#%%
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()
#%%
def fit_xgb_model(train, test, params, xgb_features, num_rounds = 10, num_rounds_es = 200000,
                  use_early_stopping = True, print_feature_imp = False,
                  random_seed = 123, calculate_rmse = True,
                  ):

    tic=timeit.default_timer()
    random.seed(random_seed)
    np.random.seed(random_seed)

    X_train, X_watch = cross_validation.train_test_split(train, test_size=0.2,random_state=random_seed)
    train_data = X_train[xgb_features].values
    train_data_full = train[xgb_features].values
    watch_data = X_watch[xgb_features].values
    train_points = X_train['points'].astype(float).values
    train_points_full = train['points'].astype(float).values
    watch_points = X_watch['points'].astype(float).values
    test_data = test[xgb_features].values


    dtrain = xgb.DMatrix(train_data, train_points)
    dtrain_full = xgb.DMatrix(train_data_full, train_points_full)

    dwatch = xgb.DMatrix(watch_data, watch_points)
    dtest = xgb.DMatrix(test_data)
    watchlist = [(dtrain, 'train'),(dwatch, 'watch')]

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
    if(is_sub_run):
        print('is a submission run')
        result_xgb_df.reset_index('id',inplace=True)
    else:
        if(calculate_rmse):
            result_xgb_df.reset_index('id',inplace=True)
            result_xgb_df = pd.merge(result_xgb_df,test[['id','points'] + xgb_features],left_on = ['id'],
                                   right_on = ['id'],how='left')

            result_xgb_df['error_sq'] = np.square((result_xgb_df['pred'] - result_xgb_df['points']))
            #reorder columns for convenience
            cols = result_xgb_df.columns.tolist()
            cols.remove('id')
            cols.remove('points')
            cols.remove('pred')
            cols.remove('error_sq')
            result_xgb_df = result_xgb_df[['id','points','pred','error_sq'] + cols]
            print('rmse',round(np.sqrt(result_xgb_df['error_sq'].mean()),5))
    toc=timeit.default_timer()
    print('xgb Time',toc - tic)
    return result_xgb_df

#%%


#base_features = []
#for column in dataset.columns:
#    base_features.append(column)
#%%
base_features = ['Week','HitterStatus','Hitter_Pos','PitcherSOTend','PitcherSide',
 'HitterGLYCategory','PitcherPcntlGroup','HitterPcntlGroup','PitcherGLYCategory',
 'PitcherGB_FLY','HitterHomeAway','PA_H2H','H_H2H','AB_H2H','BAVG_H2H','SLG_H2H',
 'HR_H2H','wOBA_H2H','SO_H2H','WHAvg_H2H','PA_PGLY','H_PGLY','AB_PGLY','BAVG_PGLY',
 'SLG_PGLY','HR_PGLY','wOBA_PGLY','SO_PGLY','WHAvg_PGLY','PA_PTier','H_PTier',
 'AB_PTier','BAVG_PTier','SLG_PTier','HR_PTier','wOBA_PTier','SO_PTier','WHAvg_PTier',
 'PA_HGLY','H_HGLY','AB_HGLY','BAVG_HGLY','SLG_HGLY','wOBA_HGLY','WHAvg_HGLY',
 'PA_Last_5','H_Last_5','AB_Last_5','BAVG_Last_5','XBH_Last_5','wOBA_Last_5',
 'WHAvg_Last_5','PA_HomeAway','wOBA_HomeAway','WHAvg_HomeAway','PA_HomeAway___Pitcher',
 'WOBA_HomeAway_Pitcher','WHAvg_HomeAway___Pitcher','PA_vs_SO_Pitcher','H_SO_Pitcher',
 'AB_SO_Pitcher','BAVG_SO_Pitcher','SLG_SO_Pitcher','wOBA_vs_SO_Pitcher',
 'WHAvg_vs_SO_Pitcher','K%_vs_SO_Pitcher','PA_vs_GB_Pitcher','H_GB_Pitcher',
 'AB_GB_Pitcher','BAVG_GB_Pitcher','SLG_GB_Pitcher','wOBA_vs_GB_Pitcher','WHAvg_vs_GB_Pitcher',
 'H2H_MU_SCORE','PGLY_MU_SCORE','PTier_MU_SCORE','HGLY_MU_SCORE','SO_Tend_MU_SCORE','GB_Tend_MU_SCORE',
 'Last5_MU_SCORE','HomeAway_MU_SCORE','HomeAway_MU_SCORE___Pitcher','PlayToday_Likelihood',
 'MatchupScore3','Normalized_MU3','PA_vs_SP_Side','Park_Adj','Expected_wOBA','Fanduel_Pos',
 'Hitter_WHRating_Avg_Last15','Hitter_SLG_Last15','PA_Last15Days','Opp._Pitcher_WHRating_Avg',
 'Opp._Pitcher_SLG','Current_O/U','Hitter_Pitcher_B_T','Hitter_WHRating_Last5Days',
 'Hitter_WHRating_Last365','Pitcher_Good_Greats_Last_4','humidity','pressure',
 'condition','temp','chanceofrain','outtoleftimpact','outtorightimpact','DayNight',
 'elevation','roof','BOP','HitterStatus_hash',
 'Hitter_Pos_hash','PitcherSOTend_hash','PitcherSide_hash',
 'HitterGLYCategory_hash','PitcherPcntlGroup_hash','HitterPcntlGroup_hash',
 'PitcherGLYCategory_hash','PitcherGB_FLY_hash','HitterHomeAway_hash','K%_vs_SO_Pitcher_hash',
 'PlayToday_Likelihood_hash','Fanduel_Pos_hash','Hitter_Pitcher_B_T_hash',
 'condition_hash','DayNight_hash','roof_hash']
base_features = [feature for feature in base_features if feature not in object_cols]
#%%
xgb_features = []
xgb_features += base_features
#xgb_features += features_by_med

params = {'learning_rate': 0.005,
              'subsample': 0.98,
              'reg_alpha': 0.5,
#              'lambda': 0.995,
              'gamma': 0.5,
              'seed': 5,
              'colsample_bytree': 0.3,
#              'n_estimators': 100,
              'objective': 'reg:linear',
              'eval_metric':'rmse',
              'min_child_weight': 2,
              'max_depth': 6,
              }
#xgb_features.remove('Week')
#xgb_features.remove('BOP')

#this is used to find early stopping rounds, then scale up when using all of dataset
#as number of rounds should be proportional to size of dataset

#result_xgb_1 = fit_xgb_model(train,test,params, xgb_features,use_early_stopping = True,
#                              print_feature_imp = True, random_seed = 6)
#
num_rounds_1 = 904
if is_sub_run:
    num_rounds_1 /= (0.8 * 0.7)
else:
    num_rounds_1 /= (0.8)
num_rounds_1 = int(num_rounds_1)
result_xgb_1 = fit_xgb_model(train,test,params,xgb_features,
                              num_rounds = num_rounds_1,
                              use_early_stopping = False,random_seed = 6)

DF_LIST_1 = []
for (inTr, inTe) in folds:
    xtr = train.iloc[inTr].copy()
    xte = train.iloc[inTe].copy()

    num_rounds_1 = 904
    if is_sub_run:
        num_rounds_1 /= (0.7)
    else:
        pass
    num_rounds_1 = int(num_rounds_1)
    result_xgb_temp = fit_xgb_model(xtr,xte,params,xgb_features,
                              num_rounds = num_rounds_1,
                              use_early_stopping = False,random_seed = 6)
    DF_LIST_1.append(result_xgb_temp)
res_oob_xgb_1 = pd.concat(DF_LIST_1,ignore_index=True)
res_oob_xgb_1.index = res_oob_xgb_1['id']
res_xgb_1 = pd.concat([result_xgb_1,res_oob_xgb_1],ignore_index=True)
res_xgb_1.index = res_xgb_1['id']
del DF_LIST_1

#%%
if is_sub_run:
    res_oob_xgb_merged = pd.merge(train[['id','points']],res_oob_xgb_1,left_on = ['id'],
                       right_on = ['id'],how='left')
    get_mse(res_oob_xgb_merged)
    print('pred mean mse',np.sqrt(np.square(train.points - train.points.mean()).mean()))

#result_xgb_samp_1 = result_xgb_1.sample(frac = 0.1,random_state = 3)
#%%
points_by_week = train.groupby('BOP')['points'].mean()
points_by_week_errors = train.groupby('BOP')['points'].sem()

plt.errorbar(points_by_week.index,points_by_week,points_by_week_errors)
#plt.savefig('fig.png',bbox_inches="tight",dpi=300)
#%%
#if is_sub_run:
#    result_xgb_1 = pd.merge(result_xgb_1,test[['id','order_by_week','outtorightimpact']],left_on = ['id'],
#                           right_on = ['id'],how='left')
#    corr_test = result_xgb_1.corr()
#%%
toc = timeit.default_timer()
print('Total Time',toc - tic0)



