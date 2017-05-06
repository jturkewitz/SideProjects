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
import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator
import pylab as p
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cross_validation import KFold



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
#tic=timeit.default_timer()
#train_orig = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/train.csv', header=0)
#test_orig = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/test.csv', header=0)
#toc=timeit.default_timer()
#print('Load Time',toc - tic)

#train_small = train_orig.sample(frac=0.01)
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
for col in combined.columns:
    if not col.startswith('cont'):
        continue
    number_unique = combined[col].nunique()
    combined_value_counts = combined[col].value_counts().to_dict()
    most_freq_number = combined[col].value_counts().max()
    col_freq = 'freq_' + col
    col_cumulative = 'cumulative_' + col
    combined[col_freq] = combined[col].map(lambda x: combined_value_counts[x])


    combined.sort(col,inplace=True)
    TEMP_VALUES = combined.drop_duplicates(col)[col_freq].cumsum().values
    TEMP_DICT = pd.Series(TEMP_VALUES,index=combined.drop_duplicates(col)[col]).to_dict()
    combined[col_cumulative] = combined[col].map(lambda x: TEMP_DICT[x])

    combined[col_freq] = combined[col_freq] / most_freq_number
#%%
freq_features_for_combine = ['freq_cont1','freq_cont10','freq_cont11','freq_cont12',
                 'freq_cont13','freq_cont2','freq_cont3','freq_cont4',
                 'freq_cont6','freq_cont7','freq_cont8','freq_cont9']

features_to_agg = ['cont1','cont10','cont11','cont12',
                 'cont13','cont2','cont3','cont4',
                 'cont6','cont7','cont8','cont9']
#%%
combined['max_cont_freq'] = combined[freq_features_for_combine].max(axis=1)
combined['min_cont_freq'] = combined[freq_features_for_combine].min(axis=1)
combined['mean_cont_freq'] = combined[freq_features_for_combine].mean(axis=1)
combined['median_cont_freq'] = combined[freq_features_for_combine].median(axis=1)
combined['std_cont_freq'] = combined[freq_features_for_combine].std(axis=1)

combined['max_cont_agg'] = combined[features_to_agg].max(axis=1)
combined['min_cont_agg'] = combined[features_to_agg].min(axis=1)
combined['mean_cont_agg'] = combined[features_to_agg].mean(axis=1)
combined['median_cont_agg'] = combined[features_to_agg].median(axis=1)
combined['std_cont_agg'] = combined[features_to_agg].std(axis=1)

combined.fillna(0,inplace=True)

combined.sort_values(by='id',inplace=True)
combined['id_shift'] = combined['id'].shift(1)
combined['id_shift'].fillna(0,inplace=True)
combined['id_diff'] = combined['id_shift'] - combined['id']

#%%

#cont_feat_list = [name for name in combined.columns if name.startswith('cont')]
#
#diff_names = []
#sum_names = []
#mult_names = []
#for i in range(1,len(cont_feat_list)+1):
#    name0 = 'cont' + str(i)
#    for j in range(1,len(cont_feat_list)+1):
#        if j <= i:
#            continue
#        else:
#            feat_name_diff = 'diff_cont_' + str(i) + '_' + str(j)
#            feat_name_sum = 'sum_cont_' + str(i) + '_' + str(j)
#            feat_name_mult = 'multcont_' + str(i) + '_' + str(j)
#            name1 = 'cont'+str(j)
#            combined[feat_name_diff] = combined[name0] - combined[name1]
#            combined[feat_name_sum] = combined[name0] + combined[name1]
#            combined[feat_name_mult] = combined[name0] * combined[name1]
#            diff_names.append(feat_name_diff)
#            sum_names.append(feat_name_sum)
#            mult_names.append(feat_name_mult)

#%%

#for feature in combined.columns:
#    print(feature,combined[feature].nunique())

#c1-72 2
#c73-76 3
#c77-c88 4
#cc89 - c98 5-9
#c99 - 108 11-20
#c109,c110 85-134
#c111 - 115 17 to 63
#c116 349

#%%
c1_to_72 = []
for i in range(1,73):
    c1_to_72.append('c'+str(i))

combined['max_c1to72'] = combined[c1_to_72].max(axis=1)
combined['min_c1to72'] = combined[c1_to_72].min(axis=1)
combined['mean_c1to72'] = combined[c1_to_72].mean(axis=1)
combined['median_c1to72'] = combined[c1_to_72].median(axis=1)
combined['std_c1to72'] = combined[c1_to_72].std(axis=1)
combined['sum_c1to72'] = combined[c1_to_72].sum(axis=1)

#%%
COLS_CLOW = []
for feature in c1_to_72:
    COLS_CLOW.append(combined[feature].values)
combined['c1to72_all'] = pd.factorize(pd.lib.fast_zip(COLS_CLOW))[0]
#%%

#if (is_sub_run):
#    train = combined.loc[combined['loss'] != -1 ]
#    test = combined.loc[combined['loss'] == -1 ]
#else:
#    train = combined.loc[(combined['loss'] != -1) & (combined['id'] > 200000)]
#    test = combined.loc[(combined['loss'] != -1) & (combined['id'] <= 200000)]

#%%
combined_small = combined.sample(frac = 0.01)
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
#%%
#train_full = combined.loc[combined.loss != -1]
#for name in train.columns:
#    if name.startswith('cont') or name == 'id' or name == 'loss':
#        continue
#    else:
#        print_mean_loss(train,name)
#%%
is_sub_run = False
#is_sub_run = True
if (is_sub_run):
    train = combined.loc[combined['loss'] != -1 ]
    test = combined.loc[combined['loss'] == -1 ]
else:
    train = combined.loc[(combined['loss'] != -1) & (combined['id'] > 200000)]
    test = combined.loc[(combined['loss'] != -1) & (combined['id'] <= 200000)]
#    train = combined.loc[(combined['loss'] != -1) & ~((combined['id'] > 200000) & (combined['id'] <= 400000))]
#    test = combined.loc[(combined['loss'] != -1) & ((combined['id'] > 200000) & (combined['id'] <= 400000))]

#%%
#corr = train.corr()
#del combined
#%%
def plot_feature_loss(input_df,feature_name = 'cont1',num_bins = 50):
    train_temp = input_df.copy()
    if feature_name.startswith('cont'):
        bins = np.linspace(0,1.0,num_bins)
        feature_name_binned = feature_name + '_binned'
        train_temp[feature_name_binned] = np.digitize(train_temp[feature_name],bins=bins,right=True)
        train_temp[feature_name_binned] = train_temp[feature_name_binned] / num_bins
        temp_dict = train_temp.groupby(feature_name_binned)['loss_orig'].mean().to_dict()
        temp_err_dict = train_temp.groupby(feature_name_binned)['loss_orig'].sem().to_dict()
    else:
        temp_dict = train_temp.groupby(feature_name)['loss_orig'].mean().to_dict()
        temp_err_dict = train_temp.groupby(feature_name)['loss_orig'].sem().to_dict()
#        bins = list(range(input_df[feature_name].min(),input_df[feature_name].max() + 1))

    lists = sorted(temp_dict.items())
    x, y = zip(*lists)
    lists_err = sorted(temp_err_dict.items())
    x_err, y_error = zip(*lists_err)

    p.figure()
    plt.errorbar(x,y,fmt = 'o',yerr = y_error,label = feature_name)
    p.xlabel(feature_name,fontsize=20)
    p.ylabel('loss',fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    p.legend(prop={'size':20},numpoints=1,loc=(0.05,0.8))
    p.xlim([train_temp[feature_name].min() - 0.02, train_temp[feature_name].max() + 0.02 ])
    plt.grid()
    ax = plt.gca()

    plt.tick_params(axis='both', which='major', labelsize=15)
    ax.yaxis.set_major_locator(MaxNLocator(prune='lower'))
    ax.xaxis.set_major_locator(MaxNLocator(prune='lower'))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    del train_temp


#for name in train.columns:
#    if name.startswith('cont'):
#        plot_feature_loss(train,feature_name = name)
#        continue
#    if name.startswith('c'):
#        if int(name[1:]) >= 130:
#            plot_feature_loss(train,feature_name = name)
#c90,c91,c92,c101,c103,c104,c105,c106,c107,c111,c113,c114,c115,c116

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
freq_features = ['freq_cont1','freq_cont10','freq_cont11','freq_cont12',
                 'freq_cont13','freq_cont14','freq_cont2','freq_cont3','freq_cont4',
                 'freq_cont5','freq_cont6','freq_cont7','freq_cont8','freq_cont9']
#%%
#may introduce slight leakage (with watchlist, etc)
#features_to_convert = ['cont14','cont7','cont6','c100']
#median_loss = train.loss.median()
#for feature in features_to_convert:
#    feature_dict = train.groupby([feature])['loss_orig'].median().to_dict()
#    train[feature + '_med'] = train[feature].map(lambda x: apply_dict(feature_dict,x,median_loss))
#    test[feature + '_med'] = test[feature].map(lambda x: apply_dict(feature_dict,x,median_loss))
#features_by_med = [name+'_med' for name in features_to_convert]
#test['pred'] = test[features_by_med].mean(axis=1)
#
#test['error'] = np.abs(test['pred'] - test['loss_orig'])
#print(test['error'].mean())
#%%
nfolds = 5
if nfolds > 1:
    folds = KFold(len(train), n_folds = nfolds, shuffle = True, random_state = 111)
else:
    folds = [(slice(None, None),slice(None,None))]
#%%

xgb_features_1 = []
xgb_features_1 += base_features
xgb_features_1 += ['sum_c1to72']
#xgb_features_1 += freq_features
#xgb_features_1 += features_by_med

params_1 = {'learning_rate': 0.01,
              'subsample': 0.99,
              'reg_alpha': 2,
              'lambda': 0.9,
              'gamma': 0.5,
              'seed': 6,
              'colsample_bytree': 0.3,
#              'n_estimators': 100,
              'objective': 'reg:linear',
              'eval_metric':'mae',
              'min_child_weight': 2,
#              'eval_metric':'rmse',
              'max_depth': 6,
              }

#this is used to find early stopping rounds, then scale up when using all of dataset
#as number of rounds should be proportional to size of dataset

#result_xgb_1 = fit_xgb_model(train,test,params_1, xgb_features_1,use_early_stopping = True,
#                              print_feature_imp = True, use_log_transform = True, random_seed = 6)
num_rounds_1 = 6165
if is_sub_run:
    num_rounds_1 /= (0.8 * 0.66)
else:
    num_rounds_1 /= (0.8)
num_rounds_1 = int(num_rounds_1)
result_xgb_1 = fit_xgb_model(train,test,params_1,xgb_features_1,
                              num_rounds = num_rounds_1, use_log_transform = True,
                              use_early_stopping = False,random_seed = 6)

result_xgb_samp_1 = result_xgb_1.sample(frac = 0.1,random_state = 3)


## cv-folds

DF_LIST_1 = []
for (inTr, inTe) in folds:
    xtr = train.iloc[inTr].copy()
    xte = train.iloc[inTe].copy()

    #for 5 fold cv don't change num rounds
    num_rounds_1 = 6165
    if is_sub_run:
        num_rounds_1 /= (0.66)
    else:
        pass
    num_rounds_1 = int(num_rounds_1)
    result_xgb_temp = fit_xgb_model(xtr,xte,params_1,xgb_features_1,
                                  num_rounds = num_rounds_1, use_log_transform = True,
                                  use_early_stopping = False,random_seed = 6)
    DF_LIST_1.append(result_xgb_temp)
res_oob_xgb_1 = pd.concat(DF_LIST_1,ignore_index=True)
res_oob_xgb_1.index = res_oob_xgb_1['id']
if not is_sub_run:
    print('mae',round(res_oob_xgb_1['error'].mean(),5))
#%%
xgb_features_2 = []
xgb_features_2 += base_features
xgb_features_2 += ['sum_c1to72']
#xgb_features += features_by_med

params_2 = {'learning_rate': 0.005,
              'subsample': 0.99,
              'reg_alpha': 0.5,
              'lambda': 0.9,
#              'gamma': 1000000,
              'seed': 6,
              'colsample_bytree': 0.2,
#              'n_estimators': 100,
              'objective': 'reg:linear',
              'eval_metric':'mae',
              'base_score':1500,
              'min_child_weight': 12,
#              'max_delta_step': 200,
#              'eval_metric':'rmse',
              'max_depth': 10,
              }

fair_constant = 200
def fair_obj(preds, dtrain):
    labels = dtrain.get_label()
    x = (preds - labels)
    den = abs(x) + fair_constant
    grad = fair_constant * x / (den)
    hess = fair_constant * fair_constant / (den * den)
    return grad, hess
#def l1_minius_l2_obj(preds, dtrain):
#    labels = dtrain.get_label()
#    x = (preds - labels)
#    den = 1 + x * x / 2
#    grad = x / np.sqrt(den)
#    hess = 1 / np.sqrt(den * den * den)
#    return grad, hess

#this is used to find early stopping rounds, then scale up when using all of dataset
#as number of rounds should be proportional to size of dataset


#result_xgb_2 = fit_xgb_model(train,test,params_2, xgb_features_2,use_early_stopping = True,
#                              print_feature_imp = True, random_seed = 6, use_custom_obj = True, obj = fair_obj)
num_rounds_2 = 8511
if is_sub_run:
    num_rounds_2 /= (0.8 * 0.66)
else:
    num_rounds_2 /= (0.8)
num_rounds_2 = int(num_rounds_2)
result_xgb_2 = fit_xgb_model(train,test,params_2,xgb_features_2,
                              num_rounds = num_rounds_2,use_early_stopping = False,
                              random_seed = 6, use_custom_obj = True, obj = fair_obj)
#result_xgb_2['pred'] = result_xgb_2['pred'].map(lambda x: 1 if x <= 0 else x)
result_xgb_2['pred'] = np.abs(result_xgb_2['pred'])
result_xgb_samp_2 = result_xgb_2.sample(frac = 0.1,random_state = 3)

DF_LIST_2 = []
for (inTr, inTe) in folds:
    xtr = train.iloc[inTr].copy()
    xte = train.iloc[inTe].copy()

    #for 5 fold cv don't change num rounds
    num_rounds_2 = 8511
    if is_sub_run:
        num_rounds_2 /= (0.66)
    else:
        pass
    num_rounds_2 = int(num_rounds_2)
    result_xgb_temp = fit_xgb_model(xtr,xte,params_2,xgb_features_2,
                              num_rounds = num_rounds_2,use_early_stopping = False,
                              random_seed = 6, use_custom_obj = True, obj = fair_obj)

    DF_LIST_2.append(result_xgb_temp)
res_oob_xgb_2 = pd.concat(DF_LIST_2,ignore_index=True)
res_oob_xgb_2.index = res_oob_xgb_2['id']
if not is_sub_run:
    print('mae',round(res_oob_xgb_2['error'].mean(),5))
#%%
#result_xgb_2_copy = result_xgb_2.copy() #16650 rounds, 1132.19 mae
#%%
xgb_features_3 = []
xgb_features_3 += base_features
xgb_features_3 += ['sum_c1to72']
#xgb_features_3 += diff_names
#xgb_features_3 += sum_names
#xgb_features_3 += mult_names

#xgb_features_3 += ['multcont_2_14']
#xgb_features_3 += ['multcont_2_7']
#xgb_features_3 += ['multcont_2_8']
#xgb_features_3 += ['sum_cont_2_7']
#xgb_features_3 += ['diff_cont_2_7']
#xgb_features_3 += ['freq_cont14']
#xgb_features_3 += ['freq_cont6']
#xgb_features_3 += ['freq_cont7']

#xgb_features_3 += freq_features

#xgb_features_3 += ['max_cont_freq','min_cont_freq','mean_cont_freq','median_cont_freq','std_cont_freq']
#xgb_features_3 += ['mean_cont_freq','median_cont_freq']
#xgb_features_3 += ['max_cont_agg','min_cont_agg','mean_cont_agg','median_cont_agg','std_cont_agg']
#xgb_features_3 += ['min_cont_agg','mean_cont_agg']

#xgb_features_3 += features_by_med
#xgb_features_3 = [name for name in xgb_features_3 if name not in cont_feat_list]
#xgb_features_3 += ['id_shift']
#xgb_features_3.remove('cont2')

params_3 = {'learning_rate': 0.005,
              'subsample': 0.95,
              'reg_alpha': 0.5,
#              'lambda': 0.995,
              'gamma': 1.5,
              'seed': 5,
              'colsample_bytree': 0.4,
#              'colsample_bylevel': 0.3,
#              'n_estimators': 100,
              'base_score': 10,
              'objective': 'reg:linear',
              'eval_metric':'mae',
#              'min_child_weight': 3,
#              'eval_metric':'rmse',
              'max_depth': 6,
              }

train_samp = train.sample(frac=0.2,random_state=3)

#result_xgb_3 = fit_xgb_model(train,test,params_3, xgb_features_3,use_early_stopping = True,
#                              print_feature_imp = True, use_log_transform = False, random_seed = 6, use_weights= False,
#                              log_const = 100, use_power_transform = True, power_constant = 0.25, power_shift = 0)
num_rounds_3 = 9710
if is_sub_run:
    num_rounds_3 /= (0.8 * 0.66)
else:
    num_rounds_3 /= (0.8)
num_rounds_3 = int(num_rounds_3)
result_xgb_3 = fit_xgb_model(train,test,params_3,xgb_features_3,
                              num_rounds = num_rounds_3, use_log_transform = False,
                              use_early_stopping = False,random_seed = 6,use_weights=False,
                               log_const = 100, use_power_transform = True, power_constant = 0.25, power_shift = 0)

result_xgb_samp_3 = result_xgb_3.sample(frac = 0.1,random_state = 3)

DF_LIST_3 = []
for (inTr, inTe) in folds:
    xtr = train.iloc[inTr].copy()
    xte = train.iloc[inTe].copy()
    num_rounds_3 = 9710
    if is_sub_run:
        num_rounds_3 /= (0.66)
    else:
        pass
    num_rounds_3 = int(num_rounds_3)
    result_xgb_temp = fit_xgb_model(xtr,xte,params_3,xgb_features_3,
                                  num_rounds = num_rounds_3, use_log_transform = False,
                                  use_early_stopping = False,random_seed = 6,use_weights=False,
                                   log_const = 100, use_power_transform = True, power_constant = 0.25, power_shift = 0)

    DF_LIST_3.append(result_xgb_temp)
res_oob_xgb_3 = pd.concat(DF_LIST_3,ignore_index=True)
res_oob_xgb_3.index = res_oob_xgb_3['id']
if not is_sub_run:
    print('mae',round(res_oob_xgb_3['error'].mean(),5))
#%%
xgb_features_4 = []
xgb_features_4 += base_features

xgb_features_4 += ['sum_c1to72']
#xgb_features_4 += ['c1to72_all']

params_4 = {'learning_rate': 0.01,
              'subsample': 0.98,
              'reg_alpha': 1.5,
              'lambda': 0.99,
              'gamma': 1.25,
              'seed': 5,
              'colsample_bytree': 0.25,
#              'colsample_bylevel': 0.9,
#              'n_estimators': 100,
              'objective': 'reg:linear',
              'eval_metric':'mae',
#              'min_child_weight': 2,
#              'eval_metric':'rmse',
              'max_depth': 10,
              }

#train_samp = train.sample(frac=0.3,random_state=3)

#result_xgb_4 = fit_xgb_model(train,test,params_4, xgb_features_4,use_early_stopping = True,
#                              print_feature_imp = True, use_log_transform = True, random_seed = 6, use_weights= False,
#                              log_const = 200)

num_rounds_4 = 4783
if is_sub_run:
    num_rounds_4 /= (0.8 * 0.66)
else:
    num_rounds_4 /= (0.8)
num_rounds_4 = int(num_rounds_4)
result_xgb_4 = fit_xgb_model(train,test,params_4,xgb_features_4,
                              num_rounds = num_rounds_4, use_log_transform = True,
                              use_early_stopping = False,random_seed = 6,use_weights=False)

result_xgb_samp_4 = result_xgb_4.sample(frac = 0.1,random_state = 3)

DF_LIST_4 = []
for (inTr, inTe) in folds:
    xtr = train.iloc[inTr].copy()
    xte = train.iloc[inTe].copy()
    num_rounds_4 = 4783
    if is_sub_run:
        num_rounds_4 /= (0.66)
    else:
        pass
    num_rounds_4 = int(num_rounds_4)
    result_xgb_temp = fit_xgb_model(xtr,xte,params_4,xgb_features_4,
                                  num_rounds = num_rounds_4, use_log_transform = True,
                                  use_early_stopping = False,random_seed = 6,use_weights=False)

    DF_LIST_4.append(result_xgb_temp)
res_oob_xgb_4 = pd.concat(DF_LIST_4,ignore_index=True)
res_oob_xgb_4.index = res_oob_xgb_4['id']
if not is_sub_run:
    print('mae',round(res_oob_xgb_4['error'].mean(),5))

#%%
#%%
xgb_features_5 = []
xgb_features_5 += base_features
xgb_features_5 += ['sum_c1to72']

params_5 = {'learning_rate': 0.005,
              'subsample': 0.95,
              'reg_alpha': 0.5,
#              'lambda': 0.995,
              'gamma': 1.5,
              'seed': 5,
              'colsample_bytree': 0.4,
#              'colsample_bylevel': 0.3,
#              'n_estimators': 100,
              'base_score': 10,
              'objective': 'reg:linear',
              'eval_metric':'mae',
#              'min_child_weight': 3,
#              'eval_metric':'rmse',
              'max_depth': 6,
              }

train_samp = train.sample(frac=0.2,random_state=3)

#result_xgb_5 = fit_xgb_model(train,test,params_5, xgb_features_5,use_early_stopping = True,
#                              print_feature_imp = True, use_log_transform = False, random_seed = 6, use_weights= False,
#                              log_const = 100, use_power_transform = True, power_constant = 0.25, power_shift = 0)
num_rounds_5 = 9710
if is_sub_run:
    num_rounds_5 /= (0.8 * 0.66)
else:
    num_rounds_5 /= (0.8)
num_rounds_5 = int(num_rounds_5)
result_xgb_5 = fit_xgb_model(train,test,params_5,xgb_features_5,
                              num_rounds = num_rounds_5, use_log_transform = False,
                              use_early_stopping = False,random_seed = 6,use_weights=False,
                               log_const = 100, use_power_transform = True, power_constant = 0.5, power_shift = 0)

result_xgb_samp_5 = result_xgb_5.sample(frac = 0.1,random_state = 3)

DF_LIST_5 = []
for (inTr, inTe) in folds:
    xtr = train.iloc[inTr].copy()
    xte = train.iloc[inTe].copy()
    num_rounds_5 = 9710
    if is_sub_run:
        num_rounds_5 /= (0.66)
    else:
        pass
    num_rounds_5 = int(num_rounds_5)
    result_xgb_temp = fit_xgb_model(xtr,xte,params_5,xgb_features_5,
                                  num_rounds = num_rounds_5, use_log_transform = False,
                                  use_early_stopping = False,random_seed = 6,use_weights=False,
                                   log_const = 100, use_power_transform = True, power_constant = 0.5, power_shift = 0)

    DF_LIST_5.append(result_xgb_temp)
res_oob_xgb_5 = pd.concat(DF_LIST_5,ignore_index=True)
res_oob_xgb_5.index = res_oob_xgb_5['id']
if not is_sub_run:
    print('mae',round(res_oob_xgb_5['error'].mean(),5))
#%%
#res_oob_xgb_1.to_csv('preds_oob_xgb_1.csv', index = False)
#res_oob_xgb_2.to_csv('preds_oob_xgb_2.csv', index = False)
#res_oob_xgb_3.to_csv('preds_oob_xgb_3.csv', index = False)
#res_oob_xgb_4.to_csv('preds_oob_xgb_4.csv', index = False)
#res_oob_xgb_5.to_csv('preds_oob_xgb_5.csv', index = False)
#%%
#result_xgb_1.to_csv('test_preds_xgb_1.csv', index = False)
#result_xgb_2.to_csv('test_preds_xgb_2.csv', index = False)
#result_xgb_3.to_csv('test_preds_xgb_3.csv', index = False)
#result_xgb_4.to_csv('test_preds_xgb_4.csv', index = False)
#result_xgb_5.to_csv('test_preds_xgb_5.csv', index = False)
#%%
#res_xgb_1 = pd.concat([result_xgb_1,res_oob_xgb_1],ignore_index=True)
#res_xgb_1.index = res_xgb_1['id']
#res_xgb_2 = pd.concat([result_xgb_2,res_oob_xgb_2],ignore_index=True)
#res_xgb_2.index = res_xgb_2['id']
#res_xgb_3 = pd.concat([result_xgb_3,res_oob_xgb_3],ignore_index=True)
#res_xgb_3.index = res_xgb_3['id']
#res_xgb_4 = pd.concat([result_xgb_4,res_oob_xgb_4],ignore_index=True)
#res_xgb_4.index = res_xgb_4['id']
#res_xgb_5 = pd.concat([result_xgb_5,res_oob_xgb_5],ignore_index=True)
#res_xgb_5.index = res_xgb_5['id']
#
#res_xgb_1.to_csv('all_preds_xgb_1.csv', index = False)
#res_xgb_2.to_csv('all_preds_xgb_2.csv', index = False)
#res_xgb_3.to_csv('all_preds_xgb_3.csv', index = False)
#res_xgb_4.to_csv('all_preds_xgb_4.csv', index = False)
#res_xgb_5.to_csv('all_preds_xgb_5.csv', index = False)
#%%
#def get_mae(df):
#    result_1 = pd.merge(df,train[['id','loss_orig']],left_on = ['id'],
#                       right_on = ['id'],how='left')
#    result_1['error'] = np.abs(result_1['loss_orig'] - result_1['pred'])
#    print('mae',round(result_1['error'].mean(),5))
#get_mae(res_oob_xgb_1)
#get_mae(res_oob_xgb_2)
#get_mae(res_oob_xgb_3)
#get_mae(res_oob_xgb_4)
#get_mae(res_oob_xgb_5)
#%%
#tic=timeit.default_timer()
#
#extc_features_1 = []
#extc_features_1 += base_features
##extc_features_1 += freq_features
#extc_features_1 += ['sum_c1to72']
#extc_features_1 += ['max_cont_freq','min_cont_freq','mean_cont_freq','std_cont_freq']
##extc_features_1 += diff_names
##extc_features_1 += sum_names
##extc_features_1 += mult_names
#
#
##train_samp = train.sample(frac=0.3,random_state=3)
##train_data = train_samp[extc_features_1].values
#train_data = train[extc_features_1].values
#log_constant = 10
##train_loss = np.log(train_samp['loss_orig'] + log_constant).astype(float).values
#train_loss = np.log(train['loss_orig'] + log_constant).astype(float).values
#test_data = test[extc_features_1].values
#
#extc_1 = ExtraTreesRegressor(n_estimators=100,
#                            max_features = None,
##                            max_features = 'sqrt',
#                            criterion= 'mse',
#                            min_samples_split= 40,
#                            max_depth= None,
##                            max_depth= 500,
#                            min_samples_leaf= 5,
##                            min_impurity_split = 0.05,
#                            random_state = 1,
#                            n_jobs = 2,
#                            verbose = 2
#                            )
#extc_1 = extc_1.fit(train_data, train_loss)
#
#y_pred = extc_1.predict(test_data)
#columns = ['pred']
#result_extc_1 = pd.DataFrame(index=test.id, columns=columns,data=y_pred)
#result_extc_1.reset_index('id',inplace=True)
#result_extc_1['pred'] = (np.exp(result_extc_1['pred']) - log_constant)
#
#if(is_sub_run):
#    print('is a submission run')
#else:
##    result_xgb_ens['id'] = result_xgb_1['id']
#    result_extc_1 = pd.merge(result_extc_1,test[['id','loss_orig']],left_on = ['id'],
#                           right_on = ['id'],how='left')
#    result_extc_1['error'] = np.abs(result_extc_1['loss_orig'] - result_extc_1['pred'])
#    result_extc_1['error_0'] = result_extc_1['loss_orig'] - result_extc_1['pred']
#    print('mae',result_extc_1.error.mean())
#
#toc=timeit.default_timer()
#print('Extc Time',toc - tic)

#%%
#cont_14_dict = train.groupby('c112')['loss_orig'].mean().to_dict()
#cont_14_err_dict = train.groupby('c112')['loss_orig'].sem().to_dict()
#lists = sorted(cont_14_dict.items())
#x, y = zip(*lists)
#lists_err = sorted(cont_14_err_dict.items())
#x_err, y_error = zip(*lists_err)
#plt.errorbar(x,y,fmt = 'o',yerr = y_error)
#%%

#%%
#for name in train_temp.columns:
#    if name.startswith('cont'):
#        print(name,train_temp[name].min(),train_temp[name].max())

#%%
#result_xgb['pred2'] = result_xgb['pred']
#result_xgb['error_3'] = np.abs(result_xgb['loss'] - (result_xgb['pred2']))
#result_xgb['error_2'] = result_xgb['loss'] - result_xgb['pred2']
#print(result_xgb.loc[result_xgb['pred'] > 5000]['error_2'].mean())
#print('error_3',result_xgb.loc[result_xgb['pred'] > 20000]['error_3'].mean())
#
#
#result_xgb_samp_2 = result_xgb.sample(frac = 0.1,random_state = 4)
#%%
#result_xgb_df['pred_2'] = result_xgb_df['pred'].map(lamb)
#result_xgb_df['error_2'] = np.abs(result_xgb_df['loss'] - result_xgb_df['pred_2'])
#print(result_xgb_df['error_2'].mean())
#%%

#cont14_dict = train.groupby(['c100'])['loss_orig'].median().to_dict()

#test_copy = test.copy()
#test_copy['pred'] = test['c100'].map(lambda x: apply_dict(cont14_dict,x,2115.5))
#test_copy['error'] = np.abs(test_copy['pred'] - test_copy['loss_orig'])
#print(test_copy['error'].mean())

#%%
#script_sub = pd.read_csv('xgstacker_starter_sub.csv', header=0)
#script_sub['loss'] = script_sub['loss'] - 24
#script_sub.to_csv('xgstacker_starter_sub_new.csv', index=False,header=True)
#%%
if is_sub_run:
    result_nn_test_3 = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/Submissions/test_keras_sub3.csv', header=0)
    result_nn_test_3.rename(columns = {'loss':'pred_nn_1'},inplace=True)
    result_nn_test_3['pred'] = result_nn_test_3['pred_nn_1']
#    result_nn_test_3.set_index('id',inplace=True)
#    result_nn_test_3.index.name = None

    result_nn_test_4 = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/Submissions/test_keras_sub_4.csv', header=0)
    result_nn_test_4.rename(columns = {'loss':'pred_nn_2'},inplace=True)
    result_nn_test_4['pred'] = result_nn_test_4['pred_nn_2']
#    result_nn_test_4.set_index('id',inplace=True)

    result_nn_test_5 = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/Submissions/test_keras_sub_4.csv', header=0)
    result_nn_test_5.rename(columns = {'loss':'pred_nn_5'},inplace=True)
    result_nn_test_5['pred'] = result_nn_test_5['pred_nn_5']
#    result_nn_test_5.set_index('id',inplace=True)
#    result_nn_test_4.index.name = None

else:
    result_nn_test_3 = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/Train_Results_Id_2e5/train_keras_5.csv', header=0)
    result_nn_test_3.set_index('id',inplace=True)
    result_nn_test_3.index.name = None
#%%
a1 = 0.5
a2 = 2.5
a3 = 0.2
a4 = 2.5

a5 = 1

a6 = 2

result_xgb_ens = (a1 * result_xgb_1 + a2 * result_xgb_2 + a3 * result_xgb_3
                  + a4 * result_xgb_4
                  + a5 * result_nn_test_3
                  + a6 * result_nn_test_4
                  )
result_xgb_ens = result_xgb_ens[['pred']]
#result_xgb_ens['pred'] = result_xgb_ens['pred'] / (a1 + a2 + a3 + a4)
#result_xgb_ens['pred'] = result_xgb_ens['pred'] / (a1 + a2 + a3 + a4 + a5)
result_xgb_ens['pred'] = result_xgb_ens['pred'] / (a1 + a2 + a3 + a4 + a5 + a6)

#result_xgb_ens['id'] = result_xgb_1['id']
#result_xgb_ens = result_xgb_ens[['id','pred']]

result_xgb_ens['pred1'] = result_xgb_1['pred']
result_xgb_ens['pred2'] = result_xgb_2['pred']
result_xgb_ens['pred3'] = result_xgb_3['pred']
result_xgb_ens['pred4'] = result_xgb_4['pred']
#result_xgb_ens['pred5'] = result_nn_test_3['pred']

#result_xgb_ens['std'] = result_xgb_ens[['pred1','pred2','pred3','pred4','pred5']].std(axis=1)
#result_xgb_ens['min'] = result_xgb_ens[['pred1','pred2','pred3','pred4','pred5']].min(axis=1)
#result_xgb_ens['max'] = result_xgb_ens[['pred1','pred2','pred3','pred4','pred5']].max(axis=1)
result_xgb_ens['std'] = result_xgb_ens[['pred1','pred2','pred3','pred4']].std(axis=1)
result_xgb_ens['min'] = result_xgb_ens[['pred1','pred2','pred3','pred4']].min(axis=1)
result_xgb_ens['max'] = result_xgb_ens[['pred1','pred2','pred3','pred4']].max(axis=1)
result_xgb_ens['max_min'] = result_xgb_ens['max'] - result_xgb_ens['min']
#result_xgb_ens['median'] = result_xgb_ens[['pred1','pred2','pred3','pred4','pred5']].median(axis=1)


#%%
#if is_sub_run:
#    result_xgb_prev = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/Submissions/test_result_xgb_ens_5.csv', header=0)
#
#result_xgb_ens = pd.merge(result_xgb_prev,result_nn_test_5[['id','pred_nn_5']],left_on = ['id'],
#                           right_on = ['id'],how='left')
#c1 = 1
#c2 = 0.3
#result_xgb_ens['pred'] = (c1*result_xgb_ens['pred'] + c2*result_xgb_ens['pred_nn_5']) / (c1 + c2)

#result_xgb_ens = result_nn_test_3.copy()
if is_sub_run:
    result_xgb_ens.to_csv('test_result_xgb_ens.csv',index=True)
else:
    result_xgb_ens['id'] = result_xgb_1['id']
    result_xgb_ens = pd.merge(result_xgb_ens,test[['id','loss_orig']],left_on = ['id'],
                           right_on = ['id'],how='left')
#    result_xgb_ens.to_csv('train_result_xgb_ens.csv',index=False)

if(is_sub_run):
    print('is a submission run')
else:
#    result_xgb_ens['id'] = result_xgb_1['id']
#    result_xgb_ens = pd.merge(result_xgb_ens,test[['id','loss_orig']],left_on = ['id'],
#                           right_on = ['id'],how='left')
    result_xgb_ens['error'] = np.abs(result_xgb_ens['loss_orig'] - result_xgb_ens['pred'])
    result_xgb_ens['error_0'] = result_xgb_ens['loss_orig'] - result_xgb_ens['pred']
    print('mae',result_xgb_ens.error.mean())

low_max_min = result_xgb_ens['max_min'] <= 50
result_xgb_ens['pred'][low_max_min] = result_xgb_ens['pred'][low_max_min] - 51
medium_low_max_min = (result_xgb_ens['max_min'] > 50) & (result_xgb_ens['max_min'] <= 200)
result_xgb_ens['pred'][medium_low_max_min] = result_xgb_ens['pred'][medium_low_max_min] - 40
medium_high_max_min = (result_xgb_ens['max_min'] > 200) & (result_xgb_ens['max_min'] <= 500)
result_xgb_ens['pred'][medium_high_max_min] = result_xgb_ens['pred'][medium_high_max_min] - 5
high_max_min = (result_xgb_ens['max_min'] > 500) & (result_xgb_ens['max_min'] <= 3000)
result_xgb_ens['pred'][high_max_min] = result_xgb_ens['pred'][high_max_min] + 110
very_high_max_min = (result_xgb_ens['max_min'] > 3000)
result_xgb_ens['pred'][very_high_max_min] = result_xgb_ens['pred'][very_high_max_min] + 1350



if(is_sub_run):
    print('is a submission run')
else:
#    result_xgb_ens['id'] = result_xgb_1['id']
#    result_xgb_ens = pd.merge(result_xgb_ens,test[['id','loss_orig']],left_on = ['id'],
#                           right_on = ['id'],how='left')
    result_xgb_ens['error'] = np.abs(result_xgb_ens['loss_orig'] - result_xgb_ens['pred'])
#    result_xgb_ens['error_0'] = result_xgb_ens['loss_orig'] - result_xgb_ens['pred']
    print('mae',result_xgb_ens.error.mean())

#%%
#result_xgb_ens = pd.read_csv('train_result_xgb_ens.csv',header=0)

#b0 = 1
#b1 = 1
#b2 = 1
#result_xgb_ens['pred'] = (b0 * result_xgb_ens['min'] + b1 * result_xgb_ens['max'] + b2 * result_xgb_ens['median']) / (b0 + b1 + b2)
#result_xgb_ens['pred'] = result_xgb_ens['median']
#median_train = train['loss_orig'].median()
#def shrink_to_median(x, median_val = median_train, shrink_factor = 100):
#    if x < median_val:
#        x += (median_val - x) / shrink_factor
#    else:
#        x += (median_val - x) / shrink_factor
#    return x
#result_xgb_ens['pred'] = result_xgb_ens['pred'].map(lambda x: shrink_to_median(x))

#xgb_1_log = np.log(result_xgb_1['pred'])
#xgb_2_log = np.log(result_xgb_2['pred'])
#xgb_3_log = np.log(result_xgb_3['pred'])
#
#combined_pred = np.exp((0.33 * xgb_1_log + 0.33 * xgb_2_log +
#                            0.34*xgb_3_log))
#result_xgb_ens['pred'] = combined_pred.values

#low_std = result_xgb_ens['std'] < 50
#result_xgb_ens['pred'][low_std] = result_xgb_ens['pred'][low_std] - 15
#medium_low_std = (result_xgb_ens['std'] > 100) & (result_xgb_ens['std'] < 1000)
#result_xgb_ens['pred'][medium_low_std] = result_xgb_ens['pred'][medium_low_std] + 40
#medium_high_std = (result_xgb_ens['std'] > 1000) & (result_xgb_ens['std'] < 5000)
#result_xgb_ens['pred'][medium_high_std] = result_xgb_ens['pred'][medium_high_std] + 80
#high_std = result_xgb_ens['std'] > 3000
#result_xgb_ens['pred'][high_std] = result_xgb_ens['pred'][high_std] + 5000

#result_xgb_load = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Allstate/Submissions/test_result_xgb_ens_2.csv', header=0)
#
#result_xgb_ens = pd.merge(result_xgb_load,result_nn_test_3,on='id')
#result_xgb_ens = pd.merge(result_xgb_ens,result_nn_test_4,on='id')
#
#b1 = 0.5
#b2 = 1
#b3 = 2
#b4 = 3
#
#b5 = 1
#b6 = 1
#b_sum = (b1 + b2 + b3 + b4 + b5 + b6)
#
#result_xgb_ens['pred'] = (result_xgb_ens['pred1'] * b1 + result_xgb_ens['pred2'] * b2 +
#                         result_xgb_ens['pred3'] * b3 + result_xgb_ens['pred4'] * b4 +
#                         result_xgb_ens['pred_nn_1'] * b5 + result_xgb_ens['pred_nn_2'] * b6) / b_sum

#result_xgb_ens.to_csv('test_result_xgb_ens.csv',index=True)
#%%
#result_xgb_ens = result_xgb_4.copy()
if(is_sub_run):
    print('is a submission run')
else:
#    result_xgb_ens['id'] = result_xgb_1['id']
#    result_xgb_ens = pd.merge(result_xgb_ens,test[['id','loss_orig']],left_on = ['id'],
#                           right_on = ['id'],how='left')
    result_xgb_ens['error'] = np.abs(result_xgb_ens['loss_orig'] - result_xgb_ens['pred'])
    result_xgb_ens['error_0'] = result_xgb_ens['loss_orig'] - result_xgb_ens['pred']
    print('mae',result_xgb_ens.error.mean())


if(is_sub_run):
    submission = result_xgb_ens.copy()
    submission.reset_index('id',inplace=True)
    submission.rename(columns = {'pred':'loss'},inplace=True)
    submission = submission[['id','loss']]
    submission.to_csv('allstate_sub.csv', index=False)
    print('xgb submission created')
#%%
#plt.scatter(result_xgb_ens['std'],result_xgb_ens['loss_orig'],alpha = 0.1)
#plt.xlim(0, 5000)
#%%
toc=timeit.default_timer()
print('Total Time',toc - tic0)