# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 01:24:08 2016

@author: Jared
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 00:46:15 2015

@author: Jared
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from sklearn import cross_validation
from sklearn.cross_validation import KFold, train_test_split, cross_val_score
#from sklearn import preprocessing
from scipy.stats import uniform as sp_rand
from scipy.stats import randint as sp_randint
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV,Lasso,ElasticNetCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble
from sklearn.metrics import make_scorer,log_loss
#import xgboost as xgb
import datetime as datetime
import xgboost as xgb
import operator

import scipy.stats as sp_stats
#import scipy.sparse
#import pickle
import timeit
#import sklearn_mod.linear_model as sk_mod_lin
#from sklearn_mod.linear_model import LassoCV as LassoCV_mod

#import pickle
#
#from lasagne.layers import InputLayer, DropoutLayer, DenseLayer
#from lasagne.updates import nesterov_momentum
#from lasagne.objectives import binary_crossentropy
#from nolearn.lasagne import NeuralNet
#import theano
#from theano import tensor as T
#from theano.tensor.nnet import sigmoid
#
#from sklearn.preprocessing import LabelEncoder
#from sklearn.cross_validation import train_test_split
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import log_loss, auc, roc_auc_score
#from sklearn import metrics
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.updates import adagrad, nesterov_momentum
from lasagne.nonlinearities import softmax
from lasagne.objectives import binary_crossentropy
#import theano

from sklearn.preprocessing import StandardScaler
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer
from lasagne.updates import nesterov_momentum
from lasagne.objectives import binary_crossentropy
from lasagne.init import Uniform
from nolearn.lasagne import NeuralNet, BatchIterator
import theano
from theano import tensor as T
from theano.tensor.nnet import sigmoid


tic0=timeit.default_timer()


pd.options.mode.chained_assignment = None  # default='warn'

#%%
tic=timeit.default_timer()
train_orig = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Telstra/train.csv', header=0)
test_orig = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Telstra/test.csv', header=0)
severity_type = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Telstra/severity_type.csv', header=0)
log_feature = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Telstra/log_feature.csv', header=0)
event_type = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Telstra/event_type.csv', header=0)
resource_type = pd.read_csv('/Users/Jared/DataAnalysis/Kaggle/Telstra/resource_type.csv', header=0)

toc=timeit.default_timer()
print('Load Time',toc - tic)
#%%
is_sub_run = False
#is_sub_run = True
#%%
def norm_rows(df):
    with np.errstate(invalid='ignore'):
        return df.div(df.sum(axis=1), axis=0).fillna(0)
def norm_row_predictions(df):
#    with np.errstate(invalid='ignore'):
    sum_value = df[['predict_0','predict_1','predict_2']].sum(axis=1)
    df['predict_0'] = df['predict_0'] / sum_value
    df['predict_1'] = df['predict_1'] / sum_value
    df['predict_2'] = df['predict_2'] / sum_value
    return df
def norm_cols(df):
    with np.errstate(invalid='ignore'):
        return df.div(df.sum(axis=0), axis=1).fillna(0)
#    return df.div(df.sum(axis=1), axis=0)
### obtain string of top10 hashes according to similarity scores for every user
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

def print_value_counts(input_df,col_name,is_normalized=False):
    for value in input_df[col_name].unique():
        print(value)
        print(input_df.loc[input_df[col_name] == value].fault_severity.value_counts(normalize=is_normalized))

def print_value_counts_spec(input_df,col_name,col_value,is_normalized=False):
    print(col_value)
    print(input_df.loc[input_df[col_name] == col_value].fault_severity.value_counts(normalize=is_normalized))

def customized_eval(preds, dtrain):
    labels = dtrain.get_label()
    top = []
    for i in range(preds.shape[0]):
        top.append(np.argsort(preds[i])[::-1][:5])
    mat = np.reshape(np.repeat(labels,np.shape(top)[1]) == np.array(top).ravel(),np.array(top).shape).astype(int)
    score = np.mean(np.sum(mat/np.log2(np.arange(2, mat.shape[1] + 2)),axis = 1))
    return 'ndcg5', score
def convert_strings_to_ints(input_df,col_name,output_col_name):
    labels, levels = pd.factorize(input_df[col_name])
    input_df[output_col_name] = labels
    output_dict = dict(zip(input_df[col_name],input_df[output_col_name]))
    return (output_dict,input_df)

def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    idea from this post:
    http://www.kaggle.com/c/emc-data-science/forums/t/2149/is-anyone-noticing-difference-betwen-validation-and-leaderboard-error/12209#post12209

    Parameters
    ----------
    y_true : array, shape = [n_samples]
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    rows = actual.shape[0]
    actual[np.arange(rows), y_true.astype(int)] = 1
    vsota = np.sum(actual * np.log(predictions))
    return -1.0 / rows * vsota

#assumes row normalized, doesn't do eps thing
def get_log_loss_row(row):
    ans = row['fault_severity']
    if (ans == 0):
        return -1.0 * np.log(row['predict_0'])
    elif (ans == 1):
        return -1.0 * np.log(row['predict_1'])
    elif (ans == 2):
        return -1.0 * np.log(row['predict_2'])
    else:
        print('not_exceptable_value')
        raise ValueError('Not of correct class')
        return -1000
def get_log_loss_row_two_classes(row):
    ans = row['fault_severity']
    if (ans == 0):
        return -1.0 * np.log(row['predict_low'])
    elif (ans == 1):
        return -1.0 * np.log(row['predict_high'])
    else:
        print('not_exceptable_value')
        raise ValueError('Not of correct class')
        return -1000

def get_second_most_common(x):
    try:
        return x.value_counts().index[1]
    except IndexError:
        return -1
#%%
test_orig['fault_severity'] = 'dummy'
#if(is_sub_run):
#    combined = pd.concat([train_users, test_users], axis=0,ignore_index=True)
#else:
#    combined = train_users

#%%
severity_type['severity_type'] = severity_type['severity_type'].map(lambda x: x.replace(' ','_'))
severity_type['severity_type'] = severity_type['severity_type'].map(lambda x: x[-1])
severity_type_value_counts = severity_type.severity_type.value_counts().to_dict()
severity_type['num_ids_with_severity_type'] = severity_type['severity_type'].map(lambda x: severity_type_value_counts[x])
severity_type_dummies = pd.get_dummies(severity_type,columns=['severity_type','num_ids_with_severity_type'])
#%%
severity_type['severity_shifted_up'] = severity_type['severity_type'].shift(1)
severity_type['severity_shifted_up'].fillna('2',inplace=True)
severity_type['is_next_severity_type_repeat'] = 0
severity_type['is_next_severity_type_different'] = 0
repeat_cond = severity_type['severity_shifted_up'] == severity_type['severity_type']
severity_type['is_next_severity_type_repeat'][repeat_cond] = 1
severity_type['is_next_severity_type_different'][~repeat_cond] = 1
#%%
event_type['event_type'] = event_type['event_type'].map(lambda x: x.replace(' ','_'))
event_type['event_type'] = event_type['event_type'].map(lambda x: x[11:])
event_type_value_counts = event_type.event_type.value_counts().to_dict()
event_type['num_ids_with_event_type'] = event_type['event_type'].map(lambda x: event_type_value_counts[x])
event_type_dummies = pd.get_dummies(event_type,columns=['event_type','num_ids_with_event_type'])
event_type_dummies_collapse = event_type_dummies.groupby('id').sum()
event_type_dummies_collapse.reset_index('id',inplace=True)
#%%
resource_type['resource_type'] = resource_type['resource_type'].map(lambda x: x.replace(' ','_'))
resource_type['resource_type'] = resource_type['resource_type'].map(lambda x: x[14:]).astype(int)
resource_type_value_counts = resource_type.resource_type.value_counts().to_dict()
resource_type['num_ids_with_resource_type'] = resource_type['resource_type'].map(lambda x: resource_type_value_counts[x])
resource_type_dummies = pd.get_dummies(resource_type,columns=['resource_type','num_ids_with_resource_type'])
resource_type_dummies_collapse = resource_type_dummies.groupby('id').sum()
resource_type_dummies_collapse.reset_index('id',inplace=True)
#%%
resource_type['resource_shifted_up'] = resource_type['resource_type'].shift(1)
resource_type['resource_shifted_up'].fillna(8,inplace=True)
resource_type['is_next_resource_type_repeat'] = 0
resource_type['is_next_resource_type_different'] = 0
repeat_cond = resource_type['resource_shifted_up'] == resource_type['resource_type']
resource_type['is_next_resource_type_repeat'][repeat_cond] = 1
resource_type['is_next_resource_type_different'][~repeat_cond] = 1

resource_type['switches_resource_type'] = resource_type['is_next_resource_type_different'].cumsum()

#resource_type_max = resource_type.groupby('id').max().reset_index('id')


#%%
log_feature['log_feature'] = log_feature['log_feature'].map(lambda x: x.replace(' ','_'))
log_feature['log_feature'] = log_feature['log_feature'].map(lambda x: x[8:]).astype(int)

log_feature.reset_index('count_of_log_feature_seen',inplace=True)
log_feature.rename(columns={'index':'count_of_log_feature_seen'},inplace=True)

log_feature_value_counts = log_feature.log_feature.value_counts().to_dict()
log_feature['num_ids_with_log_feature'] = log_feature['log_feature'].map(lambda x: log_feature_value_counts[x])

bins = [-10,0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,
        200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,
        370,380,390,400]
#bins = [-10,0,20,40,60,80,100,120,140,160,180,
#        200,220,240,260,280,300,320,340,360,
#        380,400]
#bins = [-10,0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,
#        195,200,205,210,215,220,225,230,235,240,245,250,255,260,265,270,275,280,285,290,295,300,305,310,315,320,325,330,335,340,345,350,355,360,
#        365,370,375,380,385,390,395,400]
log_feature['binned_log_feature'] = np.digitize(log_feature['log_feature'], bins, right=True)

bins_offset = list(map(lambda x:x+5, bins))
#bins_offset = list(map(lambda x:x+10, bins))
#bins_offset = list(map(lambda x:x+3, bins))
log_feature['binned_offset_log_feature'] = np.digitize(log_feature['log_feature'], bins_offset, right=True)
#%%
log_feature_small = log_feature.copy()
bins = [-10,0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,
        200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,
        370,380,390,400]
log_feature_small = log_feature_small.loc[log_feature_small['num_ids_with_log_feature'] <= 100]
log_feature_small['binned_small_log_feature'] = np.digitize(log_feature_small['log_feature'], bins, right=True)
bins_offset = list(map(lambda x:x+5, bins))
log_feature_small['binned_offset_small_log_feature'] = np.digitize(log_feature_small['log_feature'], bins_offset, right=True)
#%%
log_feature['position_of_log_feature'] = 1
log_feature['position_of_log_feature'] = log_feature.groupby(['id'])['position_of_log_feature'].cumsum()
log_feature['log_feature'] = log_feature['log_feature'].astype(int)
#%%
event_type.reset_index('count_of_event_type_seen',inplace=True)
event_type.rename(columns={'index':'count_of_event_type_seen'},inplace=True)

temp_combined = pd.concat([train_orig, test_orig], axis=0,ignore_index=True)
event_type['event_type'] = event_type['event_type'].astype(int)
event_type_combined = pd.merge(temp_combined,event_type,left_on = ['id'],
                               right_on = ['id'],how='left')
event_type_combined.sort('count_of_event_type_seen',inplace=True)

most_common_event_type_dict = event_type_combined.groupby(['location'])['event_type'].agg(lambda x: x.value_counts().index[0]).to_dict()
std_event_type_dict = event_type_combined.groupby(['location'])['event_type'].agg(lambda x: x.std()).to_dict()
event_type_no_dups = event_type_combined.drop_duplicates('location')
event_type_no_dups_ids = event_type_combined.drop_duplicates('id')
first_ids = set(event_type_no_dups['id'].unique())
#event_type_no_dups['order_by_location'] = 1
#event_type_no_dups['order_by_location'] = even.groupby(['location'])['order_by_location'].cumsum()
event_type_combined['first_id'] = event_type_combined['id'].map(lambda x: 1 if x in first_ids else 0)
event_type_first_ids = event_type_combined.loc[event_type_combined['first_id'] == 1]

min_first_event_type_dict = event_type_first_ids.groupby(['location'])['event_type'].agg(lambda x: x.min()).to_dict()
max_first_event_type_dict = event_type_first_ids.groupby(['location'])['event_type'].agg(lambda x: x.max()).to_dict()
median_first_event_type_dict = event_type_first_ids.groupby(['location'])['event_type'].agg(lambda x: x.median()).to_dict()

#%%
event_type_combined['location_number'] = event_type_combined['location'].map(lambda x: x[9:]).astype(int)
event_type_combined.sort(['location_number','count_of_event_type_seen'],inplace=True)
#%%
temp_combined = pd.concat([train_orig, test_orig], axis=0,ignore_index=True)

log_combined = pd.merge(temp_combined,log_feature,left_on = ['id'],
                               right_on = ['id'],how='left')
log_combined = pd.merge(log_combined,event_type_no_dups_ids[['id','event_type']],left_on = ['id'],
                               right_on = ['id'],how='left')
#log_combined_no_dummies = log_combined.loc[log_combined.fault_severity != 'dummy']
log_combined.sort('count_of_log_feature_seen',inplace=True)
log_feature_unique_features_per_location_dict = log_combined.groupby('location').log_feature.nunique().to_dict()
most_common_feature_dict = log_combined.groupby(['location'])['log_feature'].agg(lambda x: x.value_counts().index[0]).to_dict()

#repeated later
(location_dict,log_combined) = convert_strings_to_ints(log_combined,'location','location_order_by_log_feature')

log_combined['location_number'] = log_combined['location'].map(lambda x: x[9:]).astype(int)
#most_common_location_dict = log_combined.groupby(['log_feature'])['location_order_by_log_feature'].agg(lambda x: x.value_counts().index[0]).to_dict()
most_common_location_dict = log_combined.groupby(['log_feature'])['location_number'].agg(lambda x: x.value_counts().index[0]).to_dict()
most_common_event_dict = log_combined.groupby(['log_feature'])['event_type'].agg(lambda x: x.value_counts().index[0]).to_dict()

second_most_common_feature_dict = log_combined.groupby(['location'])['log_feature'].agg(lambda x: get_second_most_common(x)).to_dict()
mean_volume_dict = log_combined.groupby(['location'])['volume'].agg(lambda x: x.mean()).to_dict()
max_volume_dict = log_combined.groupby(['location'])['volume'].agg(lambda x: x.max()).to_dict()
std_log_feature_dict = log_combined.groupby(['location'])['log_feature'].agg(lambda x: x.std()).to_dict()

log_combined['feature_shifted_up'] = log_combined['log_feature'].shift(1)
log_combined['feature_shifted_up'].fillna(-1,inplace=True)
log_combined['is_immediate_repeat'] = 0
repeat_cond = log_combined['feature_shifted_up'] == log_combined['log_feature']
log_combined['is_immediate_repeat'][repeat_cond] = 1

unique_locations_per_feature_dict = log_combined.groupby('log_feature').location.nunique().to_dict()
#log_combined['log_feature'] = log_combined['log_feature'].astype(int)
median_ids_with_feature_number_dict = log_combined.groupby(['location'])['num_ids_with_log_feature'].agg(lambda x: x.median()).to_dict()
median_feature_number_dict = log_combined.groupby(['location'])['log_feature'].agg(lambda x: x.median()).to_dict()
max_feature_number_dict = log_combined.groupby(['location'])['log_feature'].agg(lambda x: x.max()).to_dict()
min_feature_number_dict = log_combined.groupby(['location'])['log_feature'].agg(lambda x: x.min()).to_dict()

log_combined['location_number'] = log_combined['location'].map(lambda x: x[9:]).astype(int)
log_combined.sort(['location_number','count_of_log_feature_seen'],inplace=True)
#%%
#log_combined.sort('count_of_log_feature_seen',inplace=True)
#%%
log_feature['locations_per_feature'] = log_feature['log_feature'].map(unique_locations_per_feature_dict)
log_feature['most_common_location_of_log_feature'] = log_feature['log_feature'].map(most_common_location_dict)
log_feature['most_common_event_of_log_feature'] = log_feature['log_feature'].map(most_common_event_dict)

#%%
has_event_type_dict = {}
sum_event_type_dict = {}
event_grouped = event_type_combined.groupby(['location'])
for i in event_type_combined.event_type.unique():
    has_event_type_dict[i] = event_grouped['event_type'].agg(lambda x: 1 if i in x.values else 0).to_dict()
    sum_event_type_dict[i] = event_type_combined.loc[event_type_combined.event_type == i].groupby(['location'])['event_type'].agg(lambda x: x.count()).to_dict()



event_type_number_dict = event_type_combined.groupby(['location'])['event_type'].agg(lambda x: x.nunique()).to_dict()
min_event_type_dict = event_type_combined.groupby(['location'])['event_type'].agg(lambda x: x.min()).to_dict()
max_event_type_dict = event_type_combined.groupby(['location'])['event_type'].agg(lambda x: x.max()).to_dict()
median_event_type_dict = event_type_combined.groupby(['location'])['event_type'].agg(lambda x: x.median()).to_dict()
#event_type_combined['location_most_common_event_type']
#%%
event_type_no_dups = event_type_combined.drop_duplicates('id')
#%%
resource_type.reset_index('count_of_resource_type_seen',inplace=True)
resource_type.rename(columns={'index':'count_of_resource_type_seen'},inplace=True)
#%%
resource_type_combined = pd.merge(temp_combined,resource_type,left_on = ['id'],
                               right_on = ['id'],how='left')
resource_type_combined.sort('count_of_resource_type_seen',inplace=True)
most_common_resource_type_dict = resource_type_combined.groupby(['location'])['resource_type'].agg(lambda x: x.value_counts().index[0]).to_dict()
std_resource_type_dict = resource_type_combined.groupby(['location'])['resource_type'].agg(lambda x: x.std()).to_dict()
resource_type_number_dict = resource_type_combined.groupby(['location'])['resource_type'].agg(lambda x: x.nunique()).to_dict()

has_resource_type_dict = {}
resource_grouped = resource_type_combined.groupby(['location'])
for i in resource_type_combined.resource_type.unique():
    has_resource_type_dict[i] = resource_grouped['resource_type'].agg(lambda x: 1 if i in x.values else 0).to_dict()

#%%
severity_type.reset_index('count_of_severity_type_seen',inplace=True)
severity_type.rename(columns={'index':'count_of_severity_type_seen'},inplace=True)
severity_type_combined = pd.merge(temp_combined,severity_type,left_on = ['id'],
                               right_on = ['id'],how='left')
#%%
def get_dummy_frac(x):
    try:
        return x.value_counts(normalize=True)['dummy']
    except KeyError:
        return 0
severity_type_combined.sort('count_of_severity_type_seen',inplace=True)
most_common_severity_type_dict = severity_type_combined.groupby(['location'])['severity_type'].agg(lambda x: x.value_counts().index[0]).to_dict()

location_fraction_dummy_dict = severity_type_combined.groupby(['location'])['fault_severity'].agg(lambda x: get_dummy_frac(x)).to_dict()

severity_type_number_dict = severity_type_combined.groupby(['location'])['severity_type'].agg(lambda x: x.nunique()).to_dict()
#%%
#don't forget about checking volume later
bins = [0,1,2,3,4,5,6,7,8,9,10,15,20,30,60,120,500,1000,2000,10000000]
log_feature['vol_binned'] = np.digitize(log_feature['volume'], bins, right=True)
bins = [0,1,5,10,20,40,80,160,500,1000,2000,10000000]
#bins = [0,1,2,5,10,20,40,100,200,500,10000000]
log_feature['vol_binned_coarse'] = np.digitize(log_feature['volume'], bins, right=True)
(log_feature_ordered_dups,log_feature) = convert_strings_to_ints(log_feature,'log_feature','log_feature_hash')

log_feature['combo_log_feature_and_volume_coarse'] = log_feature.apply(lambda row: str(row['log_feature']) + '_' + str(row['vol_binned_coarse']),axis=1)

log_feature_dummies = pd.get_dummies(log_feature,columns=['log_feature','vol_binned','num_ids_with_log_feature',
                                                          'combo_log_feature_and_volume_coarse','most_common_location_of_log_feature',
                                                          'binned_log_feature','binned_offset_log_feature'])
log_feature_small_dummies = pd.get_dummies(log_feature_small,columns=['binned_small_log_feature','binned_offset_small_log_feature'])
#%%
log_feature_dummies_cols = [col for col in list(log_feature_dummies) if col.startswith('log_feature_')]
log_feature_num_ids_dummies_cols = [col for col in list(log_feature_dummies) if col.startswith('num_ids_with_log_feature_')]
log_feature_by_loc_dummies_cols = [col for col in list(log_feature_dummies) if col.startswith('most_common_location_of_log_feature_')]
log_feature_dummies_binned_cols = [col for col in list(log_feature_dummies) if col.startswith('binned_log_feature_')]
log_feature_dummies_binned_offset_cols = [col for col in list(log_feature_dummies) if col.startswith('binned_offset_log_feature_')]
log_feature_dummies_cols.remove('log_feature_hash')
log_feature_dummies_by_vol = log_feature_dummies[(['id','volume'] + log_feature_dummies_cols + log_feature_dummies_binned_cols +
                                                    log_feature_dummies_binned_offset_cols + log_feature_by_loc_dummies_cols +
                                                    log_feature_num_ids_dummies_cols)]
log_feature_dummies_by_vol.rename(columns = lambda x : ('vol_' + x) if x.startswith('most_common_location_of_log_feature_') else x,inplace=True)
log_feature_dummies_by_vol.rename(columns = lambda x : ('vol_' + x) if x.startswith('log_feature_') else x,inplace=True)
log_feature_dummies_by_vol.rename(columns = lambda x : ('vol_' + x) if x.startswith('num_ids_with_log_feature_') else x,inplace=True)
log_feature_dummies_by_vol.rename(columns = lambda x : ('vol_' + x) if x.startswith('binned_log_feature_') else x,inplace=True)
log_feature_dummies_by_vol.rename(columns = lambda x : ('vol_' + x) if x.startswith('binned_offset_log_feature_') else x,inplace=True)

log_feature_dummies_by_vol_cols = [col for col in list(log_feature_dummies_by_vol) if col.startswith('vol_log_feature_')]
log_feature_num_ids_dummies_by_vol_cols = [col for col in list(log_feature_dummies_by_vol) if col.startswith('vol_num_ids_with_log_feature_')]
log_feature_by_loc_dummies_by_vol_cols = [col for col in list(log_feature_dummies_by_vol) if col.startswith('vol_most_common_location_of_log_feature_')]
log_feature_dummies_by_vol_binned_cols = [col for col in list(log_feature_dummies_by_vol) if col.startswith('vol_binned_log_feature_')]
log_feature_dummies_by_vol_binned_offset_cols = [col for col in list(log_feature_dummies_by_vol) if col.startswith('vol_binned_offset_log_feature_')]
log_feature_dummies_by_vol[log_feature_num_ids_dummies_by_vol_cols] = log_feature_dummies_by_vol[log_feature_num_ids_dummies_by_vol_cols].multiply(log_feature_dummies_by_vol['volume'],axis='index')
log_feature_dummies_by_vol[log_feature_dummies_by_vol_cols] = log_feature_dummies_by_vol[log_feature_dummies_by_vol_cols].multiply(log_feature_dummies_by_vol['volume'],axis='index')
log_feature_dummies_by_vol[log_feature_by_loc_dummies_by_vol_cols] = log_feature_dummies_by_vol[log_feature_by_loc_dummies_by_vol_cols].multiply(log_feature_dummies_by_vol['volume'],axis='index')
log_feature_dummies_by_vol[log_feature_dummies_by_vol_binned_cols] = log_feature_dummies_by_vol[log_feature_dummies_by_vol_binned_cols].multiply(log_feature_dummies_by_vol['volume'],axis='index')
log_feature_dummies_by_vol[log_feature_dummies_by_vol_binned_offset_cols] = log_feature_dummies_by_vol[log_feature_dummies_by_vol_binned_offset_cols].multiply(log_feature_dummies_by_vol['volume'],axis='index')

log_feature_dummies_by_vol_collapse = log_feature_dummies_by_vol.groupby('id').sum()
log_feature_dummies_by_vol_collapse.reset_index('id',inplace=True)

log_feature_dummies_collapse = log_feature_dummies.groupby('id').sum()
log_feature_dummies_collapse.reset_index('id',inplace=True)
#%%
log_feature_small_dummies_binned_cols = [col for col in list(log_feature_small_dummies) if col.startswith('binned_small_log_feature_')]
log_feature_small_dummies_binned_offset_cols = [col for col in list(log_feature_small_dummies) if col.startswith('binned_offset_small_log_feature_')]
log_feature_small_dummies_by_vol = log_feature_small_dummies[(['id','volume'] + log_feature_small_dummies_binned_cols +
                                                    log_feature_small_dummies_binned_offset_cols)]
log_feature_small_dummies_by_vol.rename(columns = lambda x : ('vol_' + x) if x.startswith('binned_small_log_feature_') else x,inplace=True)
log_feature_small_dummies_by_vol.rename(columns = lambda x : ('vol_' + x) if x.startswith('binned_offset_small_log_feature_') else x,inplace=True)

log_feature_small_dummies_by_vol_binned_cols = [col for col in list(log_feature_small_dummies_by_vol) if col.startswith('vol_binned_small_log_feature_')]
log_feature_small_dummies_by_vol_binned_offset_cols = [col for col in list(log_feature_small_dummies_by_vol) if col.startswith('vol_binned_offset_small_log_feature_')]
log_feature_dummies_by_vol[log_feature_small_dummies_by_vol_binned_cols] = log_feature_small_dummies_by_vol[log_feature_small_dummies_by_vol_binned_cols].multiply(log_feature_small_dummies_by_vol['volume'],axis='index')
log_feature_dummies_by_vol[log_feature_small_dummies_by_vol_binned_offset_cols] = log_feature_small_dummies_by_vol[log_feature_small_dummies_by_vol_binned_offset_cols].multiply(log_feature_small_dummies_by_vol['volume'],axis='index')

log_feature_small_dummies_by_vol_collapse = log_feature_small_dummies_by_vol.groupby('id').max()
log_feature_small_dummies_by_vol_collapse.reset_index('id',inplace=True)

#%%
temp_combined = pd.concat([train_orig, test_orig], axis=0,ignore_index=True)

log_feature_dummies_loc = pd.merge(temp_combined[['id','location']],
                                   log_feature_dummies[['id'] + log_feature_dummies_cols +
                                   log_feature_dummies_binned_cols + log_feature_dummies_binned_offset_cols],left_on = ['id'],
                               right_on = ['id'],how='left')
log_feature_dummies_loc_collapse = log_feature_dummies_loc.groupby('location').max()
log_feature_dummies_loc_collapse.reset_index('location',inplace=True)
log_feature_dummies_loc_collapse.rename(columns = lambda x : ('loc_' + x) if x.startswith('log_feature_') else x,inplace=True)
log_feature_dummies_loc_collapse.rename(columns = lambda x : ('loc_' + x) if x.startswith('binned_log_feature_') else x,inplace=True)
log_feature_dummies_loc_collapse.rename(columns = lambda x : ('loc_' + x) if x.startswith('binned_offset_log_feature_') else x,inplace=True)
log_feature_dummies_loc_collapse.drop('id',axis=1,inplace=True)

log_feature_dummies_loc_collapse_sum = log_feature_dummies_loc.groupby('location').sum()
log_feature_dummies_loc_collapse_sum.reset_index('location',inplace=True)
log_feature_dummies_loc_collapse_sum.rename(columns = lambda x : ('summed_loc_' + x) if x.startswith('log_feature_') else x,inplace=True)
log_feature_dummies_loc_collapse_sum.rename(columns = lambda x : ('summed_loc_' + x) if x.startswith('binned_log_feature_') else x,inplace=True)
log_feature_dummies_loc_collapse_sum.rename(columns = lambda x : ('summed_loc_' + x) if x.startswith('binned_offset_log_feature_') else x,inplace=True)
log_feature_dummies_loc_collapse_sum.drop('id',axis=1,inplace=True)
#%%
temp_combined = pd.concat([train_orig, test_orig], axis=0,ignore_index=True)
event_type_dummies_cols = [col for col in list(event_type_dummies) if col.startswith('event_type_')]

event_type_dummies_loc = pd.merge(temp_combined[['id','location']],event_type_dummies[['id'] + event_type_dummies_cols],left_on = ['id'],
                               right_on = ['id'],how='left')
event_type_dummies_loc_collapse = event_type_dummies_loc.groupby('location').max()
event_type_dummies_loc_collapse.reset_index('location',inplace=True)
event_type_dummies_loc_collapse.rename(columns = lambda x : ('loc_' + x) if x.startswith('event_type_') else x,inplace=True)
event_type_dummies_loc_collapse.drop('id',axis=1,inplace=True)

event_type_dummies_loc_collapse_sum = event_type_dummies_loc.groupby('location').sum()
event_type_dummies_loc_collapse_sum.reset_index('location',inplace=True)
event_type_dummies_loc_collapse_sum.rename(columns = lambda x : ('summed_loc_' + x) if x.startswith('event_type_') else x,inplace=True)
event_type_dummies_loc_collapse_sum.drop('id',axis=1,inplace=True)
#%%
log_feature_count = log_feature.groupby('id').count().reset_index('id')
log_feature_count['count_log_features'] = log_feature_count.log_feature
#%%
unique_volumes_dict = log_feature.groupby('id')['volume'].nunique().to_dict()
unique_most_common_locations_dict = log_feature.groupby('id')['most_common_location_of_log_feature'].nunique().to_dict()
unique_most_common_events_dict = log_feature.groupby('id')['most_common_event_of_log_feature'].nunique().to_dict()
#%%
log_feature_sum = log_feature.groupby('id').sum().reset_index('id')
log_feature_sum['sum_log_features_volume'] = log_feature_sum.volume
log_feature_sum['sum_num_ids_with_log_feature'] = log_feature_sum['num_ids_with_log_feature']
log_feature_sum['sum_locations_per_feature'] = log_feature_sum['locations_per_feature']

log_feature_mean = log_feature.groupby('id').mean().reset_index('id')
log_feature_mean['mean_log_features_volume'] = log_feature_mean.volume
log_feature_mean['mean_num_ids_with_log_feature'] = log_feature_mean['num_ids_with_log_feature']

log_feature_median = log_feature.groupby('id').median().reset_index('id')
log_feature_median['median_log_features_volume'] = log_feature_median.volume
log_feature_median['median_log_feature'] = log_feature_median['log_feature']
log_feature_median['median_log_feature_most_common_event'] = log_feature_median['most_common_event_of_log_feature']
log_feature_median['median_log_feature_most_common_location'] = log_feature_median['most_common_location_of_log_feature']

log_feature_max = log_feature.groupby('id').max().reset_index('id')
log_feature_max['max_log_features_volume'] = log_feature_max.volume
log_feature_max['max_num_ids_with_log_feature'] = log_feature_max['num_ids_with_log_feature']
log_feature_max['max_locations_per_feature'] = log_feature_max['locations_per_feature']
log_feature_max['max_log_feature'] = log_feature_max['log_feature']

log_feature_min = log_feature.groupby('id').min().reset_index('id')
log_feature_min['min_log_features_volume'] = log_feature_min.volume
log_feature_min['min_num_ids_with_log_feature'] = log_feature_min['num_ids_with_log_feature']
log_feature_min['min_locations_per_feature'] = log_feature_min['locations_per_feature']
log_feature_min['min_log_feature'] = log_feature_min['log_feature']

min_log_feature_dict = pd.Series(log_feature_min['min_log_feature'].values,index=log_feature_min['id']).to_dict()

log_feature_std = log_feature.groupby('id').std().reset_index('id')
log_feature_std['std_log_features_volume'] = log_feature_std.volume
log_feature_std['std_log_feature'] = log_feature_std['log_feature']
log_feature_std['std_num_ids_with_log_feature'] = log_feature_std['num_ids_with_log_feature']
#%%
event_type_median = event_type.groupby('id').median().reset_index('id')
event_type_median['median_event_type'] = event_type_median['event_type']

event_type_min = event_type.groupby('id').min().reset_index('id')
event_type_min['min_event_type'] = event_type_min['event_type']

event_type_max = event_type.groupby('id').max().reset_index('id')
event_type_max['max_event_type'] = event_type_max['event_type']
#%%
resource_type_median = resource_type.groupby('id').median().reset_index('id')
resource_type_median['median_resource_type'] = resource_type_median['resource_type']

resource_type_min = resource_type.groupby('id').min().reset_index('id')
resource_type_min['min_resource_type'] = resource_type_min['resource_type']

resource_type_max = resource_type.groupby('id').max().reset_index('id')
resource_type_max['max_resource_type'] = resource_type_max['resource_type']
resource_type_max['max_is_next_resource_type_repeat'] = resource_type_max['is_next_resource_type_repeat']
#%%
log_feature_volume_max = log_feature.copy()
log_feature_volume_max.sort('volume',ascending=False,inplace=True)
log_feature_volume_max.drop_duplicates('id',inplace=True)
log_feature_volume_max.rename(columns = {'log_feature':'max_volume_log_feature',
                                         'most_common_location_of_log_feature':'max_vol_log_feature_by_most_common_location'},inplace=True)
#%%
log_feature_volume_min = log_feature.copy()
log_feature_volume_min.sort('volume',ascending=True,inplace=True)
log_feature_volume_min.drop_duplicates('id',inplace=True)
log_feature_volume_min.rename(columns = {'log_feature':'min_volume_log_feature',
                                         'most_common_location_of_log_feature':'min_vol_log_feature_by_most_common_location'},inplace=True)
#%%
log_feature_volume_of_min_feature = log_feature.copy()
log_feature_volume_of_min_feature.sort(['id','log_feature'],ascending=True,inplace=True)
log_feature_volume_of_min_feature.drop_duplicates('id',inplace=True)
log_feature_volume_of_min_feature.rename(columns = {'volume':'volume_of_min_log_feature'},inplace=True)
#%%
log_feature_volume_of_max_feature = log_feature.copy()
log_feature_volume_of_max_feature.sort(['id','log_feature'],ascending=False,inplace=True)
log_feature_volume_of_max_feature.drop_duplicates('id',inplace=True)
log_feature_volume_of_max_feature.rename(columns = {'volume':'volume_of_max_log_feature'},inplace=True)
#%%
log_feature_volume_of_rarest_feature = log_feature.copy()
log_feature_volume_of_rarest_feature.sort(['id','num_ids_with_log_feature'],ascending=True,inplace=True)
log_feature_volume_of_rarest_feature.drop_duplicates('id',inplace=True)
log_feature_volume_of_rarest_feature.rename(columns = {'volume':'volume_of_rarest_log_feature',
                                                       'log_feature':'rarest_log_feature',
                                                      },inplace=True)

log_feature_volume_of_least_rare_feature = log_feature.copy()
log_feature_volume_of_least_rare_feature.sort(['id','num_ids_with_log_feature'],ascending=False,inplace=True)
log_feature_volume_of_least_rare_feature.drop_duplicates('id',inplace=True)
log_feature_volume_of_least_rare_feature.rename(columns = {'volume':'volume_of_least_rare_log_feature','log_feature':'least_rare_log_feature'},inplace=True)
#%%

#%%
LOG_FEATURE_DICT = {}
for i in range(1,21):
    temp_df = log_feature.loc[log_feature['position_of_log_feature'] == i]
    temp_df = temp_df[['id','log_feature','volume']]
    log_feat_name = 'position_log_feature_' + str(i)
    volume_name = 'position_volume_'+ str(i)
    temp_df.rename(columns = {'log_feature':log_feat_name,'volume':volume_name},inplace=True)
    LOG_FEATURE_DICT[i] = temp_df
#%%
resource_type['position_of_resource_type'] = 1
resource_type['position_of_resource_type'] = resource_type.groupby(['id'])['position_of_resource_type'].cumsum()

RESOURCE_TYPE_DICT = {}
for i in range(1,6):
    temp_df = resource_type.loc[resource_type['position_of_resource_type'] == i]
    temp_df = temp_df[['id','resource_type']]
    resource_type_feat_name = 'position_resource_type_' + str(i)
    temp_df.rename(columns = {'resource_type':resource_type_feat_name},inplace=True)
    RESOURCE_TYPE_DICT[i] = temp_df
#%%
event_type['position_of_event_type'] = 1
event_type['position_of_event_type'] = event_type.groupby(['id'])['position_of_event_type'].cumsum()

EVENT_TYPE_DICT = {}
for i in range(1,11):
    temp_df = event_type.loc[event_type['position_of_event_type'] == i]
    temp_df = temp_df[['id','event_type']]
    event_type_feat_name = 'position_event_type_' + str(i)
    temp_df.rename(columns = {'event_type':event_type_feat_name},inplace=True)
    EVENT_TYPE_DICT[i] = temp_df
#%%
event_type_count = event_type.groupby('id').count().reset_index('id')
event_type_count['count_event_types'] = event_type_count.event_type
#%%
resource_type_count = resource_type.groupby('id').count().reset_index('id')
resource_type_count['count_resource_types'] = resource_type_count.resource_type
#%%
log_feature.sort('count_of_log_feature_seen',inplace=True)
log_feature_no_dups = log_feature.drop_duplicates('id')

resource_type.sort('count_of_resource_type_seen',inplace=True)
resource_type_no_dups = resource_type.drop_duplicates('id')
event_type_no_dups = event_type.drop_duplicates('id')
log_feature_no_dups.reset_index(drop=True,inplace=True)
log_feature_no_dups.reset_index('order_of_log_feature',inplace=True)
log_feature_no_dups.rename(columns={'index':'order_of_log_feature','count_of_log_feature_seen':'count_of_log_feature_seen_no_dups'},inplace=True)

temp_combined = pd.concat([train_orig, test_orig], axis=0,ignore_index=True)
log_feature_no_dups_combined = pd.merge(temp_combined,log_feature_no_dups,left_on = ['id'],
                               right_on = ['id'],how='left')
log_feature_no_dups_combined = pd.merge(log_feature_no_dups_combined , severity_type[['id','severity_type','is_next_severity_type_different']],left_on = ['id'],
                               right_on = ['id'],how='left')
log_feature_no_dups_combined = pd.merge(log_feature_no_dups_combined , log_feature_sum[['id','sum_log_features_volume']],left_on = ['id'],
                               right_on = ['id'],how='left')
log_feature_no_dups_combined = pd.merge(log_feature_no_dups_combined , resource_type_no_dups[['id','switches_resource_type','resource_type',
                                             'is_next_resource_type_different']],left_on = ['id'],
                               right_on = ['id'],how='left')
log_feature_no_dups_combined = pd.merge(log_feature_no_dups_combined , event_type_no_dups[['id','event_type']],left_on = ['id'],
                               right_on = ['id'],how='left')

log_feature_no_dups_combined = pd.merge(log_feature_no_dups_combined , resource_type_count[['id','count_resource_types']],left_on = ['id'],
                               right_on = ['id'],how='left')
log_feature_no_dups_combined = pd.merge(log_feature_no_dups_combined , event_type_count[['id','count_event_types']],left_on = ['id'],
                               right_on = ['id'],how='left')

log_feature_no_dups_combined = pd.merge(log_feature_no_dups_combined , log_feature_count[['id','count_log_features']],left_on = ['id'],
                               right_on = ['id'],how='left')

log_feature_no_dups_combined.sort('order_of_log_feature',inplace=True)

(location_dict_ordered,log_feature_no_dups_combined) = convert_strings_to_ints(log_feature_no_dups_combined,'location','location_hash')

log_feature_no_dups_combined.sort('order_of_log_feature',ascending=False,inplace=True)
(location_dict_ordered,log_feature_no_dups_combined) = convert_strings_to_ints(log_feature_no_dups_combined,'location','location_hash_rev')
log_feature_no_dups_combined.sort('order_of_log_feature',inplace=True)


log_feature_mean_count_per_location_dict = log_feature_no_dups_combined.groupby('location').count_log_features.mean().to_dict()


log_feature_no_dups_combined['location_by_mean_count_features'] = (
            log_feature_no_dups_combined['location'].map(lambda x: log_feature_mean_count_per_location_dict[x]))

has_event_type_names = []
for key in has_event_type_dict:
    col_name = 'location_by_has_event_type_' + str(key)
    log_feature_no_dups_combined[col_name] = (
            log_feature_no_dups_combined['location'].map(lambda x: has_event_type_dict[key][x]))
    has_event_type_names.append(col_name)
has_resource_type_names = []
for key in has_resource_type_dict:
    col_name = 'location_by_has_resource_type_' + str(key)
    log_feature_no_dups_combined[col_name] = (
            log_feature_no_dups_combined['location'].map(lambda x: has_resource_type_dict[key][x]))
    has_resource_type_names.append(col_name)


log_feature_no_dups_combined['location_by_unique_features'] = (
            log_feature_no_dups_combined['location'].map(lambda x: log_feature_unique_features_per_location_dict[x]))
log_feature_no_dups_combined['location_by_unique_resource_types'] = (
            log_feature_no_dups_combined['location'].map(lambda x: resource_type_number_dict[x]))
log_feature_no_dups_combined['location_by_unique_event_types'] = (
            log_feature_no_dups_combined['location'].map(lambda x: event_type_number_dict[x]))
log_feature_no_dups_combined['location_by_unique_severity_types'] = (
            log_feature_no_dups_combined['location'].map(lambda x: severity_type_number_dict[x]))

log_feature_no_dups_combined['location_by_most_common_event'] = (
            log_feature_no_dups_combined['location'].map(lambda x: most_common_event_type_dict[x]))
log_feature_no_dups_combined['location_by_min_event'] = (
            log_feature_no_dups_combined['location'].map(lambda x: min_event_type_dict[x]))
log_feature_no_dups_combined['location_by_max_event'] = (
            log_feature_no_dups_combined['location'].map(lambda x: max_event_type_dict[x]))
log_feature_no_dups_combined['location_by_median_event'] = (
            log_feature_no_dups_combined['location'].map(lambda x: median_event_type_dict[x]))

log_feature_no_dups_combined['location_by_min_first_event'] = (
            log_feature_no_dups_combined['location'].map(lambda x: min_first_event_type_dict[x]))
log_feature_no_dups_combined['location_by_max_first_event'] = (
            log_feature_no_dups_combined['location'].map(lambda x: max_first_event_type_dict[x]))
log_feature_no_dups_combined['location_by_median_first_event'] = (
            log_feature_no_dups_combined['location'].map(lambda x: median_first_event_type_dict[x]))

log_feature_no_dups_combined['count_of_volumes'] = (
            log_feature_no_dups_combined['id'].map(lambda x: unique_volumes_dict[x]))
log_feature_no_dups_combined['count_of_most_common_location_log_feature'] = (
            log_feature_no_dups_combined['id'].map(lambda x: unique_most_common_locations_dict[x]))
log_feature_no_dups_combined['count_of_most_common_event_log_feature'] = (
            log_feature_no_dups_combined['id'].map(lambda x: unique_most_common_events_dict[x]))

log_feature_no_dups_combined['location_by_most_common_resource'] = (
            log_feature_no_dups_combined['location'].map(lambda x: most_common_resource_type_dict[x]))
log_feature_no_dups_combined['location_by_std_resource'] = (
            log_feature_no_dups_combined['location'].map(lambda x: std_resource_type_dict[x]))
log_feature_no_dups_combined['location_by_std_log_feature'] = (
            log_feature_no_dups_combined['location'].map(lambda x: std_log_feature_dict[x]))
log_feature_no_dups_combined['location_by_std_event_type'] = (
            log_feature_no_dups_combined['location'].map(lambda x: std_event_type_dict[x]))
log_feature_no_dups_combined['location_by_most_common_severity'] = (
            log_feature_no_dups_combined['location'].map(lambda x: most_common_severity_type_dict[x]))
log_feature_no_dups_combined['location_by_most_common_feature'] = (
            log_feature_no_dups_combined['location'].map(lambda x: most_common_feature_dict[x]))

log_feature_no_dups_combined['location_by_second_most_common_feature'] = (
            log_feature_no_dups_combined['location'].map(lambda x: second_most_common_feature_dict[x]))
log_feature_no_dups_combined['location_by_median_feature_number'] = (
            log_feature_no_dups_combined['location'].map(lambda x: median_feature_number_dict[x]))
log_feature_no_dups_combined['location_by_median_ids_with_feature'] = (
            log_feature_no_dups_combined['location'].map(lambda x: median_ids_with_feature_number_dict[x]))
log_feature_no_dups_combined['location_by_max_feature_number'] = (
            log_feature_no_dups_combined['location'].map(lambda x: max_feature_number_dict[x]))
log_feature_no_dups_combined['location_by_min_feature_number'] = (
            log_feature_no_dups_combined['location'].map(lambda x: min_feature_number_dict[x]))

log_feature_no_dups_combined['location_by_frac_dummy'] = (
            log_feature_no_dups_combined['location'].map(lambda x: location_fraction_dummy_dict[x]))

log_feature_no_dups_combined['location_by_mean_volume'] = (
            log_feature_no_dups_combined['location'].map(lambda x: mean_volume_dict[x]))
log_feature_no_dups_combined['location_by_max_volume'] = (
            log_feature_no_dups_combined['location'].map(lambda x: max_volume_dict[x]))
#log_feature_no_dups_combined['location_by_std_volume'] = (
#            log_feature_no_dups_combined['location'].map(lambda x: std_volume_dict[x]))

#(log_feature_ordered,log_feature_no_dups_combined) = convert_strings_to_ints(log_feature_no_dups_combined,'log_feature','first_log_feature_hash')
#log_feature_no_dups_combined['first_log_feature_hash'] = log_feature_no_dups_combined['log_feature'].map(lambda x: log_feature_ordered_dups[x])
log_feature_no_dups_combined['first_log_feature_hash'] = log_feature_no_dups_combined['log_feature'].astype(int)
log_feature_no_dups_combined['first_log_feature_volume'] = log_feature_no_dups_combined['volume']

log_feature_no_dups_combined['first_log_feature_hash_next'] = log_feature_no_dups_combined['first_log_feature_hash'].shift(-1)
log_feature_no_dups_combined['first_log_feature_hash_next'].fillna(-1,inplace=True)

log_feature_no_dups_combined['location_hash_next'] = log_feature_no_dups_combined['location_hash'].shift(-1)
log_feature_no_dups_combined['location_hash_next'].fillna(-1,inplace=True)


log_feature_no_dups_combined['fault_next'] = log_feature_no_dups_combined['fault_severity'].shift(-1)
log_feature_no_dups_combined['fault_next'].fillna(-1,inplace=True)
log_feature_no_dups_combined['fault_next'] = log_feature_no_dups_combined['fault_next'].map(lambda x: -1 if x == 'dummy' else x)
log_feature_no_dups_combined['fault_next2'] = log_feature_no_dups_combined['fault_severity'].shift(-2)
log_feature_no_dups_combined['fault_next2'].fillna(-1,inplace=True)
log_feature_no_dups_combined['fault_next2'] = log_feature_no_dups_combined['fault_next2'].map(lambda x: -1 if x == 'dummy' else x)
log_feature_no_dups_combined['fault_prev2'] = log_feature_no_dups_combined['fault_severity'].shift(2)
log_feature_no_dups_combined['fault_prev2'].fillna(-1,inplace=True)
log_feature_no_dups_combined['fault_prev2'] = log_feature_no_dups_combined['fault_prev2'].map(lambda x: -1 if x == 'dummy' else x)

log_feature_no_dups_combined['fault_prev'] = log_feature_no_dups_combined['fault_severity'].shift(1)
log_feature_no_dups_combined['fault_prev'].fillna(-1,inplace=True)
log_feature_no_dups_combined['fault_prev'] = log_feature_no_dups_combined['fault_prev'].map(lambda x: -1 if x == 'dummy' else x)
log_feature_no_dups_combined['fault_next3'] = log_feature_no_dups_combined['fault_severity'].shift(-3)
log_feature_no_dups_combined['fault_next3'].fillna(-1,inplace=True)
log_feature_no_dups_combined['fault_next3'] = log_feature_no_dups_combined['fault_next3'].map(lambda x: -1 if x == 'dummy' else x)
log_feature_no_dups_combined['fault_prev3'] = log_feature_no_dups_combined['fault_severity'].shift(3)
log_feature_no_dups_combined['fault_prev3'].fillna(-1,inplace=True)
log_feature_no_dups_combined['fault_prev3'] = log_feature_no_dups_combined['fault_prev3'].map(lambda x: -1 if x == 'dummy' else x)

log_feature_no_dups_combined['cumulative_feature_vol'] = log_feature_no_dups_combined['sum_log_features_volume'].cumsum()

log_feature_no_dups_combined['order_by_location'] = 1
log_feature_no_dups_combined['order_by_location'] = log_feature_no_dups_combined.groupby(['location'])['order_by_location'].cumsum()

log_feature_no_dups_combined['order_by_location_cum_vol'] = log_feature_no_dups_combined.groupby(['location'])['sum_log_features_volume'].cumsum()
log_feature_no_dups_combined['order_by_location_cum_event_type_count'] = log_feature_no_dups_combined.groupby(['location'])['count_event_types'].cumsum()
log_feature_no_dups_combined['order_by_location_cum_resource_type_count'] = log_feature_no_dups_combined.groupby(['location'])['count_resource_types'].cumsum()

log_feature_no_dups_combined['order_by_first_feature'] = 1
log_feature_no_dups_combined['order_by_first_feature'] = log_feature_no_dups_combined.groupby(['log_feature'])['order_by_first_feature'].cumsum()

log_feature_no_dups_combined['order_by_first_resource_type'] = 1
log_feature_no_dups_combined['order_by_first_resource_type'] = log_feature_no_dups_combined.groupby(['resource_type'])['order_by_first_resource_type'].cumsum()

log_feature_no_dups_combined['order_by_first_event_type'] = 1
log_feature_no_dups_combined['order_by_first_event_type'] = log_feature_no_dups_combined.groupby(['event_type'])['order_by_first_event_type'].cumsum()

log_feature_no_dups_combined['location_max'] = log_feature_no_dups_combined.groupby(['location'])['order_by_location'].transform(max)
log_feature_no_dups_combined['location_next_max'] = log_feature_no_dups_combined['location_max'] - 1
log_feature_no_dups_combined['is_final_id_at_location'] = 0
log_feature_no_dups_combined['is_next_to_final_id_at_location'] = 0

is_final_cond = log_feature_no_dups_combined['order_by_location'] == log_feature_no_dups_combined['location_max']
is_next_to_final_cond = log_feature_no_dups_combined['order_by_location'] == log_feature_no_dups_combined['location_next_max']
log_feature_no_dups_combined['is_final_id_at_location'][is_final_cond] = 1
log_feature_no_dups_combined['is_next_to_final_id_at_location'][is_next_to_final_cond] = 1

log_feature_no_dups_combined['is_next_resource_type_different_same_loc'] = log_feature_no_dups_combined['is_next_resource_type_different']
log_feature_no_dups_combined['is_next_resource_type_different_same_loc'][is_final_cond] = 0
log_feature_no_dups_combined['order_by_location_and_resource_type_change'] = \
 log_feature_no_dups_combined.groupby(['location'])['is_next_resource_type_different_same_loc'].cumsum()

log_feature_no_dups_combined['is_next_severity_type_different_same_loc'] = log_feature_no_dups_combined['is_next_severity_type_different']
log_feature_no_dups_combined['is_next_severity_type_different_same_loc'][is_final_cond] = 0
log_feature_no_dups_combined['order_by_location_and_severity_type_change'] = \
 log_feature_no_dups_combined.groupby(['location'])['is_next_severity_type_different_same_loc'].cumsum()



log_feature_no_dups_combined.sort('order_of_log_feature',ascending = False , inplace=True)
log_feature_no_dups_combined['reverse_order_by_location'] = 1
log_feature_no_dups_combined['reverse_order_by_location'] = log_feature_no_dups_combined.groupby(['location'])['reverse_order_by_location'].cumsum()
log_feature_no_dups_combined['reverse_order_by_location_and_resource_type_change'] = \
 log_feature_no_dups_combined.groupby(['location'])['is_next_resource_type_different_same_loc'].cumsum()





#is_transtion_cond = log_feature_no_dups_combined['order_by_location'] == 1
#log_feature_no_dups_combined['fault_prev'][is_transtion_cond] = -2

log_feature_no_dups_combined.sort('order_of_log_feature',inplace=True)

log_feature_no_dups_combined['vol_summed_to_loc_vol_ratio'] = (log_feature_no_dups_combined['sum_log_features_volume'] /
                                                                log_feature_no_dups_combined['location_by_mean_volume'])



log_feature_no_dups_combined.sort('order_of_log_feature',inplace=True)





#temp = log_feature_no_dups_combined.groupby(['location']).apply(lambda x: x['order_by_location'].cumsum())
#temp.reset_index(inplace=True)
#%%
log_feature_no_dups_rev = log_feature.sort(ascending = False).drop_duplicates('id')
log_feature_no_dups_rev.reset_index(drop=True,inplace=True)
log_feature_no_dups_rev.reset_index('order_of_log_feature_rev',inplace=True)
log_feature_no_dups_rev.rename(columns={'index':'order_of_log_feature_rev','count_of_log_feature_seen':'count_of_log_feature_seen_no_dups_rev'},inplace=True)
#log_feature_no_dups_rev['last_log_feature_hash'] = log_feature_no_dups_rev['log_feature'].map(lambda x: log_feature_ordered_dups[x])
log_feature_no_dups_rev['last_log_feature_hash'] = log_feature_no_dups_rev['log_feature'].astype(int)
log_feature_no_dups_rev['last_log_feature_volume'] = log_feature_no_dups_rev['volume']

#%%
#%%


#%%
tic=timeit.default_timer()
combined = pd.concat([train_orig, test_orig], axis=0,ignore_index=True)
#(location_dict,combined) = convert_strings_to_ints(combined,'location','location_hash')
#combined.reset_index(inplace=True)
#combined.rename(columns={'index':'order_of_train'},inplace=True)

combined = pd.merge(combined,severity_type_dummies,left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,severity_type[['id','severity_type','num_ids_with_severity_type']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,event_type_dummies_collapse,left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,resource_type_dummies_collapse,left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,log_feature_dummies_collapse,left_on = ['id'],
                               right_on = ['id'],how='left')

combined = pd.merge(combined,log_feature_dummies_by_vol_collapse[['id'] + log_feature_dummies_by_vol_cols + log_feature_num_ids_dummies_by_vol_cols +
                                    log_feature_dummies_by_vol_binned_cols + log_feature_dummies_by_vol_binned_offset_cols +
                                    log_feature_by_loc_dummies_by_vol_cols],left_on = ['id'],
                               right_on = ['id'],how='left')

combined = pd.merge(combined,log_feature_small_dummies_by_vol_collapse[['id'] +
                                    log_feature_small_dummies_by_vol_binned_cols + log_feature_small_dummies_by_vol_binned_offset_cols
                                    ],left_on = ['id'],
                               right_on = ['id'],how='left')

combined = pd.merge(combined,log_feature_dummies_loc_collapse,left_on = ['location'],
                               right_on = ['location'],how='left')
combined = pd.merge(combined,log_feature_dummies_loc_collapse_sum,left_on = ['location'],
                               right_on = ['location'],how='left')

combined = pd.merge(combined,event_type_dummies_loc_collapse,left_on = ['location'],
                               right_on = ['location'],how='left')
combined = pd.merge(combined,event_type_dummies_loc_collapse_sum,left_on = ['location'],
                               right_on = ['location'],how='left')

combined = pd.merge(combined,log_feature_volume_max[['id','max_volume_log_feature','max_vol_log_feature_by_most_common_location']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,log_feature_volume_min[['id','min_volume_log_feature','min_vol_log_feature_by_most_common_location']],left_on = ['id'],
                               right_on = ['id'],how='left')

combined = pd.merge(combined,log_feature_volume_of_max_feature[['id','volume_of_max_log_feature']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,log_feature_volume_of_min_feature[['id','volume_of_min_log_feature']],left_on = ['id'],
                               right_on = ['id'],how='left')

combined = pd.merge(combined,log_feature_volume_of_rarest_feature[['id','volume_of_rarest_log_feature','rarest_log_feature']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,log_feature_volume_of_least_rare_feature[['id','volume_of_least_rare_log_feature','least_rare_log_feature']],left_on = ['id'],
                               right_on = ['id'],how='left')

combined = pd.merge(combined,log_feature_count[['id','count_log_features']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,event_type_count[['id','count_event_types']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,resource_type_count[['id','count_resource_types']],left_on = ['id'],
                               right_on = ['id'],how='left')

combined = pd.merge(combined,event_type_no_dups[['id','count_of_event_type_seen']],left_on = ['id'],
                               right_on = ['id'],how='left')


combined = pd.merge(combined,log_feature_sum[['id','sum_log_features_volume',
                                    'sum_num_ids_with_log_feature','sum_locations_per_feature']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,log_feature_mean[['id','mean_log_features_volume','mean_num_ids_with_log_feature']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,log_feature_max[['id','max_log_features_volume','max_log_feature',
                                              'max_num_ids_with_log_feature','max_locations_per_feature']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,log_feature_min[['id','min_log_features_volume','min_num_ids_with_log_feature',
                                                'min_locations_per_feature','min_log_feature']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,log_feature_std[['id','std_log_features_volume','std_num_ids_with_log_feature','std_log_feature']],left_on = ['id'],
                               right_on = ['id'],how='left')

combined = pd.merge(combined,log_feature_median[['id','median_log_feature','median_log_feature_most_common_event',
                                                 'median_log_feature_most_common_location']],left_on = ['id'],
                               right_on = ['id'],how='left')

combined = pd.merge(combined,resource_type_max[['id','max_is_next_resource_type_repeat']],left_on = ['id'],
                               right_on = ['id'],how='left')

combined = pd.merge(combined,event_type_max[['id','max_event_type']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,event_type_min[['id','min_event_type']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,event_type_median[['id','median_event_type']],left_on = ['id'],
                               right_on = ['id'],how='left')

combined = pd.merge(combined,resource_type_max[['id','max_resource_type']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,resource_type_min[['id','min_resource_type']],left_on = ['id'],
                               right_on = ['id'],how='left')
combined = pd.merge(combined,resource_type_median[['id','median_resource_type']],left_on = ['id'],
                               right_on = ['id'],how='left')

combined['range_log_features_volume'] = combined['max_log_features_volume'] - combined['min_locations_per_feature']




#TODO testing - this info not available in test, but maybe able to infer it
combined = pd.merge(combined,
                    log_feature_no_dups_combined[['id','fault_next','fault_prev',
                                                  'fault_next2','fault_prev2',
                                                  'fault_next3','fault_prev3',
                                                  'first_log_feature_hash','first_log_feature_hash_next',
                                                  'first_log_feature_volume',
                                                  'vol_summed_to_loc_vol_ratio',
                                                  'location_hash','location_hash_rev','location_hash_next',
                                                  'location_by_unique_features',
                                                  'location_by_unique_event_types',
                                                  'location_by_unique_resource_types',
                                                  'location_by_unique_severity_types',
                                                  'location_by_std_resource',
                                                  'location_by_std_log_feature',
                                                  'location_by_std_event_type',
                                                  'location_by_most_common_event',
                                                  'location_by_most_common_resource',
                                                  'location_by_most_common_severity',
                                                  'location_by_most_common_feature',
                                                  'location_by_second_most_common_feature',
                                                  'location_by_median_feature_number',
                                                  'location_by_median_ids_with_feature',
                                                  'location_by_max_feature_number',
                                                  'location_by_min_feature_number',
                                                  'location_by_min_event',
                                                  'location_by_max_event',
                                                  'location_by_median_event',
                                                  'location_by_min_first_event',
                                                  'location_by_max_first_event',
                                                  'location_by_median_first_event',
                                                  'location_by_mean_volume',
                                                  'location_by_max_volume',
#                                                  'location_by_std_volume',
                                                  'location_by_frac_dummy',
                                                  'count_of_volumes',
                                                  'count_of_most_common_event_log_feature',
                                                  'count_of_most_common_location_log_feature',
                                                  'location_by_mean_count_features',
                                                  'count_of_log_feature_seen_no_dups',
                                                  'order_of_log_feature','location_max',
                                                  'cumulative_feature_vol',
                                                  'order_by_location_cum_vol',
                                                  'order_by_location_cum_event_type_count',
                                                  'order_by_location_cum_resource_type_count',
                                                  'reverse_order_by_location','order_by_location',
                                                  'order_by_location_and_resource_type_change',
                                                  'order_by_location_and_severity_type_change',
                                                  'reverse_order_by_location_and_resource_type_change',
                                                  'switches_resource_type',
                                                  'order_by_first_feature',
                                                  'order_by_first_resource_type',
                                                  'order_by_first_event_type',
                                                  'is_final_id_at_location','is_next_to_final_id_at_location']
                                                  + has_event_type_names + has_resource_type_names],left_on = ['id'],
                               right_on = ['id'],how='left')

combined = pd.merge(combined,
                    log_feature_no_dups_rev[['id','last_log_feature_hash','last_log_feature_volume']],left_on = ['id'],
                               right_on = ['id'],how='left')

#combined = pd.merge(combined,log_feature[['id']],left_on = ['id'],
#                               right_on = ['id'],how='left')
combined.fillna(0,inplace=True)
for key in LOG_FEATURE_DICT:
    temp_df = LOG_FEATURE_DICT[key]
    combined = pd.merge(combined,temp_df,left_on = ['id'],
                               right_on = ['id'],how='left')
    combined.fillna(-1,inplace=True)
for key in RESOURCE_TYPE_DICT:
    temp_df = RESOURCE_TYPE_DICT[key]
    combined = pd.merge(combined,temp_df,left_on = ['id'],
                               right_on = ['id'],how='left')
    combined.fillna(-1,inplace=True)
for key in EVENT_TYPE_DICT:
    temp_df = EVENT_TYPE_DICT[key]
    combined = pd.merge(combined,temp_df,left_on = ['id'],
                               right_on = ['id'],how='left')
    combined.fillna(-1,inplace=True)
toc=timeit.default_timer()
print('Merging Time',toc - tic)
#%%
#df['period'] = df[['Year', 'quarter']].apply(lambda x: ''.join(x), axis=1)
#%%
#combined['fault_next'] = combined['fault_severity'].shift(-1)
#combined['fault_next'].fillna(0,inplace=True)
#combined['fault_next'] = combined['fault_next'].map(lambda x: 0 if x == 'dummy' else x)
#
#combined['fault_prev'] = combined['fault_severity'].shift(1)
#combined['fault_prev'].fillna(0,inplace=True)
#combined['fault_prev'] = combined['fault_prev'].map(lambda x: 0 if x == 'dummy' else x)

#%%
tic=timeit.default_timer()
#redo_combo_log_features = False
redo_combo_log_features = True

combined['location_number'] = combined['location'].map(lambda x: x[9:]).astype(int)
#combined.sort(['location_number','order_of_log_feature'],inplace=True)
combined.sort(['median_log_feature','min_log_feature'],inplace=True)
if (redo_combo_log_features):
    log_feature_cols_all = [col for col in list(combined) if col.startswith('log_feature_')]
    log_feature_cols_all.remove('log_feature_hash')

    combined_temp = combined.copy()
    combined_temp[log_feature_cols_all] = combined_temp[log_feature_cols_all].astype(str)
    combined_temp['combo_log_features'] = combined_temp[log_feature_cols_all].sum(axis=1)
    (combo_log_feature_dict,combined_temp) = convert_strings_to_ints(combined_temp,'combo_log_features','combo_log_features_hash')

    event_type_cols_all = [col for col in list(combined) if col.startswith('event_type_')]
    resource_type_cols_all = [col for col in list(combined) if col.startswith('resource_type_')]

    combined.sort(['median_event_type','min_event_type'],inplace=True)
    combined_temp2 = combined.copy()
    combined_temp2[event_type_cols_all] = combined_temp2[event_type_cols_all].astype(str)
    combined_temp2['combo_event_types'] = combined_temp2[event_type_cols_all].sum(axis=1)
    (combo_event_dict,combined_temp2) = convert_strings_to_ints(combined_temp2,'combo_event_types','combo_event_types_hash')

    combined.sort(['median_resource_type','min_resource_type'],inplace=True)
    combined_temp3 = combined.copy()
    combined_temp3[resource_type_cols_all] = combined_temp3[resource_type_cols_all].astype(str)
    combined_temp3['combo_resource_types'] = combined_temp3[resource_type_cols_all].sum(axis=1)
    (combo_resource_dict,combined_temp3) = convert_strings_to_ints(combined_temp3,'combo_resource_types','combo_resource_types_hash')

combined['combo_log_features_hash'] = combined_temp['combo_log_features_hash']

combined['combo_event_types_hash'] = combined_temp2['combo_event_types_hash']
combined['combo_resource_types_hash'] = combined_temp3['combo_resource_types_hash']
combined['log_features_range'] = combined['max_log_feature'] - combined['min_log_feature']
combined['range_event_type'] = combined['max_event_type'] - combined['min_event_type']

toc=timeit.default_timer()
print('Combo Time',toc - tic)
#%%
combined['first_log_feature_hash'] = combined['first_log_feature_hash'].astype(int)
combined['last_log_feature_hash'] = combined['first_log_feature_hash'].astype(int)

combined['order_by_location_fraction'] = combined['order_by_location'] / combined['location_max']
combined['reverse_order_by_location_fraction'] = combined['reverse_order_by_location'] / combined['location_max']
combined['order_by_location_and_location'] = combined['location_number']*5000 + combined['order_by_location']
combined['count_of_volumes_per_feature'] = combined['count_of_volumes'] / combined['count_log_features']
combined['count_of_volumes_repeated'] = combined['count_log_features'] - combined['count_of_volumes']

event_combo_value_counts = combined['combo_event_types_hash'].value_counts().to_dict()
combined['num_ids_with_event_type_combo'] = combined['combo_event_types_hash'].map(lambda x: event_combo_value_counts[x])
resource_combo_value_counts = combined['combo_resource_types_hash'].value_counts().to_dict()
combined['num_ids_with_resource_type_combo'] = combined['combo_resource_types_hash'].map(lambda x: resource_combo_value_counts[x])
log_feature_combo_value_counts = combined['combo_log_features_hash'].value_counts().to_dict()
combined['num_ids_with_log_feature_combo'] = combined['combo_log_features_hash'].map(lambda x: log_feature_combo_value_counts[x])
most_common_combo_feature_dict = combined.groupby(['location'])['combo_log_features_hash'].agg(lambda x: x.value_counts().index[0]).to_dict()

combined['location_by_most_common_combo_feature'] = (
            combined['location'].map(lambda x: most_common_combo_feature_dict[x]))
count_log_features_dict = combined.groupby(['location'])['count_log_features'].agg(lambda x: x.mean()).to_dict()
combined['location_by_mean_count_log_features'] = combined['location'].map(count_log_features_dict)
count_median_log_features_dict = combined.groupby(['location'])['count_log_features'].agg(lambda x: x.median()).to_dict()
combined['location_by_median_count_log_features'] = combined['location'].map(count_median_log_features_dict)
combined['count_log_features_diff_to_location'] = combined['count_log_features'] - combined['location_by_median_count_log_features']
combined['volume_log_features_diff_to_location'] = combined['sum_log_features_volume'] - combined['location_by_mean_volume']
combined['median_feature_diff_to_location'] = combined['median_log_feature'] - combined['location_by_median_feature_number']

combined['log_82_203_vol_ratio'] = combined['vol_log_feature_82'] / (combined['vol_log_feature_203'])
combined['log_82_203_vol_ratio'].fillna(0,inplace=True)
combined['log_203_82_vol_diff'] = combined['vol_log_feature_203'] - (combined['vol_log_feature_82'])
combined['log_201_80_vol_diff'] = combined['vol_log_feature_201'] - (combined['vol_log_feature_80'])
combined['log_312_232_vol_diff'] = combined['vol_log_feature_312'] - (combined['vol_log_feature_232'])
combined['log_171_55_vol_diff'] = combined['vol_log_feature_171'] - (combined['vol_log_feature_55'])
combined['log_170_54_vol_diff'] = combined['vol_log_feature_170'] - (combined['vol_log_feature_54'])
combined['log_193_71_vol_diff'] = combined['vol_log_feature_193'] - (combined['vol_log_feature_71'])

combined['count_features_event_types_ratio'] = combined['count_log_features'] / combined['count_event_types']
combined['count_resouce_types_event_types_ratio'] = combined['count_resource_types'] / combined['count_event_types']

combined_non_dummies = combined.loc[combined['fault_severity'] != 'dummy'].copy()

location_non_dummies_dict = combined_non_dummies.groupby('location')['id'].agg('count').to_dict()

def get_non_dummies_count(x):
    try:
        return location_non_dummies_dict[x]
    except KeyError:
        return 0
combined['location_by_non_dummies_count'] = combined['location'].map(lambda x: get_non_dummies_count(x))
#%%
combo_feature_most_common_severity_type_dict = combined.groupby(['combo_log_features_hash'])['severity_type'].agg(lambda x: x.value_counts().index[0]).to_dict()
combo_feature_severity_type_number_dict = combined.groupby(['combo_log_features_hash'])['severity_type'].agg(lambda x: x.nunique()).to_dict()
combo_feature_median_location_dict = combined.groupby(['combo_log_features_hash'])['location_number'].agg(lambda x: x.median()).to_dict()
combo_feature_nunique_location_dict = combined.groupby(['combo_log_features_hash'])['location_number'].agg(lambda x: x.nunique()).to_dict()
combo_feature_mean_volume_dict = combined.groupby(['combo_log_features_hash'])['sum_log_features_volume'].agg(lambda x: x.mean()).to_dict()

combined['combo_log_features_by_most_common_severity'] = combined['combo_log_features_hash'].map(lambda x: combo_feature_most_common_severity_type_dict[x])
combined['combo_log_features_by_nunique_severity'] = combined['combo_log_features_hash'].map(lambda x: combo_feature_severity_type_number_dict[x])
combined['combo_log_features_by_median_loc'] = combined['combo_log_features_hash'].map(lambda x: combo_feature_median_location_dict[x])
combined['combo_log_features_by_nunique_loc'] = combined['combo_log_features_hash'].map(lambda x: combo_feature_nunique_location_dict[x])
combined['combo_log_features_by_nunique_loc_per_ids'] = combined['combo_log_features_by_nunique_loc'] / combined['num_ids_with_log_feature_combo']
combined['combo_log_features_by_mean_volume'] = combined['combo_log_features_hash'].map(lambda x: combo_feature_mean_volume_dict[x])
#%%
#combined['dummies_location_ordered'] = combined['location_hash']
#location_dummies = pd.get_dummies(combined[['id','order_by_location','dummies_location_ordered']],columns=['dummies_location_ordered'])
#loc_dummies_ordered_cols = [col for col in list(location_dummies) if col.startswith('dummies_location_ordered_')]
#
#location_dummies[loc_dummies_ordered_cols] = location_dummies[loc_dummies_ordered_cols].multiply(location_dummies['order_by_location'],axis='index')
#combined = pd.merge(combined,
#                    location_dummies[['id'] + loc_dummies_ordered_cols],left_on = ['id'],
#                               right_on = ['id'],how='left')
#%%
if (is_sub_run):
    train = combined.loc[combined['fault_severity'] != 'dummy' ]
    test = combined.loc[combined['fault_severity'] == 'dummy' ]
else:
#    train = combined.loc[(combined['fault_severity'] != 'dummy') &
#                         (combined['id'] > 5000)]
#    test = combined.loc[(combined['fault_severity'] != 'dummy') &
#                         (combined['id'] <= 5000)]
#    train = combined.loc[(combined['fault_severity'] != 'dummy') &
#                         ((combined['id'] > 10000) | (combined['id'] <= 5000))]
#    test = combined.loc[(combined['fault_severity'] != 'dummy') &
#                         (combined['id'] > 5000) & (combined['id'] <= 10000)]
    train = combined.loc[(combined['fault_severity'] != 'dummy') &
                         ((combined['id'] > 15000) | (combined['id'] <= 10000))]
    test = combined.loc[(combined['fault_severity'] != 'dummy') &
                         (combined['id'] > 10000) & (combined['id'] <= 15000)]
#    train = combined.loc[(combined['fault_severity'] != 'dummy') &
#                         (combined['id'] < 15000)]
#    test = combined.loc[(combined['fault_severity'] != 'dummy') &
#                         (combined['id'] >= 15000)]


#%%
def fit_xgb_model(train, test, params, xgb_features, num_rounds = 10, num_boost_rounds = 10,
                  do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
                  random_seed = 5, reweight_probs = True, calculate_log_loss = True, use_two_classes = False):

    tic=timeit.default_timer()
    random.seed(random_seed)
    np.random.seed(random_seed)

    X_train, X_watch = cross_validation.train_test_split(train, test_size=0.2,random_state=random_seed)
    train_data = X_train[xgb_features].values
    train_data_full = train[xgb_features].values
    watch_data = X_watch[xgb_features].values
    train_fault_severity = X_train['fault_severity'].astype(int).values
    train_fault_severity_full = train['fault_severity'].astype(int).values
    watch_fault_severity = X_watch['fault_severity'].astype(int).values
    test_data = test[xgb_features].values
    dtrain = xgb.DMatrix(train_data, train_fault_severity)
    dtrain_full = xgb.DMatrix(train_data_full, train_fault_severity_full)
    dwatch = xgb.DMatrix(watch_data, watch_fault_severity)
    dtest = xgb.DMatrix(test_data)
    watchlist = [(dtrain, 'train'),(dwatch, 'watch')]
    if(do_grid_search):
        print('Random search cv')
        gbm_search = xgb.XGBClassifier()
#        num_features = len(xgb_features)
        clf = RandomizedSearchCV(gbm_search,
                                 {'max_depth': sp_randint(1,20), 'learning_rate':sp_rand(0.005,0.2),
                                  'objective':['multi:softprob'],
                                  'subsample':sp_rand(0.1,0.99),
                                  'colsample_bytree':sp_rand(0.1,0.99),'seed':[random_seed],
#                                  'gamma':sp_rand(0,0.5),
                                  'min_child_weight':sp_randint(1,20),
#                                  'max_delta_step':sp_rand(0,1),
                                  'reg_alpha':sp_rand(0,0.3),
                                  'n_estimators': [300,500,600,750,850,1000]},
                                  verbose=10, n_jobs=1, cv = 4, scoring='log_loss', n_iter = 30,
                                  refit=False)
        clf.fit(train_data_full, train_fault_severity_full)
        print('best clf score',clf.best_score_)
        print('best params:', clf.best_params_)
        toc=timeit.default_timer()
        print('Grid search time',toc - tic)
    if (use_early_stopping):
        xgb_classifier = xgb.train(params, dtrain, num_boost_round=num_rounds, evals=watchlist,
                            early_stopping_rounds=100, verbose_eval=50)
        y_pred = xgb_classifier.predict(dtest,ntree_limit=xgb_classifier.best_iteration)
    else:
        xgb_classifier = xgb.train(params, dtrain_full, num_boost_round=num_boost_rounds, evals=[(dtrain_full,'train')],
                            verbose_eval=50)
        y_pred = xgb_classifier.predict(dtest)


    if(print_feature_imp):
        create_feature_map(xgb_features)
        imp_dict = xgb_classifier.get_fscore(fmap='xgb.fmap')
        imp_dict = sorted(imp_dict.items(), key=operator.itemgetter(1),reverse=True)
        print('{0:<20} {1:>5}'.format('Feature','Imp'))
        print("--------------------------------------")
        num_to_print = 30
        num_printed = 0
        for i in imp_dict:
            num_printed = num_printed + 1
            if (num_printed > num_to_print):
                continue
            print ('{0:20} {1:5.0f}'.format(i[0], i[1]))
    columns = ['predict_0','predict_1','predict_2']

    if (use_two_classes):
        columns = ['predict_low','predict_high']
    result_xgb_df = pd.DataFrame(index=test.id, columns=columns,data=y_pred)

    result_xgb_df = norm_rows(result_xgb_df)

    if(is_sub_run):
        print('creating xgb output')
    else:
        if(calculate_log_loss):
            result_xgb_df.reset_index('id',inplace=True)
            result_xgb_df = pd.merge(result_xgb_df,test[['id','fault_severity']],left_on = ['id'],
                                   right_on = ['id'],how='left')
            if(not use_two_classes):
                result_xgb_df['log_loss'] = result_xgb_df.apply(lambda row: get_log_loss_row(row),axis=1)
            else:
                result_xgb_df['log_loss'] = result_xgb_df.apply(lambda row: get_log_loss_row_two_classes(row),axis=1)
            print('log_loss',round(result_xgb_df['log_loss'].mean(),5))
#        print('log_loss 0',round(result_xgb_df.loc[result_xgb_df['fault_severity'] == 0].log_loss.sum(),1))
#        print('log_loss 1',round(result_xgb_df.loc[result_xgb_df['fault_severity'] == 1].log_loss.sum(),1))
#        print('log_loss 2',round(result_xgb_df.loc[result_xgb_df['fault_severity'] == 2].log_loss.sum(),1))
#        output_truth = output_xgb_df.groupby('country_destination')
#        for name,group in output_truth:
#            print(name,'mean','{0:.3f}'.format(group['ndcg_val'].mean()),'count',group['ndcg_val'].count())
    toc=timeit.default_timer()
    print('xgb Time',toc - tic)
    return result_xgb_df
#%%
severity_type_cols = [col for col in list(train) if col.startswith('severity_type_')]
event_type_cols = [col for col in list(train) if col.startswith('event_type_')]
resource_type_cols = [col for col in list(train) if col.startswith('resource_type_')]
log_feature_cols = [col for col in list(train) if col.startswith('log_feature_')]
log_feature_cols.remove('log_feature_hash')

log_feature_vol_combo_cols = [col for col in list(train) if col.startswith('combo_log_feature_and_volume_coarse_')]

vol_log_feature_cols = [col for col in list(train) if col.startswith('vol_log_feature_')]
vol_num_ids_with_log_feature_cols = [col for col in list(train) if col.startswith('vol_num_ids_with_log_feature_')]
vol_log_feature_by_loc_cols = [col for col in list(train) if col.startswith('vol_most_common_location_of_log_feature_')]
vol_binned_log_feature_cols = [col for col in list(train) if col.startswith('vol_binned_log_feature_')]
vol_binned_offset_log_feature_cols = [col for col in list(train) if col.startswith('vol_binned_offset_log_feature_')]

vol_binned_small_log_feature_cols = [col for col in list(train) if col.startswith('vol_binned_small_log_feature_')]
vol_binned_offset_small_log_feature_cols = [col for col in list(train) if col.startswith('vol_binned_offset_small_log_feature_')]

num_ids_with_log_feature_cols = [col for col in list(train) if col.startswith('num_ids_with_log_feature_')]
num_ids_with_resource_type_cols = [col for col in list(train) if col.startswith('num_ids_with_resource_type_')]
num_ids_with_event_type_cols = [col for col in list(train) if col.startswith('num_ids_with_event_type_')]
num_ids_with_severity_type_cols = [col for col in list(train) if col.startswith('num_ids_with_severity_type_')]

log_vol_feature_cols = [col for col in list(train) if col.startswith('vol_binned_')]
log_vol_feature_cols.remove('vol_binned_coarse')


resource_type_position_cols = [col for col in list(train) if col.startswith('position_resource_type_')]
event_type_position_cols = [col for col in list(train) if col.startswith('position_event_type_')]

log_feature_position_cols = [col for col in list(train) if col.startswith('position_log_feature_')]
log_volume_position_cols = [col for col in list(train) if col.startswith('position_volume_')]

log_by_loc_cols = [col for col in list(train) if col.startswith('most_common_location_of_log_feature_')]


loc_by_logs_cols = [col for col in list(train) if col.startswith('loc_log_feature_')]
loc_by_logs_summed_cols = [col for col in list(train) if col.startswith('summed_loc_log_feature_')]

loc_by_binned_logs_cols = [col for col in list(train) if col.startswith('loc_binned_log_feature_')]
loc_by_binned_logs_summed_cols = [col for col in list(train) if col.startswith('summed_loc_binned_log_feature_')]

loc_by_binned_offset_logs_cols = [col for col in list(train) if col.startswith('loc_binned_offset_log_feature_')]
loc_by_binned_offset_logs_summed_cols = [col for col in list(train) if col.startswith('summed_loc_binned_offset_log_feature_')]

loc_by_event_types_cols = [col for col in list(train) if col.startswith('loc_event_type_')]
loc_by_event_types_summed_cols = [col for col in list(train) if col.startswith('summed_loc_event_type_')]


#%%
#for col in log_feature_cols:
#for col in vol_binned_small_log_feature_cols:
#    print(col)
#    print_value_counts_spec(combined,col,1)
#105,108,118,155,
#170
#203,227,232,233,82
#%%
xgb_features = [
                'count_log_features','count_event_types',
#                'count_log_features_diff_to_location',
                'count_resource_types',
                'count_of_volumes',
#                'count_features_event_types_ratio'
#                'count_of_volumes_per_feature',
#                'count_of_volumes_repeated',
#                'count_of_most_common_event_log_feature',
#                'count_of_most_common_location_log_feature'
#                'range_log_features_volume'
                ]
xgb_features = (xgb_features + ['location_by_unique_features'])
xgb_features = (xgb_features + ['location_by_unique_resource_types'])
#xgb_features = (xgb_features + ['location_by_unique_severity_types'])
xgb_features = (xgb_features + ['location_by_unique_event_types'])
xgb_features = (xgb_features + ['location_by_most_common_event'])
xgb_features = (xgb_features + ['location_by_most_common_resource'])
#xgb_features = (xgb_features + ['location_by_most_common_severity'])
xgb_features = (xgb_features + ['location_by_most_common_feature'])
#xgb_features = (xgb_features + ['location_by_most_common_combo_feature'])



#xgb_features = (xgb_features + ['location_by_median_first_event'])
#xgb_features = (xgb_features + ['location_by_min_first_event'])
#xgb_features = (xgb_features + ['location_by_max_first_event'])

#xgb_features = (xgb_features + ['location_by_median_event'])
#xgb_features = (xgb_features + ['location_by_min_event'])
#xgb_features = (xgb_features + ['location_by_max_event'])
xgb_features = (xgb_features + has_event_type_names)
xgb_features = (xgb_features + has_resource_type_names)

xgb_features = (xgb_features + ['location_by_std_resource'])

#xgb_features = (xgb_features + ['location_by_median_ids_with_feature'])
xgb_features = (xgb_features + ['location_by_median_feature_number'])
#xgb_features = (xgb_features + ['location_by_max_feature_number'])

#xgb_features = (xgb_features + ['location_by_min_feature_number'])

#xgb_features = (xgb_features + ['location_by_mean_count_log_features'])
xgb_features = (xgb_features + ['location_by_median_count_log_features'])

#xgb_features = (xgb_features + ['location_by_second_most_common_feature'])

#xgb_features = (xgb_features + ['location_by_mean_count_features'])

xgb_features = (xgb_features + ['location_by_mean_volume'])
xgb_features = (xgb_features + ['location_by_max_volume'])

#xgb_features = (xgb_features + ['location_by_frac_dummy'])

#xgb_features = (xgb_features + ['location_by_non_dummies_count']) #by number of ids
xgb_features = (xgb_features + ['location_max']) #by number of ids
#xgb_features = (xgb_features + ['location_hash']) #by order of log feature
xgb_features = (xgb_features + ['location_number'])

xgb_features = (xgb_features + loc_by_logs_cols)
#xgb_features = (xgb_features + loc_by_logs_summed_cols)

#xgb_features = (xgb_features + loc_by_binned_logs_cols)
#xgb_features = (xgb_features + loc_by_binned_offset_logs_cols)
#xgb_features = (xgb_features + loc_by_binned_logs_summed_cols)

#xgb_features = (xgb_features + loc_by_event_types_cols)
#xgb_features = (xgb_features + loc_by_event_types_summed_cols)

#xgb_features = (xgb_features + ['location_hash_rev']) #by rev order of log feature

xgb_features = (xgb_features + ['sum_log_features_volume'])
#xgb_features = (xgb_features + ['volume_log_features_diff_to_location'])
xgb_features = (xgb_features + ['mean_log_features_volume'])
xgb_features = (xgb_features + ['max_log_features_volume'])
xgb_features = (xgb_features + ['min_num_ids_with_log_feature'])
#xgb_features = (xgb_features + ['std_log_features_volume','std_num_ids_with_log_feature'])
#xgb_features = (xgb_features + ['std_num_ids_with_log_feature'])

#xgb_features = (xgb_features + ['vol_summed_to_loc_vol_ratio'])

#xgb_features = (xgb_features + ['first_log_feature_hash'])
#xgb_features = (xgb_features + ['first_log_feature_volume'])
#xgb_features = (xgb_features + ['last_log_feature_hash']) #this feature may be bad
#xgb_features = (xgb_features + ['last_log_feature_volume'])

#xgb_features = (xgb_features + ['median_feature_diff_to_location'])

#xgb_features = (xgb_features + ['log_82_203_vol_ratio'])
xgb_features = (xgb_features + ['log_203_82_vol_diff'])
xgb_features = (xgb_features + ['log_201_80_vol_diff'])
xgb_features = (xgb_features + ['log_312_232_vol_diff'])
xgb_features = (xgb_features + ['log_170_54_vol_diff'])
xgb_features = (xgb_features + ['log_171_55_vol_diff'])
xgb_features = (xgb_features + ['log_193_71_vol_diff'])



xgb_features = (xgb_features + ['std_log_feature'])
xgb_features = (xgb_features + ['median_log_feature'])
xgb_features = (xgb_features + ['min_log_feature'])
xgb_features = (xgb_features + ['max_log_feature'])
xgb_features = (xgb_features + ['log_features_range'])
xgb_features = (xgb_features + ['volume_of_min_log_feature'])
xgb_features = (xgb_features + ['volume_of_max_log_feature'])


#xgb_features = (xgb_features + ['first_log_feature_hash_next'])

#xgb_features = (xgb_features + ['max_is_next_resource_type_repeat'])

#xgb_features = (xgb_features + resource_type_position_cols)
#xgb_features = (xgb_features + event_type_position_cols)

xgb_features = (xgb_features + ['position_event_type_1','position_event_type_2'])
#xgb_features = (xgb_features + ['position_resource_type_1','position_resource_type_2'])

#xgb_features = (xgb_features + ['position_log_feature_1'])
#xgb_features = (xgb_features + log_feature_position_cols)
#xgb_features = (xgb_features + log_volume_position_cols)


#xgb_features = (xgb_features + ['switches_resource_type'])

#xgb_features = (xgb_features + ['order_of_log_feature'])


#xgb_features = (xgb_features + ['cumulative_feature_vol'])
#xgb_features = (xgb_features + ['count_of_log_feature_seen_no_dups'])
#xgb_features = (xgb_features + ['count_of_event_type_seen'])

xgb_features = (xgb_features + ['order_by_location'])
xgb_features = (xgb_features + ['reverse_order_by_location'])
#xgb_features = (xgb_features + ['order_by_location_cum_vol'])

xgb_features = (xgb_features + ['order_by_location_fraction'])
xgb_features = (xgb_features + ['reverse_order_by_location_fraction'])

#xgb_features = (xgb_features + loc_dummies_ordered_cols)

#xgb_features = (xgb_features + ['order_by_location_and_location'])

#xgb_features = (xgb_features + ['order_by_location_cum_event_type_count'])
#xgb_features = (xgb_features + ['order_by_location_cum_resource_type_count'])

#xgb_features = (xgb_features + ['order_by_location_and_resource_type_change'])
#xgb_features = (xgb_features + ['order_by_location_and_severity_type_change'])
#xgb_features = (xgb_features + ['reverse_order_by_location_and_resource_type_change'])

#xgb_features = (xgb_features + ['order_by_first_feature']) #doesn't help

#xgb_features = (xgb_features + ['order_by_first_resource_type'])
#xgb_features = (xgb_features + ['order_by_first_event_type'])

#xgb_features = (xgb_features + ['is_final_id_at_location','is_next_to_final_id_at_location'])

#xgb_features = (xgb_features + ['severity_type','num_ids_with_severity_type'])
xgb_features = (xgb_features + ['severity_type'])


xgb_features = (xgb_features + vol_log_feature_cols)
xgb_features = (xgb_features + vol_binned_log_feature_cols) #controversial feature (good on some holdouts, not on others)
xgb_features = (xgb_features + vol_binned_offset_log_feature_cols) #controversial feature (good on some holdouts, not on others)


#xgb_features = (xgb_features + log_feature_vol_combo_cols)
#xgb_features = (xgb_features + log_vol_feature_cols)


#xgb_features = (xgb_features + severity_type_cols)
#xgb_features = (xgb_features + event_type_cols)
#xgb_features = (xgb_features + resource_type_cols)
#xgb_features = (xgb_features + ['resource_type_5','resource_type_1','resource_type_9','resource_type_6','resource_type_7','resource_type_3'])
#xgb_features = (xgb_features + log_feature_cols)



xgb_features = (xgb_features + log_by_loc_cols)
#xgb_features = (xgb_features + vol_log_feature_by_loc_cols)

xgb_features = (xgb_features + ['median_log_feature_most_common_location'])
xgb_features = (xgb_features + ['median_log_feature_most_common_event'])
#xgb_features = (xgb_features + ['log_feature_hash']) #not sure what this is



xgb_features = (xgb_features + ['combo_log_features_hash'])
xgb_features = (xgb_features + ['combo_resource_types_hash'])
xgb_features = (xgb_features + ['combo_event_types_hash'])

xgb_features = (xgb_features + ['num_ids_with_log_feature_combo'])
xgb_features = (xgb_features + ['num_ids_with_resource_type_combo'])
xgb_features = (xgb_features + ['num_ids_with_event_type_combo'])

xgb_features = (xgb_features + ['median_event_type'])
xgb_features = (xgb_features + ['max_event_type'])
xgb_features = (xgb_features + ['min_event_type'])
#xgb_features = (xgb_features + ['range_event_type'])

#xgb_features = (xgb_features + ['median_resource_type'])
#xgb_features = (xgb_features + ['max_resource_type'])
xgb_features = (xgb_features + ['min_resource_type'])

#xgb_features = (xgb_features + ['id'])

#xgb_features = (xgb_features + num_ids_with_log_feature_cols)

#xgb_features = (xgb_features + num_ids_with_resource_type_cols)
#xgb_features = (xgb_features + num_ids_with_event_type_cols)
#xgb_features = (xgb_features + num_ids_with_severity_type_cols)


#xgb_features.remove('log_feature_hash')

params = {'learning_rate': 0.01,
              'subsample': 0.95,
              'reg_alpha': 0.5,
              'lambda': 0.95,
              'gamma': 1.0,
              'seed': 6,
              'colsample_bytree': 0.4,
              'n_estimators': 100,
              'objective': 'multi:softprob',
              'eval_metric':'mlogloss',
#              'num_parallel_tree':5,
#              'max_delta_step': 0.5,
              'max_depth': 8,
#              'min_child_weight': 2,
              'num_class':3}
num_rounds = 10000
num_boost_rounds = 150

#result_xgb_df = fit_xgb_model(train,test,params,xgb_features,
#                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
#                                               random_seed = 6)
result_xgb_df = fit_xgb_model(train,test,params,xgb_features,
                              num_rounds = num_rounds, num_boost_rounds = 2300,
                              do_grid_search = False,
                              use_early_stopping = False, print_feature_imp = False,random_seed = 6)
#%%
train_low = train.copy()
train_low['fault_severity'] = train_low['fault_severity'].map(lambda x: 0 if x == 0 else 1)
test_low = test.copy()
if(not is_sub_run):
    test_low['fault_severity'] = test_low['fault_severity'].map(lambda x: 0 if x == 0 else 1)
train_high = train.copy()
train_high['fault_severity'] = train_high['fault_severity'].map(lambda x: 0 if x < 2 else 1)
test_high = test.copy()
if(not is_sub_run):
    test_high['fault_severity'] = test_high['fault_severity'].map(lambda x: 0 if x < 2 else 1)
#%%
params_low = {'learning_rate': 0.01,
              'subsample': 0.95,
              'reg_alpha': 0.5,
              'lambda': 0.95,
              'gamma': 1.0,
              'seed': 6,
              'colsample_bytree': 0.4,
              'n_estimators': 100,
              'objective': 'multi:softprob',
              'eval_metric':'mlogloss',
#              'num_parallel_tree':5,
#              'max_delta_step': 0.5,
              'max_depth': 8,
#              'min_child_weight': 2,
              'num_class':2}
num_rounds = 10000
num_boost_rounds = 150



#result_xgb_df_low = fit_xgb_model(train_low,test_low,params_low,xgb_features,
#                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
#                                               random_seed = 6,use_two_classes=True)
result_xgb_df_low = fit_xgb_model(train_low,test_low,params_low,xgb_features,
                              num_rounds = num_rounds, num_boost_rounds = 1600,
                              do_grid_search = False,
                              use_early_stopping = False, print_feature_imp = False,random_seed = 6,
                              use_two_classes=True)
#%%
params_high = {'learning_rate': 0.01,
              'subsample': 0.95,
              'reg_alpha': 0.5,
#              'lambda': 0.95,
              'gamma': 0.5,
              'seed': 6,
              'colsample_bytree': 0.8,
              'n_estimators': 100,
              'objective': 'multi:softprob',
              'eval_metric':'mlogloss',
#              'num_parallel_tree':5,
#              'max_delta_step': 0.5,
              'max_depth': 6,
#              'min_child_weight': 2,
              'num_class':2}
num_rounds = 10000
num_boost_rounds = 150



#result_xgb_df_high = fit_xgb_model(train_high,test_high,params_high,xgb_features,
#                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
#                                               random_seed = 6,use_two_classes=True)
result_xgb_df_high = fit_xgb_model(train_high,test_high,params_high,xgb_features,
                              num_rounds = num_rounds, num_boost_rounds = 1600,
                              do_grid_search = False,
                              use_early_stopping = False, print_feature_imp = False,random_seed = 6,
                              use_two_classes=True)
#%%
result_xgb_df_low_high = result_xgb_df.copy()
result_xgb_df_low_high['predict_0'] = result_xgb_df_low['predict_low']
result_xgb_df_low_high['predict_2'] = result_xgb_df_high['predict_high']
result_xgb_df_low_high['predict_1'] = result_xgb_df_low['predict_high'] * result_xgb_df_high['predict_low']
result_xgb_df_low_high[['predict_0','predict_1','predict_2']] = norm_rows(result_xgb_df_low_high[['predict_0','predict_1','predict_2']])
#%%
#%%
xgb_features_2 = [
                'count_log_features','count_event_types',
#                'count_log_features_diff_to_location',
                'count_resource_types',
                'count_of_volumes',
                ]
xgb_features_2 = (xgb_features_2 + ['location_by_unique_features'])
xgb_features_2 = (xgb_features_2 + ['location_by_unique_resource_types'])
#xgb_features_2 = (xgb_features_2 + ['location_by_unique_severity_types'])
xgb_features_2 = (xgb_features_2 + ['location_by_unique_event_types'])
xgb_features_2 = (xgb_features_2 + ['location_by_most_common_event'])
xgb_features_2 = (xgb_features_2 + ['location_by_most_common_resource'])
#xgb_features_2 = (xgb_features_2 + ['location_by_most_common_severity'])
xgb_features_2 = (xgb_features_2 + ['location_by_most_common_feature'])
#xgb_features_2 = (xgb_features_2 + ['location_by_most_common_combo_feature'])

#xgb_features_2 = (xgb_features_2 + ['location_by_median_first_event'])
#xgb_features_2 = (xgb_features_2 + ['location_by_min_first_event'])
#xgb_features_2 = (xgb_features_2 + ['location_by_max_first_event'])

xgb_features_2 = (xgb_features_2 + ['location_by_median_event'])
#xgb_features_2 = (xgb_features_2 + ['location_by_min_event'])
#xgb_features_2 = (xgb_features_2 + ['location_by_max_event'])
#xgb_features_2 = (xgb_features_2 + has_event_type_names)
xgb_features_2 = (xgb_features_2 + has_resource_type_names)

#xgb_features_2 = (xgb_features_2 + ['location_by_median_ids_with_feature'])
xgb_features_2 = (xgb_features_2 + ['location_by_median_feature_number'])
#xgb_features_2 = (xgb_features_2 + ['location_by_max_feature_number'])

#xgb_features_2 = (xgb_features_2 + ['location_by_min_feature_number'])

#xgb_features_2 = (xgb_features_2 + ['location_by_mean_count_log_features'])
xgb_features_2 = (xgb_features_2 + ['location_by_median_count_log_features'])

xgb_features_2 = (xgb_features_2 + ['location_by_second_most_common_feature'])

#xgb_features_2 = (xgb_features_2 + ['location_by_mean_count_features'])

xgb_features_2 = (xgb_features_2 + ['location_by_mean_volume'])
#xgb_features_2 = (xgb_features_2 + ['location_by_max_volume'])

#xgb_features_2 = (xgb_features_2 + ['location_by_frac_dummy'])

xgb_features_2 = (xgb_features_2 + ['location_max']) #by number of ids
#xgb_features_2 = (xgb_features_2 + ['location_hash']) #by order of log feature
xgb_features_2 = (xgb_features_2 + ['location_number'])

xgb_features_2 = (xgb_features_2 + loc_by_logs_cols)
#xgb_features_2 = (xgb_features_2 + loc_by_logs_summed_cols)

#xgb_features_2 = (xgb_features_2 + loc_by_binned_logs_cols)
#xgb_features_2 = (xgb_features_2 + loc_by_binned_offset_logs_cols)
#xgb_features_2 = (xgb_features_2 + loc_by_binned_logs_summed_cols)

#xgb_features_2 = (xgb_features_2 + loc_by_event_types_cols)
#xgb_features_2 = (xgb_features_2 + loc_by_event_types_summed_cols)

#xgb_features_2 = (xgb_features_2 + ['location_hash_rev']) #by rev order of log feature

#xgb_features_2 = (xgb_features_2 + ['sum_log_features_volume'])
#xgb_features_2 = (xgb_features_2 + ['volume_log_features_diff_to_location'])
xgb_features_2 = (xgb_features_2 + ['mean_log_features_volume'])
#xgb_features_2 = (xgb_features_2 + ['max_log_features_volume'])
xgb_features_2 = (xgb_features_2 + ['min_num_ids_with_log_feature'])
#xgb_features_2 = (xgb_features_2 + ['std_log_features_volume','std_num_ids_with_log_feature'])
xgb_features_2 = (xgb_features_2 + ['std_num_ids_with_log_feature'])

#xgb_features_2 = (xgb_features_2 + ['log_82_203_vol_ratio'])
#xgb_features_2 = (xgb_features_2 + ['log_203_82_vol_diff'])
#xgb_features_2 = (xgb_features_2 + ['log_201_80_vol_diff'])
#xgb_features_2 = (xgb_features_2 + ['log_312_232_vol_diff'])
#xgb_features_2 = (xgb_features_2 + ['log_170_54_vol_diff'])
#xgb_features_2 = (xgb_features_2 + ['log_171_55_vol_diff'])
#xgb_features_2 = (xgb_features_2 + ['log_193_71_vol_diff'])

xgb_features_2 = (xgb_features_2 + ['std_log_feature'])
xgb_features_2 = (xgb_features_2 + ['median_log_feature'])
xgb_features_2 = (xgb_features_2 + ['min_log_feature'])
xgb_features_2 = (xgb_features_2 + ['max_log_feature'])
xgb_features_2 = (xgb_features_2 + ['volume_of_min_log_feature'])
xgb_features_2 = (xgb_features_2 + ['volume_of_max_log_feature'])
xgb_features_2 = (xgb_features_2 + ['rarest_log_feature'])
xgb_features_2 = (xgb_features_2 + ['volume_of_rarest_log_feature'])
xgb_features_2 = (xgb_features_2 + ['least_rare_log_feature'])
xgb_features_2 = (xgb_features_2 + ['volume_of_least_rare_log_feature'])
xgb_features_2 = (xgb_features_2 + ['log_features_range'])

xgb_features_2 = (xgb_features_2 + ['order_by_location'])
xgb_features_2 = (xgb_features_2 + ['reverse_order_by_location'])
#xgb_features_2 = (xgb_features_2 + ['order_by_location_cum_vol'])

#xgb_features_2 = (xgb_features_2 + ['order_by_location_fraction'])
#xgb_features_2 = (xgb_features_2 + ['reverse_order_by_location_fraction'])

xgb_features_2 = (xgb_features_2 + ['severity_type'])


xgb_features_2 = (xgb_features_2 + vol_log_feature_cols)
xgb_features_2 = (xgb_features_2 + vol_binned_log_feature_cols) #controversial feature (good on some holdouts, not on others)
xgb_features_2 = (xgb_features_2 + vol_binned_offset_log_feature_cols) #controversial feature (good on some holdouts, not on others)

#xgb_features_2 = (xgb_features_2 + severity_type_cols)
#xgb_features_2 = (xgb_features_2 + event_type_cols)
xgb_features_2 = (xgb_features_2 + resource_type_cols)
xgb_features_2 = (xgb_features_2 + ['position_resource_type_1'])
xgb_features_2 = (xgb_features_2 + ['position_event_type_1'])
#xgb_features_2 = (xgb_features_2 + ['resource_type_2','resource_type_1','resource_type_9','resource_type_2','resource_type_2','resource_type_3'])
#xgb_features_2 = (xgb_features_2 + log_feature_cols)



xgb_features_2 = (xgb_features_2 + log_by_loc_cols)
#xgb_features_2 = (xgb_features_2 + vol_log_feature_by_loc_cols)

xgb_features_2 = (xgb_features_2 + ['median_log_feature_most_common_location'])
xgb_features_2 = (xgb_features_2 + ['median_log_feature_most_common_event'])
#xgb_features_2 = (xgb_features_2 + ['log_feature_hash']) #not sure what this is



xgb_features_2 = (xgb_features_2 + ['combo_log_features_hash'])
xgb_features_2 = (xgb_features_2 + ['combo_resource_types_hash'])
xgb_features_2 = (xgb_features_2 + ['combo_event_types_hash'])

xgb_features_2 = (xgb_features_2 + ['num_ids_with_log_feature_combo'])
xgb_features_2 = (xgb_features_2 + ['num_ids_with_resource_type_combo'])
xgb_features_2 = (xgb_features_2 + ['num_ids_with_event_type_combo'])

xgb_features_2 = (xgb_features_2 + ['median_event_type'])
xgb_features_2 = (xgb_features_2 + ['max_event_type'])
xgb_features_2 = (xgb_features_2 + ['min_event_type'])

xgb_features_2 = (xgb_features_2 + ['max_resource_type'])
xgb_features_2 = (xgb_features_2 + ['min_resource_type'])

params_2 = {'learning_rate': 0.05,
              'subsample': 0.98,
              'reg_alpha': 0.5,
#              'lambda': 0.9,
              'gamma': 0.5,
              'seed': 9,
              'colsample_bytree': 0.2,
              'n_estimators': 100,
              'objective': 'multi:softprob',
              'eval_metric':'mlogloss',
#              'num_parallel_tree':5,
#              'max_delta_step': 0.5,
              'max_depth': 10,
              'min_child_weight':2,
              'num_class':2}
num_rounds = 10000
num_boost_rounds = 150

#result_xgb_df_2 = fit_xgb_model(train_low,test_low,params_2,xgb_features_2,
#                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
#                                               random_seed = 9, use_two_classes=True)
#result_xgb_df_2 = fit_xgb_model(train_low,test_low,params_2,xgb_features_2,
#                              num_rounds = num_rounds, num_boost_rounds = 1000,
#                              do_grid_search = False,
#                              use_early_stopping = False, print_feature_imp = False,random_seed = 9,use_two_classes=True)
#%%
xgb_features_3 = [
                'count_log_features','count_event_types',
                'count_resource_types',
                'count_of_volumes',
                ]
xgb_features_3 = (xgb_features_3 + ['location_by_unique_features'])
xgb_features_3 = (xgb_features_3 + ['location_by_unique_resource_types'])
#xgb_features_3 = (xgb_features_3 + ['location_by_unique_severity_types'])
xgb_features_3 = (xgb_features_3 + ['location_by_unique_event_types'])
xgb_features_3 = (xgb_features_3 + ['location_by_most_common_event'])
xgb_features_3 = (xgb_features_3 + ['location_by_most_common_resource'])
#xgb_features_3 = (xgb_features_3 + ['location_by_most_common_severity'])
xgb_features_3 = (xgb_features_3 + ['location_by_most_common_feature'])
#xgb_features_3 = (xgb_features_3 + ['location_by_most_common_combo_feature'])


xgb_features_3 = (xgb_features_3 + ['location_by_median_event'])
#xgb_features_3 = (xgb_features_3 + ['location_by_min_event'])
#xgb_features_3 = (xgb_features_3 + ['location_by_max_event'])
#xgb_features_3 = (xgb_features_3 + has_event_type_names)
xgb_features_3 = (xgb_features_3 + has_resource_type_names)

#xgb_features_3 = (xgb_features_3 + ['location_by_median_ids_with_feature'])
xgb_features_3 = (xgb_features_3 + ['location_by_median_feature_number'])
#xgb_features_3 = (xgb_features_3 + ['location_by_max_feature_number'])

xgb_features_3 = (xgb_features_3 + ['location_by_min_feature_number'])

#xgb_features_3 = (xgb_features_3 + ['location_by_mean_count_log_features'])
xgb_features_3 = (xgb_features_3 + ['location_by_median_count_log_features'])

xgb_features_3 = (xgb_features_3 + ['location_by_second_most_common_feature'])

xgb_features_3 = (xgb_features_3 + ['location_by_mean_volume'])
xgb_features_3 = (xgb_features_3 + ['location_by_std_log_feature'])
xgb_features_3 = (xgb_features_3 + ['location_by_std_event_type'])
xgb_features_3 = (xgb_features_3 + ['location_by_std_resource'])
#xgb_features_3 = (xgb_features_3 + ['location_by_max_volume'])

#xgb_features_3 = (xgb_features_3 + ['location_by_frac_dummy'])

xgb_features_3 = (xgb_features_3 + ['location_max']) #by number of ids
xgb_features_3 = (xgb_features_3 + ['location_number'])

xgb_features_3 = (xgb_features_3 + loc_by_logs_cols)
#xgb_features_3 = (xgb_features_3 + loc_by_logs_summed_cols)

#xgb_features_3 = (xgb_features_3 + loc_by_binned_logs_cols)
#xgb_features_3 = (xgb_features_3 + loc_by_binned_offset_logs_cols)
#xgb_features_3 = (xgb_features_3 + loc_by_binned_logs_summed_cols)

#xgb_features_3 = (xgb_features_3 + loc_by_event_types_cols)
#xgb_features_3 = (xgb_features_3 + loc_by_event_types_summed_cols)

xgb_features_3 = (xgb_features_3 + ['sum_log_features_volume'])
xgb_features_3 = (xgb_features_3 + ['mean_log_features_volume'])
xgb_features_3 = (xgb_features_3 + ['max_log_features_volume'])
xgb_features_3 = (xgb_features_3 + ['min_num_ids_with_log_feature'])
xgb_features_3 = (xgb_features_3 + ['std_num_ids_with_log_feature'])

xgb_features_3 = (xgb_features_3 + ['std_log_feature'])
xgb_features_3 = (xgb_features_3 + ['median_log_feature'])
xgb_features_3 = (xgb_features_3 + ['min_log_feature'])
xgb_features_3 = (xgb_features_3 + ['max_log_feature'])
xgb_features_3 = (xgb_features_3 + ['volume_of_min_log_feature'])
xgb_features_3 = (xgb_features_3 + ['volume_of_max_log_feature'])
xgb_features_3 = (xgb_features_3 + ['rarest_log_feature'])
xgb_features_3 = (xgb_features_3 + ['volume_of_rarest_log_feature'])
xgb_features_3 = (xgb_features_3 + ['least_rare_log_feature'])
xgb_features_3 = (xgb_features_3 + ['volume_of_least_rare_log_feature'])
xgb_features_3 = (xgb_features_3 + ['log_features_range'])

xgb_features_3 = (xgb_features_3 + ['order_by_location'])
xgb_features_3 = (xgb_features_3 + ['reverse_order_by_location'])
#xgb_features_3 = (xgb_features_3 + ['order_by_location_cum_vol'])

#xgb_features_3 = (xgb_features_3 + ['order_by_location_fraction'])
#xgb_features_3 = (xgb_features_3 + ['reverse_order_by_location_fraction'])

xgb_features_3 = (xgb_features_3 + ['severity_type'])


xgb_features_3 = (xgb_features_3 + vol_log_feature_cols)
xgb_features_3 = (xgb_features_3 + vol_binned_log_feature_cols) #controversial feature (good on some holdouts, not on others)
xgb_features_3 = (xgb_features_3 + vol_binned_offset_log_feature_cols) #controversial feature (good on some holdouts, not on others)

xgb_features_3 = (xgb_features_3 + vol_binned_small_log_feature_cols) #controversial feature (good on some holdouts, not on others)
xgb_features_3 = (xgb_features_3 + vol_binned_offset_small_log_feature_cols) #controversial feature (good on some holdouts, not on others)


#xgb_features_3 = (xgb_features_3 + log_feature_vol_combo_cols)
#xgb_features_3 = (xgb_features_3 + log_vol_feature_cols)



xgb_features_3 = (xgb_features_3 + log_by_loc_cols)
#xgb_features_3 = (xgb_features_3 + vol_log_feature_by_loc_cols)

#xgb_features_3 = (xgb_features_3 + ['median_log_feature_most_common_location'])
#xgb_features_3 = (xgb_features_3 + ['median_log_feature_most_common_event'])



xgb_features_3 = (xgb_features_3 + ['combo_log_features_hash'])
xgb_features_3 = (xgb_features_3 + ['combo_resource_types_hash'])
xgb_features_3 = (xgb_features_3 + ['combo_event_types_hash'])

xgb_features_3 = (xgb_features_3 + ['num_ids_with_log_feature_combo'])
xgb_features_3 = (xgb_features_3 + ['num_ids_with_resource_type_combo'])
xgb_features_3 = (xgb_features_3 + ['num_ids_with_event_type_combo'])

xgb_features_3 = (xgb_features_3 + ['median_event_type'])
xgb_features_3 = (xgb_features_3 + ['max_event_type'])
xgb_features_3 = (xgb_features_3 + ['min_event_type'])

#xgb_features_3 = (xgb_features_3 + ['max_resource_type'])
#xgb_features_3 = (xgb_features_3 + ['min_resource_type'])



params_3 = {'learning_rate': 0.05,
              'subsample': 0.95,
              'reg_alpha': 0.05,
#              'lambda': 0.99,
#              'gamma': 1,
              'seed': 11,
              'colsample_bytree': 0.3,
              'n_estimators': 100,
              'objective': 'multi:softprob',
              'eval_metric':'mlogloss',
#              'num_parallel_tree':1,
#              'max_delta_step': 0.8,
              'max_depth': 5,
#              'min_child_weight':20,
              'num_class':3}
num_rounds = 10000
num_boost_rounds = 150

#result_xgb_df_3 = fit_xgb_model(train,test,params_3,xgb_features_3,
#                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
#                                               random_seed = 11)
result_xgb_df_3 = fit_xgb_model(train,test,params_3,xgb_features_3,
                              num_rounds = num_rounds, num_boost_rounds = 450,
                              do_grid_search = False,
                              use_early_stopping = False, print_feature_imp = False,random_seed = 11)

#%%
if (not is_sub_run):
    ans_xgb_df = pd.merge(result_xgb_df,combined[['id','order_by_location','location_number','median_log_feature','combo_log_features_hash','num_ids_with_log_feature_combo',
                                                  'severity_type','location_max','median_event_type','combo_resource_types_hash'] + vol_log_feature_cols],
                          left_on = ['id'],
                          right_on = ['id'],how='left')
    ans_bad_xgb_df = ans_xgb_df.loc[ans_xgb_df.log_loss >= 2.5]
#    ans_rare = ans_xgb_df.loc[ans_xgb_df['num_ids_with_log_feature_combo'] <= 30]
    ans_rare = ans_xgb_df.loc[ans_xgb_df['severity_type'] == '1']
#    correlation_matrix = result_xgb_df.corr()
#%%
#test_rare_combos = test.loc[test['num_ids_with_log_feature_combo'] <= 30].copy()
#train_rare_combos = train.loc[train['num_ids_with_log_feature_combo'] <= 30].copy()
test_rare_combos = test.loc[test['severity_type'] == '1'].copy()
train_rare_combos = train.loc[train['severity_type'] == '1'].copy()

#%%
xgb_features_9 = [
                'count_log_features','count_event_types',
#                'count_log_features_diff_to_location',
                'count_resource_types',
                'count_of_volumes',
                'count_of_volumes_per_feature',
#                'count_of_volumes_repeated',
#                'count_of_most_common_event_log_feature',
#                'count_of_most_common_location_log_feature'
#                'range_log_features_volume'
                ]
#xgb_features_9 = (xgb_features_9 + ['location_by_unique_features'])
xgb_features_9 = (xgb_features_9 + ['location_by_unique_resource_types'])
xgb_features_9 = (xgb_features_9 + ['location_by_unique_severity_types'])
xgb_features_9 = (xgb_features_9 + ['location_by_unique_event_types'])
xgb_features_9 = (xgb_features_9 + ['location_by_most_common_event'])
xgb_features_9 = (xgb_features_9 + ['location_by_most_common_resource'])
xgb_features_9 = (xgb_features_9 + ['location_by_most_common_severity'])
xgb_features_9 = (xgb_features_9 + ['location_by_most_common_feature'])
xgb_features_9 = (xgb_features_9 + ['location_by_most_common_combo_feature'])

xgb_features_9 = (xgb_features_9 + ['location_by_std_resource'])

xgb_features_9 = (xgb_features_9 + has_event_type_names)
xgb_features_9 = (xgb_features_9 + has_resource_type_names)

#xgb_features_9 = (xgb_features_9 + ['location_by_median_ids_with_feature'])
xgb_features_9 = (xgb_features_9 + ['location_by_median_feature_number'])
#xgb_features_9 = (xgb_features_9 + ['location_by_max_feature_number'])
#xgb_features_9 = (xgb_features_9 + ['location_by_min_feature_number'])

xgb_features_9 = (xgb_features_9 + ['location_by_median_count_log_features'])

#xgb_features_9 = (xgb_features_9 + ['location_by_second_most_common_feature'])

xgb_features_9 = (xgb_features_9 + ['location_by_mean_volume'])
xgb_features_9 = (xgb_features_9 + ['location_by_max_volume'])

#xgb_features_9 = (xgb_features_9 + ['location_by_frac_dummy'])

xgb_features_9 = (xgb_features_9 + ['location_max']) #by number of ids
#xgb_features_9 = (xgb_features_9 + ['location_hash']) #by order of log feature
xgb_features_9 = (xgb_features_9 + ['location_number'])

xgb_features_9 = (xgb_features_9 + loc_by_logs_cols)
#xgb_features_9 = (xgb_features_9 + loc_by_logs_summed_cols)

#xgb_features_9 = (xgb_features_9 + loc_by_binned_logs_cols)
#xgb_features_9 = (xgb_features_9 + loc_by_binned_offset_logs_cols)
#xgb_features_9 = (xgb_features_9 + loc_by_binned_logs_summed_cols)

#xgb_features_9 = (xgb_features_9 + loc_by_event_types_cols)
#xgb_features_9 = (xgb_features_9 + loc_by_event_types_summed_cols)

#xgb_features_9 = (xgb_features_9 + ['location_hash_rev']) #by rev order of log feature

xgb_features_9 = (xgb_features_9 + ['std_log_feature'])
xgb_features_9 = (xgb_features_9 + ['median_log_feature'])
xgb_features_9 = (xgb_features_9 + ['min_log_feature'])
xgb_features_9 = (xgb_features_9 + ['max_log_feature'])
xgb_features_9 = (xgb_features_9 + ['rarest_log_feature'])
xgb_features_9 = (xgb_features_9 + ['volume_of_rarest_log_feature'])
xgb_features_9 = (xgb_features_9 + ['least_rare_log_feature'])
xgb_features_9 = (xgb_features_9 + ['volume_of_least_rare_log_feature'])
xgb_features_9 = (xgb_features_9 + ['volume_of_min_log_feature'])
xgb_features_9 = (xgb_features_9 + ['volume_of_max_log_feature'])
xgb_features_9 = (xgb_features_9 + ['log_features_range'])

#xgb_features_9 = (xgb_features_9 + ['max_volume_log_feature'])
#xgb_features_9 = (xgb_features_9 + ['max_vol_log_feature_by_most_common_location'])
#xgb_features_9 = (xgb_features_9 + ['min_vol_log_feature_by_most_common_location'])

#xgb_features_9 = (xgb_features_9 + ['log_82_203_vol_ratio'])
xgb_features_9 = (xgb_features_9 + ['log_203_82_vol_diff'])
xgb_features_9 = (xgb_features_9 + ['log_201_80_vol_diff'])
xgb_features_9 = (xgb_features_9 + ['log_312_232_vol_diff'])
xgb_features_9 = (xgb_features_9 + ['log_170_54_vol_diff'])
xgb_features_9 = (xgb_features_9 + ['log_171_55_vol_diff'])
xgb_features_9 = (xgb_features_9 + ['log_193_71_vol_diff'])

xgb_features_9 = (xgb_features_9 + ['order_by_location'])
xgb_features_9 = (xgb_features_9 + ['reverse_order_by_location'])
#xgb_features_9 = (xgb_features_9 + ['order_by_location_cum_vol'])

xgb_features_9 = (xgb_features_9 + ['order_by_location_fraction'])
xgb_features_9 = (xgb_features_9 + ['reverse_order_by_location_fraction'])

xgb_features_9 = (xgb_features_9 + ['severity_type'])


xgb_features_9 = (xgb_features_9 + vol_log_feature_cols)
xgb_features_9 = (xgb_features_9 + vol_binned_log_feature_cols) #controversial feature (good on some holdouts, not on others)
xgb_features_9 = (xgb_features_9 + vol_binned_offset_log_feature_cols) #controversial feature (good on some holdouts, not on others)
xgb_features_9 = (xgb_features_9 + vol_binned_small_log_feature_cols) #controversial feature (good on some holdouts, not on others)
xgb_features_9 = (xgb_features_9 + vol_binned_offset_small_log_feature_cols) #controversial feature (good on some holdouts, not on others)

xgb_features_9 = (xgb_features_9 + log_by_loc_cols)
xgb_features_9 = (xgb_features_9 + vol_log_feature_by_loc_cols)

xgb_features_9 = (xgb_features_9 + ['median_log_feature_most_common_location'])
xgb_features_9 = (xgb_features_9 + ['median_log_feature_most_common_event'])
#xgb_features_9 = (xgb_features_9 + ['log_feature_hash']) #not sure what this is



xgb_features_9 = (xgb_features_9 + ['combo_log_features_hash'])
xgb_features_9 = (xgb_features_9 + ['combo_resource_types_hash'])
xgb_features_9 = (xgb_features_9 + ['combo_event_types_hash'])

xgb_features_9 = (xgb_features_9 + ['num_ids_with_log_feature_combo'])
xgb_features_9 = (xgb_features_9 + ['num_ids_with_resource_type_combo'])
xgb_features_9 = (xgb_features_9 + ['num_ids_with_event_type_combo'])

#xgb_features_9 = (xgb_features_9 + ['median_event_type'])
#xgb_features_9 = (xgb_features_9 + ['max_event_type'])
#xgb_features_9 = (xgb_features_9 + ['min_event_type'])
#xgb_features_9 = (xgb_features_9 + ['range_event_type'])

#xgb_features_9 = (xgb_features_9 + ['median_resource_type'])
#xgb_features_9 = (xgb_features_9 + ['max_resource_type'])
#xgb_features_9 = (xgb_features_9 + ['min_resource_type'])

#xgb_features_9 = (xgb_features_9 + ['id'])

#xgb_features_9 = (xgb_features_9 + num_ids_with_log_feature_cols)

params_9 = {'learning_rate': 0.01,
              'subsample': 0.98,
              'reg_alpha': 0.25,
#              'lambda': 0.9,
#              'gamma': 0.5,
              'seed': 8,
              'colsample_bytree': 0.6,
              'n_estimators': 100,
              'objective': 'multi:softprob',
              'eval_metric':'mlogloss',
#              'num_parallel_tree':5,
#              'max_delta_step': 0.5,
              'max_depth': 6,
              'min_child_weight': 1,
              'num_class':3}
num_rounds = 10000
num_boost_rounds = 150



#result_xgb_df_9 = fit_xgb_model(train,test_rare_combos,params_9,xgb_features_9,
#                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
#                                               random_seed = 6)
#result_xgb_df_9 = fit_xgb_model(train,test_rare_combos,params_9,xgb_features_9,
#                              num_rounds = num_rounds, num_boost_rounds = 2100,
#                              do_grid_search = False,
#                              use_early_stopping = False, print_feature_imp = False,random_seed = 6)
#%%
#if (not is_sub_run):
#    ans_xgb_df_9 = pd.merge(result_xgb_df_9,combined[['id','order_by_location','location_number','median_log_feature','combo_log_features_hash',
#                                                  'severity_type','location_max','median_event_type','combo_resource_types_hash'] + vol_log_feature_cols],
#                          left_on = ['id'],
#                          right_on = ['id'],how='left')
#    ans_bad_xgb_df_9 = ans_xgb_df_9.loc[ans_xgb_df_9.log_loss >= 2.5]

#%%
xgb_features_5 = [
                'count_log_features','count_event_types',
#                'count_log_features_diff_to_location',
                'count_resource_types',
                'count_of_volumes',
#                'count_of_volumes_per_feature',
#                'count_of_volumes_repeated',
#                'count_of_most_common_event_log_feature',
#                'count_of_most_common_location_log_feature'
#                'range_log_features_volume'
                ]
xgb_features_5 = (xgb_features_5 + ['location_by_unique_features'])
xgb_features_5 = (xgb_features_5 + ['location_by_unique_resource_types'])
#xgb_features_5 = (xgb_features_5 + ['location_by_unique_severity_types'])
xgb_features_5 = (xgb_features_5 + ['location_by_unique_event_types'])
xgb_features_5 = (xgb_features_5 + ['location_by_most_common_event'])
xgb_features_5 = (xgb_features_5 + ['location_by_most_common_resource'])
#xgb_features_5 = (xgb_features_5 + ['location_by_most_common_severity'])
xgb_features_5 = (xgb_features_5 + ['location_by_most_common_feature'])
#xgb_features_5 = (xgb_features_5 + ['location_by_most_common_combo_feature'])

xgb_features_5 = (xgb_features_5 + ['location_by_std_resource'])

#xgb_features_5 = (xgb_features_5 + ['location_by_median_first_event'])
#xgb_features_5 = (xgb_features_5 + ['location_by_min_first_event'])
#xgb_features_5 = (xgb_features_5 + ['location_by_max_first_event'])

#xgb_features_5 = (xgb_features_5 + ['location_by_median_event'])
#xgb_features_5 = (xgb_features_5 + ['location_by_min_event'])
#xgb_features_5 = (xgb_features_5 + ['location_by_max_event'])
xgb_features_5 = (xgb_features_5 + has_event_type_names)

#xgb_features_5 = (xgb_features_5 + ['location_by_median_ids_with_feature'])
xgb_features_5 = (xgb_features_5 + ['location_by_median_feature_number'])
#xgb_features_5 = (xgb_features_5 + ['location_by_max_feature_number'])

#xgb_features_5 = (xgb_features_5 + ['location_by_min_feature_number'])

#xgb_features_5 = (xgb_features_5 + ['location_by_mean_count_log_features'])
xgb_features_5 = (xgb_features_5 + ['location_by_median_count_log_features'])

#xgb_features_5 = (xgb_features_5 + ['location_by_second_most_common_feature'])

#xgb_features_5 = (xgb_features_5 + ['location_by_mean_count_features'])

xgb_features_5 = (xgb_features_5 + ['location_by_mean_volume'])
xgb_features_5 = (xgb_features_5 + ['location_by_max_volume'])

#xgb_features_5 = (xgb_features_5 + ['location_by_frac_dummy'])

xgb_features_5 = (xgb_features_5 + ['location_max']) #by number of ids
#xgb_features_5 = (xgb_features_5 + ['location_hash']) #by order of log feature
xgb_features_5 = (xgb_features_5 + ['location_number'])

xgb_features_5 = (xgb_features_5 + loc_by_logs_cols)
#xgb_features_5 = (xgb_features_5 + loc_by_logs_summed_cols)

#xgb_features_5 = (xgb_features_5 + loc_by_binned_logs_cols)
#xgb_features_5 = (xgb_features_5 + loc_by_binned_offset_logs_cols)
#xgb_features_5 = (xgb_features_5 + loc_by_binned_logs_summed_cols)

#xgb_features_5 = (xgb_features_5 + loc_by_event_types_cols)
#xgb_features_5 = (xgb_features_5 + loc_by_event_types_summed_cols)

#xgb_features_5 = (xgb_features_5 + ['location_hash_rev']) #by rev order of log feature

xgb_features_5 = (xgb_features_5 + ['sum_log_features_volume'])
#xgb_features_5 = (xgb_features_5 + ['volume_log_features_diff_to_location'])
xgb_features_5 = (xgb_features_5 + ['mean_log_features_volume'])
xgb_features_5 = (xgb_features_5 + ['max_log_features_volume'])
xgb_features_5 = (xgb_features_5 + ['min_num_ids_with_log_feature'])
#xgb_features_5 = (xgb_features_5 + ['std_log_features_volume','std_num_ids_with_log_feature'])
#xgb_features_5 = (xgb_features_5 + ['std_num_ids_with_log_feature'])

#xgb_features_5 = (xgb_features_5 + ['max_volume_log_feature'])
#xgb_features_5 = (xgb_features_5 + ['max_vol_log_feature_by_most_common_location'])
#xgb_features_5 = (xgb_features_5 + ['min_vol_log_feature_by_most_common_location'])



#xgb_features_5 = (xgb_features_5 + ['first_log_feature_hash'])
#xgb_features_5 = (xgb_features_5 + ['first_log_feature_volume'])
#xgb_features_5 = (xgb_features_5 + ['last_log_feature_hash']) #this feature may be bad
#xgb_features_5 = (xgb_features_5 + ['last_log_feature_volume'])

#xgb_features_5 = (xgb_features_5 + ['median_feature_diff_to_location'])

#xgb_features_5 = (xgb_features_5 + ['log_82_203_vol_ratio'])
xgb_features_5 = (xgb_features_5 + ['log_203_82_vol_diff'])
xgb_features_5 = (xgb_features_5 + ['log_201_80_vol_diff'])
xgb_features_5 = (xgb_features_5 + ['log_312_232_vol_diff'])
xgb_features_5 = (xgb_features_5 + ['log_170_54_vol_diff'])
xgb_features_5 = (xgb_features_5 + ['log_171_55_vol_diff'])
xgb_features_5 = (xgb_features_5 + ['log_193_71_vol_diff'])



xgb_features_5 = (xgb_features_5 + ['std_log_feature'])
xgb_features_5 = (xgb_features_5 + ['median_log_feature'])
xgb_features_5 = (xgb_features_5 + ['min_log_feature'])
xgb_features_5 = (xgb_features_5 + ['max_log_feature'])
xgb_features_5 = (xgb_features_5 + ['log_features_range'])
xgb_features_5 = (xgb_features_5 + ['volume_of_min_log_feature'])
xgb_features_5 = (xgb_features_5 + ['volume_of_max_log_feature'])



#xgb_features_5 = (xgb_features_5 + ['first_log_feature_hash_next'])

#xgb_features_5 = (xgb_features_5 + ['max_is_next_resource_type_repeat'])

#xgb_features_5 = (xgb_features_5 + resource_type_position_cols)
#xgb_features_5 = (xgb_features_5 + event_type_position_cols)

xgb_features_5 = (xgb_features_5 + ['position_event_type_1','position_event_type_2'])
#xgb_features_5 = (xgb_features_5 + ['position_resource_type_1','position_resource_type_2'])

#xgb_features_5 = (xgb_features_5 + ['position_log_feature_1'])
#xgb_features_5 = (xgb_features_5 + log_feature_position_cols)
#xgb_features_5 = (xgb_features_5 + log_volume_position_cols)


#xgb_features_5 = (xgb_features_5 + ['switches_resource_type'])

#xgb_features_5 = (xgb_features_5 + ['order_of_log_feature'])


#xgb_features_5 = (xgb_features_5 + ['cumulative_feature_vol'])
#xgb_features_5 = (xgb_features_5 + ['count_of_log_feature_seen_no_dups'])
#xgb_features_5 = (xgb_features_5 + ['count_of_event_type_seen'])

xgb_features_5 = (xgb_features_5 + ['order_by_location'])
xgb_features_5 = (xgb_features_5 + ['reverse_order_by_location'])
#xgb_features_5 = (xgb_features_5 + ['order_by_location_cum_vol'])

xgb_features_5 = (xgb_features_5 + ['order_by_location_fraction'])
xgb_features_5 = (xgb_features_5 + ['reverse_order_by_location_fraction'])

#xgb_features_5 = (xgb_features_5 + loc_dummies_ordered_cols)

#xgb_features_5 = (xgb_features_5 + ['order_by_location_and_location'])

#xgb_features_5 = (xgb_features_5 + ['order_by_location_cum_event_type_count'])
#xgb_features_5 = (xgb_features_5 + ['order_by_location_cum_resource_type_count'])

#xgb_features_5 = (xgb_features_5 + ['order_by_location_and_resource_type_change'])
#xgb_features_5 = (xgb_features_5 + ['order_by_location_and_severity_type_change'])
#xgb_features_5 = (xgb_features_5 + ['reverse_order_by_location_and_resource_type_change'])

#xgb_features_5 = (xgb_features_5 + ['order_by_first_feature']) #doesn't help

#xgb_features_5 = (xgb_features_5 + ['order_by_first_resource_type'])
#xgb_features_5 = (xgb_features_5 + ['order_by_first_event_type'])

#xgb_features_5 = (xgb_features_5 + ['is_final_id_at_location','is_next_to_final_id_at_location'])

#xgb_features_5 = (xgb_features_5 + ['severity_type','num_ids_with_severity_type'])
xgb_features_5 = (xgb_features_5 + ['severity_type'])


xgb_features_5 = (xgb_features_5 + vol_log_feature_cols)
xgb_features_5 = (xgb_features_5 + vol_binned_log_feature_cols) #controversial feature (good on some holdouts, not on others)
xgb_features_5 = (xgb_features_5 + vol_binned_offset_log_feature_cols) #controversial feature (good on some holdouts, not on others)


#xgb_features_5 = (xgb_features_5 + log_feature_vol_combo_cols)
#xgb_features_5 = (xgb_features_5 + log_vol_feature_cols)


#xgb_features_5 = (xgb_features_5 + severity_type_cols)
#xgb_features_5 = (xgb_features_5 + event_type_cols)
#xgb_features_5 = (xgb_features_5 + resource_type_cols)
#xgb_features_5 = (xgb_features_5 + ['resource_type_5','resource_type_1','resource_type_9','resource_type_6','resource_type_7','resource_type_3'])
#xgb_features_5 = (xgb_features_5 + log_feature_cols)



xgb_features_5 = (xgb_features_5 + log_by_loc_cols)
#xgb_features_5 = (xgb_features_5 + vol_log_feature_by_loc_cols)

xgb_features_5 = (xgb_features_5 + ['median_log_feature_most_common_location'])
xgb_features_5 = (xgb_features_5 + ['median_log_feature_most_common_event'])
#xgb_features_5 = (xgb_features_5 + ['log_feature_hash']) #not sure what this is



xgb_features_5 = (xgb_features_5 + ['combo_log_features_hash'])
xgb_features_5 = (xgb_features_5 + ['combo_resource_types_hash'])
xgb_features_5 = (xgb_features_5 + ['combo_event_types_hash'])

xgb_features_5 = (xgb_features_5 + ['num_ids_with_log_feature_combo'])
xgb_features_5 = (xgb_features_5 + ['num_ids_with_resource_type_combo'])
xgb_features_5 = (xgb_features_5 + ['num_ids_with_event_type_combo'])

xgb_features_5 = (xgb_features_5 + ['median_event_type'])
xgb_features_5 = (xgb_features_5 + ['max_event_type'])
xgb_features_5 = (xgb_features_5 + ['min_event_type'])
#xgb_features_5 = (xgb_features_5 + ['range_event_type'])

#xgb_features_5 = (xgb_features_5 + ['median_resource_type'])
#xgb_features_5 = (xgb_features_5 + ['max_resource_type'])
xgb_features_5 = (xgb_features_5 + ['min_resource_type'])

#xgb_features_5 = (xgb_features_5 + ['id'])

#xgb_features_5 = (xgb_features_5 + num_ids_with_log_feature_cols)

#xgb_features_5 = (xgb_features_5 + num_ids_with_resource_type_cols)
#xgb_features_5 = (xgb_features_5 + num_ids_with_event_type_cols)
#xgb_features_5 = (xgb_features_5 + num_ids_with_severity_type_cols)


#xgb_features_5.remove('log_feature_hash')

params_5 = {'learning_rate': 0.01,
              'subsample': 0.95,
              'reg_alpha': 0.4,
              'lambda': 0.9,
              'gamma': 2.0,
              'seed': 8,
              'colsample_bytree': 0.7,
              'n_estimators': 100,
              'objective': 'multi:softprob',
              'eval_metric':'mlogloss',
#              'num_parallel_tree':5,
#              'max_delta_step': 0.5,
              'max_depth': 10,
              'min_child_weight': 4,
              'num_class':3}
num_rounds = 10000
num_boost_rounds = 150

#result_xgb_df_5 = fit_xgb_model(train,test,params_5,xgb_features_5,
#                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
#                                               random_seed = 6)
result_xgb_df_5 = fit_xgb_model(train,test,params_5,xgb_features_5,
                              num_rounds = num_rounds, num_boost_rounds = 3400,
                              do_grid_search = False,
                              use_early_stopping = False, print_feature_imp = False,random_seed = 6)
#%%
xgb_features_6 = [
                'count_log_features','count_event_types',
#                'count_log_features_diff_to_location',
                'count_resource_types',
                'count_of_volumes',
#                'count_of_volumes_per_feature',
#                'count_of_volumes_repeated',
#                'count_of_most_common_event_log_feature',
#                'count_of_most_common_location_log_feature'
#                'range_log_features_volume'
                ]
xgb_features_6 = (xgb_features_6 + ['location_by_unique_features'])
xgb_features_6 = (xgb_features_6 + ['location_by_unique_resource_types'])
#xgb_features_6 = (xgb_features_6 + ['location_by_unique_severity_types'])
xgb_features_6 = (xgb_features_6 + ['location_by_unique_event_types'])
xgb_features_6 = (xgb_features_6 + ['location_by_most_common_event'])
xgb_features_6 = (xgb_features_6 + ['location_by_most_common_resource'])
#xgb_features_6 = (xgb_features_6 + ['location_by_most_common_severity'])
xgb_features_6 = (xgb_features_6 + ['location_by_most_common_feature'])
#xgb_features_6 = (xgb_features_6 + ['location_by_most_common_combo_feature'])

#xgb_features_6 = (xgb_features_6 + ['location_by_median_first_event'])
#xgb_features_6 = (xgb_features_6 + ['location_by_min_first_event'])
#xgb_features_6 = (xgb_features_6 + ['location_by_max_first_event'])

#xgb_features_6 = (xgb_features_6 + ['location_by_median_event'])
#xgb_features_6 = (xgb_features_6 + ['location_by_min_event'])
#xgb_features_6 = (xgb_features_6 + ['location_by_max_event'])
#xgb_features_6 = (xgb_features_6 + has_event_type_names)

#xgb_features_6 = (xgb_features_6 + ['location_by_median_ids_with_feature'])
xgb_features_6 = (xgb_features_6 + ['location_by_median_feature_number'])
#xgb_features_6 = (xgb_features_6 + ['location_by_max_feature_number'])

#xgb_features_6 = (xgb_features_6 + ['location_by_min_feature_number'])

#xgb_features_6 = (xgb_features_6 + ['location_by_mean_count_log_features'])
xgb_features_6 = (xgb_features_6 + ['location_by_median_count_log_features'])

#xgb_features_6 = (xgb_features_6 + ['location_by_second_most_common_feature'])

#xgb_features_6 = (xgb_features_6 + ['location_by_mean_count_features'])

xgb_features_6 = (xgb_features_6 + ['location_by_mean_volume'])
xgb_features_6 = (xgb_features_6 + ['location_by_max_volume'])

#xgb_features_6 = (xgb_features_6 + ['location_by_frac_dummy'])

xgb_features_6 = (xgb_features_6 + ['location_max']) #by number of ids
#xgb_features_6 = (xgb_features_6 + ['location_hash']) #by order of log feature
xgb_features_6 = (xgb_features_6 + ['location_number'])

xgb_features_6 = (xgb_features_6 + loc_by_logs_cols)
xgb_features_6 = (xgb_features_6 + loc_by_logs_summed_cols)

#xgb_features_6 = (xgb_features_6 + loc_by_binned_logs_cols)
#xgb_features_6 = (xgb_features_6 + loc_by_binned_offset_logs_cols)
#xgb_features_6 = (xgb_features_6 + loc_by_binned_logs_summed_cols)

#xgb_features_6 = (xgb_features_6 + loc_by_event_types_cols)
#xgb_features_6 = (xgb_features_6 + loc_by_event_types_summed_cols)

#xgb_features_6 = (xgb_features_6 + ['location_hash_rev']) #by rev order of log feature

#xgb_features_6 = (xgb_features_6 + ['sum_log_features_volume'])
#xgb_features_6 = (xgb_features_6 + ['volume_log_features_diff_to_location'])
xgb_features_6 = (xgb_features_6 + ['mean_log_features_volume'])
#xgb_features_6 = (xgb_features_6 + ['max_log_features_volume'])
xgb_features_6 = (xgb_features_6 + ['min_num_ids_with_log_feature'])
#xgb_features_6 = (xgb_features_6 + ['std_log_features_volume','std_num_ids_with_log_feature'])
#xgb_features_6 = (xgb_features_6 + ['std_num_ids_with_log_feature'])

#xgb_features_6 = (xgb_features_6 + ['vol_summed_to_loc_vol_ratio'])

#xgb_features_6 = (xgb_features_6 + ['max_volume_log_feature'])
#xgb_features_6 = (xgb_features_6 + ['max_vol_log_feature_by_most_common_location'])
#xgb_features_6 = (xgb_features_6 + ['min_vol_log_feature_by_most_common_location'])



#xgb_features_6 = (xgb_features_6 + ['first_log_feature_hash'])
#xgb_features_6 = (xgb_features_6 + ['first_log_feature_volume'])
#xgb_features_6 = (xgb_features_6 + ['last_log_feature_hash']) #this feature may be bad
#xgb_features_6 = (xgb_features_6 + ['last_log_feature_volume'])

#xgb_features_6 = (xgb_features_6 + ['median_feature_diff_to_location'])

#xgb_features_6 = (xgb_features_6 + ['log_82_203_vol_ratio'])
#xgb_features_6 = (xgb_features_6 + ['log_203_82_vol_diff'])
#xgb_features_6 = (xgb_features_6 + ['log_201_80_vol_diff'])
#xgb_features_6 = (xgb_features_6 + ['log_312_232_vol_diff'])
#xgb_features_6 = (xgb_features_6 + ['log_170_54_vol_diff'])
#xgb_features_6 = (xgb_features_6 + ['log_171_55_vol_diff'])
#xgb_features_6 = (xgb_features_6 + ['log_193_71_vol_diff'])



xgb_features_6 = (xgb_features_6 + ['std_log_feature'])
xgb_features_6 = (xgb_features_6 + ['median_log_feature'])
xgb_features_6 = (xgb_features_6 + ['min_log_feature'])
xgb_features_6 = (xgb_features_6 + ['max_log_feature'])
xgb_features_6 = (xgb_features_6 + ['rarest_log_feature'])
xgb_features_6 = (xgb_features_6 + ['volume_of_rarest_log_feature'])
xgb_features_6 = (xgb_features_6 + ['least_rare_log_feature'])
xgb_features_6 = (xgb_features_6 + ['volume_of_least_rare_log_feature'])
xgb_features_6 = (xgb_features_6 + ['volume_of_min_log_feature'])
xgb_features_6 = (xgb_features_6 + ['volume_of_max_log_feature'])
xgb_features_6 = (xgb_features_6 + ['log_features_range'])



#xgb_features_6 = (xgb_features_6 + ['first_log_feature_hash_next'])

#xgb_features_6 = (xgb_features_6 + ['max_is_next_resource_type_repeat'])

#xgb_features_6 = (xgb_features_6 + resource_type_position_cols)
#xgb_features_6 = (xgb_features_6 + event_type_position_cols)

#xgb_features_6 = (xgb_features_6 + ['position_event_type_1','position_event_type_2'])
#xgb_features_6 = (xgb_features_6 + ['position_resource_type_1','position_resource_type_2'])

#xgb_features_6 = (xgb_features_6 + ['position_log_feature_1'])
#xgb_features_6 = (xgb_features_6 + log_feature_position_cols)
#xgb_features_6 = (xgb_features_6 + log_volume_position_cols)


#xgb_features_6 = (xgb_features_6 + ['switches_resource_type'])

#xgb_features_6 = (xgb_features_6 + ['order_of_log_feature'])


#xgb_features_6 = (xgb_features_6 + ['cumulative_feature_vol'])
#xgb_features_6 = (xgb_features_6 + ['count_of_log_feature_seen_no_dups'])
#xgb_features_6 = (xgb_features_6 + ['count_of_event_type_seen'])

xgb_features_6 = (xgb_features_6 + ['order_by_location'])
xgb_features_6 = (xgb_features_6 + ['reverse_order_by_location'])
#xgb_features_6 = (xgb_features_6 + ['order_by_location_cum_vol'])

#xgb_features_6 = (xgb_features_6 + ['order_by_location_fraction'])
#xgb_features_6 = (xgb_features_6 + ['reverse_order_by_location_fraction'])

#xgb_features_6 = (xgb_features_6 + loc_dummies_ordered_cols)

#xgb_features_6 = (xgb_features_6 + ['order_by_location_and_location'])

#xgb_features_6 = (xgb_features_6 + ['order_by_location_cum_event_type_count'])
#xgb_features_6 = (xgb_features_6 + ['order_by_location_cum_resource_type_count'])

#xgb_features_6 = (xgb_features_6 + ['order_by_location_and_resource_type_change'])
#xgb_features_6 = (xgb_features_6 + ['order_by_location_and_severity_type_change'])
#xgb_features_6 = (xgb_features_6 + ['reverse_order_by_location_and_resource_type_change'])

#xgb_features_6 = (xgb_features_6 + ['order_by_first_feature']) #doesn't help

#xgb_features_6 = (xgb_features_6 + ['order_by_first_resource_type'])
#xgb_features_6 = (xgb_features_6 + ['order_by_first_event_type'])

#xgb_features_6 = (xgb_features_6 + ['is_final_id_at_location','is_next_to_final_id_at_location'])

#xgb_features_6 = (xgb_features_6 + ['severity_type','num_ids_with_severity_type'])
xgb_features_6 = (xgb_features_6 + ['severity_type'])


xgb_features_6 = (xgb_features_6 + vol_log_feature_cols)
xgb_features_6 = (xgb_features_6 + vol_binned_log_feature_cols) #controversial feature (good on some holdouts, not on others)
xgb_features_6 = (xgb_features_6 + vol_binned_offset_log_feature_cols) #controversial feature (good on some holdouts, not on others)


#xgb_features_6 = (xgb_features_6 + log_feature_vol_combo_cols)
#xgb_features_6 = (xgb_features_6 + log_vol_feature_cols)


#xgb_features_6 = (xgb_features_6 + severity_type_cols)
#xgb_features_6 = (xgb_features_6 + event_type_cols)
#xgb_features_6 = (xgb_features_6 + resource_type_cols)
#xgb_features_6 = (xgb_features_6 + ['resource_type_6','resource_type_1','resource_type_9','resource_type_6','resource_type_7','resource_type_3'])
#xgb_features_6 = (xgb_features_6 + log_feature_cols)



xgb_features_6 = (xgb_features_6 + log_by_loc_cols)
#xgb_features_6 = (xgb_features_6 + vol_log_feature_by_loc_cols)

xgb_features_6 = (xgb_features_6 + ['median_log_feature_most_common_location'])
xgb_features_6 = (xgb_features_6 + ['median_log_feature_most_common_event'])
#xgb_features_6 = (xgb_features_6 + ['log_feature_hash']) #not sure what this is



#xgb_features_6 = (xgb_features_6 + ['combo_log_features_hash'])
xgb_features_6 = (xgb_features_6 + ['combo_resource_types_hash'])
xgb_features_6 = (xgb_features_6 + ['combo_event_types_hash'])

xgb_features_6 = (xgb_features_6 + ['num_ids_with_log_feature_combo'])
xgb_features_6 = (xgb_features_6 + ['num_ids_with_resource_type_combo'])
xgb_features_6 = (xgb_features_6 + ['num_ids_with_event_type_combo'])

xgb_features_6 = (xgb_features_6 + ['median_event_type'])
xgb_features_6 = (xgb_features_6 + ['max_event_type'])
xgb_features_6 = (xgb_features_6 + ['min_event_type'])
#xgb_features_6 = (xgb_features_6 + ['range_event_type'])

#xgb_features_6 = (xgb_features_6 + ['median_resource_type'])
#xgb_features_6 = (xgb_features_6 + ['max_resource_type'])
xgb_features_6 = (xgb_features_6 + ['min_resource_type'])

#xgb_features_6 = (xgb_features_6 + ['id'])

#xgb_features_6 = (xgb_features_6 + num_ids_with_log_feature_cols)

#xgb_features_6 = (xgb_features_6 + num_ids_with_resource_type_cols)
#xgb_features_6 = (xgb_features_6 + num_ids_with_event_type_cols)
#xgb_features_6 = (xgb_features_6 + num_ids_with_severity_type_cols)


#xgb_features_6.remove('log_feature_hash')

params_6 = {'learning_rate': 0.005,
              'subsample': 0.95,
              'reg_alpha': 0.05,
              'lambda': 0.98,
              'gamma': 2.0,
              'seed': 8,
              'colsample_bytree': 0.6,
              'n_estimators': 100,
              'objective': 'multi:softprob',
              'eval_metric':'mlogloss',
#              'num_parallel_tree':5,
#              'max_delta_step': 0.5,
              'max_depth': 11,
              'min_child_weight': 2,
              'num_class':3}
num_rounds = 10000
num_boost_rounds = 150

#result_xgb_df_6 = fit_xgb_model(train,test,params_6,xgb_features_6,
#                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
#                                               random_seed = 6)
result_xgb_df_6 = fit_xgb_model(train,test,params_6,xgb_features_6,
                              num_rounds = num_rounds, num_boost_rounds = 6000,
                              do_grid_search = False,
                              use_early_stopping = False, print_feature_imp = False,random_seed = 6)
#%%
xgb_features_7 = [
                'count_log_features','count_event_types',
#                'count_log_features_diff_to_location',
                'count_resource_types',
                'count_of_volumes',
#                'count_of_volumes_per_feature',
#                'count_of_volumes_repeated',
#                'count_of_most_common_event_log_feature',
#                'count_of_most_common_location_log_feature'
#                'range_log_features_volume'
                ]
xgb_features_7 = (xgb_features_7 + ['location_by_unique_features'])
xgb_features_7 = (xgb_features_7 + ['location_by_unique_resource_types'])
#xgb_features_7 = (xgb_features_7 + ['location_by_unique_severity_types'])
xgb_features_7 = (xgb_features_7 + ['location_by_unique_event_types'])
xgb_features_7 = (xgb_features_7 + ['location_by_most_common_event'])
xgb_features_7 = (xgb_features_7 + ['location_by_most_common_resource'])
#xgb_features_7 = (xgb_features_7 + ['location_by_most_common_severity'])
xgb_features_7 = (xgb_features_7 + ['location_by_most_common_feature'])
#xgb_features_7 = (xgb_features_7 + ['location_by_most_common_combo_feature'])

xgb_features_7 = (xgb_features_7 + ['location_by_median_event'])
#xgb_features_7 = (xgb_features_7 + ['location_by_min_event'])
#xgb_features_7 = (xgb_features_7 + ['location_by_max_event'])
#xgb_features_7 = (xgb_features_7 + has_event_type_names)
xgb_features_7 = (xgb_features_7 + has_resource_type_names)

#xgb_features_7 = (xgb_features_7 + ['location_by_median_ids_with_feature'])
xgb_features_7 = (xgb_features_7 + ['location_by_median_feature_number'])
#xgb_features_7 = (xgb_features_7 + ['location_by_max_feature_number'])

#xgb_features_7 = (xgb_features_7 + ['location_by_min_feature_number'])

#xgb_features_7 = (xgb_features_7 + ['location_by_mean_count_log_features'])
xgb_features_7 = (xgb_features_7 + ['location_by_median_count_log_features'])

xgb_features_7 = (xgb_features_7 + ['location_by_second_most_common_feature'])

#xgb_features_7 = (xgb_features_7 + ['location_by_mean_count_features'])

xgb_features_7 = (xgb_features_7 + ['location_by_mean_volume'])
#xgb_features_7 = (xgb_features_7 + ['location_by_max_volume'])

#xgb_features_7 = (xgb_features_7 + ['location_by_frac_dummy'])

xgb_features_7 = (xgb_features_7 + ['location_max']) #by number of ids
#xgb_features_7 = (xgb_features_7 + ['location_hash']) #by order of log feature
xgb_features_7 = (xgb_features_7 + ['location_number'])

xgb_features_7 = (xgb_features_7 + loc_by_logs_cols)
xgb_features_7 = (xgb_features_7 + loc_by_logs_summed_cols)

#xgb_features_7 = (xgb_features_7 + loc_by_binned_logs_cols)
#xgb_features_7 = (xgb_features_7 + loc_by_binned_offset_logs_cols)
#xgb_features_7 = (xgb_features_7 + loc_by_binned_logs_summed_cols)

#xgb_features_7 = (xgb_features_7 + loc_by_event_types_cols)
#xgb_features_7 = (xgb_features_7 + loc_by_event_types_summed_cols)

#xgb_features_7 = (xgb_features_7 + ['location_hash_rev']) #by rev order of log feature

#xgb_features_7 = (xgb_features_7 + ['sum_log_features_volume'])
#xgb_features_7 = (xgb_features_7 + ['volume_log_features_diff_to_location'])
xgb_features_7 = (xgb_features_7 + ['mean_log_features_volume'])
#xgb_features_7 = (xgb_features_7 + ['max_log_features_volume'])
xgb_features_7 = (xgb_features_7 + ['min_num_ids_with_log_feature'])
#xgb_features_7 = (xgb_features_7 + ['std_log_features_volume','std_num_ids_with_log_feature'])
#xgb_features_7 = (xgb_features_7 + ['std_num_ids_with_log_feature'])

#xgb_features_7 = (xgb_features_7 + ['max_volume_log_feature'])
#xgb_features_7 = (xgb_features_7 + ['max_vol_log_feature_by_most_common_location'])
#xgb_features_7 = (xgb_features_7 + ['min_vol_log_feature_by_most_common_location'])

#xgb_features_7 = (xgb_features_7 + ['log_82_203_vol_ratio'])
#xgb_features_7 = (xgb_features_7 + ['log_203_82_vol_diff'])
#xgb_features_7 = (xgb_features_7 + ['log_201_80_vol_diff'])
#xgb_features_7 = (xgb_features_7 + ['log_312_232_vol_diff'])
#xgb_features_7 = (xgb_features_7 + ['log_170_54_vol_diff'])
#xgb_features_7 = (xgb_features_7 + ['log_171_55_vol_diff'])
#xgb_features_7 = (xgb_features_7 + ['log_193_71_vol_diff'])



xgb_features_7 = (xgb_features_7 + ['std_log_feature'])
xgb_features_7 = (xgb_features_7 + ['median_log_feature'])
xgb_features_7 = (xgb_features_7 + ['min_log_feature'])
xgb_features_7 = (xgb_features_7 + ['max_log_feature'])
xgb_features_7 = (xgb_features_7 + ['volume_of_min_log_feature'])
xgb_features_7 = (xgb_features_7 + ['volume_of_max_log_feature'])
xgb_features_7 = (xgb_features_7 + ['rarest_log_feature'])
xgb_features_7 = (xgb_features_7 + ['volume_of_rarest_log_feature'])
xgb_features_7 = (xgb_features_7 + ['least_rare_log_feature'])
xgb_features_7 = (xgb_features_7 + ['volume_of_least_rare_log_feature'])
xgb_features_7 = (xgb_features_7 + ['log_features_range'])

#xgb_features_7 = (xgb_features_7 + resource_type_position_cols)
#xgb_features_7 = (xgb_features_7 + event_type_position_cols)

#xgb_features_7 = (xgb_features_7 + ['position_event_type_1','position_event_type_2'])
#xgb_features_7 = (xgb_features_7 + ['position_resource_type_1','position_resource_type_2'])

#xgb_features_7 = (xgb_features_7 + ['position_log_feature_1'])
#xgb_features_7 = (xgb_features_7 + log_feature_position_cols)
#xgb_features_7 = (xgb_features_7 + log_volume_position_cols)

#xgb_features_7 = (xgb_features_7 + ['cumulative_feature_vol'])
#xgb_features_7 = (xgb_features_7 + ['count_of_log_feature_seen_no_dups'])
#xgb_features_7 = (xgb_features_7 + ['count_of_event_type_seen'])

xgb_features_7 = (xgb_features_7 + ['order_by_location'])
xgb_features_7 = (xgb_features_7 + ['reverse_order_by_location'])
#xgb_features_7 = (xgb_features_7 + ['order_by_location_cum_vol'])

#xgb_features_7 = (xgb_features_7 + ['order_by_location_fraction'])
#xgb_features_7 = (xgb_features_7 + ['reverse_order_by_location_fraction'])

#xgb_features_7 = (xgb_features_7 + loc_dummies_ordered_cols)

#xgb_features_7 = (xgb_features_7 + ['order_by_location_and_location'])

#xgb_features_7 = (xgb_features_7 + ['order_by_location_cum_event_type_count'])
#xgb_features_7 = (xgb_features_7 + ['order_by_location_cum_resource_type_count'])

#xgb_features_7 = (xgb_features_7 + ['order_by_location_and_resource_type_change'])
#xgb_features_7 = (xgb_features_7 + ['order_by_location_and_severity_type_change'])
#xgb_features_7 = (xgb_features_7 + ['reverse_order_by_location_and_resource_type_change'])

#xgb_features_7 = (xgb_features_7 + ['order_by_first_feature']) #doesn't help

#xgb_features_7 = (xgb_features_7 + ['order_by_first_resource_type'])
#xgb_features_7 = (xgb_features_7 + ['order_by_first_event_type'])

#xgb_features_7 = (xgb_features_7 + ['is_final_id_at_location','is_next_to_final_id_at_location'])

#xgb_features_7 = (xgb_features_7 + ['severity_type','num_ids_with_severity_type'])
xgb_features_7 = (xgb_features_7 + ['severity_type'])


xgb_features_7 = (xgb_features_7 + vol_log_feature_cols)
xgb_features_7 = (xgb_features_7 + vol_binned_log_feature_cols) #controversial feature (good on some holdouts, not on others)
xgb_features_7 = (xgb_features_7 + vol_binned_offset_log_feature_cols) #controversial feature (good on some holdouts, not on others)
xgb_features_7 = (xgb_features_7 + vol_binned_small_log_feature_cols) #controversial feature (good on some holdouts, not on others)
xgb_features_7 = (xgb_features_7 + vol_binned_offset_small_log_feature_cols) #controversial feature (good on some holdouts, not on others)


#xgb_features_7 = (xgb_features_7 + log_feature_vol_combo_cols)
#xgb_features_7 = (xgb_features_7 + log_vol_feature_cols)


#xgb_features_7 = (xgb_features_7 + severity_type_cols)
#xgb_features_7 = (xgb_features_7 + event_type_cols)
xgb_features_7 = (xgb_features_7 + resource_type_cols)
#xgb_features_7 = (xgb_features_7 + ['resource_type_7','resource_type_1','resource_type_9','resource_type_7','resource_type_7','resource_type_3'])
#xgb_features_7 = (xgb_features_7 + log_feature_cols)



xgb_features_7 = (xgb_features_7 + log_by_loc_cols)
#xgb_features_7 = (xgb_features_7 + vol_log_feature_by_loc_cols)

xgb_features_7 = (xgb_features_7 + ['median_log_feature_most_common_location'])
xgb_features_7 = (xgb_features_7 + ['median_log_feature_most_common_event'])
#xgb_features_7 = (xgb_features_7 + ['log_feature_hash']) #not sure what this is



#xgb_features_7 = (xgb_features_7 + ['combo_log_features_hash'])
xgb_features_7 = (xgb_features_7 + ['combo_resource_types_hash'])
xgb_features_7 = (xgb_features_7 + ['combo_event_types_hash'])

xgb_features_7 = (xgb_features_7 + ['num_ids_with_log_feature_combo'])
xgb_features_7 = (xgb_features_7 + ['num_ids_with_resource_type_combo'])
xgb_features_7 = (xgb_features_7 + ['num_ids_with_event_type_combo'])

xgb_features_7 = (xgb_features_7 + ['median_event_type'])
xgb_features_7 = (xgb_features_7 + ['max_event_type'])
xgb_features_7 = (xgb_features_7 + ['min_event_type'])
#xgb_features_7 = (xgb_features_7 + ['range_event_type'])

#xgb_features_7 = (xgb_features_7 + ['median_resource_type'])
xgb_features_7 = (xgb_features_7 + ['max_resource_type'])
xgb_features_7 = (xgb_features_7 + ['min_resource_type'])

#xgb_features_7 = (xgb_features_7 + ['id'])

#xgb_features_7 = (xgb_features_7 + num_ids_with_log_feature_cols)

#xgb_features_7 = (xgb_features_7 + num_ids_with_resource_type_cols)
#xgb_features_7 = (xgb_features_7 + num_ids_with_event_type_cols)
#xgb_features_7 = (xgb_features_7 + num_ids_with_severity_type_cols)


#xgb_features_7.remove('log_feature_hash')

params_7 = {'learning_rate': 0.01,
              'subsample': 0.95,
              'reg_alpha': 0.6,
              'lambda': 0.8,
              'gamma': 1.5,
              'seed': 9,
              'colsample_bytree': 0.7,
              'n_estimators': 100,
              'objective': 'multi:softprob',
              'eval_metric':'mlogloss',
#              'num_parallel_tree':5,
#              'max_delta_step': 0.5,
              'max_depth': 15,
              'min_child_weight':4,
              'num_class':3}
num_rounds = 10000
num_boost_rounds = 150

#result_xgb_df_7 = fit_xgb_model(train,test,params_7,xgb_features_7,
#                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
#                                               random_seed = 9)
result_xgb_df_7 = fit_xgb_model(train,test,params_7,xgb_features_7,
                              num_rounds = num_rounds, num_boost_rounds = 1700,
                              do_grid_search = False,
                              use_early_stopping = False, print_feature_imp = False,random_seed = 9)
#%%
xgb_features_8 = [
                'count_log_features','count_event_types',
#                'count_log_features_diff_to_location',
                'count_resource_types',
                'count_of_volumes',
#                'count_of_volumes_per_feature',
#                'count_of_volumes_repeated',
#                'count_of_most_common_event_log_feature',
#                'count_of_most_common_location_log_feature'
#                'range_log_features_volume'
                ]
xgb_features_8 = (xgb_features_8 + ['location_by_unique_features'])
xgb_features_8 = (xgb_features_8 + ['location_by_unique_resource_types'])
xgb_features_8 = (xgb_features_8 + ['location_by_unique_severity_types'])
xgb_features_8 = (xgb_features_8 + ['location_by_unique_event_types'])
xgb_features_8 = (xgb_features_8 + ['location_by_most_common_event'])
xgb_features_8 = (xgb_features_8 + ['location_by_most_common_resource'])
xgb_features_8 = (xgb_features_8 + ['location_by_most_common_severity'])
xgb_features_8 = (xgb_features_8 + ['location_by_most_common_feature'])
#xgb_features_8 = (xgb_features_8 + ['location_by_most_common_combo_feature'])

#xgb_features_8 = (xgb_features_8 + ['location_by_median_first_event'])
#xgb_features_8 = (xgb_features_8 + ['location_by_min_first_event'])
#xgb_features_8 = (xgb_features_8 + ['location_by_max_first_event'])

#xgb_features_8 = (xgb_features_8 + ['location_by_median_event'])
#xgb_features_8 = (xgb_features_8 + ['location_by_min_event'])
#xgb_features_8 = (xgb_features_8 + ['location_by_max_event'])
xgb_features_8 = (xgb_features_8 + has_event_type_names)
xgb_features_8 = (xgb_features_8 + has_resource_type_names)

#xgb_features_8 = (xgb_features_8 + ['location_by_median_ids_with_feature'])
xgb_features_8 = (xgb_features_8 + ['location_by_median_feature_number'])
#xgb_features_8 = (xgb_features_8 + ['location_by_max_feature_number'])

#xgb_features_8 = (xgb_features_8 + ['location_by_min_feature_number'])

#xgb_features_8 = (xgb_features_8 + ['location_by_mean_count_log_features'])
xgb_features_8 = (xgb_features_8 + ['location_by_median_count_log_features'])

xgb_features_8 = (xgb_features_8 + ['location_by_second_most_common_feature'])

#xgb_features_8 = (xgb_features_8 + ['location_by_mean_count_features'])

#xgb_features_8 = (xgb_features_8 + ['location_by_std_resource'])
xgb_features_8 = (xgb_features_8 + ['location_by_mean_volume'])
#xgb_features_8 = (xgb_features_8 + ['location_by_max_volume'])

#xgb_features_8 = (xgb_features_8 + ['location_by_frac_dummy'])

xgb_features_8 = (xgb_features_8 + ['location_max']) #by number of ids
#xgb_features_8 = (xgb_features_8 + ['location_hash']) #by order of log feature
xgb_features_8 = (xgb_features_8 + ['location_number'])

xgb_features_8 = (xgb_features_8 + loc_by_logs_cols)
xgb_features_8 = (xgb_features_8 + loc_by_logs_summed_cols)

#xgb_features_8 = (xgb_features_8 + loc_by_binned_logs_cols)
#xgb_features_8 = (xgb_features_8 + loc_by_binned_offset_logs_cols)
#xgb_features_8 = (xgb_features_8 + loc_by_binned_logs_summed_cols)

#xgb_features_8 = (xgb_features_8 + loc_by_event_types_cols)
#xgb_features_8 = (xgb_features_8 + loc_by_event_types_summed_cols)

#xgb_features_8 = (xgb_features_8 + ['location_hash_rev']) #by rev order of log feature

#xgb_features_8 = (xgb_features_8 + ['sum_log_features_volume'])
#xgb_features_8 = (xgb_features_8 + ['volume_log_features_diff_to_location'])
xgb_features_8 = (xgb_features_8 + ['mean_log_features_volume'])
#xgb_features_8 = (xgb_features_8 + ['max_log_features_volume'])
xgb_features_8 = (xgb_features_8 + ['min_num_ids_with_log_feature'])
#xgb_features_8 = (xgb_features_8 + ['std_log_features_volume','std_num_ids_with_log_feature'])
#xgb_features_8 = (xgb_features_8 + ['std_num_ids_with_log_feature'])

#xgb_features_8 = (xgb_features_8 + ['vol_summed_to_loc_vol_ratio'])


#xgb_features_8 = (xgb_features_8 + ['first_log_feature_hash'])
#xgb_features_8 = (xgb_features_8 + ['first_log_feature_volume'])
#xgb_features_8 = (xgb_features_8 + ['last_log_feature_hash']) #this feature may be bad
#xgb_features_8 = (xgb_features_8 + ['last_log_feature_volume'])

#xgb_features_8 = (xgb_features_8 + ['median_feature_diff_to_location'])

#xgb_features_8 = (xgb_features_8 + ['log_82_203_vol_ratio'])
#xgb_features_8 = (xgb_features_8 + ['log_203_82_vol_diff'])
#xgb_features_8 = (xgb_features_8 + ['log_201_80_vol_diff'])
#xgb_features_8 = (xgb_features_8 + ['log_312_232_vol_diff'])
#xgb_features_8 = (xgb_features_8 + ['log_170_54_vol_diff'])
#xgb_features_8 = (xgb_features_8 + ['log_171_55_vol_diff'])
#xgb_features_8 = (xgb_features_8 + ['log_193_71_vol_diff'])



xgb_features_8 = (xgb_features_8 + ['std_log_feature'])
xgb_features_8 = (xgb_features_8 + ['median_log_feature'])
xgb_features_8 = (xgb_features_8 + ['min_log_feature'])
xgb_features_8 = (xgb_features_8 + ['max_log_feature'])
xgb_features_8 = (xgb_features_8 + ['log_features_range'])



#xgb_features_8 = (xgb_features_8 + ['first_log_feature_hash_next'])

#xgb_features_8 = (xgb_features_8 + ['max_is_next_resource_type_repeat'])

#xgb_features_8 = (xgb_features_8 + resource_type_position_cols)
#xgb_features_8 = (xgb_features_8 + event_type_position_cols)

#xgb_features_8 = (xgb_features_8 + ['position_event_type_1','position_event_type_2'])
#xgb_features_8 = (xgb_features_8 + ['position_resource_type_1','position_resource_type_2'])

#xgb_features_8 = (xgb_features_8 + ['position_log_feature_1'])
#xgb_features_8 = (xgb_features_8 + log_feature_position_cols)
#xgb_features_8 = (xgb_features_8 + log_volume_position_cols)


#xgb_features_8 = (xgb_features_8 + ['switches_resource_type'])

#xgb_features_8 = (xgb_features_8 + ['order_of_log_feature'])


#xgb_features_8 = (xgb_features_8 + ['cumulative_feature_vol'])
#xgb_features_8 = (xgb_features_8 + ['count_of_log_feature_seen_no_dups'])
#xgb_features_8 = (xgb_features_8 + ['count_of_event_type_seen'])

xgb_features_8 = (xgb_features_8 + ['order_by_location'])
xgb_features_8 = (xgb_features_8 + ['reverse_order_by_location'])
#xgb_features_8 = (xgb_features_8 + ['order_by_location_cum_vol'])

#xgb_features_8 = (xgb_features_8 + ['order_by_location_fraction'])
#xgb_features_8 = (xgb_features_8 + ['reverse_order_by_location_fraction'])

#xgb_features_8 = (xgb_features_8 + loc_dummies_ordered_cols)

#xgb_features_8 = (xgb_features_8 + ['order_by_location_and_location'])

#xgb_features_8 = (xgb_features_8 + ['order_by_location_cum_event_type_count'])
#xgb_features_8 = (xgb_features_8 + ['order_by_location_cum_resource_type_count'])

#xgb_features_8 = (xgb_features_8 + ['order_by_location_and_resource_type_change'])
#xgb_features_8 = (xgb_features_8 + ['order_by_location_and_severity_type_change'])
#xgb_features_8 = (xgb_features_8 + ['reverse_order_by_location_and_resource_type_change'])

#xgb_features_8 = (xgb_features_8 + ['order_by_first_feature']) #doesn't help

#xgb_features_8 = (xgb_features_8 + ['order_by_first_resource_type'])
#xgb_features_8 = (xgb_features_8 + ['order_by_first_event_type'])

#xgb_features_8 = (xgb_features_8 + ['is_final_id_at_location','is_next_to_final_id_at_location'])

#xgb_features_8 = (xgb_features_8 + ['severity_type','num_ids_with_severity_type'])
xgb_features_8 = (xgb_features_8 + ['severity_type'])


xgb_features_8 = (xgb_features_8 + vol_log_feature_cols)
xgb_features_8 = (xgb_features_8 + vol_binned_log_feature_cols) #controversial feature (good on some holdouts, not on others)
xgb_features_8 = (xgb_features_8 + vol_binned_offset_log_feature_cols) #controversial feature (good on some holdouts, not on others)


#xgb_features_8 = (xgb_features_8 + log_feature_vol_combo_cols)
#xgb_features_8 = (xgb_features_8 + log_vol_feature_cols)


#xgb_features_8 = (xgb_features_8 + severity_type_cols)
#xgb_features_8 = (xgb_features_8 + event_type_cols)
xgb_features_8 = (xgb_features_8 + resource_type_cols)
#xgb_features_8 = (xgb_features_8 + ['resource_type_8','resource_type_1','resource_type_9','resource_type_8','resource_type_8','resource_type_3'])
#xgb_features_8 = (xgb_features_8 + log_feature_cols)



xgb_features_8 = (xgb_features_8 + log_by_loc_cols)
#xgb_features_8 = (xgb_features_8 + vol_log_feature_by_loc_cols)

xgb_features_8 = (xgb_features_8 + ['median_log_feature_most_common_location'])
xgb_features_8 = (xgb_features_8 + ['median_log_feature_most_common_event'])
#xgb_features_8 = (xgb_features_8 + ['log_feature_hash']) #not sure what this is



xgb_features_8 = (xgb_features_8 + ['combo_log_features_hash'])
xgb_features_8 = (xgb_features_8 + ['combo_resource_types_hash'])
xgb_features_8 = (xgb_features_8 + ['combo_event_types_hash'])

xgb_features_8 = (xgb_features_8 + ['num_ids_with_log_feature_combo'])
xgb_features_8 = (xgb_features_8 + ['num_ids_with_resource_type_combo'])
xgb_features_8 = (xgb_features_8 + ['num_ids_with_event_type_combo'])

xgb_features_8 = (xgb_features_8 + ['median_event_type'])
xgb_features_8 = (xgb_features_8 + ['max_event_type'])
xgb_features_8 = (xgb_features_8 + ['min_event_type'])
#xgb_features_8 = (xgb_features_8 + ['range_event_type'])

#xgb_features_8 = (xgb_features_8 + ['median_resource_type'])
xgb_features_8 = (xgb_features_8 + ['max_resource_type'])
xgb_features_8 = (xgb_features_8 + ['min_resource_type'])

#xgb_features_8 = (xgb_features_8 + ['id'])

#xgb_features_8 = (xgb_features_8 + num_ids_with_log_feature_cols)

#xgb_features_8 = (xgb_features_8 + num_ids_with_resource_type_cols)
#xgb_features_8 = (xgb_features_8 + num_ids_with_event_type_cols)
#xgb_features_8 = (xgb_features_8 + num_ids_with_severity_type_cols)


#xgb_features_8.remove('log_feature_hash')

params_8 = {'learning_rate': 0.01,
              'subsample': 0.98,
              'reg_alpha': 0.05,
#              'lambda': 0.8,
#              'gamma': 1.5,
              'seed': 10,
              'colsample_bytree': 0.2,
              'n_estimators': 100,
              'objective': 'multi:softprob',
              'eval_metric':'mlogloss',
#              'num_parallel_tree':5,
#              'max_delta_step': 0.5,
              'max_depth': 4,
#              'min_child_weight': 5,
              'num_class':3}
num_rounds = 10000
num_boost_rounds = 150

#result_xgb_df_8 = fit_xgb_model(train,test,params_8,xgb_features_8,
#                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
#                                               random_seed = 10)
#result_xgb_df_8 = fit_xgb_model(train,test,params_8,xgb_features_8,
#                              num_rounds = num_rounds, num_boost_rounds = 4800,
#                              do_grid_search = False,
#                              use_early_stopping = False, print_feature_imp = False,random_seed = 9)
#%%
xgb_features_4 = [
                'count_log_features','count_event_types',
                'count_resource_types',
                'count_of_volumes',
                ]
xgb_features_4 = (xgb_features_4 + ['location_by_unique_features'])
xgb_features_4 = (xgb_features_4 + ['location_by_unique_resource_types'])
#xgb_features_4 = (xgb_features_4 + ['location_by_unique_severity_types'])
xgb_features_4 = (xgb_features_4 + ['location_by_unique_event_types'])
xgb_features_4 = (xgb_features_4 + ['location_by_most_common_event'])
xgb_features_4 = (xgb_features_4 + ['location_by_most_common_resource'])
#xgb_features_4 = (xgb_features_4 + ['location_by_most_common_severity'])
xgb_features_4 = (xgb_features_4 + ['location_by_most_common_feature'])
#xgb_features_4 = (xgb_features_4 + ['location_by_most_common_combo_feature'])

#xgb_features_4 = (xgb_features_4 + ['location_by_median_event'])
#xgb_features_4 = (xgb_features_4 + ['location_by_min_event'])
#xgb_features_4 = (xgb_features_4 + ['location_by_max_event'])
xgb_features_4 = (xgb_features_4 + has_event_type_names)
#xgb_features_4 = (xgb_features_4 + has_resource_type_names)

#xgb_features_4 = (xgb_features_4 + ['location_by_median_ids_with_feature'])
xgb_features_4 = (xgb_features_4 + ['location_by_median_feature_number'])
#xgb_features_4 = (xgb_features_4 + ['location_by_max_feature_number'])

#xgb_features_4 = (xgb_features_4 + ['location_by_min_feature_number'])

#xgb_features_4 = (xgb_features_4 + ['location_by_mean_count_log_features'])
xgb_features_4 = (xgb_features_4 + ['location_by_median_count_log_features'])

#xgb_features_4 = (xgb_features_4 + ['location_by_second_most_common_feature'])

xgb_features_4 = (xgb_features_4 + ['location_by_mean_volume'])
xgb_features_4 = (xgb_features_4 + ['location_by_max_volume'])

#xgb_features_4 = (xgb_features_4 + ['location_by_std_event_type'])
xgb_features_4 = (xgb_features_4 + ['location_by_std_resource'])
xgb_features_4 = (xgb_features_4 + ['location_by_std_log_feature'])

#xgb_features_4 = (xgb_features_4 + ['location_by_frac_dummy'])

xgb_features_4 = (xgb_features_4 + ['location_max']) #by number of ids
#xgb_features_4 = (xgb_features_4 + ['location_hash']) #by order of log feature
xgb_features_4 = (xgb_features_4 + ['location_number'])

xgb_features_4 = (xgb_features_4 + loc_by_logs_cols)
#xgb_features_4 = (xgb_features_4 + loc_by_logs_summed_cols)

#xgb_features_4 = (xgb_features_4 + loc_by_binned_logs_cols)
#xgb_features_4 = (xgb_features_4 + loc_by_binned_offset_logs_cols)
#xgb_features_4 = (xgb_features_4 + loc_by_binned_logs_summed_cols)

#xgb_features_4 = (xgb_features_4 + loc_by_event_types_cols)
#xgb_features_4 = (xgb_features_4 + loc_by_event_types_summed_cols)

xgb_features_4 = (xgb_features_4 + ['sum_log_features_volume'])
xgb_features_4 = (xgb_features_4 + ['mean_log_features_volume'])
xgb_features_4 = (xgb_features_4 + ['max_log_features_volume'])
xgb_features_4 = (xgb_features_4 + ['min_num_ids_with_log_feature'])
#xgb_features_4 = (xgb_features_4 + ['std_log_features_volume','std_num_ids_with_log_feature'])
#xgb_features_4 = (xgb_features_4 + ['std_num_ids_with_log_feature'])

#xgb_features_4 = (xgb_features_4 + ['vol_summed_to_loc_vol_ratio'])

#xgb_features_4 = (xgb_features_4 + ['max_volume_log_feature'])
#xgb_features_4 = (xgb_features_4 + ['max_vol_log_feature_by_most_common_location'])
#xgb_features_4 = (xgb_features_4 + ['min_vol_log_feature_by_most_common_location'])

#xgb_features_4 = (xgb_features_4 + ['median_feature_diff_to_location'])

#xgb_features_4 = (xgb_features_4 + ['log_82_203_vol_ratio'])
#xgb_features_4 = (xgb_features_4 + ['log_203_82_vol_diff'])
#xgb_features_4 = (xgb_features_4 + ['log_201_80_vol_diff'])
#xgb_features_4 = (xgb_features_4 + ['log_312_232_vol_diff'])
#xgb_features_4 = (xgb_features_4 + ['log_170_54_vol_diff'])
#xgb_features_4 = (xgb_features_4 + ['log_171_55_vol_diff'])
#xgb_features_4 = (xgb_features_4 + ['log_193_71_vol_diff'])



xgb_features_4 = (xgb_features_4 + ['std_log_feature'])
xgb_features_4 = (xgb_features_4 + ['median_log_feature'])
xgb_features_4 = (xgb_features_4 + ['min_log_feature'])
#xgb_features_4 = (xgb_features_4 + ['volume_of_max_log_feature'])
xgb_features_4 = (xgb_features_4 + ['volume_of_min_log_feature'])
#xgb_features_4 = (xgb_features_4 + ['max_log_feature'])
#xgb_features_4 = (xgb_features_4 + ['log_features_range'])



#xgb_features_4 = (xgb_features_4 + ['first_log_feature_hash_next'])

#xgb_features_4 = (xgb_features_4 + ['max_is_next_resource_type_repeat'])

#xgb_features_4 = (xgb_features_4 + resource_type_position_cols)
#xgb_features_4 = (xgb_features_4 + event_type_position_cols)

xgb_features_4 = (xgb_features_4 + ['position_event_type_1'])
xgb_features_4 = (xgb_features_4 + ['position_resource_type_1'])
#xgb_features_4 = (xgb_features_4 + ['position_log_feature_1'])


xgb_features_4 = (xgb_features_4 + ['switches_resource_type'])

#xgb_features_4 = (xgb_features_4 + ['order_of_log_feature'])


#xgb_features_4 = (xgb_features_4 + ['cumulative_feature_vol'])
#xgb_features_4 = (xgb_features_4 + ['count_of_log_feature_seen_no_dups'])
#xgb_features_4 = (xgb_features_4 + ['count_of_event_type_seen'])

#xgb_features_4 = (xgb_features_4 + ['order_by_location'])
xgb_features_4 = (xgb_features_4 + ['reverse_order_by_location'])
#xgb_features_4 = (xgb_features_4 + ['order_by_location_cum_vol'])

#xgb_features_4 = (xgb_features_4 + ['order_by_location_fraction'])
xgb_features_4 = (xgb_features_4 + ['reverse_order_by_location_fraction'])

xgb_features_4 = (xgb_features_4 + ['severity_type'])
#xgb_features_4 = (xgb_features_4 + ['combo_log_features_by_most_common_severity'])
#xgb_features_4 = (xgb_features_4 + ['combo_log_features_by_nunique_severity'])
#xgb_features_4 = (xgb_features_4 + ['combo_log_features_by_nunique_loc'])
xgb_features_4 = (xgb_features_4 + ['combo_log_features_by_nunique_loc_per_ids'])
#xgb_features_4 = (xgb_features_4 + ['combo_log_features_by_median_loc'])
#xgb_features_4 = (xgb_features_4 + ['combo_log_features_by_mean_volume'])

xgb_features_4 = (xgb_features_4 + vol_log_feature_cols)
xgb_features_4 = (xgb_features_4 + vol_binned_log_feature_cols) #controversial feature (good on some holdouts, not on others)
xgb_features_4 = (xgb_features_4 + vol_binned_offset_log_feature_cols) #controversial feature (good on some holdouts, not on others)
#xgb_features_4 = (xgb_features_4 + vol_num_ids_with_log_feature_cols) #controversial feature (good on some holdouts, not on others)
#xgb_features_4 = (xgb_features_4 + vol_binned_small_log_feature_cols) #controversial feature (good on some holdouts, not on others)
#xgb_features_4 = (xgb_features_4 + vol_binned_offset_small_log_feature_cols) #controversial feature (good on some holdouts, not on others)


#xgb_features_4 = (xgb_features_4 + log_feature_vol_combo_cols)
#xgb_features_4 = (xgb_features_4 + log_vol_feature_cols)


#xgb_features_4 = (xgb_features_4 + event_type_cols)
#xgb_features_4 = (xgb_features_4 + resource_type_cols)
#xgb_features_4 = (xgb_features_4 + log_feature_cols)



#xgb_features_4 = (xgb_features_4 + log_by_loc_cols)


#xgb_features_4 = (xgb_features_4 + vol_log_feature_by_loc_cols)

xgb_features_4 = (xgb_features_4 + ['median_log_feature_most_common_location'])
#xgb_features_4 = (xgb_features_4 + ['median_log_feature_most_common_event'])


#xgb_features_4 = (xgb_features_4 + ['combo_log_features_hash'])
xgb_features_4 = (xgb_features_4 + ['combo_resource_types_hash'])
#xgb_features_4 = (xgb_features_4 + ['combo_event_types_hash'])

#xgb_features_4 = (xgb_features_4 + ['num_ids_with_log_feature_combo'])
xgb_features_4 = (xgb_features_4 + ['num_ids_with_resource_type_combo'])
xgb_features_4 = (xgb_features_4 + ['num_ids_with_event_type_combo'])

#xgb_features_4 = (xgb_features_4 + ['median_event_type'])
#xgb_features_4 = (xgb_features_4 + ['max_event_type'])
#xgb_features_4 = (xgb_features_4 + ['min_event_type'])
#xgb_features_4 = (xgb_features_4 + ['range_event_type'])

#xgb_features_4 = (xgb_features_4 + ['median_resource_type'])
#xgb_features_4 = (xgb_features_4 + ['max_resource_type'])
#xgb_features_4 = (xgb_features_4 + ['min_resource_type'])

#xgb_features_4 = (xgb_features_4 + ['id'])

#xgb_features_4 = (xgb_features_4 + num_ids_with_log_feature_cols)

#xgb_features_4 = (xgb_features_4 + num_ids_with_resource_type_cols)
#xgb_features_4 = (xgb_features_4 + num_ids_with_event_type_cols)
#xgb_features_4 = (xgb_features_4 + num_ids_with_severity_type_cols)


#xgb_features_4.remove('log_feature_hash')

params_4 = {'learning_rate': 0.015,
              'subsample': 0.95,
              'reg_alpha': 0.5,
#              'lambda': 0.99,
              'gamma': 0.7,
              'seed': 6,
              'colsample_bytree': 0.4,
              'n_estimators': 100,
              'objective': 'multi:softprob',
              'eval_metric':'mlogloss',
#              'num_parallel_tree':1,
#              'max_delta_step': 0.5,
              'max_depth': 7,
#              'min_child_weight': 2,
              'num_class':3}
num_rounds = 10000
num_boost_rounds = 150

#result_xgb_df_4 = fit_xgb_model(train,test,params_4,xgb_features_4,
#                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
#                                               random_seed = 6)
result_xgb_df_4 = fit_xgb_model(train,test,params_4,xgb_features_4,
                              num_rounds = num_rounds, num_boost_rounds = 1200,
                              do_grid_search = False,
                              use_early_stopping = False, print_feature_imp = False,random_seed = 6)
#%%

##let's try a neural net, from BNP script
class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler



def getDummiesInplace(columnList, train, test = None):
    #Takes in a list of column names and one or two pandas dataframes
    #One-hot encodes all indicated columns inplace
    columns = []

    if test is not None:
        df = pd.concat([train,test], axis= 0)
    else:
        df = train

    for columnName in df.columns:
        index = df.columns.get_loc(columnName)
        if columnName in columnList:
            dummies = pd.get_dummies(df.ix[:,index], prefix = columnName, prefix_sep = ".")
            columns.append(dummies)
        else:
            columns.append(df.ix[:,index])
    df = pd.concat(columns, axis = 1)

    if test is not None:
        train = df[:train.shape[0]]
        test = df[train.shape[0]:]
        return train, test
    else:
        train = df
        return train

def pdFillNAN(df, strategy = "mean"):
    #Fills empty values with either the mean value of each feature, or an indicated number
    if strategy == "mean":
        return df.fillna(df.mean())
    elif type(strategy) == int:
        return df.fillna(strategy)
#%%
tic=timeit.default_timer()
train_nn = train_low.copy()
test_nn = test_low.copy()

labels = train_nn['fault_severity']
labels_2 = pd.get_dummies(train_nn['fault_severity']).astype(np.int32).values
train_nn_fault_severity = train_nn['fault_severity'].astype(np.int32).values
train_nn_id = train_nn['id']
test_nn_id = test_nn['id']

train_nn = train_nn[xgb_features]
test_nn = test_nn[xgb_features]

#find categorical variables
categorical_variables = []
for var in train_nn.columns:
    vector=pd.concat([train_nn[var],test_nn[var]], axis=0)
    typ=str(train_nn[var].dtype)
    if (typ=='object'):
        categorical_variables.append(var)

train_nn, test_nn = getDummiesInplace(categorical_variables, train_nn, test_nn)

#Remove sparse columns
cls = train_nn.sum(axis=0)
train_nn = train_nn.drop(train_nn.columns[cls<10], axis=1)
test_nn = test_nn.drop(test_nn.columns[cls<10], axis=1)

print ("Scaling...")
train_nn, scaler = preprocess_data(train_nn)
test_nn, scaler = preprocess_data(test_nn, scaler)

train_nn = np.asarray(train_nn, dtype=np.float64)
labels = np.asarray(labels, dtype=np.int32).reshape(-1,1)
#labels = labels.astype(np.int32).values
#%%
#net = NeuralNet(
#    layers=[
#        ('input', InputLayer),
#        ('dropout0', DropoutLayer),
#        ('hidden1', DenseLayer),
#        ('dropout1', DropoutLayer),
#        ('hidden2', DenseLayer),
#        ('output', DenseLayer),
#        ],
#
#    input_shape=(None, len(train_nn[1])),
#    dropout0_p=0.04,
#    hidden1_num_units=1000,
#    hidden1_W=Uniform(),
#    dropout1_p=0.05,
#    hidden2_num_units=1000,
#    #hidden2_W=Uniform(),
#
#    output_nonlinearity=sigmoid,
#    output_num_units=1,
#    update=nesterov_momentum,
#    update_learning_rate=theano.shared(np.float32(0.0001)),
#    update_momentum=theano.shared(np.float32(0.9)),
#    # Decay the learning rate
#    on_epoch_finished=[AdjustVariable('update_learning_rate', start=0.0001, stop=0.00001),
#                       AdjustVariable('update_momentum', start=0.9, stop=0.99),
#                       ],
#    regression=True,
#    y_tensor_type = T.imatrix,
#    objective_loss_function = binary_crossentropy,
#    #batch_iterator_train = BatchIterator(batch_size = 256),
#    max_epochs=50,
#    eval_size=0.1,
#    #train_split =0.0,
#    verbose=2,
#    )
#
#
#seednumber=1235
#np.random.seed(seednumber)
#net.fit(train_nn, labels)
#
#
#preds = net.predict(test_nn)[:,0]
#
##preds_nn = net.predict(test_nn)[:,0]
#preds_nn = net.predict_proba(test_nn)[:,0]
#
#columns = ['predict']
#result_nn_df = pd.DataFrame(index=test_nn_id, columns=columns,data=preds_nn)
#
#result_nn_df['predict_low'] = 1 - result_nn_df['predict']
#result_nn_df['predict_high'] = result_nn_df['predict']
#result_nn_df.drop(['predict'],axis=1,inplace=True)
#result_nn_df.replace([0],[0.000001],inplace=True)
#
#result_nn_df = norm_rows(result_nn_df)
#
#
#if(is_sub_run):
#    print('is a sub run')
#else:
#    result_nn_df.reset_index('id',inplace=True)
#    result_nn_df = pd.merge(result_nn_df,test_low[['id','fault_severity']],left_on = ['id'],
#                           right_on = ['id'],how='left')
#    result_nn_df['log_loss'] = result_nn_df.apply(lambda row: get_log_loss_row_two_classes(row),axis=1)
#    print('log_loss',round(result_nn_df['log_loss'].mean(),5))
#
#toc=timeit.default_timer()
#print('nn Time',toc - tic)
#%%
#
#nn_features = []
#nn_features = (nn_features + log_feature_vol_combo_cols)
#
##nn_features = (nn_features + severity_type_cols)
##nn_features = (nn_features + event_type_cols)
#nn_features = (nn_features + resource_type_cols)
#nn_features = (nn_features + log_feature_cols)
#nn_features = (nn_features + log_vol_feature_cols)
#nn_features = (nn_features + num_ids_with_log_feature_cols)
#nn_features = (nn_features + num_ids_with_resource_type_cols)
#nn_features = (nn_features + num_ids_with_event_type_cols)
#nn_features = (nn_features + num_ids_with_severity_type_cols)
#nn_features = (nn_features + ['order_by_location'])
#
#num_classes = 3
#num_features = len(nn_features)
#
#scaler = StandardScaler()
#combined_3 = combined.copy()
#if (is_sub_run):
#    train_nn = combined_3.loc[combined['fault_severity'] != 'dummy' ]
#    test_nn = combined_3.loc[combined['fault_severity'] == 'dummy' ]
#else:
#    train_nn = ((combined_3['fault_severity'] != 'dummy') &
#                         ((combined_3['id'] > 10000) | (combined_3['id'] <= 5000))) == 1
#    test_nn = ((combined_3['fault_severity'] != 'dummy') &
#                         (combined_3['id'] > 5000) & (combined_3['id'] <= 10000)) == 1
## Convert to np.array to make lasagne happy
##combined_data = combined_3[nn_features]
#train_data_full = np.array(combined_3[nn_features])
#train_data_full = train_data_full.astype(np.float32)
#scaler = StandardScaler()
#train_data = scaler.fit_transform(train_data_full)
#
#train_data = train_data_full[(np.array(train_nn.values)),:]
#test_data = train_data_full[(np.array(test_nn.values)),:]
#
#train_fault_severity = combined_3[train_nn]['fault_severity'].astype(np.int32).values
#tic=timeit.default_timer()
#epochs = 50
#val_auc = np.zeros(epochs)
#%%
#
## Comment out second layer for run time.
#layers = [('input', InputLayer),
#           ('dense0', DenseLayer),
#           ('dropout0', DropoutLayer),
#           ('dense1', DenseLayer),
#           ('dropout1', DropoutLayer),
##           ('dense2', DenseLayer),
##           ('dropout2', DropoutLayer),
##           ('dense3', DenseLayer),
##           ('dropout3', DropoutLayer),
##           ('dense4', DenseLayer),
##           ('dropout4', DropoutLayer),
##           ('dense5', DenseLayer),
##           ('dropout5', DropoutLayer),
#           ('output', DenseLayer)
#           ]
#
#net1 = NeuralNet(layers=layers,
#                 input_shape=(None, len(train_nn[1])),
#                 dense0_num_units=50, # 512, - reduce num units to make faster
#                 dropout0_p=0.5,
#                 dense1_num_units=50,
#                 dropout1_p=0.5,
##                 dense2_num_units=256,
##                 dropout2_p=0.1,
##                 dense3_num_units=256,
##                 dropout3_p=0.1,
##                 dense4_num_units=256,
##                 dropout4_p=0.1,
##                 dense5_num_units=256,
##                 dropout5_p=0.1,
#
#                 output_num_units=3,
#                 output_nonlinearity=softmax,
#                 update=adagrad,
#                 update_learning_rate=0.005,
##                 eval_size=0.0,
#
#                 # objective_loss_function = binary_accuracy,
#                 verbose=1,
##                 allow_input_downcast=True,
#                 max_epochs=1)
#epochs = 10
#for i in range(epochs):
#    net1.fit(train_nn, train_nn_fault_severity)
#    pred = net1.predict_proba(test_nn)
##    val_auc[i] = roc_auc_score(y[split:],pred)
#
#columns = ['predict_0','predict_1','predict_2']
#result_nn_df = pd.DataFrame(index=test.id, columns=columns,data=pred)
#result_nn_df.replace([0],[0.000001],inplace=True)
#
#result_nn_df = norm_rows(result_nn_df)
#
#
#if(is_sub_run):
#    print('is a sub run')
#else:
#    result_nn_df.reset_index('id',inplace=True)
#    result_nn_df = pd.merge(result_nn_df,test[['id','fault_severity']],left_on = ['id'],
#                           right_on = ['id'],how='left')
#    result_nn_df['log_loss'] = result_nn_df.apply(lambda row: get_log_loss_row(row),axis=1)
#    print('log_loss',round(result_nn_df['log_loss'].mean(),5))
#
#toc=timeit.default_timer()
#print('nn Time',toc - tic)
#%%


#%%
#
##lets try stacking
#train_1 = combined.loc[(combined['fault_severity'] != 'dummy') &
#                     (combined['id'] > 4000)]
#test_1 = combined.loc[(combined['fault_severity'] != 'dummy') &
#                     (combined['id'] <= 4000)]
#train_2 = combined.loc[(combined['fault_severity'] != 'dummy') &
#                     ((combined['id'] < 4000) | (combined['id'] > 8000))]
#test_2 = combined.loc[(combined['fault_severity'] != 'dummy') &
#                     (combined['id'] > 4000) & (combined['id'] <= 8000)]
#train_3 = combined.loc[(combined['fault_severity'] != 'dummy') &
#                     ((combined['id'] < 8000) | (combined['id'] > 12000))]
#test_3 = combined.loc[(combined['fault_severity'] != 'dummy') &
#                     (combined['id'] > 8000) & (combined['id'] <= 12000)]
#train_4 = combined.loc[(combined['fault_severity'] != 'dummy') &
#                     ((combined['id'] < 12000) | (combined['id'] > 16000))]
#test_4 = combined.loc[(combined['fault_severity'] != 'dummy') &
#                     (combined['id'] > 12000) & (combined['id'] <= 16000)]
#train_5 = combined.loc[(combined['fault_severity'] != 'dummy') &
#                     ((combined['id'] < 16000) | (combined['id'] > 20000))]
#test_5 = combined.loc[(combined['fault_severity'] != 'dummy') &
#                     (combined['id'] > 16000) & (combined['id'] <= 20000)]
#
#train_dummies = combined.loc[(combined['fault_severity'] != 'dummy')]
#test_dummies = combined.loc[(combined['fault_severity'] == 'dummy')]
#def get_predictions_to_stack(params,xgb_features,num_boost_rounds,num_boost_rounds_full,random_seed = 6):
#    tic = timeit.default_timer()
#    result_xgb_df_1a = fit_xgb_model(train_1,test_1,params,xgb_features,
#                              num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                              do_grid_search = False,
#                              use_early_stopping = False, print_feature_imp = False,random_seed = random_seed)
#    result_xgb_df_2a = fit_xgb_model(train_2,test_2,params,xgb_features,
#                              num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                              do_grid_search = False,
#                              use_early_stopping = False, print_feature_imp = False,random_seed = random_seed)
#    result_xgb_df_3a = fit_xgb_model(train_3,test_3,params,xgb_features,
#                              num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                              do_grid_search = False,
#                              use_early_stopping = False, print_feature_imp = False,random_seed = random_seed)
#    result_xgb_df_4a = fit_xgb_model(train_4,test_4,params,xgb_features,
#                              num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                              do_grid_search = False,
#                              use_early_stopping = False, print_feature_imp = False,random_seed = random_seed)
#    result_xgb_df_5a = fit_xgb_model(train_5,test_5,params,xgb_features,
#                              num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
#                              do_grid_search = False,
#                              use_early_stopping = False, print_feature_imp = False,random_seed = random_seed)
#    result_xgb_df_dummies_a = fit_xgb_model(train_dummies,test_dummies,params,xgb_features,
#                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds_full,
#                                               do_grid_search = False, use_early_stopping = False, print_feature_imp = False,
#                                               random_seed = random_seed, calculate_log_loss=False)
#    if (is_sub_run):
#        result_xgb_df_1a.reset_index('id',inplace=True)
#        result_xgb_df_2a.reset_index('id',inplace=True)
#        result_xgb_df_3a.reset_index('id',inplace=True)
#        result_xgb_df_4a.reset_index('id',inplace=True)
#        result_xgb_df_5a.reset_index('id',inplace=True)
#    result_xgb_df_dummies_a.reset_index('id',inplace=True)
#    result_xgb_to_stack_a = pd.concat([result_xgb_df_1a,result_xgb_df_2a,result_xgb_df_3a,result_xgb_df_4a,result_xgb_df_5a,
#                                 result_xgb_df_dummies_a])
#    toc=timeit.default_timer()
#    print('Stacking Time',toc - tic)
#    return result_xgb_to_stack_a
#%%
#result_xgb_df_to_stack = get_predictions_to_stack(params,xgb_features,1650,2300,6)
#result_xgb_df_to_stack_3 = get_predictions_to_stack(params_3,xgb_features_3,500,800,7)
#result_xgb_df_to_stack_4 = get_predictions_to_stack(params_4,xgb_features_4,3200,4500,8)
#result_xgb_df_to_stack_5 = get_predictions_to_stack(params_5,xgb_features_5,2400,3400,9)
#result_xgb_df_to_stack_6 = get_predictions_to_stack(params_6,xgb_features_6,4300,6000,10)
#result_xgb_df_to_stack_7 = get_predictions_to_stack(params_7,xgb_features_7,1200,1700,11)
#%%
#submission = submission[['id','predict_0','predict_1','predict_2']]
#result_xgb_df_to_stack_3.to_csv('result_xgb_df_to_stack_3.csv', index=False)
#result_xgb_df_to_stack_4.to_csv('result_xgb_df_to_stack_4.csv', index=False)
#result_xgb_df_to_stack_5.to_csv('result_xgb_df_to_stack_5.csv', index=False)
#result_xgb_df_to_stack_6.to_csv('result_xgb_df_to_stack_6.csv', index=False)
#result_xgb_df_to_stack_7.to_csv('result_xgb_df_to_stack_7.csv', index=False)
#%%
#result_xgb_df_to_stack_3.rename(columns={'predict_0':'predict_0_3','predict_1':'predict_1_3','predict_2':'predict_2_3'},inplace=True)
#result_xgb_df_to_stack_4.rename(columns={'predict_0':'predict_0_4','predict_1':'predict_1_4','predict_2':'predict_2_4'},inplace=True)
#result_xgb_df_to_stack_5.rename(columns={'predict_0':'predict_0_5','predict_1':'predict_1_5','predict_2':'predict_2_5'},inplace=True)
#result_xgb_df_to_stack_6.rename(columns={'predict_0':'predict_0_6','predict_1':'predict_1_6','predict_2':'predict_2_6'},inplace=True)
#result_xgb_df_to_stack_7.rename(columns={'predict_0':'predict_0_7','predict_1':'predict_1_7','predict_2':'predict_2_7'},inplace=True)
#%%
#combined_3 = pd.merge(result_xgb_df_to_stack,result_xgb_df_to_stack_3[['id','predict_0_3','predict_1_3','predict_2_3']],left_on = ['id'],
#                               right_on = ['id'],how='left')
#combined_3 = pd.merge(combined_3,result_xgb_df_to_stack_4[['id','predict_0_4','predict_1_4','predict_2_4']],left_on = ['id'],
#                               right_on = ['id'],how='left')
#combined_3 = pd.merge(combined_3,result_xgb_df_to_stack_5[['id','predict_0_5','predict_1_5','predict_2_5']],left_on = ['id'],
#                               right_on = ['id'],how='left')
#combined_3 = pd.merge(combined_3,result_xgb_df_to_stack_6[['id','predict_0_6','predict_1_6','predict_2_6']],left_on = ['id'],
#                               right_on = ['id'],how='left')
#combined_3 = pd.merge(combined_3,result_xgb_df_to_stack_7[['id','predict_0_7','predict_1_7','predict_2_7']],left_on = ['id'],
#                               right_on = ['id'],how='left')
#combined_3.fillna('dummy',inplace=True)
#%%
#combined_2 = pd.merge(combined,result_xgb_to_stack_a[['id','predict_0','predict_1','predict_2']],left_on = ['id'],
#                               right_on = ['id'],how='left')
#combined_2 = pd.merge(combined_2,result_xgb_to_stack_b[['id','predict_0_b','predict_1_b','predict_2_b']],left_on = ['id'],
#                               right_on = ['id'],how='left')
#
#bins = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
#combined_2['predict_0_binned'] = np.digitize(combined_2['predict_0'], bins, right=True)
#combined_2['predict_1_binned'] = np.digitize(combined_2['predict_1'], bins, right=True)
#combined_2['predict_2_binned'] = np.digitize(combined_2['predict_2'], bins, right=True)
#combined_2.sort('order_of_log_feature',inplace=True)
#
#combined_2['pred_0_up1'] = combined_2['predict_0_binned'].shift(-1)
#combined_2['pred_1_up1'] = combined_2['predict_1_binned'].shift(-1)
#combined_2['pred_2_up1'] = combined_2['predict_2_binned'].shift(-1)
#combined_2['pred_0_up1'].fillna(-1,inplace=True)
#combined_2['pred_1_up1'].fillna(-1,inplace=True)
#combined_2['pred_2_up1'].fillna(-1,inplace=True)
#
#mean_predict0_location_dict = combined_2.groupby(['location'])['predict_0'].agg(lambda x: x.mean()).to_dict()
#mean_predict1_location_dict = combined_2.groupby(['location'])['predict_1'].agg(lambda x: x.mean()).to_dict()
#mean_predict2_location_dict = combined_2.groupby(['location'])['predict_2'].agg(lambda x: x.mean()).to_dict()
#
#combined_2['location_by_mean_pred0'] = combined_2['location'].map(mean_predict0_location_dict)
#combined_2['location_by_mean_pred1'] = combined_2['location'].map(mean_predict1_location_dict)
#combined_2['location_by_mean_pred2'] = combined_2['location'].map(mean_predict2_location_dict)
#is_sub_run = True
#if (is_sub_run):
#    train_stacked = combined_3.loc[combined_3['fault_severity'] != 'dummy' ]
#    test_stacked = combined_3.loc[combined_3['fault_severity'] == 'dummy' ]
#else:
#    train_stacked = combined_3.loc[(combined_3['fault_severity'] != 'dummy') &
#                         (combined_3['id'] > 5000)]
#    test_stacked = combined_3.loc[(combined_3['fault_severity'] != 'dummy') &
#                         (combined_3['id'] <= 5000)]
#%%
##xgb_features_stacked = xgb_features
##xgb_features_stacked = xgb_features_stacked + ['location_by_mean_pred0','location_by_mean_pred1','location_by_mean_pred2']
##xgb_features_stacked = xgb_features_stacked + ['predict_0_binned','predict_1_binned','predict_2_binned']
##xgb_features_stacked = xgb_features_stacked + ['pred_0_up1','pred_1_up1','pred_2_up1']
##xgb_features_stacked = ['pred_0_up1','pred_1_up1','pred_2_up1']
##xgb_features_stacked = ['order_by_location']
##xgb_features_stacked = xgb_features_stacked + ['predict_0','predict_1','predict_2']
##xgb_features_stacked = ['predict_0','predict_1','predict_2']
##xgb_features_stacked = (xgb_features_stacked + ['location_hash']) #by order of log feature
##xgb_features_stacked = (xgb_features_stacked + ['order_by_location'])
##xgb_features_stacked = (xgb_features_stacked + ['reverse_order_by_location'])
##
##xgb_features_stacked = (xgb_features_stacked + ['order_by_location_cum_vol'])
#
#
#xgb_features_stacked = ['predict_0','predict_1','predict_2']
#xgb_features_stacked = (xgb_features_stacked + ['predict_0_3','predict_1_3','predict_2_3'])
#xgb_features_stacked = (xgb_features_stacked + ['predict_0_4','predict_1_4','predict_2_4'])
#xgb_features_stacked = (xgb_features_stacked + ['predict_0_5','predict_1_5','predict_2_5'])
#xgb_features_stacked = (xgb_features_stacked + ['predict_0_6','predict_1_6','predict_2_6'])
#xgb_features_stacked = (xgb_features_stacked + ['predict_0_7','predict_1_7','predict_2_7'])
#params_stacked = {'learning_rate': 0.01,
#              'subsample': 0.9,
##              'reg_alpha': 0.5,
#              'gamma': 1,
#              'seed': 6,
#              'colsample_bytree': 0.4,
#              'n_estimators': 100,
#              'objective': 'multi:softprob',
#              'eval_metric':'mlogloss',
##              'max_delta_step': 0.2,
#              'max_depth': 5,
#              'min_child_weight': 3,
#              'num_class':3}
#num_rounds = 10000
#num_boost_rounds = 150
#
##result_xgb_df_stacked = fit_xgb_model(train_stacked,test_stacked,params_stacked,xgb_features_stacked,
##                                               num_rounds = num_rounds, num_boost_rounds = num_boost_rounds,
##                                               do_grid_search = False, use_early_stopping = True, print_feature_imp = False,
##                                               random_seed = 5)
#result_xgb_df_stacked = fit_xgb_model(train_stacked,test_stacked,params_stacked,xgb_features_stacked,
#                                               num_rounds = num_rounds, num_boost_rounds = 1000,
#                                               do_grid_search = False, use_early_stopping = False, print_feature_imp = False,
#                                               random_seed = 5)
#%%
#tic = timeit.default_timer()
#train_data = train[xgb_features].values
#train_fault_severity = train['fault_severity'].astype(int).values
#test_data = test[xgb_features].values
#
##from scipy import sparse
##test_data = sparse.csr_matrix(test_data)
##train_data = sparse.csr_matrix(train_data)
#
#clf = RandomForestClassifier(n_estimators=1000,random_state = 2, n_jobs = -1,
#                             oob_score = True,
#                             max_features = 50,
#                             min_samples_leaf = 3,
##                             class_weight = 'balanced',
##                             max_leaf_nodes = 100,
##                             min_samples_split = 5,
##                             max_depth = 50,
##                             criterion = 'entropy',
##                             min_samples_split = 1
#                             )
#clf = clf.fit(train_data, train_fault_severity)
#print('oob_score',clf.oob_score_)
#
#
#y_pred = clf.predict_proba(test_data)
#columns = ['predict_0','predict_1','predict_2']
#result_rf_df = pd.DataFrame(index=test.id, columns=columns,data=y_pred)
#result_rf_df.replace([0],[0.000001],inplace=True)
#
#result_rf_df = norm_rows(result_rf_df)
#
#if(is_sub_run):
#    print('is a sub run')
#else:
#    result_rf_df.reset_index('id',inplace=True)
#    result_rf_df = pd.merge(result_rf_df,test[['id','fault_severity']],left_on = ['id'],
#                           right_on = ['id'],how='left')
#    result_rf_df['log_loss'] = result_rf_df.apply(lambda row: get_log_loss_row(row),axis=1)
#    print('log_loss',round(result_rf_df['log_loss'].mean(),5))
#
#toc=timeit.default_timer()
#print('rf Time',toc - tic)
#%%
tic = timeit.default_timer()
train_data = train[xgb_features_6].values
train_fault_severity = train['fault_severity'].astype(int).values
test_data = test[xgb_features_6].values

#from scipy import sparse
#test_data = sparse.csr_matrix(test_data)
#train_data = sparse.csr_matrix(train_data)

extc = ExtraTreesClassifier(
                            n_estimators=2000,
#                            n_estimators=200,
#                            max_features= 1000,
                            max_features = None,
                            criterion= 'entropy',
#                            min_samples_split= 5,
                            min_samples_split= 1,
                            max_depth= None,
#                            min_weight_fraction_leaf=0.5,
#                            max_depth= 500,
                            min_samples_leaf= 4,
                            random_state = 1,
                            n_jobs = 2
                            )
extc = extc.fit(train_data, train_fault_severity)
#print('oob_score',clf.oob_score_)


y_pred = extc.predict_proba(test_data)
columns = ['predict_0','predict_1','predict_2']
result_extc_df = pd.DataFrame(index=test.id, columns=columns,data=y_pred)
result_extc_df.replace([0],[0.00001],inplace=True)

result_extc_df = norm_rows(result_extc_df)

if(is_sub_run):
    print('is a sub run')
else:
    result_extc_df.reset_index('id',inplace=True)
    result_extc_df = pd.merge(result_extc_df,test[['id','fault_severity']],left_on = ['id'],
                           right_on = ['id'],how='left')
    result_extc_df['log_loss'] = result_extc_df.apply(lambda row: get_log_loss_row(row),axis=1)
    print('log_loss',round(result_extc_df['log_loss'].mean(),5))

toc=timeit.default_timer()
print('extc Time',toc - tic)
#%%
tic = timeit.default_timer()
extc_features_2 = [
                'count_log_features','count_event_types',
                'count_resource_types',
                'count_of_volumes',
                ]
extc_features_2 = (extc_features_2 + ['location_by_unique_features'])
extc_features_2 = (extc_features_2 + ['location_by_unique_resource_types'])
#extc_features_2 = (extc_features_2 + ['location_by_unique_severity_types'])
extc_features_2 = (extc_features_2 + ['location_by_unique_event_types'])
extc_features_2 = (extc_features_2 + ['location_by_most_common_event'])
extc_features_2 = (extc_features_2 + ['location_by_most_common_resource'])
#extc_features_2 = (extc_features_2 + ['location_by_most_common_severity'])
extc_features_2 = (extc_features_2 + ['location_by_most_common_feature'])
#extc_features_2 = (extc_features_2 + ['location_by_most_common_combo_feature'])

#extc_features_2 = (extc_features_2 + ['location_by_median_event'])
#extc_features_2 = (extc_features_2 + ['location_by_min_event'])
#extc_features_2 = (extc_features_2 + ['location_by_max_event'])
extc_features_2 = (extc_features_2 + has_event_type_names)
#extc_features_2 = (extc_features_2 + has_resource_type_names)

#extc_features_2 = (extc_features_2 + ['location_by_median_ids_with_feature'])
extc_features_2 = (extc_features_2 + ['location_by_median_feature_number'])
#extc_features_2 = (extc_features_2 + ['location_by_max_feature_number'])

#extc_features_2 = (extc_features_2 + ['location_by_min_feature_number'])

#extc_features_2 = (extc_features_2 + ['location_by_mean_count_log_features'])
extc_features_2 = (extc_features_2 + ['location_by_median_count_log_features'])

#extc_features_2 = (extc_features_2 + ['location_by_second_most_common_feature'])

extc_features_2 = (extc_features_2 + ['location_by_mean_volume'])
extc_features_2 = (extc_features_2 + ['location_by_max_volume'])

#extc_features_2 = (extc_features_2 + ['location_by_std_event_type'])
extc_features_2 = (extc_features_2 + ['location_by_std_resource'])
extc_features_2 = (extc_features_2 + ['location_by_std_log_feature'])

#extc_features_2 = (extc_features_2 + ['location_by_frac_dummy'])

extc_features_2 = (extc_features_2 + ['location_max']) #by number of ids
#extc_features_2 = (extc_features_2 + ['location_hash']) #by order of log feature
extc_features_2 = (extc_features_2 + ['location_number'])

extc_features_2 = (extc_features_2 + loc_by_logs_cols)
#extc_features_2 = (extc_features_2 + loc_by_logs_summed_cols)

#extc_features_2 = (extc_features_2 + loc_by_binned_logs_cols)
#extc_features_2 = (extc_features_2 + loc_by_binned_offset_logs_cols)
#extc_features_2 = (extc_features_2 + loc_by_binned_logs_summed_cols)

#extc_features_2 = (extc_features_2 + loc_by_event_types_cols)
#extc_features_2 = (extc_features_2 + loc_by_event_types_summed_cols)

extc_features_2 = (extc_features_2 + ['sum_log_features_volume'])
extc_features_2 = (extc_features_2 + ['mean_log_features_volume'])
extc_features_2 = (extc_features_2 + ['max_log_features_volume'])
extc_features_2 = (extc_features_2 + ['min_num_ids_with_log_feature'])
#extc_features_2 = (extc_features_2 + ['std_log_features_volume','std_num_ids_with_log_feature'])
#extc_features_2 = (extc_features_2 + ['std_num_ids_with_log_feature'])

#extc_features_2 = (extc_features_2 + ['vol_summed_to_loc_vol_ratio'])

#extc_features_2 = (extc_features_2 + ['max_volume_log_feature'])
#extc_features_2 = (extc_features_2 + ['max_vol_log_feature_by_most_common_location'])
#extc_features_2 = (extc_features_2 + ['min_vol_log_feature_by_most_common_location'])

#extc_features_2 = (extc_features_2 + ['median_feature_diff_to_location'])

#extc_features_2 = (extc_features_2 + ['log_82_203_vol_ratio'])
#extc_features_2 = (extc_features_2 + ['log_203_82_vol_diff'])
#extc_features_2 = (extc_features_2 + ['log_201_80_vol_diff'])
#extc_features_2 = (extc_features_2 + ['log_312_232_vol_diff'])
#extc_features_2 = (extc_features_2 + ['log_170_54_vol_diff'])
#extc_features_2 = (extc_features_2 + ['log_171_55_vol_diff'])
#extc_features_2 = (extc_features_2 + ['log_193_71_vol_diff'])



extc_features_2 = (extc_features_2 + ['std_log_feature'])
extc_features_2 = (extc_features_2 + ['median_log_feature'])
extc_features_2 = (extc_features_2 + ['min_log_feature'])
#extc_features_2 = (extc_features_2 + ['volume_of_max_log_feature'])
extc_features_2 = (extc_features_2 + ['volume_of_min_log_feature'])
#extc_features_2 = (extc_features_2 + ['max_log_feature'])
#extc_features_2 = (extc_features_2 + ['log_features_range'])



#extc_features_2 = (extc_features_2 + ['first_log_feature_hash_next'])

#extc_features_2 = (extc_features_2 + ['max_is_next_resource_type_repeat'])

#extc_features_2 = (extc_features_2 + resource_type_position_cols)
#extc_features_2 = (extc_features_2 + event_type_position_cols)

extc_features_2 = (extc_features_2 + ['position_event_type_1'])
extc_features_2 = (extc_features_2 + ['position_resource_type_1'])
#extc_features_2 = (extc_features_2 + ['position_log_feature_1'])


extc_features_2 = (extc_features_2 + ['switches_resource_type'])

#extc_features_2 = (extc_features_2 + ['order_of_log_feature'])


#extc_features_2 = (extc_features_2 + ['cumulative_feature_vol'])
#extc_features_2 = (extc_features_2 + ['count_of_log_feature_seen_no_dups'])
#extc_features_2 = (extc_features_2 + ['count_of_event_type_seen'])

#extc_features_2 = (extc_features_2 + ['order_by_location'])
extc_features_2 = (extc_features_2 + ['reverse_order_by_location'])
#extc_features_2 = (extc_features_2 + ['order_by_location_cum_vol'])

#extc_features_2 = (extc_features_2 + ['order_by_location_fraction'])
extc_features_2 = (extc_features_2 + ['reverse_order_by_location_fraction'])

extc_features_2 = (extc_features_2 + ['severity_type'])
#extc_features_2 = (extc_features_2 + ['combo_log_features_by_most_common_severity'])
#extc_features_2 = (extc_features_2 + ['combo_log_features_by_nunique_severity'])
#extc_features_2 = (extc_features_2 + ['combo_log_features_by_nunique_loc'])
extc_features_2 = (extc_features_2 + ['combo_log_features_by_nunique_loc_per_ids'])
#extc_features_2 = (extc_features_2 + ['combo_log_features_by_median_loc'])
#extc_features_2 = (extc_features_2 + ['combo_log_features_by_mean_volume'])

extc_features_2 = (extc_features_2 + vol_log_feature_cols)
extc_features_2 = (extc_features_2 + vol_binned_log_feature_cols) #controversial feature (good on some holdouts, not on others)
extc_features_2 = (extc_features_2 + vol_binned_offset_log_feature_cols) #controversial feature (good on some holdouts, not on others)
#extc_features_2 = (extc_features_2 + vol_num_ids_with_log_feature_cols) #controversial feature (good on some holdouts, not on others)
#extc_features_2 = (extc_features_2 + vol_binned_small_log_feature_cols) #controversial feature (good on some holdouts, not on others)
#extc_features_2 = (extc_features_2 + vol_binned_offset_small_log_feature_cols) #controversial feature (good on some holdouts, not on others)


#extc_features_2 = (extc_features_2 + log_feature_vol_combo_cols)
#extc_features_2 = (extc_features_2 + log_vol_feature_cols)


#extc_features_2 = (extc_features_2 + event_type_cols)
#extc_features_2 = (extc_features_2 + resource_type_cols)
#extc_features_2 = (extc_features_2 + log_feature_cols)

#extc_features_2 = (extc_features_2 + log_by_loc_cols)


#extc_features_2 = (extc_features_2 + vol_log_feature_by_loc_cols)

extc_features_2 = (extc_features_2 + ['median_log_feature_most_common_location'])
#extc_features_2 = (extc_features_2 + ['median_log_feature_most_common_event'])


#extc_features_2 = (extc_features_2 + ['combo_log_features_hash'])
#extc_features_2 = (extc_features_2 + ['combo_resource_types_hash'])
#extc_features_2 = (extc_features_2 + ['combo_event_types_hash'])

#extc_features_2 = (extc_features_2 + ['num_ids_with_log_feature_combo'])
#extc_features_2 = (extc_features_2 + ['num_ids_with_resource_type_combo'])
#extc_features_2 = (extc_features_2 + ['num_ids_with_event_type_combo'])

extc_features_2 = (extc_features_2 + ['median_event_type'])
extc_features_2 = (extc_features_2 + ['max_event_type'])
extc_features_2 = (extc_features_2 + ['min_event_type'])
#extc_features_2 = (extc_features_2 + ['range_event_type'])

#extc_features_2 = (extc_features_2 + ['median_resource_type'])
#extc_features_2 = (extc_features_2 + ['max_resource_type'])
#extc_features_2 = (extc_features_2 + ['min_resource_type'])

#extc_features_2 = (extc_features_2 + ['id'])

#extc_features_2 = (extc_features_2 + num_ids_with_log_feature_cols)

#extc_features_2 = (extc_features_2 + num_ids_with_resource_type_cols)
#extc_features_2 = (extc_features_2 + num_ids_with_event_type_cols)
#extc_features_2 = (extc_features_2 + num_ids_with_severity_type_cols)

train_data = train[extc_features_2].values
train_fault_severity = train['fault_severity'].astype(int).values
test_data = test[extc_features_2].values

extc_2 = ExtraTreesClassifier(n_estimators=2000,
#                            max_features= 1000,
                            max_features = None,
                            criterion= 'entropy',
#                            min_samples_split= 5,
                            min_samples_split= 1,
                            max_depth= None,
#                            max_depth= 500,
                            min_samples_leaf= 5,
                            random_state = 1,
                            n_jobs = 2
                            )
extc_2 = extc_2.fit(train_data, train_fault_severity)
#print('oob_score',clf.oob_score_)


y_pred = extc_2.predict_proba(test_data)
columns = ['predict_0','predict_1','predict_2']
result_extc_df_2 = pd.DataFrame(index=test.id, columns=columns,data=y_pred)
result_extc_df_2.replace([0],[0.00001],inplace=True)

result_extc_df_2 = norm_rows(result_extc_df_2)

if(is_sub_run):
    print('is a sub run')
else:
    result_extc_df_2.reset_index('id',inplace=True)
    result_extc_df_2 = pd.merge(result_extc_df_2,test[['id','fault_severity']],left_on = ['id'],
                           right_on = ['id'],how='left')
    result_extc_df_2['log_loss'] = result_extc_df_2.apply(lambda row: get_log_loss_row(row),axis=1)
    print('log_loss',round(result_extc_df_2['log_loss'].mean(),5))

toc=timeit.default_timer()
print('extc Time',toc - tic)
#%%
#result_xgb_ens = 0.9 * result_xgb_df + 0.0 * result_xgb_df_2 + 0.5 * result_xgb_df_3
#result_xgb_ens = (1.0 * result_xgb_df + 1.0 * result_xgb_df_4 + 1.0 * result_xgb_df_5 + 1.0 * result_xgb_df_6 +
#                 1.0 * result_xgb_df_7 + 1.0 * result_xgb_df_8)
#result_xgb_df.reset_index('id',inplace=True)
#result_xgb_df_4.reset_index('id',inplace=True)
#result_xgb_df_5.reset_index('id',inplace=True)
#result_xgb_df_6.reset_index('id',inplace=True)
#result_xgb_df_7.reset_index('id',inplace=True)
#result_xgb_df_8.reset_index('id',inplace=True)
#result_xgb_df.rename(columns={'predict_0':'predict_0_0','predict_1':'predict_1_0','predict_2':'predict_2_0'},inplace=True)
#result_xgb_df_4.rename(columns={'predict_0':'predict_0_4','predict_1':'predict_1_4','predict_2':'predict_2_4'},inplace=True)
#result_xgb_df_5.rename(columns={'predict_0':'predict_0_5','predict_1':'predict_1_5','predict_2':'predict_2_5'},inplace=True)
#result_xgb_df_6.rename(columns={'predict_0':'predict_0_6','predict_1':'predict_1_6','predict_2':'predict_2_6'},inplace=True)
#result_xgb_df_7.rename(columns={'predict_0':'predict_0_7','predict_1':'predict_1_7','predict_2':'predict_2_7'},inplace=True)
#result_xgb_df_8.rename(columns={'predict_0':'predict_0_8','predict_1':'predict_1_8','predict_2':'predict_2_8'},inplace=True)
#result_xgb_ens = pd.merge(result_xgb_df,result_xgb_df_4)
#result_xgb_ens = pd.merge(result_xgb_ens,result_xgb_df_5)
#result_xgb_ens = pd.merge(result_xgb_ens,result_xgb_df_6)
#result_xgb_ens = pd.merge(result_xgb_ens,result_xgb_df_7)
#result_xgb_ens = pd.merge(result_xgb_ens,result_xgb_df_8)

def get_pred(row,pred_number):
    str_0 = 'predict_'+str(pred_number)+'_0'
    str_4 = 'predict_'+str(pred_number)+'_4'
    str_5 = 'predict_'+str(pred_number)+'_5'
    str_6 = 'predict_'+str(pred_number)+'_6'
    str_7 = 'predict_'+str(pred_number)+'_7'
    str_8 = 'predict_'+str(pred_number)+'_8'
    ans_0 = row[str_0]
    ans_4 = row[str_4]
    ans_5 = row[str_5]
    ans_6 = row[str_6]
    ans_7 = row[str_7]
    ans_8 = row[str_8]
    ans_array = np.array([ans_0,ans_4,ans_5,ans_6,ans_7,ans_8])
#    if(np.mean(ans_array) < 0.5):
#        return np.max(ans_array)
#    else:
#        return np.min(ans_array)
    return np.median(ans_array)

#result_xgb_ens['predict_0'] = result_xgb_ens.apply(lambda row: get_pred(row,0),axis=1)
#result_xgb_ens['predict_1'] = result_xgb_ens.apply(lambda row: get_pred(row,1),axis=1)
#result_xgb_ens['predict_2'] = result_xgb_ens.apply(lambda row: get_pred(row,2),axis=1)

#result_xgb_ens = (1.0 * result_xgb_df_4 + 1.0 * result_xgb_df_5 + 1.0 * result_xgb_df_6 )
#result_xgb_ens = ( 1.0 * result_xgb_df_5)
#result_xgb_ens = 0.6 * result_xgb_df + 0.4 * result_xgb_df_4

#result_xgb_df_log = np.log(result_xgb_df[['predict_0','predict_1','predict_2']])
#result_xgb_df_4_log = np.log(result_xgb_df_4[['predict_0','predict_1','predict_2']])
#result_xgb_df_5_log = np.log(result_xgb_df_5[['predict_0','predict_1','predict_2']])
#result_xgb_df_6_log = np.log(result_xgb_df_6[['predict_0','predict_1','predict_2']])
#result_xgb_df_7_log = np.log(result_xgb_df_7[['predict_0','predict_1','predict_2']])
#
#result_xgb_ens = np.exp((0.1 * result_xgb_df_log + 0.1 * result_xgb_df_4_log +
#                        0.6 * result_xgb_df_5_log + 0.1 * result_xgb_df_6_log + 0.1 * result_xgb_df_7_log))

#%%
#TODO check that this works
#result_xgb_df_9_full = result_xgb_df.copy()
#if(not is_sub_run):
#    result_xgb_df_9_full.set_index('id',inplace=True)
#    result_xgb_df_9.set_index('id',inplace=True)
#result_xgb_df_9_full['predict_0'] = result_xgb_df_9['predict_0']
#result_xgb_df_9_full['predict_1'] = result_xgb_df_9['predict_1']
#result_xgb_df_9_full['predict_2'] = result_xgb_df_9['predict_2']
#result_xgb_df_9_full.fillna(0,inplace=True)
#if(not is_sub_run):
#    result_xgb_df_9_full.reset_index('id',inplace=True)
#%%

result_xgb_ens = (2.0 * result_xgb_df + 0.0 * result_xgb_df_3 + 1.0 * result_xgb_df_4 + 1.0 * result_xgb_df_5 + 5.0 * result_xgb_df_6 +
                 0.0 * result_xgb_df_7 + 3.0 * result_xgb_df_low_high
                 + 1.0 * result_extc_df
                 + 2.0 * result_extc_df_2
#                 1.0 * result_xgb_df_8
#                 + 0.0 * result_xgb_df_9_full
                 )

#result_xgb_ens = result_xgb_ens[['id','predict_0','predict_1','predict_2']]
result_xgb_ens = result_xgb_ens[['predict_0','predict_1','predict_2']]
result_xgb_ens = norm_rows(result_xgb_ens)


if(is_sub_run):
    print('is a submission run')
else:
    result_xgb_ens['id'] = result_xgb_df['id']
    result_xgb_ens = pd.merge(result_xgb_ens,test[['id','fault_severity']],left_on = ['id'],
                           right_on = ['id'],how='left')
    result_xgb_ens['log_loss'] = result_xgb_ens.apply(lambda row: get_log_loss_row(row),axis=1)
    print('log_loss',round(result_xgb_ens['log_loss'].mean(),5))
#    print('log_loss 0',round(result_xgb_ens.loc[result_xgb_ens['fault_severity'] == 0].log_loss.sum(),1))
#    print('log_loss 1',round(result_xgb_ens.loc[result_xgb_ens['fault_severity'] == 1].log_loss.sum(),1))
#    print('log_loss 2',round(result_xgb_ens.loc[result_xgb_ens['fault_severity'] == 2].log_loss.sum(),1))

#0-5k log_loss 0.40516
#5k-10k log_loss 0.41689
#10k-15k 0.41343
#%%
if (is_sub_run):
    result_xgb_ens_2 = result_xgb_ens.copy()
    result_xgb_ens_2.reset_index('id',inplace=True)
    ans_xgb_df = pd.merge(result_xgb_ens_2,combined[['id','order_by_location','location_number','median_log_feature','combo_log_features_hash',
                                                  'severity_type','location_max','location_by_non_dummies_count','median_event_type','combo_resource_types_hash'] + vol_log_feature_cols],
                          left_on = ['id'],
                          right_on = ['id'],how='left')
#    ans_bad_xgb_df = ans_xgb_df.loc[ans_xgb_df.log_loss >= 2.5]

if(is_sub_run):
#    submission = result_xgb_df.copy()
#    submission = result_xgb_df_stacked.copy()
    submission = result_xgb_ens.copy()
    submission.reset_index('id',inplace=True)
    submission = submission[['id','predict_0','predict_1','predict_2']]
    submission.to_csv('telstra_2.csv', index=False)
    print('xgb submission created')

#%%
toc=timeit.default_timer()
print('Total Time',toc - tic0)